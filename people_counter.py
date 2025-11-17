"""People counting script using YOLOv8 and supervision.

Этот скрипт открывает поток с камеры или видеофайла, отслеживает людей,
и подсчитывает время их пребывания в кадре.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv


# ----------------------------- Константы ------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4



def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Создаёт и обрабатывает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Отслеживание людей и времени их пребывания в кадре"
    )
    parser.add_argument("--source", default=1, help="Источник видео: ID камеры или путь к файлу")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Порог уверенности детектора",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Вычислительное устройство для модели",
    )
    parser.add_argument(
        "--save-csv",
        default="people_counts.csv",
        help="Путь для сохранения CSV сессий появления людей в кадре",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> str:
    """Возвращает выбранное устройство для инференса.

    Если указан режим ``auto``, используется GPU при наличии CUDA,
    иначе выполняется на CPU.
    """

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg



def prepare_video_capture(source: str | int) -> cv2.VideoCapture:
    """Открывает источник видео и возвращает объект ``VideoCapture``."""

    # сначала пытаемся через CAP_DSHOW, если source - индекс
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # fallback на дефолтный backend
            cap = cv2.VideoCapture(source)
    else:
        # если это строка (путь/rtsp), обычный вызов
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть источник видео")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Используемое разрешение видео: {int(width)}x{int(height)}")

    return cap


def ensure_integer_source(source: str) -> str | int:
    """Преобразует числовую строку в целое, иначе возвращает исходное значение."""

    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def empty_detections() -> sv.Detections:
    """Создаёт пустой объект ``Detections`` для удобства работы."""

    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
        tracker_id=np.empty((0,), dtype=int),
    )


def ensure_csv_header(csv_path: str) -> None:
    """Гарантирует наличие заголовка в CSV-файле сессий."""

    path = Path(csv_path)
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "first_seen_iso", "last_seen_iso", "duration_sec"])


def log_person_session(csv_path: str, session: PersonSession) -> None:
    """Логирует завершённую сессию пребывания человека в кадре."""

    duration = (session.last_seen - session.first_seen).total_seconds()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                session.track_id,
                session.first_seen.isoformat(),
                session.last_seen.isoformat(),
                f"{duration:.2f}",
            ]
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Точка входа в программу."""

    args = parse_arguments(argv)
    source = ensure_integer_source(args.source)
    device = resolve_device(args.device)
    model = YOLO(DEFAULT_MODEL)

    cap = prepare_video_capture(source)
    success, frame = cap.read()
    if not success:
        print("Не удалось прочитать первый кадр", file=sys.stderr)
        cap.release()
        return 1
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_color=sv.Color.from_hex("#FFFFFF"),
    )

    ensure_csv_header(args.save_csv)

    print(f"Используется устройство: {device}")

    while True:
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,
            device=device,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        if not results:
            original_frame = frame
            detections = empty_detections()
        else:
            result = results[0]
            original_frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            if len(detections):
                if detections.class_id is None:
                    mask = np.ones(len(detections), dtype=bool)
                else:
                    mask = detections.class_id == 0
                detections = detections[mask]
        annotated_frame = box_annotator.annotate(
            scene=original_frame.copy(), detections=detections
        )

        now = dt.datetime.utcnow()

        tracker_ids = np.empty((len(detections),), dtype=int)
        if detections is not None and len(detections):
            tracker_ids = (
                detections.tracker_id
                if detections.tracker_id is not None
                else np.empty((len(detections),), dtype=int)
            )

        current_ids = set()
        if detections is not None and len(detections):
            for tid in tracker_ids:
                if tid is None:
                    continue
                current_ids.add(int(tid))

        for tid in current_ids:
            session = person_sessions.get(tid)
            if session is None:
                person_sessions[tid] = PersonSession(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now,
                    active=True,
                    total_time_sec=0.0,
                )
                continue

            session.last_seen = now
            if not session.active:
                session.first_seen = now
                session.active = True

        for tid, session in person_sessions.items():
            if tid in current_ids:
                continue

            if session.active:
                delta = (now - session.last_seen).total_seconds()
                if delta >= MISSING_THRESHOLD_SEC:
                    session.total_time_sec += (
                        session.last_seen - session.first_seen
                    ).total_seconds()
                    log_person_session(args.save_csv, session)
                    session.active = False

        labels = []
        if len(detections):
            for tracker_id in tracker_ids:
                if tracker_id is None:
                    labels.append("ID ?")
                    continue

                tid = int(tracker_id)
                session = person_sessions.get(tid)
                current_duration = session.total_time_sec if session else 0.0
                if session and session.active:
                    current_duration += (now - session.first_seen).total_seconds()
                labels.append(f"ID {tid} ({current_duration:.1f}s)")

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        cv2.imshow("People Counter", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        success, next_frame = cap.read()
        if not success:
            break
        frame = next_frame

    cap.release()
    cv2.destroyAllWindows()
    return 0


INSTRUCTIONS = """
Инструкция по использованию people_counter.py
=============================================

1. Установка зависимостей::

       pip install ultralytics supervision opencv-python torch

2. Запуск со встроенной камерой (по умолчанию камера с индексом 0)::

       python people_counter.py

3. Запуск с видеофайлом::

       python people_counter.py --source path/to/video.mp4

4. Для сохранения отчёта с временными сессиями укажите путь к CSV::

       python people_counter.py --save-csv sessions.csv

5. Автоопределение устройства: если доступен GPU (torch.cuda.is_available()),
   используется режим "cuda", иначе выполняется на CPU.
"""


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print(INSTRUCTIONS)
    sys.exit(exit_code)
