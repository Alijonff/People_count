"""People counting script using YOLOv8 and supervision.

Этот скрипт открывает поток с камеры или видеофайла, отслеживает людей,
подсчитывает входы через линию у входа и отображает занятость рабочих мест.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv


# ----------------------------- Константы ------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4

SEAT_ZONES = {
    "seat_1": ((100, 150), (450, 520)),
    "seat_2": ((520, 150), (900, 520)),
    "seat_3": ((940, 150), (1320, 520)),
}

# Линия входа в департамент (вертикальная, слева).
ENTRANCE_LINE_START = (40, 0)
ENTRANCE_LINE_END = (40, 1080)


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Создаёт и обрабатывает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Подсчёт входов и занятости мест")
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


def calculate_point_side_for_line(p1, p2, point):
    """Определяет, по какую сторону от направленной линии (p1 -> p2) находится точка."""

    x1, y1 = p1
    x2, y2 = p2
    px, py = point

    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def empty_detections() -> sv.Detections:
    """Создаёт пустой объект ``Detections`` для удобства работы."""

    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
        tracker_id=np.empty((0,), dtype=int),
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

    entrance_track_last_side: Dict[int, int] = {}
    entrance_count = 0

    seat_occupied = {name: False for name in SEAT_ZONES.keys()}

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
            annotated_frame = frame.copy()
            detections = empty_detections()
            tracker_ids = np.empty((len(detections),), dtype=int)
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

            tracker_ids = (
                detections.tracker_id
                if detections.tracker_id is not None
                else np.empty((len(detections),), dtype=int)
            )

            labels = [f"ID {int(tid)}" for tid in tracker_ids] if len(detections) else []
            annotated_frame = box_annotator.annotate(
                scene=original_frame.copy(), detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

        # -----------------------------------------
        # Подсчёт входов через линию ENTRANCE_LINE
        # -----------------------------------------
        p1 = ENTRANCE_LINE_START
        p2 = ENTRANCE_LINE_END

        for bbox, tracker_id in zip(detections.xyxy, tracker_ids):
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            current_side = calculate_point_side_for_line(p1, p2, (cx, cy))
            tid = int(tracker_id)

            last_side = entrance_track_last_side.get(tid)

            if last_side is None:
                entrance_track_last_side[tid] = current_side
                continue

            if current_side == 0 or current_side == last_side:
                continue

            if last_side < current_side:
                entrance_count += 1

            entrance_track_last_side[tid] = current_side

        # Сбрасываем занятость на каждый кадр
        seat_occupied = {name: False for name in SEAT_ZONES.keys()}

        # Для каждого человека вычисляем центр
        for bbox, tracker_id in zip(detections.xyxy, tracker_ids):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Проверяем попадание центра в каждый seat
            for seat_name, ((sx1, sy1), (sx2, sy2)) in SEAT_ZONES.items():
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    seat_occupied[seat_name] = True

        for seat_name, ((sx1, sy1), (sx2, sy2)) in SEAT_ZONES.items():
            color = (0, 255, 0) if seat_occupied[seat_name] else (0, 0, 255)
            cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), color, 2)

            status_text = f"{seat_name}: {'occupied' if seat_occupied[seat_name] else 'free'}"

            cv2.putText(
                annotated_frame,
                status_text,
                (sx1, sy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Рисуем зелёную линию входа
        cv2.line(
            annotated_frame,
            ENTRANCE_LINE_START,
            ENTRANCE_LINE_END,
            (0, 255, 0),
            2,
        )

        # Пишем счётчик входов
        cv2.putText(
            annotated_frame,
            f"ENTRANCES: {entrance_count}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
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

4. Автоопределение устройства: если доступен GPU (torch.cuda.is_available()),
   используется режим "cuda", иначе выполняется на CPU.
"""


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print(INSTRUCTIONS)
    sys.exit(exit_code)
