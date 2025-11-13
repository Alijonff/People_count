"""People counting script using YOLOv8 and supervision.

Этот скрипт открывает поток с камеры или видеофайла, отслеживает людей и
подсчитывает пересечения виртуальной линии. Результаты визуализируются и
при необходимости сохраняются в CSV-лог.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv


# ----------------------------- Константы ------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4
DEFAULT_CSV_PATH = "people_counts.csv"
IN_LABEL = "IN"
OUT_LABEL = "OUT"


@dataclass
class LineDefinition:
    """Структура для хранения координат линии подсчёта."""

    point1: Tuple[int, int]
    point2: Tuple[int, int]

    @property
    def vector(self) -> np.ndarray:
        """Возвращает вектор линии в виде numpy массива."""

        return np.array([self.point2[0] - self.point1[0], self.point2[1] - self.point1[1]])


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Создаёт и обрабатывает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Подсчёт людей по пересечению линии")
    parser.add_argument("--source", default=0, help="Источник видео: ID камеры или путь к файлу")
    parser.add_argument(
        "--line",
        nargs=4,
        metavar=("x1", "y1", "x2", "y2"),
        type=int,
        help="Координаты линии подсчёта (в пикселях)",
    )
    parser.add_argument(
        "--interactive-line",
        action="store_true",
        help="Настроить линию мышкой перед запуском отслеживания",
    )
    parser.add_argument(
        "--save-csv",
        default=DEFAULT_CSV_PATH,
        help="Путь к CSV-файлу с логом пересечений",
    )
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


def ensure_csv_header(csv_path: str) -> None:
    """Создаёт CSV-файл и записывает заголовок, если файл отсутствует."""

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_iso", "track_id", "direction", "total_in", "total_out"])


def select_line_interactively(source: str | int) -> Optional[LineDefinition]:
    """Позволяет пользователю задать линию кликами мыши.

    Возвращает ``LineDefinition`` или ``None``, если выбрать линию не удалось.
    """

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Не удалось открыть источник для интерактивного выбора линии", file=sys.stderr)
        return None

    line_points: list[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _: int, __: int) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            line_points.append((x, y))

    cv2.namedWindow("Выбор линии")
    cv2.setMouseCallback("Выбор линии", on_mouse)

    print("Кликните две точки для задания линии. Нажмите 'q' для выхода.")
    while True:
        success, frame = cap.read()
        if not success:
            break

        for point in line_points:
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
        if len(line_points) == 2:
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)

        cv2.imshow("Выбор линии", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            line_points = []
            break
        if len(line_points) >= 2:
            break

    cap.release()
    cv2.destroyWindow("Выбор линии")

    if len(line_points) == 2:
        return LineDefinition(line_points[0], line_points[1])
    return None


def compute_default_line(frame_shape: Tuple[int, int, int]) -> LineDefinition:
    """Создаёт вертикальную линию по центру кадра."""

    height, width = frame_shape[:2]
    x = width // 2
    return LineDefinition(point1=(x, 0), point2=(x, height))


def prepare_video_capture(source: str | int) -> cv2.VideoCapture:
    """Открывает источник видео и возвращает объект ``VideoCapture``."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть источник видео")
    return cap


def ensure_integer_source(source: str) -> str | int:
    """Преобразует числовую строку в целое, иначе возвращает исходное значение."""

    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def calculate_point_side(line: LineDefinition, point: Tuple[float, float]) -> int:
    """Определяет положение точки относительно линии.

    Возвращает -1, 0 или 1 в зависимости от знака псевдовекторного произведения.
    -1 означает, что точка находится справа от линии (если смотреть от ``point1`` к ``point2``),
    1 — слева, 0 — на линии.
    """

    x1, y1 = line.point1
    x2, y2 = line.point2
    px, py = point
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def log_crossing(
    csv_path: str,
    track_id: int,
    direction: str,
    total_in: int,
    total_out: int,
) -> None:
    """Записывает событие пересечения в CSV-файл."""

    timestamp = dt.datetime.utcnow().isoformat()
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, track_id, direction, total_in, total_out])


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

    ensure_csv_header(args.save_csv)

    if args.interactive_line:
        line = select_line_interactively(source)
        if line is None:
            print("Линия не выбрана, используется значение по умолчанию.")
    else:
        line = None

    model = YOLO(DEFAULT_MODEL)

    cap = prepare_video_capture(source)
    success, frame = cap.read()
    if not success:
        print("Не удалось прочитать первый кадр", file=sys.stderr)
        cap.release()
        return 1

    if line is None:
        if args.line:
            line = LineDefinition(point1=(args.line[0], args.line[1]), point2=(args.line[2], args.line[3]))
        else:
            line = compute_default_line(frame.shape)

    # Используем позиционные аргументы, чтобы избежать несовместимости
    # между версиями supervision, где названия параметров могли отличаться.
    line_zone = sv.LineZone(line.point1, line.point2)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1.0)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    track_last_side: Dict[int, int] = {}
    total_in = 0
    total_out = 0

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
                scene=original_frame.copy(), detections=detections, labels=labels
            )

            for bbox, tracker_id in zip(detections.xyxy, tracker_ids):
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                current_side = calculate_point_side(line, center)
                last_side = track_last_side.get(int(tracker_id))
                if last_side is None:
                    track_last_side[int(tracker_id)] = current_side
                    continue
                if current_side == 0:
                    continue
                if last_side == 0:
                    track_last_side[int(tracker_id)] = current_side
                    continue
                if current_side == last_side:
                    continue

                # Принимаем направление "IN" как переход из отрицательной полуплоскости
                # в положительную относительно ориентации линии (point1 -> point2).
                if last_side < current_side:
                    direction = IN_LABEL
                    total_in += 1
                else:
                    direction = OUT_LABEL
                    total_out += 1

                track_last_side[int(tracker_id)] = current_side
                log_crossing(args.save_csv, int(tracker_id), direction, total_in, total_out)

        line_zone.in_count = total_in
        line_zone.out_count = total_out
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)

        cv2.putText(
            annotated_frame,
            f"IN: {total_in}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"OUT: {total_out}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
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

3. Задание собственной линии через аргументы::

       python people_counter.py --line 100 50 500 400

4. Интерактивный режим выбора линии::

       python people_counter.py --interactive-line

5. Запуск с видеофайлом::

       python people_counter.py --source path/to/video.mp4

6. CSV-лог с пересечениями создаётся в файле people_counts.csv (или по пути из --save-csv).

7. Автоопределение устройства: если доступен GPU (torch.cuda.is_available()),
   используется режим "cuda", иначе выполняется на CPU.
"""


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print(INSTRUCTIONS)
    sys.exit(exit_code)
