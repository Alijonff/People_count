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

# Три рабочих места — координаты временные, я их позже заменю.
SEAT_ZONES = {
    "seat_1": ((100, 100), (300, 300)),  # Левое место
    "seat_2": ((400, 100), (600, 300)),  # Центральное место
    "seat_3": ((700, 100), (900, 300)),  # Правое место
}

# Линия входа в департамент (вертикальная, слева).
# Координаты примерные, я потом подправлю.
ENTRANCE_LINE_START = (40, 0)  # x = 40, верх кадра
ENTRANCE_LINE_END = (40, 720)  # x = 40, низ кадра (подставь реальную высоту)


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

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Не удалось открыть источник для интерактивного выбора линии", file=sys.stderr)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Используемое разрешение видео: {int(width)}x{int(height)}")

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


def compute_default_line(frame_shape):
    height, width = frame_shape[:2]
    x = int(width * 0.25)       # линия на 25% от левого края
    return LineDefinition(point1=(x, 0), point2=(x, height))



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


def create_line_zone(line: LineDefinition) -> sv.LineZone:
    """Создаёт ``LineZone`` используя актуальный API ``supervision``."""

    start_point = sv.Point(*line.point1)
    end_point = sv.Point(*line.point2)
    return sv.LineZone(start=start_point, end=end_point)


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

    line_zone = create_line_zone(line)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=1.0)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_color=sv.Color.from_hex("#FFFFFF"),
    )

    track_last_side: Dict[int, int] = {}
    entrance_track_last_side: Dict[int, int] = {}
    entrance_count = 0
    total_in = 0
    total_out = 0

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

        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)

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
