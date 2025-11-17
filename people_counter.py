"""People counting script using YOLOv8 and supervision.

Этот скрипт открывает поток с камеры или видеофайла, отслеживает людей,
и подсчитывает время их пребывания в кадре.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv

try:
    import face_recognition
except Exception:
    face_recognition = None


# ----------------------------- Константы ------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4
DEFAULT_GALLERY_DIR = "gallery"
DEFAULT_PROFILE_REFRESH_HOURS = 24
PROFILE_MATCH_THRESHOLD = 0.3


@dataclass
class FaceProfile:
    """Профиль пользователя с эмбеддингом и сохранённым снимком."""

    person_id: int
    embedding: np.ndarray
    image_path: Path
    updated_at: dt.datetime
    best_face_area: float = 0.0

    def to_dict(self, base_dir: Path) -> Dict[str, str | int | list[float]]:
        return {
            "person_id": self.person_id,
            "embedding": self.embedding.tolist(),
            "image_path": str(self.image_path.relative_to(base_dir)),
            "updated_at": self.updated_at.isoformat(),
            "best_face_area": self.best_face_area,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object], base_dir: Path) -> "FaceProfile":
        return cls(
            person_id=int(data["person_id"]),
            embedding=np.array(data["embedding"], dtype=np.float32),
            image_path=base_dir / str(data["image_path"]),
            updated_at=dt.datetime.fromisoformat(str(data["updated_at"])),
            best_face_area=float(data.get("best_face_area", 0.0)),
        )


@dataclass
class PersonSession:
    """Состояние присутствия человека в кадре."""

    track_id: int
    person_id: int
    first_seen: dt.datetime
    last_seen: dt.datetime
    active: bool
    total_time_sec: float = 0.0


person_sessions: Dict[int, PersonSession] = {}
track_to_person: Dict[int, int] = {}
MISSING_THRESHOLD_SEC = 3.0


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Нормализует эмбеддинг к единичной норме."""

    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return embedding
    return embedding / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет косинусное расстояние между двумя эмбеддингами."""

    a_n = normalize_embedding(a)
    b_n = normalize_embedding(b)
    return float(1 - np.dot(a_n, b_n))


class ProfileStore:
    """Хранилище профилей с эмбеддингами и снимками на диске."""

    def __init__(self, gallery_dir: str | Path):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.date_dir = self.gallery_dir / dt.date.today().isoformat()
        self.person_dir = self.date_dir / "person"
        self.tracker_dir = self.date_dir / "tracker"
        self.person_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.date_dir / "profiles.json"
        self.profiles: Dict[int, FaceProfile] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Не удалось прочитать profiles.json, создаётся новая галерея")
            return

        for item in data:
            profile = FaceProfile.from_dict(item, self.date_dir)
            self.profiles[profile.person_id] = profile

    def _save_metadata(self) -> None:
        payload = [p.to_dict(self.date_dir) for p in self.profiles.values()]
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _next_id(self) -> int:
        if not self.profiles:
            return 1
        return max(self.profiles.keys()) + 1

    def _save_person_snapshot(self, person_id: int, face_image: np.ndarray) -> Path:
        timestamp = int(dt.datetime.utcnow().timestamp())
        file_path = self.person_dir / f"person_{person_id}_{timestamp}.jpg"
        cv2.imwrite(str(file_path), face_image)
        return file_path

    def _save_tracker_snapshot(self, person_id: int, face_image: np.ndarray) -> Path:
        timestamp = int(dt.datetime.utcnow().timestamp())
        file_path = self.tracker_dir / f"person_{person_id}_{timestamp}.jpg"
        cv2.imwrite(str(file_path), face_image)
        return file_path

    def match_profile(self, embedding: np.ndarray) -> tuple[Optional[FaceProfile], Optional[float]]:
        best_match: Optional[FaceProfile] = None
        best_distance: Optional[float] = None
        for profile in self.profiles.values():
            distance = cosine_distance(profile.embedding, embedding)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = profile
        if best_distance is not None and best_distance <= PROFILE_MATCH_THRESHOLD:
            return best_match, best_distance
        return None, None

    def create_profile(self, embedding: np.ndarray, face_image: np.ndarray) -> int:
        person_id = self._next_id()
        normalized_embedding = normalize_embedding(embedding)
        face_area = float(face_image.shape[0] * face_image.shape[1])
        self._save_tracker_snapshot(person_id, face_image)
        snapshot_path = self._save_person_snapshot(person_id, face_image)
        profile = FaceProfile(
            person_id=person_id,
            embedding=normalized_embedding,
            image_path=snapshot_path,
            updated_at=dt.datetime.utcnow(),
            best_face_area=face_area,
        )
        self.profiles[person_id] = profile
        self._save_metadata()
        print(f"Создан новый профиль: person_id={person_id}, снимок={snapshot_path}")
        return person_id

    def touch_profile(
        self,
        person_id: int,
        embedding: Optional[np.ndarray] = None,
        face_image: Optional[np.ndarray] = None,
    ) -> None:
        profile = self.profiles.get(person_id)
        if profile is None:
            return
        updated = False
        if embedding is not None:
            profile.embedding = normalize_embedding(embedding)
            updated = True
        if face_image is not None:
            self._save_tracker_snapshot(person_id, face_image)
            face_area = float(face_image.shape[0] * face_image.shape[1])
            if face_area > profile.best_face_area:
                profile.image_path = self._save_person_snapshot(person_id, face_image)
                profile.best_face_area = face_area
                updated = True
            else:
                updated = True

        if updated:
            profile.updated_at = dt.datetime.utcnow()
            self._save_metadata()
            print(f"Профиль {person_id} обновлён")

    def cleanup_expired(self, ttl_hours: float) -> None:
        now = dt.datetime.utcnow()
        removed = []
        for pid, profile in list(self.profiles.items()):
            if now - profile.updated_at > dt.timedelta(hours=ttl_hours):
                removed.append(pid)
                try:
                    if profile.image_path.exists():
                        profile.image_path.unlink()
                    for tracker_snapshot in self.tracker_dir.glob(
                        f"person_{pid}_*.jpg"
                    ):
                        tracker_snapshot.unlink()
                except OSError:
                    pass
                del self.profiles[pid]
        if removed:
            self._save_metadata()
            print(f"Очищены устаревшие профили: {removed}")


def extract_face_roi(frame: np.ndarray, bbox: np.ndarray, face_detector: cv2.CascadeClassifier) -> Optional[np.ndarray]:
    """Находит лицо внутри прямоугольника и возвращает ROI."""

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if face_detector.empty():
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return crop

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return crop[y : y + fh, x : x + fw]


def compute_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Возвращает эмбеддинг лица через доступную библиотеку либо простой дескриптор."""

    if face_recognition is not None:
        rgb = face_roi[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            return normalize_embedding(np.array(encodings[0], dtype=np.float32))

    resized = cv2.resize(face_roi, (32, 32), interpolation=cv2.INTER_AREA)
    embedding = resized.flatten().astype(np.float32)
    return normalize_embedding(embedding)



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
    parser.add_argument(
        "--gallery-dir",
        default=DEFAULT_GALLERY_DIR,
        help="Каталог для сохранения снимков лиц и метаданных",
    )
    parser.add_argument(
        "--profile-refresh-hours",
        type=float,
        default=DEFAULT_PROFILE_REFRESH_HOURS,
        help="Через сколько часов обновлять/сбрасывать устаревшие эмбеддинги",
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
        writer.writerow(
            [
                "track_id",
                "person_id",
                "first_seen_iso",
                "last_seen_iso",
                "duration_sec",
                "active",
            ]
        )


def log_person_session(csv_path: str, session: PersonSession) -> None:
    """Логирует завершённую сессию пребывания человека в кадре."""

    duration = (session.last_seen - session.first_seen).total_seconds()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                session.track_id,
                session.person_id,
                session.first_seen.isoformat(),
                session.last_seen.isoformat(),
                f"{duration:.2f}",
                session.active,
            ]
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Точка входа в программу."""

    args = parse_arguments(argv)
    source = ensure_integer_source(args.source)
    device = resolve_device(args.device)
    model = YOLO(DEFAULT_MODEL)
    profile_store = ProfileStore(args.gallery_dir)
    profile_store.cleanup_expired(args.profile_refresh_hours)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    last_profile_cleanup = dt.datetime.utcnow()

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

        if now - last_profile_cleanup >= dt.timedelta(hours=args.profile_refresh_hours):
            profile_store.cleanup_expired(args.profile_refresh_hours)
            last_profile_cleanup = now

        tracker_ids = np.empty((len(detections),), dtype=int)
        if detections is not None and len(detections):
            tracker_ids = (
                detections.tracker_id
                if detections.tracker_id is not None
                else np.empty((len(detections),), dtype=int)
            )

        person_assignments: Dict[int, int] = {}
        if detections is not None and len(detections):
            for det_idx, tracker_id in enumerate(tracker_ids):
                if tracker_id is None:
                    continue
                tid = int(tracker_id)
                bbox = detections.xyxy[det_idx]
                face_roi = extract_face_roi(original_frame, bbox, face_detector)
                if face_roi is None:
                    continue
                embedding = compute_embedding(face_roi)
                if embedding is None:
                    continue

                person_id = track_to_person.get(tid)
                if person_id is None:
                    match, distance = profile_store.match_profile(embedding)
                    if match is not None:
                        person_id = match.person_id
                        profile_store.touch_profile(person_id, embedding, face_roi)
                        print(
                            f"Трек {tid} сопоставлен с person_id={person_id} (dist={distance:.3f})"
                        )
                    else:
                        person_id = profile_store.create_profile(embedding, face_roi)
                else:
                    profile_store.touch_profile(person_id, embedding)

                if person_id is not None:
                    track_to_person[tid] = person_id
                    person_assignments[tid] = person_id

        current_ids = set()
        if detections is not None and len(detections):
            for tid in tracker_ids:
                if tid is None:
                    continue
                current_ids.add(int(tid))

        for tid in current_ids:
            person_id = person_assignments.get(tid, track_to_person.get(tid, -1))
            session = person_sessions.get(tid)
            if session is None:
                person_sessions[tid] = PersonSession(
                    track_id=tid,
                    person_id=person_id,
                    first_seen=now,
                    last_seen=now,
                    active=True,
                    total_time_sec=0.0,
                )
                continue

            session.person_id = person_id
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
                    session.active = False
                    log_person_session(args.save_csv, session)

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
                pid = session.person_id if session else track_to_person.get(tid, -1)
                pid_label = f"PID {pid}" if pid is not None and pid != -1 else "PID ?"
                state_label = "active" if session and session.active else "inactive"
                labels.append(
                    f"{pid_label} | TID {tid} | {state_label} | {current_duration:.1f}s"
                )

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

5. Параметр ``--gallery-dir`` задаёт папку с эталонными снимками и profiles.json,
   а ``--profile-refresh-hours`` определяет период очистки устаревших эмбеддингов.

6. Автоопределение устройства: если доступен GPU (torch.cuda.is_available()),
   используется режим "cuda", иначе выполняется на CPU.
"""


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print(INSTRUCTIONS)
    sys.exit(exit_code)
