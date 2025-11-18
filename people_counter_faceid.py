"""Отслеживание присутствия сотрудников с Face ID и отчётом в Excel.

Пример установки зависимостей::
    pip install ultralytics opencv-python numpy pandas insightface

Скрипт использует YOLOv8 для детекции людей, ByteTrack для трекинга и
эмбеддинги лиц из внешней базы сотрудников.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from face_db import load_face_db

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4
DEFAULT_SIMILARITY_THRESHOLD = 0.45
FACE_CHECK_INTERVAL = 5
MISSING_THRESHOLD_SEC = 3.0
SNAPSHOT_DIR = Path("snapshots")


@dataclass
class IdentityInfo:
    identity_id: str
    type: str
    track_ids: set[int] = field(default_factory=set)
    first_seen: dt.datetime = field(default_factory=dt.datetime.utcnow)
    last_seen: dt.datetime = field(default_factory=dt.datetime.utcnow)
    total_time_sec: float = 0.0
    snapshot_path: Optional[str] = None
    name: str | None = None
    employee_id: str | None = None


@dataclass
class TrackState:
    track_id: int
    identity_id: str
    last_seen: dt.datetime


class FaceEncoder:
    """Извлекает эмбеддинги лиц, используя доступные библиотеки."""

    def __init__(self, device: str):
        self.device = device
        self.haar_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_app = self._init_insightface()
        self.face_recognition = self._init_face_recognition()

    def _init_insightface(self):
        if importlib.util.find_spec("insightface") is None:
            return None
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if self.device.startswith("cuda") else -1
        app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        return app

    def _init_face_recognition(self):
        if importlib.util.find_spec("face_recognition") is None:
            return None
        import face_recognition

        return face_recognition

    def extract_face(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        crop = frame[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            return None

        if self.face_app is not None:
            faces = self.face_app.get(crop)
            if faces:
                largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                fx1, fy1, fx2, fy2 = largest.bbox.astype(int)
                fx1 = max(0, fx1)
                fy1 = max(0, fy1)
                fx2 = min(crop.shape[1] - 1, fx2)
                fy2 = min(crop.shape[0] - 1, fy2)
                return crop[fy1 : fy2 + 1, fx1 : fx2 + 1]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            return crop
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return crop[fy : fy + fh, fx : fx + fw]

    def compute_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if face_img is None or face_img.size == 0:
            return None
        if self.face_app is not None:
            faces = self.face_app.get(face_img)
            if faces:
                return faces[0].normed_embedding.astype(np.float32)
        if self.face_recognition is not None:
            rgb = face_img[:, :, ::-1]
            encodings = self.face_recognition.face_encodings(rgb)
            if encodings:
                emb = np.asarray(encodings[0], dtype=np.float32)
                return emb / np.linalg.norm(emb)
        resized = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
        emb = resized.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm != 0 else emb


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Отслеживание сотрудников с Face ID и подсчётом времени присутствия"
    )
    parser.add_argument("--source", default=0, help="Источник видео: ID камеры или путь")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Порог YOLO")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Устройство"
    )
    parser.add_argument(
        "--save-xlsx",
        default="people_report.xlsx",
        help="Путь для сохранения итогового отчёта",
    )
    parser.add_argument(
        "--face-similarity", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Порог"
    )
    parser.add_argument(
        "--face-db", default="face_db", help="Каталог с базой эмбеддингов сотрудников"
    )
    return parser.parse_args(argv)


def prepare_video_capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть источник видео")
    return cap


def ensure_snapshot_dir() -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    return SNAPSHOT_DIR


def save_snapshot(identity_id: str, frame: np.ndarray, bbox: np.ndarray) -> str:
    folder = ensure_snapshot_dir()
    ts = int(dt.datetime.utcnow().timestamp())
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)
    crop = frame[y1 : y2 + 1, x1 : x2 + 1]
    path = folder / f"identity_{identity_id}_{ts}.jpg"
    cv2.imwrite(str(path), crop)
    return str(path)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def match_employee(embedding: np.ndarray, face_db: Dict[str, dict], threshold: float) -> str | None:
    best_id: str | None = None
    best_score = -1.0
    second_score = -1.0
    for employee_id, info in face_db.items():
        emb = np.asarray(info.get("embedding", []), dtype=np.float32)
        if emb.size == 0:
            continue
        score = similarity(embedding, emb)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_id = employee_id
        elif score > second_score:
            second_score = score
    if best_score >= threshold and best_score - second_score >= 0.05:
        return best_id
    if best_score >= threshold and second_score < threshold:
        return best_id
    return None


def merge_identities(
    current_id: str, new_id: str, identities: Dict[str, IdentityInfo], track_states: Dict[int, TrackState]
) -> None:
    if current_id == new_id:
        return
    if new_id not in identities:
        identities[new_id] = IdentityInfo(identity_id=new_id, type="employee")
    target = identities[new_id]
    source = identities.get(current_id)
    if source is not None:
        target.track_ids.update(source.track_ids)
        target.total_time_sec += source.total_time_sec
        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        if target.snapshot_path is None:
            target.snapshot_path = source.snapshot_path
        identities.pop(current_id, None)
    for tid, state in list(track_states.items()):
        if state.identity_id == current_id:
            state.identity_id = new_id
            target.track_ids.add(tid)


def update_time(identity_id: str, now: dt.datetime, track_state: TrackState, identities: Dict[str, IdentityInfo]) -> None:
    identity = identities.setdefault(identity_id, IdentityInfo(identity_id=identity_id, type="unknown"))
    if identity.first_seen > now:
        identity.first_seen = now
    if track_state.last_seen:
        delta = (now - track_state.last_seen).total_seconds()
        if delta > 0:
            identity.total_time_sec += delta
    identity.last_seen = now
    identity.track_ids.add(track_state.track_id)


def process_frame(
    frame: np.ndarray,
    detections,
    face_encoder: FaceEncoder,
    face_db: Dict[str, dict],
    identities: Dict[str, IdentityInfo],
    track_states: Dict[int, TrackState],
    frame_idx: int,
    threshold: float,
    unknown_counter: int,
) -> int:
    now = dt.datetime.utcnow()
    seen_tracks: set[int] = set()
    if detections is None or len(detections) == 0:
        return unknown_counter

    for det_idx in range(len(detections)):
        if detections.class_id[det_idx] != 0:
            continue
        tracker_id = detections.tracker_id[det_idx]
        if tracker_id is None:
            continue
        tid = int(tracker_id)
        bbox = detections.xyxy[det_idx]
        seen_tracks.add(tid)
        state = track_states.get(tid)
        if state is None:
            identity_id = f"unknown_{unknown_counter}"
            unknown_counter += 1
            state = TrackState(track_id=tid, identity_id=identity_id, last_seen=now)
            track_states[tid] = state
            identities.setdefault(
                identity_id,
                IdentityInfo(identity_id=identity_id, type="unknown", snapshot_path=save_snapshot(identity_id, frame, bbox)),
            )
        update_time(state.identity_id, now, state, identities)
        state.last_seen = now

        should_check_face = frame_idx % FACE_CHECK_INTERVAL == 0 or identities[state.identity_id].type == "unknown"
        if should_check_face:
            face_img = face_encoder.extract_face(frame, bbox)
            embedding = face_encoder.compute_embedding(face_img) if face_img is not None else None
            if embedding is not None:
                employee_id = match_employee(embedding, face_db, threshold)
                if employee_id is not None:
                    merge_identities(state.identity_id, employee_id, identities, track_states)
                    identity = identities[employee_id]
                    identity.type = "employee"
                    identity.employee_id = employee_id
                    identity.name = str(face_db.get(employee_id, {}).get("name", ""))
                    if identity.snapshot_path is None:
                        identity.snapshot_path = save_snapshot(employee_id, frame, bbox)
                    state.identity_id = employee_id
    for tid, state in list(track_states.items()):
        if tid not in seen_tracks:
            if (now - state.last_seen).total_seconds() > MISSING_THRESHOLD_SEC:
                track_states.pop(tid, None)
    return unknown_counter


def build_report(identities: Dict[str, IdentityInfo]) -> pd.DataFrame:
    rows = []
    for identity in identities.values():
        row = {
            "identity_id": identity.identity_id,
            "type": identity.type,
            "employee_id": identity.employee_id or "",
            "name": identity.name or "",
            "track_ids": ",".join(str(t) for t in sorted(identity.track_ids)),
            "first_seen": identity.first_seen.isoformat(),
            "last_seen": identity.last_seen.isoformat(),
            "total_time_sec": round(identity.total_time_sec, 2),
            "snapshot_path": identity.snapshot_path or "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    source = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source
    device = resolve_device(args.device)
    print(f"Используется устройство: {device}")

    model = YOLO(DEFAULT_MODEL)
    face_db = load_face_db(args.face_db)
    face_encoder = FaceEncoder(device)

    cap = prepare_video_capture(source)
    track_states: Dict[int, TrackState] = {}
    identities: Dict[str, IdentityInfo] = {}
    frame_idx = 0
    unknown_counter = 1

    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model.track(
            frame,
            conf=args.conf,
            device=device,
            verbose=False,
            tracker="bytetrack.yaml",
            persist=True,
            classes=0,
        )
        detections = results[0].boxes if results else None
        unknown_counter = process_frame(
            frame,
            detections,
            face_encoder,
            face_db,
            identities,
            track_states,
            frame_idx,
            args.face_similarity,
            unknown_counter,
        )
        for det_idx in range(len(detections) if detections is not None else 0):
            if detections.class_id[det_idx] != 0:
                continue
            tracker_id = detections.tracker_id[det_idx]
            if tracker_id is None:
                continue
            tid = int(tracker_id)
            bbox = detections.xyxy[det_idx].astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            identity_id = track_states.get(tid, TrackState(tid, "?", dt.datetime.utcnow())).identity_id
            label = identity_id
            if identity_id in identities and identities[identity_id].name:
                label = f"{identity_id}:{identities[identity_id].name}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        cv2.imshow("People Counter FaceID", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    report = build_report(identities)
    save_path = Path(args.save_xlsx)
    report.to_excel(save_path, index=False)
    print(f"Отчёт сохранён: {save_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
