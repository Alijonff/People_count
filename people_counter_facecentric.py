"""
Требуемые зависимости:
  pip install ultralytics opencv-python insightface numpy pandas
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO


BBox = Tuple[float, float, float, float]
TrackState = Dict[str, object]
IdentityStore = Dict[str, Dict[str, object]]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def open_video_source(source_arg: str) -> cv2.VideoCapture:
    source: int | str
    if isinstance(source_arg, str) and source_arg.isdigit():
        source = int(source_arg)
    else:
        source = source_arg
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source_arg}")
    return cap


def init_face_analysis(device: str) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if device.startswith("cuda") else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.ndim == 1:
        vec_a = vec_a[np.newaxis, :]
    if vec_b.ndim == 1:
        vec_b = vec_b[np.newaxis, :]
    a_norm = vec_a / (np.linalg.norm(vec_a, axis=1, keepdims=True) + 1e-8)
    b_norm = vec_b / (np.linalg.norm(vec_b, axis=1, keepdims=True) + 1e-8)
    sim = np.sum(a_norm * b_norm, axis=1)
    return float(sim.max())


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def update_identity(
    face_embedding: np.ndarray,
    known_identities: IdentityStore,
    threshold: float = 0.35,
) -> Tuple[str, bool]:
    best_identity = None
    best_score = -1.0
    for identity_id, data in known_identities.items():
        ref_embedding = data["embedding"]
        score = cosine_similarity(face_embedding, ref_embedding)
        if score > best_score and score >= threshold:
            best_score = score
            best_identity = identity_id
    if best_identity is not None:
        return best_identity, False
    new_id = f"person_{len(known_identities) + 1}"
    known_identities[new_id] = {
        "embedding": face_embedding,
        "created_ts": datetime.utcnow(),
        "snapshot_path": None,
        "observations": 1,
    }
    return new_id, True


def save_snapshot(
    frame: np.ndarray,
    bbox: BBox,
    snapshots_dir: str,
    identity_id: str,
    prefix: str = "face",
) -> str:
    os.makedirs(snapshots_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1i, y1i, x2i, y2i = (
        max(0, int(x1)),
        max(0, int(y1)),
        min(w - 1, int(x2)),
        min(h - 1, int(y2)),
    )
    crop = frame[y1i:y2i, x1i:x2i]
    filename = f"{identity_id}_{prefix}_{int(time.time())}.jpg"
    path = os.path.join(snapshots_dir, filename)
    if crop.size > 0:
        cv2.imwrite(path, crop)
    return path


def find_face_for_bbox(faces, person_bbox: BBox, iou_threshold: float = 0.1):
    best_face = None
    best_iou = 0.0
    for face in faces:
        bbox = face.bbox
        face_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        iou = compute_iou(person_bbox, face_bbox)
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_face = face
    return best_face


def update_track_time(track: TrackState, now_ts: float) -> None:
    last_ts = track.get("last_update_ts", track["first_seen_ts"])
    delta = max(0.0, now_ts - float(last_ts))
    track["total_time_sec"] = float(track.get("total_time_sec", 0.0)) + delta
    track["last_seen_ts"] = now_ts
    track["last_update_ts"] = now_ts


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    face_app: FaceAnalysis,
    tracks: Dict[int, TrackState],
    known_identities: IdentityStore,
    snapshots_dir: str,
    device: str,
    face_threshold: float = 0.35,
    iou_threshold: float = 0.1,
) -> np.ndarray:
    now_ts = time.time()
    faces = face_app.get(frame)
    results = model.track(
        source=frame,
        device=device,
        conf=0.25,
        verbose=False,
        tracker="bytetrack.yaml",
        persist=True,
    )
    annotated = frame.copy()
    if not results or results[0].boxes is None:
        return annotated

    boxes = results[0].boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i]) if boxes.cls is not None else -1
        if cls_id != 0:
            continue
        if boxes.id is None:
            continue
        track_id = int(boxes.id[i])
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        bbox = (x1, y1, x2, y2)
        track = tracks.get(track_id)
        if track is None:
            track = {
                "identity_id": None,
                "first_seen_ts": now_ts,
                "last_seen_ts": now_ts,
                "total_time_sec": 0.0,
                "last_update_ts": now_ts,
                "bbox": bbox,
            }
            tracks[track_id] = track
        else:
            track["bbox"] = bbox
        update_track_time(track, now_ts)

        matched_face = find_face_for_bbox(faces, bbox, iou_threshold=iou_threshold)
        if matched_face is not None:
            embedding = matched_face.embedding
            identity_id = track.get("identity_id")
            if identity_id is None:
                identity_id, is_new = update_identity(embedding, known_identities, threshold=face_threshold)
                track["identity_id"] = identity_id
                if is_new:
                    snapshot_path = save_snapshot(frame, matched_face.bbox, snapshots_dir, identity_id, prefix="face")
                    known_identities[identity_id]["snapshot_path"] = snapshot_path
            else:
                known_identity = known_identities.get(identity_id)
                if known_identity is not None:
                    known_identity["observations"] = int(known_identity.get("observations", 0)) + 1

        identity_to_draw = track.get("identity_id") or f"track_{track_id}"
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            identity_to_draw,
            (int(x1), max(0, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated


def save_report_to_excel(
    tracks: Dict[int, TrackState],
    known_identities: IdentityStore,
    output_path: str,
) -> None:
    aggregated: Dict[str, Dict[str, object]] = {}
    for track in tracks.values():
        identity_id = track.get("identity_id") or "unknown"
        entry = aggregated.get(identity_id)
        first_seen = datetime.fromtimestamp(float(track["first_seen_ts"]))
        last_seen = datetime.fromtimestamp(float(track["last_seen_ts"]))
        total_time = float(track.get("total_time_sec", 0.0))
        if entry is None:
            aggregated[identity_id] = {
                "identity_id": identity_id,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "total_time_sec": total_time,
                "snapshot_path": None,
                "type": "known" if identity_id != "unknown" else "unknown",
            }
        else:
            entry["first_seen"] = min(entry["first_seen"], first_seen)
            entry["last_seen"] = max(entry["last_seen"], last_seen)
            entry["total_time_sec"] += total_time
    for identity_id, info in known_identities.items():
        entry = aggregated.get(identity_id)
        if entry:
            entry["snapshot_path"] = info.get("snapshot_path")
    df = pd.DataFrame(aggregated.values())
    df.sort_values(by="first_seen", inplace=True)
    df.to_excel(output_path, index=False)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face-centric people counter")
    parser.add_argument("--source", default="1", help="Источник видео (камера ID или путь)")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Устройство для инференса",
    )
    parser.add_argument(
        "--save-xlsx",
        default="facecentric_report.xlsx",
        help="Путь для сохранения Excel отчета",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="snapshots",
        help="Директория для сохранения snapshot'ов",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.35,
        help="Порог cosine similarity для совпадения лица",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="Минимальный IoU для сопоставления лица и бокса",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model = YOLO("yolov8n.pt")
    face_app = init_face_analysis(device)
    cap = open_video_source(args.source)

    tracks: Dict[int, TrackState] = {}
    known_identities: IdentityStore = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = process_frame(
                frame,
                model,
                face_app,
                tracks,
                known_identities,
                args.snapshots_dir,
                device,
                face_threshold=args.face_threshold,
                iou_threshold=args.iou_threshold,
            )
            cv2.imshow("Face-centric counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_report_to_excel(tracks, known_identities, args.save_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
