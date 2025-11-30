import cv2
import numpy as np
import argparse
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Константы
FACE_MATCH_THRESHOLD = 0.30  # Порог схожести для узнавания того же человека
IOU_THRESHOLD = 0.3

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)

class DynamicTracker:
    def __init__(self, snapshots_dir="snapshots"):
        self.identities = {} # { id: { "embedding": emb, "first_seen": ts, "last_seen": ts, "total_time": float, "name": str } }
        self.track_to_identity = {} # { track_id: person_id } - связь между YOLO track ID и Person ID
        self.next_id = 1
        self.snapshots_dir = Path(snapshots_dir)

    def save_snapshot(self, frame, bbox, person_name):
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.snapshots_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            filename = f"{person_name}_{int(time.time())}.jpg"
            cv2.imwrite(str(day_dir / filename), face_img)

    def update_with_face(self, embedding, frame, bbox, track_id):
        # Попытка найти существующего
        best_id = None
        best_sim = 0.0
        
        for pid, data in self.identities.items():
            sim = cosine_similarity(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        
        now = time.time()
        
        # Выводим в консоль для отладки
        if best_id is not None:
            print(f"Track {track_id}: Лучшее совпадение = Person_{best_id}, similarity = {best_sim:.3f} (порог {FACE_MATCH_THRESHOLD})")
        
        if best_sim > FACE_MATCH_THRESHOLD and best_id is not None:
            # Обновляем существующего
            identity = self.identities[best_id]
            
            # Усредняем эмбеддинг для улучшения качества распознавания
            identity["embedding"] = (identity["embedding"] * 0.7 + embedding * 0.3)
            identity["embedding"] = identity["embedding"] / np.linalg.norm(identity["embedding"])
            
            if now - identity["last_seen"] < 5.0:
                identity["total_time"] += (now - identity["last_seen"])
            identity["last_seen"] = now
            
            # Запоминаем связь track_id -> person_id
            self.track_to_identity[track_id] = best_id
            return best_id, identity["name"], identity["total_time"]
        else:
            # Создаем нового
            new_id = self.next_id
            self.next_id += 1
            name = f"Person_{new_id}"
            print(f"Track {track_id}: Создан новый {name} (similarity слишком низкая: {best_sim:.3f})")
            self.identities[new_id] = {
                "embedding": embedding,
                "first_seen": now,
                "last_seen": now,
                "total_time": 0.0,
                "name": name
            }
            # Сохраняем фото нового человека
            self.save_snapshot(frame, bbox, name)
            
            # Запоминаем связь track_id -> person_id
            self.track_to_identity[track_id] = new_id
            return new_id, name, 0.0

    def update_without_face(self, track_id):
        # Обновляем время для известного track_id, даже если лицо не видно
        if track_id in self.track_to_identity:
            person_id = self.track_to_identity[track_id]
            if person_id in self.identities:
                identity = self.identities[person_id]
                now = time.time()
                
                if now - identity["last_seen"] < 5.0:
                    identity["total_time"] += (now - identity["last_seen"])
                identity["last_seen"] = now
                return person_id, identity["name"], identity["total_time"]
        return None, None, 0.0

    def cleanup_old_tracks(self, current_track_ids):
        # Удаляем track_id, которых больше нет в кадре
        old_tracks = [tid for tid in self.track_to_identity.keys() if tid not in current_track_ids]
        for tid in old_tracks:
            del self.track_to_identity[tid]

    def save_report(self, output_path="dynamic_report.xlsx"):
        rows = []
        for pid, data in self.identities.items():
            rows.append({
                "Person ID": pid,
                "Name": data["name"],
                "First Seen": datetime.fromtimestamp(data["first_seen"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Last Seen": datetime.fromtimestamp(data["last_seen"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Total Time (sec)": round(data["total_time"], 2)
            })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(output_path, index=False)
            print(f"Отчет сохранен в {output_path}")
        else:
            print("Нет данных для отчета.")

def main(source=0, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    yolo_model = YOLO("yolov8n.pt")
    
    ctx_id = 0 if device == "cuda" else -1
    face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    tracker = DynamicTracker()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False, device=device)
            faces = face_app.get(frame)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                # Собираем текущие track_id для cleanup
                current_track_ids = set([int(tid) for tid in track_ids])

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    person_bbox = [x1, y1, x2, y2]
                    track_id_int = int(track_id)
                    
                    # Ищем лицо внутри бокса человека
                    best_face = None
                    best_iou = 0.0
                    for face in faces:
                        face_bbox = face.bbox.astype(int)
                        iou = compute_iou(person_bbox, face_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_face = face
                    
                    # Сначала проверяем, знаем ли мы уже этот track_id
                    if track_id_int in tracker.track_to_identity:
                        # Продолжаем отслеживать известного человека
                        pid, name, total_time = tracker.update_without_face(track_id_int)
                        if pid is not None:
                            label = f"{name} ({total_time:.1f}s)"
                            color = (0, 255, 0)  # Green
                        else:
                            label = f"Track ID: {track_id_int}"
                            color = (255, 0, 0)  # Blue
                    elif best_face is not None and best_iou > IOU_THRESHOLD:
                        # Новый человек с видимым лицом - идентифицируем
                        pid, name, total_time = tracker.update_with_face(best_face.embedding, frame, person_bbox, track_id_int)
                        label = f"{name} ({total_time:.1f}s)"
                        color = (0, 255, 0)  # Green
                    else:
                        # Новый человек без видимого лица
                        label = f"Track ID: {track_id_int}"
                        color = (255, 0, 0)  # Blue
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Очищаем старые track_id
                tracker.cleanup_old_tracks(current_track_ids)

            cv2.imshow("Dynamic Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.save_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="ID камеры")
    parser.add_argument("--device", default="auto", help="cpu или cuda")
    args = parser.parse_args()
    
    src = int(args.source) if args.source.isdigit() else args.source
    main(src, args.device)
