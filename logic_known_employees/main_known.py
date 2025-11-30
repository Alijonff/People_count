import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from typing import Dict, List
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from face_db import load_face_db

# Константы
FACE_MATCH_THRESHOLD = 0.45  # Порог схожести лиц (cosine similarity)
IOU_THRESHOLD = 0.3        # Порог перекрытия бокса лица и человека

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)

class EmployeeTracker:
    def __init__(self, db_dir="face_db"):
        self.face_db = load_face_db(db_dir)
        self.active_sessions = {} # { employee_id: { "start_time": ts, "last_seen": ts, "total_time": float } }
        print(f"Загружено {len(self.face_db)} сотрудников.")

    def update(self, employee_id):
        now = time.time()
        if employee_id not in self.active_sessions:
            self.active_sessions[employee_id] = {
                "start_time": now,
                "last_seen": now,
                "total_time": 0.0
            }
        else:
            session = self.active_sessions[employee_id]
            # Если прошло меньше 5 секунд с последнего появления, считаем это продолжением сессии
            if now - session["last_seen"] < 5.0:
                session["total_time"] += (now - session["last_seen"])
            session["last_seen"] = now

    def get_info(self, employee_id):
        if employee_id in self.face_db:
            name = self.face_db[employee_id]["name"]
            total_time = 0.0
            if employee_id in self.active_sessions:
                total_time = self.active_sessions[employee_id]["total_time"]
            return name, total_time
        return "Unknown", 0.0

def main(source=0, device="auto"):
    # Настройка устройства
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    # Инициализация моделей
    print("Загрузка YOLO...")
    yolo_model = YOLO("yolov8n.pt")
    
    print("Загрузка InsightFace...")
    ctx_id = 0 if device == "cuda" else -1
    face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    tracker = EmployeeTracker()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Детекция людей (YOLO)
        results = yolo_model.track(frame, persist=True, classes=[0], verbose=False, device=device)
        
        # 2. Детекция лиц (InsightFace)
        faces = face_app.get(frame)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                person_bbox = [x1, y1, x2, y2]
                
                # Ищем лицо внутри бокса человека
                best_face = None
                best_iou = 0.0
                
                for face in faces:
                    face_bbox = face.bbox.astype(int)
                    iou = compute_iou(person_bbox, face_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_face = face
                
                label = f"ID: {int(track_id)}"
                color = (0, 0, 255) # Red by default (Unknown)

                if best_face is not None and best_iou > IOU_THRESHOLD:
                    # Лицо найдено, пытаемся узнать
                    embedding = best_face.embedding
                    
                    best_match_id = None
                    best_sim = 0.0
                    
                    for emp_id, data in tracker.face_db.items():
                        sim = cosine_similarity(embedding, data["embedding"])
                        if sim > best_sim:
                            best_sim = sim
                            best_match_id = emp_id
                    
                    if best_sim > FACE_MATCH_THRESHOLD and best_match_id:
                        tracker.update(best_match_id)
                        name, total_time = tracker.get_info(best_match_id)
                        label = f"{name} ({total_time:.1f}s)"
                        color = (0, 255, 0) # Green (Known)
                    else:
                        label = "Unknown"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Known Employee Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="ID камеры или путь к видео")
    parser.add_argument("--device", default="auto", help="cpu или cuda")
    args = parser.parse_args()
    
    src = int(args.source) if args.source.isdigit() else args.source
    main(src, args.device)
