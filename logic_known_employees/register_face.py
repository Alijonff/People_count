import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from insightface.app import FaceAnalysis
from face_db import load_face_db, save_face_db

def init_face_app(ctx_id=0):
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

def register_new_face(source=0, db_dir="face_db"):
    print("Инициализация модели лиц...")
    # Попробуем использовать GPU (ctx_id=0), если не получится - будет warning от insightface
    app = init_face_app(ctx_id=0)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {source}")
        return

    face_db = load_face_db(db_dir)
    print(f"Загружено {len(face_db)} сотрудников из базы.")
    print("Нажмите 'SPACE' чтобы сделать снимок и сохранить лицо.")
    print("Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        faces = app.get(frame)

        # Рисуем рамки вокруг найденных лиц
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow("Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if len(faces) == 0:
                print("Лиц не найдено! Попробуйте еще раз.")
                continue
            elif len(faces) > 1:
                print("Найдено больше одного лица! Оставьте в кадре только одного человека.")
                continue
            
            # Одно лицо найдено
            face = faces[0]
            embedding = face.embedding
            
            # Запрос имени
            # Чтобы консоль была активна, можно временно закрыть окно или просто переключиться
            print("Введите Имя Сотрудника (Employee Name): ", end='', flush=True)
            # cv2.waitKey(1) # костыль чтобы пробросить события, но input блокирует
            name = input()
            if not name:
                print("Имя не введено, отмена.")
                continue

            # Генерируем ID
            new_id = str(len(face_db) + 1)
            while new_id in face_db:
                new_id = str(int(new_id) + 1)
            
            face_db[new_id] = {
                "name": name,
                "embedding": embedding
            }
            
            save_face_db(face_db, db_dir)
            print(f"Сотрудник '{name}' успешно добавлен (ID: {new_id})!")
            print("Можете добавить следующего или нажать 'q' для выхода.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="ID камеры")
    parser.add_argument("--db-dir", default="face_db", help="Папка базы данных")
    args = parser.parse_args()
    
    src = int(args.source) if args.source.isdigit() else args.source
    register_new_face(src, args.db_dir)
