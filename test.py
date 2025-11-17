import cv2

INDEX = 1  # или 0, если хочешь проверить обе

cap = cv2.VideoCapture(INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Камера {INDEX} недоступна")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать кадр")
        break

    cv2.imshow(f"Camera {INDEX}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
