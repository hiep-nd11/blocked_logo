import cv2
from ultralytics import YOLO

MODEL_PATH = "models/20260304_gg_endcart_640n.pt"
INPUT_VIDEO = "Endcard-20260303T102443Z-3-001/endcart2/1027(14).mp4"

CONF_THRES = 0.25

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

found = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    results = model(frame, conf=CONF_THRES)[0]

    if len(results.boxes) > 0:
        second = frame_index / fps

        print("========== DETECTED ==========")
        print(f"Frame index: {frame_index}")
        print(f"Time (seconds): {second:.2f}s")

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

        found = True
        break   

cap.release()

if not found:
    print("No object detected in video.")