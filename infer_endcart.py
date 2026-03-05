import cv2
from ultralytics import YOLO

MODEL_PATH = "models/20260304_gg_endcart_640n.pt"
INPUT_VIDEO = "Endcard-20260303T102443Z-3-001/endcart2/1027(14).mp4"

CONF_THRES = 0.55

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
half_frame = total_frames // 2

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
        in_second_half = frame_index > half_frame

        if in_second_half:
            print("========== DETECTED ==========")
            print(f"Frame index: {frame_index} / {total_frames}")
            print(f"Time (seconds): {second:.2f}s")
            print(f"Half point: frame {half_frame} ({half_frame / fps:.2f}s)")

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                print(f"Box: [{x1}, {y1}, {x2}, {y2}]  label: {label}")

            found = True
            break

cap.release()

if not found:
    print("No object detected in second half of video.")