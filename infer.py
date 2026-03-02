import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/hiepnd72/Documents/work/blocked/best.pt"  
INPUT_VIDEO = "/home/hiepnd72/Documents/work/blocked/logo3232/logo/Bản sao của App10_ Mincraft4_170924_9.16.mp4"
OUTPUT_VIDEO = "output.mp4"

CONF_THRES = 0.25


model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(INPUT_VIDEO)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"Tọa độ: ({x1}, {y1}, {x2}, {y2})")
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    out.write(frame)


cap.release()
out.release()

print("Done! Saved to:", OUTPUT_VIDEO)