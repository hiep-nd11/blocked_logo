import torch
import json
import re
import numpy as np
import cv2
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import supervision as sv

MODEL_ID     = "Qwen/Qwen3-VL-4B-Instruct"
TEMPERATURE  = 0.1
INPUT_VIDEO  = ""
OUTPUT_VIDEO = ""
FRAME_SKIP   = 20                          

PROMPT = (
    "Look at this image and find the most prominent app or brand logo visible. "
    "A logo is a visual symbol or icon that represents an app or brand. "
    "Logos at the bottom area are usually the advertised app's main icon or brand symbol. "
    "Do NOT detect buttons, text labels, or UI elements such as 'Try Now', 'Open in App', "
    "'Download', 'Sign In', banners, or any plain text even if it mentions a brand name. "
    "Only detect actual graphical logo symbols or icons. "
    "Output a JSON list with exactly one object containing: "
    "\"box_2d\": [x1, y1, x2, y2] (bounding box coordinates), "
    "\"label\": the name of the app or brand. "
    "If no actual logo symbol is visible, return an empty list: []. "
    "Do not include any explanation, only output the JSON list."
)

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print("Model loaded.")

def generate_response(image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=TEMPERATURE,
        do_sample=True if TEMPERATURE > 0 else False
    )
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]
    return generated_text


def parse_detections(response_text, width, height):
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not data:
        return None

    xyxy, class_name = [], []
    for item in data:
        if "box_2d" in item and "label" in item:
            x1, y1, x2, y2 = item["box_2d"]

            x1 = int(x1 / 1000 * width)
            y1 = int(y1 / 1000 * height)
            x2 = int(x2 / 1000 * width)
            y2 = int(y2 / 1000 * height)

            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            xyxy.append([x1, y1, x2, y2])
            class_name.append(item["label"])

    if not xyxy:
        return None

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.arange(len(xyxy)),
        data={"class_name": np.array(class_name)}
    )
    return detections


def annotate_frame(frame_bgr, detections):
    h, w = frame_bgr.shape[:2]
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        smart_position=True,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.CENTER,
    )
    annotated = frame_bgr.copy()
    for annotator in (box_annotator, label_annotator):
        annotated = annotator.annotate(scene=annotated, detections=detections)
    return annotated


def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Cannot open video {INPUT_VIDEO}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print(f"Video: {width}x{height} @ {fps:.1f}fps | {total} frames")
    print(f"Processing every {FRAME_SKIP} frames...")

    frame_idx = 0
    last_detections = None  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target_height = int(1024 * height / width)
            resized = pil_image.resize((1024, target_height), Image.Resampling.LANCZOS)

            response_text = generate_response(resized, PROMPT)
            print(f"[Frame {frame_idx}/{total}] Response: {response_text[:100]}")

            last_detections = parse_detections(response_text, width, height)

        if last_detections is not None:
            annotated = annotate_frame(frame, last_detections)
        else:
            annotated = frame

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()