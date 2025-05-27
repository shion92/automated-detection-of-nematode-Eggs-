import os
import json
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# === Config ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_DIR = "dataset/test/images"
OUTPUT_IMG_DIR = "Processed_Image/yolo"
OUTPUT_JSON_DIR = "Processed_Image/yolo"
CONFIDENCE_THRESHOLD = 0.5

# === Create output folders ===
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)  
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# === Load trained model ===
model = YOLO(MODEL_PATH)

# === Run prediction and save output ===
for image_name in tqdm(os.listdir(IMAGE_DIR), desc="Predicting"):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)
    result = model(image_path)[0]

    image = cv2.imread(image_path)
    pred_boxes = []
    pred_scores = []

    for box in result.boxes:
        score = float(box.conf)
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pred_boxes.append([x1, y1, x2, y2])
        pred_scores.append(round(score, 4))

        # Draw box on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{result.names[int(box.cls)]} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save annotated image
    output_image_path = os.path.join(OUTPUT_IMG_DIR, image_name)
    cv2.imwrite(output_image_path, image)

    # Save prediction JSON
    json_output = {
        "boxes": pred_boxes,
        "scores": pred_scores
    }
    output_json_path = os.path.join(OUTPUT_JSON_DIR, image_name.replace(".tif", ".json"))
    with open(output_json_path, "w") as f:
        json.dump(json_output, f, indent=2)
