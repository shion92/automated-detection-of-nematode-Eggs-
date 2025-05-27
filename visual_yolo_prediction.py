# -------------------------
# Imports
# -------------------------
import os
import json
import cv2
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "yolov8s_sgd_lr0001_xmosaic_cutout"  # change as needed
IMAGE_DIR = "dataset/test/images"
LABEL_DIR = f"Processed_Images/YOLO/{MODEL_NAME}/test/labels"
OUTPUT_IMG_DIR = os.path.join(os.path.dirname(LABEL_DIR),"images")
# OUTPUT_JSON_DIR = f"Processed_Images/YOLO/{MODEL_NAME}/json_predictions"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (608, 608)  # used for scaling YOLO format to absolute pixel coordinates

# -------------------------
# Directory Setup
# -------------------------
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# -------------------------
# Draw Predictions from Label Files
# -------------------------
for image_name in tqdm(os.listdir(IMAGE_DIR), desc="Rendering predictions"):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue

    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(label_path):
        continue  # skip if prediction not available

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    pred_boxes = []
    pred_scores = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            class_id, cx, cy, w, h, conf = map(float, parts)
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Convert YOLO format to (x1, y1, x2, y2)
            cx, cy, w, h = cx * width, cy * height, w * width, h * height
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(round(conf, 4))

            # Draw bounding box
            label = f"class_{int(class_id)} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # Save annotated image
    output_img_path = os.path.join(OUTPUT_IMG_DIR, image_name)
    cv2.imwrite(output_img_path, image)
    

    # # Save predictions in JSON format
    # json_output = {
    #     "boxes": pred_boxes,
    #     "scores": pred_scores
    # }
    # output_json_path = os.path.join(OUTPUT_JSON_DIR, base_name + ".json")
    # with open(output_json_path, "w") as f:
    #     json.dump(json_output, f, indent=2)
