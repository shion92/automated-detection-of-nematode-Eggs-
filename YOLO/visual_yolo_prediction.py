# -------------------------
# Imports
# -------------------------
import os
import json
import cv2
from tqdm import tqdm
from yolov8s import predict_model  

# -------------------------
# Configuration
# -------------------------
SPLIT = "test"
MODEL_NAME = "yolov8s_sgd_lr0001"  # change as needed
MODEL_PATH = os.path.join("model", "YOLO", MODEL_NAME, "weights", "best.pt")
IMAGE_DIR = f"dataset/{SPLIT}/images"
LABEL_DIR = f"Processed_Images/YOLO/{MODEL_NAME}/{SPLIT}/labels"
ANNOTATION_DIR = f"dataset/{SPLIT}/labels"  # Ground truth annotations
OUTPUT_IMG_DIR = os.path.join(os.path.dirname(LABEL_DIR), "images")
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (608, 608)  # used for scaling YOLO format to pixel coordinates

# -------------------------
# Directory Setup
# -------------------------
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# -------------------------
# Step 0: Run prediction if missing
# -------------------------
def prediction_needed(image_dir, label_dir):
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(label_dir, f"{base}.txt")
            if not os.path.exists(label_path):
                return True
    return False

if prediction_needed(IMAGE_DIR, LABEL_DIR):
    print(f"Warning: Prediction results missing. Running prediction with model: {MODEL_NAME}")
    task = "segment" if "_seg" in MODEL_NAME else "detect"
    predict_model(weight_path= MODEL_PATH, config_name=MODEL_NAME, task = task)
    print("Prediction complete.")

# -------------------------
# Helper Function
# -------------------------
def parse_yolo_line(line, img_width, img_height):
    """Parse a YOLO format line and return bounding box coordinates"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    
    class_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    
    # Convert YOLO format to (x1, y1, x2, y2)
    cx, cy, w, h = cx * img_width, cy * img_height, w * img_width, h * img_height
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    
    confidence = float(parts[5]) if len(parts) > 5 else 1.0
    
    return {
        'class_id': class_id,
        'bbox': [x1, y1, x2, y2],
        'confidence': confidence
    }

# -------------------------
# Draw Predictions and Annotations
# -------------------------
for image_name in tqdm(os.listdir(IMAGE_DIR), desc="Rendering predictions and annotations"):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        print(f"Warning: not valid files .png', '.jpg', '.jpeg', '.tif', '.tiff'")
        continue
        
    
    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    annotation_path = os.path.join(ANNOTATION_DIR, f"{base_name}.txt")
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load {image_name}")
        continue
    
    height, width = image.shape[:2]
    
    pred_boxes = []
    pred_scores = []
    
    # Draw ground truth annotations in BLUE
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            for line in f:
                annotation = parse_yolo_line(line, width, height)
                if annotation is None:
                    print(f"Warning: Annotation not found for {image_name}")
                    continue
                
                x1, y1, x2, y2 = annotation['bbox']
                class_id = annotation['class_id']
                
                # Draw blue bounding box for ground truth
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 2)  # Blue color
                label_gt = f"GT_class_{class_id}"
                cv2.putText(image, label_gt, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 128, 0), 1)  # Blue text
    
    # Draw predictions in GREEN
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                pred = parse_yolo_line(line, width, height)
                if pred is None or pred['confidence'] < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = pred['bbox']
                pred_boxes.append([x1, y1, x2, y2])
                pred_scores.append(round(pred['confidence'], 4))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 128), 2)
                cv2.putText(image, f"Pred_class_{pred['class_id']} {pred['confidence']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

    # Add legend
    legend_y = 30
    cv2.putText(image, "Blue = Ground Truth", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    cv2.putText(image, "Green = Predictions", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 225, 128), 2)

    # Save image
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, image_name), image)

print(f"✅ Done! Annotated images saved to: {OUTPUT_IMG_DIR}")
