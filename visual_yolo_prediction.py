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
MODEL_NAME = "yolov8s_sgd_lr0001_xmosaic_cutout" # change as needed
IMAGE_DIR = "dataset/test/images"
LABEL_DIR = f"Processed_Images/YOLO/{MODEL_NAME}/test/labels"
ANNOTATION_DIR = "dataset/test/labels"  # Ground truth annotations
OUTPUT_IMG_DIR = os.path.join(os.path.dirname(LABEL_DIR), "images")
# OUTPUT_JSON_DIR = f"Processed_Images/YOLO/{MODEL_NAME}/json_predictions"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (608, 608) # used for scaling YOLO format to absolute pixel coordinates

# -------------------------
# Directory Setup
# -------------------------
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# -------------------------
# Helper Function to Parse YOLO Format
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
        continue
        print(f"Warning: not valid files .png', '.jpg', '.jpeg', '.tif', '.tiff'")
    
    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    annotation_path = os.path.join(ANNOTATION_DIR, f"{base_name}.txt")
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        continue
        print(f"Warning: Image not found for {image_name}")
        
    height, width = image.shape[:2]
    
    pred_boxes = []
    pred_scores = []
    
    # Draw ground truth annotations in BLUE
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            for line in f:
                annotation = parse_yolo_line(line, width, height)
                if annotation is None:
                    continue
                    print(f"Warning: Annotation not found for {image_name}")
                
                x1, y1, x2, y2 = annotation['bbox']
                class_id = annotation['class_id']
                
                # Draw blue bounding box for ground truth
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color
                label_gt = f"GT_class_{class_id}"
                cv2.putText(image, label_gt, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 0), 1)  # Blue text
    
    # Draw predictions in GREEN
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                prediction = parse_yolo_line(line, width, height)
                if prediction is None:
                    continue
                
                confidence = prediction['confidence']
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = prediction['bbox']
                class_id = prediction['class_id']
                
                pred_boxes.append([x1, y1, x2, y2])
                pred_scores.append(round(confidence, 4))
                
                # Draw green bounding box for predictions
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color
                label_pred = f"Pred_class_{class_id} {confidence:.2f}"
                cv2.putText(image, label_pred, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)  # Green text
    
    # Add legend to the image
    legend_y = 30
    cv2.putText(image, "Blue = Ground Truth", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 0, 0), 2)
    cv2.putText(image, "Green = Predictions", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 255, 0), 2)
    
    # Save annotated image
    output_img_path = os.path.join(OUTPUT_IMG_DIR, image_name)
    cv2.imwrite(output_img_path, image)

print(f"Visualization complete! Images saved to: {OUTPUT_IMG_DIR}")

# # Save predictions in JSON format
# json_output = {
#     "boxes": pred_boxes,
#     "scores": pred_scores
# }
# output_json_path = os.path.join(OUTPUT_JSON_DIR, base_name + ".json")
# with open(output_json_path, "w") as f:
#     json.dump(json_output, f, indent=2)