import os
import glob
import json
import cv2
import numpy as np
import colorsys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------
# CONFIGURATION
# -------------------------
SPLIT = "test"  # change if needed
IMAGE_DIR = f"dataset/{SPLIT}/images"
OUTPUT_DIR = os.path.join("evaluation", "overlays", SPLIT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASS = 0  # Assuming class ID for eggs is 0


class ModelConfig:
    """Configuration for different model types"""
    def __init__(self, model_name: str, model_type: str, pred_folder: str, 
                 gt_folder: str, gt_format: str = 'yolo'):
        self.model_name = model_name
        self.model_type = model_type  # 'yolo', 'faster_rcnn', 'segmentation'
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.gt_format = gt_format  # 'yolo', 'pascal_voc', 'mask'


# List of model configurations
MODEL_CONFIGS = [
    ModelConfig(
        model_name="YOLOv8s-max",
        model_type="yolo",
        pred_folder=f"Processed_Images/YOLO/yolov8s_sgd_lr0001_max/{SPLIT}/labels",
        gt_folder=f"dataset/{SPLIT}/labels",
        gt_format="yolo"
    ),
    ModelConfig(
        model_name="YOLOv8m",
        model_type="yolo",
        pred_folder=f"Processed_Images/YOLO/yolov8m_sgd_lr0001/{SPLIT}/labels",
        gt_folder=f"dataset/{SPLIT}/labels",
        gt_format="yolo"
    ),
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.005",
        model_type="faster_rcnn",
        pred_folder=f"Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.005/{SPLIT}",
        gt_folder=f"dataset/{SPLIT}/annotations",
        gt_format="pascal_voc"
    ),
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.001",
        model_type="faster_rcnn",
        pred_folder=f"Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.001/{SPLIT}",
        gt_folder=f"dataset/{SPLIT}/annotations",
        gt_format="pascal_voc"
    ),
]

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def get_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct BGR colours using HSV space.
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert to 0–255 and BGR order for OpenCV
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors

def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]

def load_yolo_ground_truth(folder: str) -> Dict[str, List[List[float]]]:
    """Load YOLO format ground truth."""
    gt_data = {}
    for file_path in glob.glob(os.path.join(folder, '*.txt')):
        filename = Path(file_path).stem
        boxes = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == TARGET_CLASS:
                    coords = list(map(float, parts[1:5]))
                    boxes.append(xywh_to_xyxy(coords))
        gt_data[filename] = boxes
    return gt_data

def load_pascal_voc_ground_truth(folder: str) -> Dict[str, List[List[float]]]:
    """Load Pascal VOC XML format ground truth."""
    import xml.etree.ElementTree as ET
    gt_data = {}
    for xml_file in glob.glob(os.path.join(folder, '*.xml')):
        filename = Path(xml_file).stem
        boxes = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            if obj.find('name').text.lower() == 'nematode egg':
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                boxes.append([x1, y1, x2, y2])
        gt_data[filename] = boxes
    return gt_data

def load_yolo_predictions(folder: str, image_dir: str) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Load YOLO format predictions from .txt files.  
    Assumes values are normalized [0,1].  
    Each line: class_id x_center y_center width height confidence  
    Returns dict: {image_name: [([x1,y1,x2,y2], confidence), ...]}
    """
    pred_data = {}
    for txt_file in glob.glob(os.path.join(folder, '*.txt')):
        image_name = Path(txt_file).stem
        image_path = os.path.join(image_dir, f"{image_name}.tif")
        if not os.path.exists(image_path):
            for ext in ['.jpg', '.png']:
                alt = os.path.join(image_dir, f"{image_name}{ext}")
                if os.path.exists(alt):
                    image_path = alt
                    break
            else:
                continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        detections = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                if class_id != TARGET_CLASS:
                    continue
                x_center_rel, y_center_rel, w_rel, h_rel = map(float, parts[1:5])
                confidence = float(parts[5])
                x_center = x_center_rel * w
                y_center = y_center_rel * h
                width_abs = w_rel * w
                height_abs = h_rel * h
                bbox = xywh_to_xyxy([x_center, y_center, width_abs, height_abs])
                bbox = [
                    max(0, min(bbox[0], w - 1)),
                    max(0, min(bbox[1], h - 1)),
                    max(0, min(bbox[2], w - 1)),
                    max(0, min(bbox[3], h - 1))
                ]
                detections.append((bbox, confidence))
        pred_data[image_name] = sorted(detections, key=lambda x: -x[1])
    return pred_data

def load_faster_rcnn_predictions(folder: str) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Load Faster R-CNN predictions from .json files.
    Supports COCO-style (list of dicts) and custom (single dict per image).
    Returns dict: {image_name: [([x1,y1,x2,y2], score), ...]}
    """
    pred_data = {}
    for json_file in glob.glob(os.path.join(folder, '*.json')):
        base = os.path.basename(json_file).lower()
        if "image" not in base:
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                img_id = Path(item['image_id']).stem
                if item.get("category_id", 1) != 1:
                    continue
                x, y, w, h = item['bbox']
                bbox = [x, y, x + w, y + h]
                score = item['score']
                pred_data.setdefault(img_id, []).append((bbox, score))
        elif isinstance(data, dict) and 'boxes' in data and 'scores' in data:
            img_id = Path(json_file).stem
            for box, score in zip(data['boxes'], data['scores']):
                if len(box) == 4:
                    pred_data.setdefault(img_id, []).append(([float(box[0]), float(box[1]), float(box[2]), float(box[3])], float(score)))
    for img in pred_data:
        pred_data[img] = sorted(pred_data[img], key=lambda x: -x[1])
    return pred_data

# -------------------------
# MAIN VISUALISATION SCRIPT
# -------------------------
def main():
    colours = get_distinct_colors(len(MODEL_CONFIGS))
    model_predictions = {}
    gt_data_dict = {}

    # Pre-load ground truth for each model format (choose first model's gt_folder to load GT once)
    first_cfg = MODEL_CONFIGS[0]
    if first_cfg.gt_format == "yolo":
        gt_data = load_yolo_ground_truth(first_cfg.gt_folder)
    else:
        gt_data = load_pascal_voc_ground_truth(first_cfg.gt_folder)

    # Load predictions
    for idx, cfg in enumerate(MODEL_CONFIGS):
        if cfg.model_type == "yolo":
            preds = load_yolo_predictions(cfg.pred_folder, IMAGE_DIR)
        else:
            preds = load_faster_rcnn_predictions(cfg.pred_folder)
        model_predictions[cfg.model_name] = preds

    # Iterate over each image in the test set
    for img_path in glob.glob(os.path.join(IMAGE_DIR, "*.tif")):
        img_name = Path(img_path).stem
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Create a list to hold per-model annotated images
        annotated_imgs = []

        for idx, cfg in enumerate(MODEL_CONFIGS):
            base_copy = image.copy()

            # Draw ground truth on base_copy (in green)
            for gt_box in gt_data.get(img_name, []):
                x1, y1, x2, y2 = map(int, gt_box)
                cv2.rectangle(base_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            # Draw that model's predictions (in its distinct colour)
            preds = model_predictions[cfg.model_name].get(img_name, [])
            for (bbox, score) in preds:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(base_copy, (x1, y1), (x2, y2), colours[idx], thickness=3)
                text = f"{score:.2f}"
                cv2.putText(
                    base_copy,
                    text,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    colours[idx],
                    2,
                    lineType=cv2.LINE_AA
                )

            # Add model name label at top-left
            label = cfg.model_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3 , 2)
            cv2.rectangle(base_copy, (5, 5), (5 + tw + 10, 5 + th + 10), (255, 255, 255), -1)
            cv2.putText(
                base_copy,
                label,
                (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 0),
                3,
                lineType=cv2.LINE_AA
            )

            annotated_imgs.append(base_copy)

        # Concatenate all annotated images horizontally
        row = cv2.hconcat(annotated_imgs)

        # Save the concatenated output
        output_path = os.path.join(OUTPUT_DIR, f"{img_name}_comparison.png")
        cv2.imwrite(output_path, row)
        print(f"Saved comparison for {img_name} to {output_path}")


if __name__ == "__main__":
    main()
