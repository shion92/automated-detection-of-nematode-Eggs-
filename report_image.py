import os
import glob
import json
import cv2
import numpy as np
import colorsys
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------
# CONFIGURATION
# -------------------------
SPLIT = "test"  # change if needed

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
    # ModelConfig(
    #     model_name="YOLOv8s",
    #     model_type="yolo",
    #     pred_folder=f"Processed_Images/YOLO/yolov8s_sgd_lr0001/{SPLIT}/labels",
    #     gt_folder=f"dataset/{SPLIT}/labels",
    #     gt_format="yolo"
    # ),
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
        model_name="YOLOv8m-max",
        model_type="yolo",
        pred_folder=f"Processed_Images/YOLO/yolov8m_sgd_lr0001_max/{SPLIT}/labels",
        gt_folder=f"dataset/{SPLIT}/labels",
        gt_format="yolo"
    ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet50-lr0.005",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.005/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet50-lr0.001",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.001/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet50-lr0.0001",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.0001/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet34-lr0.005",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.005/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet34-lr0.001",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.0001/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="Faster-RCNN-resnet34-lr0.0001",
    #     model_type="faster_rcnn",
    #     pred_folder=f"Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.0001/{SPLIT}",
    #     gt_folder=f"dataset/{SPLIT}/annotations",
    #     gt_format="pascal_voc"
    # ),
    # ModelConfig(
    #     model_name="DeepLabV3+-lr0.0001",
    #     model_type="segmentation",
    #     pred_folder=f"Processed_Images/deeplab/Predictions/lr_0.0001/{SPLIT}",  # .json or PNG masks
    #     gt_folder=f"dataset/{SPLIT}/masks",
    #     gt_format="mask"
    # ),
    # ModelConfig(
    #     model_name="DeepLabV3+-lr0.0005",
    #     model_type="segmentation",
    #     pred_folder=f"Processed_Images/deeplab/Predictions/lr_0.0005/{SPLIT}",  # .json or PNG masks
    #     gt_folder=f"dataset/{SPLIT}/masks",
    #     gt_format="mask"
    # ),
    # ModelConfig(
    #     model_name="YOLOv8s-seg",
    #     model_type="segmentation",
    #     pred_folder=f"Processed_Images/YOLO/yolov8s_seg_lr0001/{SPLIT}/labels",
    #     gt_folder=f"dataset/{SPLIT}/masks",
    #     gt_format="mask"
    # ),
    # ModelConfig(
    #     model_name="YOLOv8s-seg-max",
    #     model_type="segmentation",
    #     pred_folder=f"Processed_Images/YOLO/yolov8s_seg_lr0001_eras/{SPLIT}/labels",
    #     gt_folder=f"dataset/{SPLIT}/masks",
    #     gt_format="mask"
    # ),
]

OUTPUT_DIR = os.path.join("evaluation", "overlays")
IMAGE_DIR = f"dataset/{SPLIT}/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASS = 0  # Assuming class ID for eggs is 0

# -------------------------
# HELPER FUNCTIONS
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

def load_yolo_predictions(folder: str) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Load YOLO format predictions from .txt files.
    Each line: class_id x_center y_center width height confidence
    Returns dict: {image_name: [([x1,y1,x2,y2], confidence), ...]}
    """
    pred_data = {}
    for txt_file in glob.glob(os.path.join(folder, '*.txt')):
        image_name = os.path.splitext(os.path.basename(txt_file))[0]
        detections = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                if class_id != TARGET_CLASS:
                    continue
                coords = list(map(float, parts[1:5]))
                confidence = float(parts[5])
                bbox = xywh_to_xyxy(coords)
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
        # Skip files unless they contain 'image' in the filename
        if "image" not in base:
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            # COCO-style list of dicts
            for item in data:
                img_id = os.path.splitext(item['image_id'])[0]
                if item.get("category_id", 1) != 1:
                    continue
                x, y, w, h = item['bbox']
                bbox = [x, y, x + w, y + h]
                score = item['score']
                pred_data.setdefault(img_id, []).append((bbox, score))
        elif isinstance(data, dict) and 'boxes' in data and 'scores' in data:
            img_id = Path(json_file).stem
            boxes = data['boxes']
            scores = data['scores']
            for box, score in zip(boxes, scores):
                # Box format assumed [x1, y1, x2, y2]
                if len(box) == 4:
                    bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                    pred_data.setdefault(img_id, []).append((bbox, float(score)))
        # Sort by score descending
    for img in pred_data:
        pred_data[img] = sorted(pred_data[img], key=lambda x: -x[1])
    return pred_data

def load_binary_mask_png(folder: str) -> Dict[str, np.ndarray]:
    """
    Load binary masks from PNG files.
    Returns dict: {image_name: mask_array}
    """
    mask_data = {}
    for mask_file in glob.glob(os.path.join(folder, '*.png')):
        key = Path(mask_file).stem
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask > 127).astype(np.uint8)
        mask_data[key] = binary
    return mask_data

def load_binary_mask_json(folder: str) -> Dict[str, np.ndarray]:
    """
    Load segmentation masks stored as float arrays in JSON.
    Returns dict: {image_name: binary_mask_array}
    """
    mask_data = {}
    for json_file in glob.glob(os.path.join(folder, '*.json')):
        key = Path(json_file).stem
        with open(json_file, 'r') as f:
            data = json.load(f)
        if "mask" in data:
            arr = np.array(data["mask"])
            binary = (arr >= 0.5).astype(np.uint8)
            mask_data[key] = binary
    return mask_data

def load_yolo_segmentation_txt(folder: str, image_dir: str, confidence_thresh=0.5) -> Dict[str, np.ndarray]:
    """
    Load YOLOv8-seg polygon predictions from .txt files.
    Each line: class_id x1 y1 x2 y2 ... xn yn confidence
    Returns dict: {image_name: binary_mask_array}
    """
    mask_data = {}
    for txt_file in glob.glob(os.path.join(folder, '*.txt')):
        base_name = Path(txt_file).stem
        image_path = os.path.join(image_dir, f"{base_name}.tif")
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7 or (len(parts) - 2) % 2 != 0:
                    continue
                class_id = int(float(parts[0]))
                confidence = float(parts[-1])
                if class_id != TARGET_CLASS or confidence < confidence_thresh:
                    continue
                coords = list(map(float, parts[1:-1]))
                pts = np.array(
                    [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)],
                    dtype=np.int32
                )
                cv2.fillPoly(mask, [pts], color=1)
        mask_data[base_name] = mask
    return mask_data

# -------------------------
# MAIN VISUALISATION SCRIPT
# -------------------------
def main():
    # Generate distinct colours for each model
    colours = get_distinct_colors(len(MODEL_CONFIGS))

    # Mapping from model_name to its predictions
    model_predictions = {}

    # Load predictions for each model ahead of time
    for idx, cfg in enumerate(MODEL_CONFIGS):
        if cfg.model_type == "yolo":
            preds = load_yolo_predictions(cfg.pred_folder)
        elif cfg.model_type == "faster_rcnn":
            preds = load_faster_rcnn_predictions(cfg.pred_folder)
        elif cfg.model_type == "segmentation":
            # Two possible mask formats: PNG or JSON or YOLO-seg txt
            # Try PNG first, then JSON, then YOLO-seg txt
            png_masks = load_binary_mask_png(cfg.pred_folder)
            if png_masks:
                preds = png_masks
            else:
                json_masks = load_binary_mask_json(cfg.pred_folder)
                if json_masks:
                    preds = json_masks
                else:
                    # YOLOv8-seg style
                    preds = load_yolo_segmentation_txt(cfg.pred_folder, IMAGE_DIR)
        else:
            preds = {}
        model_predictions[cfg.model_name] = preds

    # Iterate over each image in the test set
    for img_path in glob.glob(os.path.join(IMAGE_DIR, "*.tif")):
        img_name = Path(img_path).stem
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Create an overlay copy
        overlay = image.copy()

        # Draw each model's predictions
        for idx, cfg in enumerate(MODEL_CONFIGS):
            colour = colours[idx]
            preds = model_predictions.get(cfg.model_name, {})

            if cfg.model_type in ["yolo", "faster_rcnn"]:
                detections = preds.get(img_name, [])
                for (bbox, score) in detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw bounding box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
                    # Put confidence text
                    text = f"{cfg.model_name}: {score:.2f}"
                    cv2.putText(
                        overlay,
                        text,
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        1,
                        lineType=cv2.LINE_AA
                    )
            elif cfg.model_type == "segmentation":
                mask = model_predictions[cfg.model_name].get(img_name)
                if mask is None:
                    continue
                # Resize mask if needed to match image
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Create coloured mask
                coloured_mask = np.zeros_like(image, dtype=np.uint8)
                coloured_mask[mask == 1] = colour
                # Overlay mask with transparency
                alpha = 0.4
                overlay = cv2.addWeighted(overlay, 1.0, coloured_mask, alpha, 0)

        # Blend overlay with original image for final view
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        # Draw legend (model colour mapping) at the top-left
        legend_x, legend_y = 10, 25
        for idx, cfg in enumerate(MODEL_CONFIGS):
            colour = colours[idx]
            # Draw small filled rectangle
            cv2.rectangle(
                blended,
                (legend_x, legend_y - 15),
                (legend_x + 20, legend_y + 5),
                colour,
                -1
            )
            # Put model name next to rectangle
            cv2.putText(
                blended,
                cfg.model_name,
                (legend_x + 25, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )
            legend_y += 25  # Move down for next legend entry

        # Save the output image
        output_path = os.path.join(OUTPUT_DIR, f"{img_name}_overlay.png")
        cv2.imwrite(output_path, blended)
        print(f"Saved overlay for {img_name} to {output_path}")


if __name__ == "__main__":
    main()
