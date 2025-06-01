import os
import json
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
  
# -------------------------
# CONFIGURATION
# -------------------------
class ModelConfig:
    """Configuration for different model types"""
    def __init__(self, model_name: str, model_type: str, pred_folder: str, 
                 gt_folder: str, gt_format: str = 'yolo'):
        self.model_name = model_name
        self.model_type = model_type  # 'yolo', 'yolo_seg', 'faster_rcnn'
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.gt_format = gt_format  # 'yolo', 'pascal_voc'

# Example configurations 
MODEL_CONFIGS = [
    ModelConfig(
        model_name="YOLOv8s",
        model_type="yolo",
        pred_folder="Processed_Images/YOLO/yolov8s_sgd_lr0001/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    ModelConfig(
        model_name="YOLOv8s-max",
        model_type="yolo",
        pred_folder="Processed_Images/YOLO/yolov8s_sgd_lr0001_max/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    ModelConfig(
        model_name="YOLOv8s-seg",
        model_type="yolo_seg", 
        pred_folder="Processed_Images/YOLO/yolov8s_seg/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    
    ModelConfig(
        model_name="YOLOv8s-seg-max",
        model_type="yolo_seg", 
        pred_folder="Processed_Images/YOLO/yolov8s_seg_eras/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    
    ModelConfig(
        model_name="YOLOv8m",
        model_type="yolo", 
        pred_folder="Processed_Images/YOLO/yolov8m_sgd_lr0001/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    
    ModelConfig(
        model_name="YOLOv8m-max",
        model_type="yolo", 
        pred_folder="Processed_Images/YOLO/yolov8m_sgd_lr0001_max/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.005",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.005/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.001",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.001/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.0001",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.0001/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet34-lr0.005",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.005/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet34-lr0.001",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.0001/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    ),
    
    ModelConfig(
        model_name="Faster-RCNN-resnet34-lr0.0001",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet34/Predictions/lr_0.0001/test", 
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc"
    )
]

OUTPUT_DIR = os.path.join("evaluation")
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard_logs')
TARGET_CLASS = 0
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert YOLO format to xyxy format"""
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# -------------------------
# DATA LOADING FUNCTIONS
# -------------------------
def load_yolo_ground_truth(folder: str) -> Dict[str, List[List[float]]]:
    """Load YOLO format ground truth"""
    print(f"Loading YOLO ground truth from: {folder}")
    gt_data = {}
    
    for file_path in glob.glob(os.path.join(folder, '*.txt')):
        filename = os.path.splitext(os.path.basename(file_path))[0]
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
    """Load Pascal VOC XML format ground truth"""
    print(f"Loading Pascal VOC ground truth from: {folder}")
    gt_data = {}
    
    for xml_file in glob.glob(os.path.join(folder, '*.xml')):
        filename = os.path.splitext(os.path.basename(xml_file))[0].lower()
        
        boxes = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                if obj.find('name').text == 'nematode egg':
                    bbox = obj.find('bndbox')
                    x1 = int(bbox.find('xmin').text)
                    y1 = int(bbox.find('ymin').text)
                    x2 = int(bbox.find('xmax').text)
                    y2 = int(bbox.find('ymax').text)
                    boxes.append([x1, y1, x2, y2])
        
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
        
        # Convert .xml filename to .tif for consistency
        if filename.endswith('.xml'):
            filename = filename[:-4] + '.tif'
        gt_data[filename] = boxes
    
    return gt_data

def load_yolo_predictions(folder: str) -> Dict[str, List[Tuple[List[float], float]]]:
    """Load YOLO format predictions"""
    print(f"Loading YOLO predictions from: {folder}")
    pred_data = {}
    
    for file_path in glob.glob(os.path.join(folder, '*.txt')):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        detections = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and int(parts[0]) == TARGET_CLASS:
                    bbox = list(map(float, parts[1:5]))
                    confidence = float(parts[5])
                    detections.append((xywh_to_xyxy(bbox), confidence))
        
        pred_data[filename] = sorted(detections, key=lambda x: -x[1])
    
    return pred_data

def load_faster_rcnn_predictions(folder: str) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Load .json predictions from a folder for Faster RCNN. Supports:
    1. COCO-style: List of dicts with 'image_id', 'bbox' ([x, y, w, h]), 'score'
    2. Custom: File per image, with {'boxes': [...], 'scores': [...]}, boxes in [x1, y1, x2, y2] or [x, y, w, h]
    
    """
    print(f"Loading Faster R-CNN predictions from: {folder}")
    pred_data = {}

    for json_file in glob.glob(os.path.join(folder, '*.json')):
        if "image" not in os.path.basename(json_file).lower():
            continue
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                for item in data:
                    # Extract clean filename (e.g., Image_62)
                    filename = os.path.splitext(item['image_id'])[0]

                    # Skip if not the target class (optional, if you're filtering)
                    if item.get("category_id", 1) != 1:
                        continue

                    # Convert bbox format from [x, y, width, height] 
                    x, y, w, h = item["bbox"]
                    bbox = [x, y, x + w, y + h]
                    score = item["score"]

                    if filename not in pred_data:
                        pred_data[filename] = []
                    pred_data.setdefault(filename, []).append((bbox, score))

            elif isinstance(data, dict) and 'boxes' in data and 'scores' in data:
                filename = Path(json_file).stem.lower()
                boxes = data['boxes']
                scores = data['scores']

                if len(boxes) != len(scores):
                    print(f"Warning: Mismatch between boxes and scores in {filename}")
                    continue

                entries = []
                for box, score in zip(boxes, scores):
                    # If box is [x, y, w, h] → convert to [x1, y1, x2, y2]
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        bbox = [x1, y1, x2, y2]
                        entries.append((bbox, score))
                    else:
                        print(f"Invalid box in {filename}: {box}")
                        continue

                pred_data[filename] = sorted(entries, key=lambda x: -x[1])
                
            
            else:
                print(f"Warning: Unknown format in file: {json_file}")
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return pred_data

# -------------------------
# EVALUATION FUNCTIONS
# -------------------------
def compute_detection_metrics(gt_data: Dict, pred_data: Dict, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """Compute precision, recall, F1-score"""
    tp = fp = fn = 0
    
    for img_name, gt_boxes in gt_data.items():
        pred_detections = pred_data.get(img_name, [])
        pred_boxes = [bbox for bbox, _ in pred_detections]
        
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def compute_average_precision(gt_data: Dict, pred_data: Dict, iou_threshold: float = 0.5) -> float:
    """Compute Average Precision using 11-point interpolation"""
    all_detections = []
    total_gt = sum(len(boxes) for boxes in gt_data.values())
    
    for img_name, detections in pred_data.items():
        for bbox, confidence in detections:
            all_detections.append((img_name, bbox, confidence))
    
    all_detections.sort(key=lambda x: -x[2])
    
    matched_gt = {img: set() for img in gt_data}
    tp_list = []
    fp_list = []
    
    for img_name, pred_box, _ in all_detections:
        gt_boxes = gt_data.get(img_name, [])
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt[img_name]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img_name].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # 11-point interpolation
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 11):
        precision_at_recall = precisions[recalls >= recall_threshold]
        max_precision = precision_at_recall.max() if len(precision_at_recall) > 0 else 0
        ap += max_precision / 11
    
    return ap

def evaluate_model(config: ModelConfig) -> Dict[str, float]:
    """Evaluate a single model configuration"""
    print(f"Evaluating {config.model_name} ({config.model_type})")
    
    # Load ground truth
    gt_folder = os.path.join(config.gt_folder)
    if config.gt_format == 'yolo':
        gt_data = load_yolo_ground_truth(gt_folder)
    else:
        gt_data = load_pascal_voc_ground_truth(gt_folder)
    
    # Load predictions
    pred_folder = os.path.join(config.pred_folder)
    if config.model_type in ['yolo', 'yolo_seg']:
        pred_data = load_yolo_predictions(pred_folder)
    elif config.model_type in ['faster_rcnn']:
        pred_data = load_faster_rcnn_predictions(pred_folder)
    
    if not gt_data or not pred_data:
        print(f"Warning: No data found for {config.model_name}")
        return {}
    
    # Compute metrics
    precision, recall, f1 = compute_detection_metrics(gt_data, pred_data, 0.5)
    
    # Compute AP at different IoU thresholds
    ap_metrics = {}
    for iou_thresh in IOU_THRESHOLDS:
        ap = compute_average_precision(gt_data, pred_data, iou_thresh)
        ap_metrics[f'AP@{iou_thresh:.2f}'] = ap
    
    # Compute mAP@0.5:0.95
    map_5095 = np.mean(list(ap_metrics.values()))
    
    results = {
        'Model': config.model_name,
        'Type': config.model_type,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'mAP@0.5': ap_metrics['AP@0.50'],
        'mAP@0.5:0.95': map_5095,
        'Total_GT': sum(len(boxes) for boxes in gt_data.values()),
        'Total_Predictions': sum(len(dets) for dets in pred_data.values())
    }
    
    # Add individual AP scores
    results.update(ap_metrics)
    
    return results

# -------------------------
# TENSORBOARD LOGGING
# -------------------------
def log_to_tensorboard(all_results: List[Dict], writer: SummaryWriter):
    """Log results to TensorBoard"""
    print("Logging results to TensorBoard...")
    
    for i, result in enumerate(all_results):
        model_name = result['Model']
        
        # Log main metrics
        writer.add_scalar(f'Precision/{model_name}', result['Precision'], 0)
        writer.add_scalar(f'Recall/{model_name}', result['Recall'], 0)
        writer.add_scalar(f'F1/{model_name}', result['F1'], 0)
        writer.add_scalar(f'mAP@0.5/{model_name}', result['mAP@0.5'], 0)
        writer.add_scalar(f'mAP@0.5:0.95/{model_name}', result['mAP@0.5:0.95'], 0)
        
        # Log AP at different IoU thresholds
        for iou_thresh in IOU_THRESHOLDS:
            ap_key = f'AP@{iou_thresh:.2f}'
            if ap_key in result:
                writer.add_scalar(f'AP_IoU_Curve/{model_name}', result[ap_key], int(iou_thresh * 100))
    
    # Create comparison charts
    models = [r['Model'] for r in all_results]
    precisions = [r['Precision'] for r in all_results]
    recalls = [r['Recall'] for r in all_results]
    f1s = [r['F1'] for r in all_results]
    map50s = [r['mAP@0.5'] for r in all_results]
    map5095s = [r['mAP@0.5:0.95'] for r in all_results]
    
    # Log bar charts for comparison
    for i, (model, prec, rec, f1, map50, map5095) in enumerate(zip(models, precisions, recalls, f1s, map50s, map5095s)):
        writer.add_scalar('Comparison/Precision', prec, i)
        writer.add_scalar('Comparison/Recall', rec, i)
        writer.add_scalar('Comparison/F1', f1, i)
        writer.add_scalar('Comparison/mAP@0.5', map50, i)
        writer.add_scalar('Comparison/mAP@0.5:0.95', map5095, i)
    
    writer.close()
    print(f"TensorBoard logs saved to: {TENSORBOARD_DIR}")
    print(f"To view: tensorboard --logdir {TENSORBOARD_DIR}")

# -------------------------
# RESULTS SAVING
# -------------------------
def save_results(all_results: List[Dict]):
    """Save results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save summary CSV
    summary_columns = ['Model', 'Type', 'Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']
    summary_csv = os.path.join(OUTPUT_DIR, f'model_comparison_summary_{timestamp}.csv')
    df[summary_columns].to_csv(summary_csv, index=False)
    
    # Save JSON for programmatic access
    json_file = os.path.join(OUTPUT_DIR, f'results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Summary results saved to: {summary_csv}")
    print(f"JSON results saved to: {json_file}")
    
    return summary_csv

def print_comparison_table(all_results: List[Dict]):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    summary_cols = ['Model', 'Type', 'Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']
    summary_df = df[summary_cols].round(4)
    
    print(summary_df.to_string(index=False))
    
    # Find best performers (with tolerance to floating point equality)
    def get_best_models(metric):
        max_val = df[metric].max()
        best = df[df[metric] == max_val]['Model'].tolist()
        return best, max_val

    print(f"\nBEST PERFORMERS:")

    for metric in ['Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']:
        best_models, best_val = get_best_models(metric)
        best_str = ', '.join(best_models)
        print(f"Best {metric}: {best_str} ({best_val:.4f})")
# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    print("Universal Model Evaluation and Comparison")
    print("="*60)
    
    # Initialize TensorBoard writer
    writer = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(TENSORBOARD_DIR, f'comparison_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # Evaluate all models
    all_results = []
    for config in MODEL_CONFIGS:
        try:
            results = evaluate_model(config)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {config.model_name}: {e}")
            continue
    
    if not all_results:
        print("No models were successfully evaluated!")
        return
    
    # Display and save results
    print_comparison_table(all_results)
    csv_file = save_results(all_results)
    
    # Log to TensorBoard
    if writer:
        log_to_tensorboard(all_results, writer)
    
    print(f"✅ Comparison completed! Check results in: {OUTPUT_DIR}")
    
    print(f"To view TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

if __name__ == '__main__':
    main()