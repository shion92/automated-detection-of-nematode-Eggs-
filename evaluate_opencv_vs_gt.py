import os
import json
import cv2
import xml.etree.ElementTree as ET
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
GT_DIR = "labels/train"
PRED_JSON_DIR = "Processed_Images/with_fastNIMeansDenoising/Predictions"
IMAGE_DIR = "images/train"
IOU_THRESHOLD = 0.5
OUTPUT_DIR = "evaluation_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "vis"), exist_ok=True)

# === Parse Pascal VOC XML for egg annotations ===
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label != "nematode egg":
            continue
        bndbox = obj.find('bndbox')
        box = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ]
        boxes.append(box)
    return boxes

# === Compute IoU between two boxes ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0

# === Evaluation Loop ===
metrics = {}
all_tp, all_fp, all_fn = 0, 0, 0
all_ious = []

for xml_path in glob(os.path.join(GT_DIR, "*.xml")):
    filename = os.path.basename(xml_path).replace(".xml", ".tif")
    gt_boxes = parse_voc_xml(xml_path)

    # === Load predicted boxes from JSON ===
    json_path = os.path.join(PRED_JSON_DIR, f"{os.path.splitext(filename)[0]}.json")
    if not os.path.exists(json_path):
        print(f"⚠️ Skipping {filename}: prediction JSON not found.")
        continue

    with open(json_path, "r") as jf:
        data = json.load(jf)
        pred_boxes = data.get("boxes", [])

    # === Load image for visualisation ===
    image_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(image_path):
        print(f"⚠️ Skipping {filename}: original image not found.")
        continue

    vis = cv2.imread(image_path)
    matched_gt = set()
    matched_pred = set()
    ious = []

    for i, pb in enumerate(pred_boxes):
        best_iou = 0
        best_gt = -1
        for j, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt)
            matched_pred.add(i)
            ious.append(best_iou)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(ious) if ious else 0

    all_tp += tp
    all_fp += fp
    all_fn += fn
    all_ious.extend(ious)

    metrics[filename] = {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Mean IoU": round(mean_iou, 4),
    }

    # === Visualise matched/unmatched boxes ===
    for gb in gt_boxes:
        cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (255, 0, 0), 2)  # Blue = GT
    for i, pb in enumerate(pred_boxes):
        color = (0, 255, 0) if i in matched_pred else (0, 0, 255)  # Green TP / Red FP
        cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), color, 2)

    vis_path = os.path.join(OUTPUT_DIR, "vis", f"eval_{filename.replace('.tif', '.png')}")
    cv2.imwrite(vis_path, vis)

# === Dataset Summary ===
summary = {
    "Total Images": len(metrics),
    "Total TP": all_tp,
    "Total FP": all_fp,
    "Total FN": all_fn,
    "Precision": round(all_tp / (all_tp + all_fp), 4) if (all_tp + all_fp) > 0 else 0,
    "Recall": round(all_tp / (all_tp + all_fn), 4) if (all_tp + all_fn) > 0 else 0,
    "F1": round(2 * all_tp * all_tp / ((all_tp + all_fp) * (all_tp + all_fn) + 1e-6), 4),
    "Mean IoU": round(np.mean(all_ious), 4) if all_ious else 0
}

# === Save Evaluation Results ===
with open(os.path.join(OUTPUT_DIR, "metrics_imagewise.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Evaluation complete. Results saved to:")
print(f" - {os.path.join(OUTPUT_DIR, 'metrics_imagewise.json')}")
print(f" - {os.path.join(OUTPUT_DIR, 'metrics_summary.json')}")
print(f" - Visuals saved in: {os.path.join(OUTPUT_DIR, 'vis/')}")
