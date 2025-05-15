import os
import json
import xml.etree.ElementTree as ET
import cv2
from glob import glob
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# -------------------------
# this script evaluate the predictions of Faster R-CNN/YOLO for the following:
# -------------------------
# Ground-truth loader (from Pascal-VOC XMLs)
# Prediction loader (from your JSONs)
# IoU matching & TP/FP/FN logic
# Precision, Recall, F1 at a fixed threshold
# Precision–Recall curve and AUC-PR
# COCO mAP@0.5 and mAP@[.5:.95]


# -------------------------
# Configuration
# -------------------------
SPLIT = "val"
IMAGE_DIR = f"dataset/{SPLIT}/images"
PRED_DIR = VIS_DIR = f"Processed_Images/faster_rcnn/Predictions/{SPLIT}"
# PRED_DIR = VIS_DIR = f"Processed_Images/faster_rcnn_ResNet-34/Predictions/{SPLIT}"
ANN_DIR = f"dataset/{SPLIT}/annotations"
IOU_THRESH = 0.5

# -------------------------
# IoU Helper 
# -------------------------
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea) if (box1Area+box2Area-interArea)>0 else 0

# -------------------------
# 1) Ground-truth loader
# -------------------------
def load_ground_truth(ann_dir):
    gt = {}
    for xmlf in glob(os.path.join(ann_dir, "*.xml")):
        img_id = os.path.basename(xmlf).replace(".xml", ".tif")
        boxes = []
        tree, root = ET.parse(xmlf), ET.parse(xmlf).getroot()
        for obj in root.findall("object"):
            if obj.find("name").text != "nematode egg": 
                continue
            b = obj.find("bndbox")
            boxes.append([
                int(b.find("xmin").text), int(b.find("ymin").text),
                int(b.find("xmax").text), int(b.find("ymax").text)
            ])
        gt[img_id] = np.array(boxes)  # shape [N,4] or empty
    return gt

# -------------------------
# 2) Prediction loader
# -------------------------
def load_predictions(pred_dir):
    preds = {}
    for jf in glob(os.path.join(pred_dir, "*.json")):
        img_id = os.path.basename(jf).replace(".json", ".tif")
        obj = json.load(open(jf))
        preds[img_id] = {
            "boxes":  np.array(obj.get("boxes", [])),
            "scores": np.array(obj.get("scores", []))
        }
    return preds

# -------------------------
# 3) Matching logic for one threshold
# -------------------------
def match_image(gt_boxes, pred_boxes, pred_scores, iou_thresh, conf_thresh):
    # filter by confidence
    keep = pred_scores >= conf_thresh
    boxes = pred_boxes[keep]
    scores= pred_scores[keep]
    # sort by descending score
    order = scores.argsort()[::-1]
    boxes, scores = boxes[order], scores[order]

    matched_gt = set()
    tp = fp = 0
    for box in boxes:
        ious = np.array([compute_iou(box, gt) for gt in gt_boxes])
        best_iou_idx = ious.argmax() if len(ious)>0 else -1
        if len(ious)>0 and ious[best_iou_idx] >= iou_thresh and best_iou_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_iou_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, scores.tolist()

# -------------------------
# 4) Aggregate metrics at single threshold
# -------------------------
def evaluate_threshold(gt, preds, iou_thresh, conf_thresh):
    total_tp = total_fp = total_fn = 0
    all_scores, all_labels = [], []
    for img_id, gt_boxes in gt.items():
        pred = preds.get(img_id, {"boxes":np.empty((0,4)), "scores":np.array([])})
        tp, fp, fn, scores = match_image(
            gt_boxes, pred["boxes"], pred["scores"], iou_thresh, conf_thresh
        )
        total_tp += tp; total_fp += fp; total_fn += fn
        # For PR-curve: label each kept prediction as 1 (TP) or 0 (FP)
        # We know match_image processes in descending order,
        # but for PR curve we need per-detection labels:
        # so we repeat match logic here:
        # (alternatively you could collect inside match_image)
        for box, score in zip(pred["boxes"], pred["scores"]):
            if score < conf_thresh: continue
            ious = np.array([compute_iou(box, gt) for gt in gt_boxes])
            if len(ious)>0 and ious.max()>=iou_thresh:
                all_labels.append(1)
            else:
                all_labels.append(0)
            all_scores.append(score)
    precision = total_tp / (total_tp + total_fp) if (total_tp+total_fp)>0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp+total_fn)>0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    return precision, recall, f1, np.array(all_scores), np.array(all_labels)

# -------------------------
# 5) Precision–Recall curve & AUC-PR
# -------------------------
def compute_pr_curve(all_scores, all_labels):
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_scores)
    auc_pr = auc(recalls, precisions)
    return precisions, recalls, thresholds, auc_pr

# -------------------------
# 6) COCO mAP evaluation
# -------------------------
def write_coco_gt(gt, out_path):
    coco = {"images":[], "annotations":[], "categories":[{"id":1,"name":"nematode egg"}]}
    ann_id = 1
    for img_id, boxes in gt.items():
        coco["images"].append({"file_name":img_id, "id":img_id})
        for b in boxes:
            x1,y1,x2,y2 = map(int, b)
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id":ann_id, 
                "image_id":img_id, 
                "category_id":1,
                "bbox":[x1, y1, w, h],
                "area":int(w * h), 
                "iscrowd":0
            })
            ann_id += 1
    with open(out_path,"w") as f:
        json.dump(coco, f)

def write_coco_preds(preds, out_path):
    coco_results = []
    for img_id, p in preds.items():
        for box, score in zip(p["boxes"], p["scores"]):
            x1,y1,x2,y2 = map(int, box)
            coco_results.append({
                "image_id":img_id, 
                "category_id":1,
                "bbox":[x1,y1, x2-x1, y2-y1], 
                "score":float(score)
            })
    with open(out_path,"w") as f:
        json.dump(coco_results, f)

def compute_coco_map(gt_json, pred_json, out_json=None):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    evaler = COCOeval(coco_gt, coco_dt, "bbox")
    evaler.evaluate(); evaler.accumulate(); evaler.summarize()
    
    stats = evaler.stats       # works whether stats is list or ndarray

    # 4. If requested, dump them to JSON
    if out_json:
        keys = [
            "AP@[.50:.95]_all",   "AP@.50_all",    "AP@.75_all",
            "AP@[.50:.95]_small", "AP@[.50:.95]_medium", "AP@[.50:.95]_large",
            "AR@[.50:.95]_maxDets1",  "AR@[.50:.95]_maxDets10",
            "AR@[.50:.95]_maxDets100",
            "AR@[.50:.95]_small", "AR@[.50:.95]_medium", "AR@[.50:.95]_large"
        ]
        metrics = dict(zip(keys, stats))

        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved COCO metrics to {out_json}")

    return stats

# -------------------------
# 7a) Visualisation
# -------------------------

def draw_legend(image):
    cv2.rectangle(image, (10, 10), (300, 90), (255, 255, 255), -1)  # white background box
    cv2.putText(image, 'Legend:', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, 'Ground Truth (Blue)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(image, 'Faster R-CNN Prediction (Green, IoU, Conf)', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


def draw_boxes(image_path, pred_boxes, pred_scores, gt_boxes):
    img = cv2.imread(image_path)

    # Draw GT boxes in BLUE
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw predicted boxes in GREEN
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compute best IoU against GT boxes
        iou = max([compute_iou(box, gt_box) for gt_box in gt_boxes], default=0)
        label = f"IoU: {iou:.2f}, Conf: {pred_scores[i]:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)

    return draw_legend(img)


# -------------------------
# 7b) Run Evaluation
# -------------------------
if __name__ == "__main__":
    # 7.1 Load data
    gt    = load_ground_truth(ANN_DIR)
    preds = load_predictions(PRED_DIR)

    # 7.2 Evaluate at conf=0.5
    P, R, F1, scores, labels = evaluate_threshold(gt, preds, IOU_THRESH, 0.5)
    print(f"Fixed-threshold (0.5) → Precision: {P:.3f}, Recall: {R:.3f}, F1: {F1:.3f}")

    # 7.3 PR curve + AUC-PR
    precs, recs, threshs, aucpr = compute_pr_curve(scores, labels)
    print(f"AUC-PR: {aucpr:.3f}")
    
    pr_path   = os.path.join(VIS_DIR, "pr_data.json")
    # 7.3.1 Write out raw PR data for plotting
    pr_data = {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "auc_pr": float(aucpr),
        "precisions": precs.tolist(),
        "recalls": recs.tolist(),
        "thresholds": threshs.tolist(),
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "iou_thresh": IOU_THRESH,
        "conf_thresh": 0.5,
    }
    with open(pr_path, "w") as f:
        json.dump(pr_data, f, indent=2)
    print("✅ Saved PR data to pr_data.json")

    # 7.4 Write COCO files & compute mAP
    gt_path   = os.path.join(VIS_DIR, "gt_coco.json")
    pred_path = os.path.join(VIS_DIR, "preds_coco.json")
    coco_path = os.path.join(VIS_DIR, "coco_metrics.json")
    
    write_coco_gt(gt,    gt_path)
    write_coco_preds(preds, pred_path)
    compute_coco_map(gt_path, pred_path, out_json = coco_path)
    
    print(f"✅ Saved GT to {gt_path}")
    print(f"✅ Saved predictions to {pred_path}")   
    
# -------------------------
#  Main Visualisation Loop
# -------------------------
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')]

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        json_path = os.path.join(PRED_DIR, img_name.replace(".tif", ".json"))

        if not os.path.exists(json_path):
            print(f"⚠️ Skipping {img_name}: no prediction found.")
            continue

        with open(json_path, 'r') as f:
            pred_data = json.load(f)
        pred_boxes = pred_data.get("boxes", [])
        pred_scores = pred_data.get("scores", [0.0] * len(pred_boxes))  # fallback if scores not saved

        gt_boxes = gt.get(img_name, np.empty((0,4))).tolist()

        visual_img = draw_boxes(img_path, pred_boxes, pred_scores, gt_boxes)
        save_path = os.path.join(VIS_DIR, img_name.replace(".tif", ".jpg"))
        cv2.imwrite(save_path, visual_img)
        print(f"✅ Saved with IoU & confidence: {save_path}")