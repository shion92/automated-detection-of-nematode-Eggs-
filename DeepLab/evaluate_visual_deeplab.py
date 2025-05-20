import os
import json
import cv2
from glob import glob
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

# -------------------------
# this script evaluates the predictions of DeepLabV3+ segmentation for the following:
# -------------------------
# Ground-truth loader (from mask PNGs)
# Prediction loader (from your JSONs)
# Pixel‐level IoU helper
# Pixel‐wise TP/FP/FN logic
# Pixel Accuracy, Mean Accuracy
# Precision, Recall, F1 at a fixed threshold
# Per‐class IoU & Mean IoU
# Precision–Recall curve and AUC-PR
# Visualisation of GT vs. Pred overlay

# -------------------------
# Configuration
# -------------------------
SPLIT = "test"
IMAGE_DIR = f"dataset/{SPLIT}/images"
PRED_DIR = f"Processed_Images/deeplab/Predictions/{SPLIT}"
VIS_DIR = f"Processed_Images/deeplab/Predictions/{SPLIT}"
EVAL_DIR = f"evaluation/deeplab/{SPLIT}"
os.makedirs(EVAL_DIR, exist_ok=True)
GT_MASK_DIR  = f"dataset/{SPLIT}/masks"
THRESH = 0.5   # binarisation threshold for predicted masks

# -------------------------
# IoU Helper 
# -------------------------
def compute_pixel_iou(gt_mask, pred_mask):
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return inter / union if union > 0 else 1.0

# -------------------------
# 1) Ground-truth loader
# -------------------------
def load_ground_truth(gt_dir):
    gt = {}
    for png in glob(os.path.join(gt_dir, "*.png")):
        name = os.path.basename(png).replace(".png", "")
        mask = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        gt[name] = (mask > 127).astype(np.uint8)
    return gt

# -------------------------
# 2) Prediction loader
# -------------------------
def load_predictions(pred_dir):
    preds = {}
    for jf in glob(os.path.join(pred_dir, "*.json")):
        name = os.path.basename(jf).replace(".json", "")
        obj = json.load(open(jf))
        if "mask" not in obj:
            raise KeyError(
                f"No 'mask' key in {jf}; found keys: {list(obj.keys())}"
            )
        pm = np.array(obj["mask"], dtype=np.float32)
        preds[name] = pm
    return preds

# -------------------------
# 3) Matching logic for one threshold
# -------------------------
def pixel_counts(gt_mask, pred_mask):
    tp = int(np.logical_and(gt_mask==1, pred_mask==1).sum())
    fp = int(np.logical_and(gt_mask==0, pred_mask==1).sum())
    fn = int(np.logical_and(gt_mask==1, pred_mask==0).sum())
    tn = int(np.logical_and(gt_mask==0, pred_mask==0).sum())
    return tp, fp, fn, tn

# -------------------------
# 4) Aggregate metrics at single threshold
# -------------------------
def evaluate_threshold(gt, preds, thresh):
    total_tp = total_fp = total_fn = total_tn = 0
    inters = {0:0, 1:0}
    unions = {0:0, 1:0}
    all_scores, all_labels = [], []
    for name, gt_mask in gt.items():
        pm = preds.get(name, np.zeros_like(gt_mask))
        if pm.shape != gt_mask.shape:
            pm = cv2.resize(pm,
                            (gt_mask.shape[1], gt_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        pm_bin = (pm > thresh).astype(np.uint8)
        tp, fp, fn, tn = pixel_counts(gt_mask, pm_bin)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

        # per-class IoU accumulators
        for cls in [0, 1]:
            inter = int(np.logical_and(gt_mask==cls, pm_bin==cls).sum())
            union = int(np.logical_or(gt_mask==cls, pm_bin==cls).sum())
            inters[cls] += inter
            unions[cls] += union

        # for PR-curve at pixel-level
        flat_gt   = gt_mask.flatten()
        flat_pred = pm.flatten()
        all_labels.extend(flat_gt.tolist())
        all_scores.extend(flat_pred.tolist())

    # pixel accuracy & mean accuracy
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    acc_cls0  = total_tn / (total_tn + total_fp) if (total_tn + total_fp)>0 else 0
    acc_cls1  = total_tp / (total_tp + total_fn) if (total_tp + total_fn)>0 else 0
    mean_acc  = (acc_cls0 + acc_cls1) / 2

    # precision, recall, f1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp)>0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn)>0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall)>0 else 0

    # per-class IoU & mean IoU
    iou0 = inters[0] / unions[0] if unions[0]>0 else 1.0
    iou1 = inters[1] / unions[1] if unions[1]>0 else 1.0
    miou = (iou0 + iou1) / 2

    return pixel_acc, mean_acc, precision, recall, f1, iou0, iou1, miou, np.array(all_scores), np.array(all_labels)

# -------------------------
# 5) Precision–Recall curve & AUC-PR
# -------------------------
def compute_pr_curve(all_scores, all_labels):
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_scores)
    auc_pr = auc(recalls, precisions)
    return precisions, recalls, thresholds, auc_pr

# -------------------------
# 6) Visualisation
# -------------------------
def draw_legend(image):
    cv2.rectangle(image, (10, 10), (300, 70), (255, 255, 255), -1)
    cv2.putText(image, 'Legend:', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, 'Background (no egg): no overlay', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, 'Egg (GT=Red, Pred=Green)', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def draw_overlay(image_path, gt_mask, pred_mask):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # resize pred_mask 
    pred_mask = cv2.resize(pred_mask,(w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[gt_mask == 1]   = (0,   0, 255)
    overlay[pred_mask == 1] = (0, 255,   0)
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return draw_legend(blended)

# -------------------------
# 7) Run Evaluation
# -------------------------
if __name__ == "__main__":
    # 7.1 Load data
    gt = load_ground_truth(GT_MASK_DIR)
    preds = load_predictions(PRED_DIR)

    # 7.2 Evaluate at fixed threshold
    PA, mAcc, P, R, F1, IoU_bg, IoU_egg, mIoU, scores, labels = evaluate_threshold(gt, preds, THRESH)
    print(f"Fixed-threshold ({THRESH}) → Precision: {P:.3f}, Recall: {R:.3f}, F1: {F1:.3f}")
    print(f"Pixel Accuracy: {PA:.3f}, Mean Accuracy: {mAcc:.3f}")
    print(f"IoU Background: {IoU_bg:.3f}, IoU Egg: {IoU_egg:.3f}, Mean IoU: {mIoU:.3f}")

    # 7.3 PR curve + AUC-PR
    precs, recs, threshs, aucpr = compute_pr_curve(scores, labels)
    print(f"AUC-PR: {aucpr:.3f}")
    pr_path = os.path.join(EVAL_DIR, "pr_data.json")
    with open(pr_path, "w") as f:
        json.dump({
            "auc_pr": float(aucpr),
            "precisions": precs.tolist(),
            "recalls": recs.tolist(),
            "thresholds": threshs.tolist(),
            "fixed_threshold": THRESH,
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
        }, f, indent=2)
    print(f"✅ Saved PR data to pr_data.json")

# -------------------------
#  Main Visualisation Loop
# -------------------------
    for name, gt_mask in tqdm(gt.items(), desc="Visualising"):
        img_path = os.path.join(IMAGE_DIR, name + ".tif")
        pm = preds.get(name, np.zeros_like(gt_mask))
        pm_bin = (pm > THRESH).astype(np.uint8)

        vis = draw_overlay(img_path, gt_mask, pm_bin)
        output = os.path.join(VIS_DIR, name + ".jpg")
        cv2.imwrite(output, vis)
        print(f"✅ Saved overlay: {output}")

    print("\n✅ All evaluation and visualisations complete.")
