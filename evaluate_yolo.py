import os
import glob
import numpy as np

# ——— USER CONFIGURATION —————————————————————————————
PRED_FOLDER = '/Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/YOLO/nematode_yolov8s/test/labels'
GT_FOLDER   = '/Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/dataset/test/labels'
TARGET_CLASS = 0
# ———————————————————————————————————————————————————————

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def iou(box1, box2):
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter_w = max(0, xi2 - xi1); inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area1 = max(0, (box1[2]-box1[0])*(box1[3]-box1[1]))
    area2 = max(0, (box2[2]-box2[0])*(box2[3]-box2[1]))
    union = area1 + area2 - inter
    return inter/union if union>0 else 0

def load_ground_truth(folder):
    gt = {}
    for fn in glob.glob(os.path.join(folder, '*.txt')):
        name = os.path.splitext(os.path.basename(fn))[0]
        boxes = []
        for line in open(fn):
            cls, *coords = line.split()
            if int(cls)!=TARGET_CLASS: continue
            boxes.append(xywh_to_xyxy(list(map(float, coords[:4]))))
        gt[name] = boxes
    return gt

def load_predictions(folder):
    preds = {}
    for fn in glob.glob(os.path.join(folder, '*.txt')):
        name = os.path.splitext(os.path.basename(fn))[0]
        recs = []
        for line in open(fn):
            cls, *rest = line.split()
            if int(cls)!=TARGET_CLASS: continue
            bbox = list(map(float, rest[:4])); conf = float(rest[4])
            recs.append((xywh_to_xyxy(bbox), conf))
        preds[name] = sorted(recs, key=lambda x:-x[1])
    return preds

def compute_detection_metrics(gt, preds, iou_thr=0.5):
    tp = fp = fn = 0
    for img, gt_boxes in gt.items():
        pred_boxes = [b for b,_ in preds.get(img,[])]
        matched = set()
        for pb in pred_boxes:
            best_iou, best_j = 0, -1
            for j, gb in enumerate(gt_boxes):
                if j in matched: continue
                cur = iou(pb, gb)
                if cur>best_iou:
                    best_iou, best_j = cur, j
            if best_iou>=iou_thr:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return prec, rec, f1

def compute_ap(gt, preds, iou_thr=0.5):
    # gather all preds
    items = []
    total_gt = sum(len(boxes) for boxes in gt.values())
    for img, recs in preds.items():
        for box, conf in recs:
            items.append((img, box, conf))
    items.sort(key=lambda x:-x[2])
    matched = {img:set() for img in gt}
    tp_list, fp_list = [], []
    for img, box, _ in items:
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt.get(img,[])):
            if j in matched[img]: continue
            cur = iou(box, gb)
            if cur>best_iou:
                best_iou, best_j = cur, j
        if best_iou>=iou_thr:
            tp_list.append(1); fp_list.append(0)
            matched[img].add(best_j)
        else:
            tp_list.append(0); fp_list.append(1)
    tp_cum = np.cumsum(tp_list); fp_cum = np.cumsum(fp_list)
    recs = tp_cum/total_gt if total_gt>0 else np.zeros_like(tp_cum)
    precs= tp_cum/(tp_cum+fp_cum+1e-8)
    # 11-point interpolation
    ap=0
    for t in np.linspace(0,1,11):
        p = precs[recs>=t].max() if np.any(recs>=t) else 0
        ap += p/11
    return ap

if __name__=='__main__':
    gt   = load_ground_truth(GT_FOLDER)
    pred = load_predictions(PRED_FOLDER)

    # simple detection metrics @ IoU=0.5
    prec, rec, f1 = compute_detection_metrics(gt, pred, 0.5)
    # mAP@0.5
    ap50 = compute_ap(gt, pred, 0.5)

    print(f"{'Metric':<15}{'Value'}")
    print(f"{'-'*25}")
    print(f"Precision        {prec:.4f}")
    print(f"Recall           {rec:.4f}")
    print(f"F1               {f1:.4f}")
    print(f"mAP@0.5          {ap50:.4f}")
