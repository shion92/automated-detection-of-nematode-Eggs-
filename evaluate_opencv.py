import os
import glob
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SPLIT = "train"  # or "val", "test"
GT_ANNOTATIONS = f"dataset/{SPLIT}/annotations"
PRED_JSON_DIR  = f"/Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/Processed_Images/faster_rcnn/Predictions/{SPLIT}"
COCO_GT_JSON   = f"coco_gt_{SPLIT}.json"
COCO_PRED_JSON = f"coco_pred_{SPLIT}.json"
# ──────────────────────────────────────────────────────────────────────────────

def voc_to_coco(gt_dir, out_json):
    images, annotations = [], []
    ann_id = 1
    for img_id, xml_file in enumerate(sorted(glob.glob(os.path.join(gt_dir, "*.xml"))), 1):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        fname = root.findtext("filename")
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            x1, y1 = float(bbox.findtext("xmin")), float(bbox.findtext("ymin"))
            x2, y2 = float(bbox.findtext("xmax")), float(bbox.findtext("ymax"))
            w_box, h_box = x2-x1, y2-y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": [{"id":1,"name":"egg"}]}
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"[+] Wrote {len(images)} images and {len(annotations)} GT boxes → {out_json}")

def preds_to_coco(pred_dir, gt_json, out_json):
    coco = COCO(gt_json)
    pred_list = []
    for img in coco.dataset["images"]:
        base = os.path.splitext(img["file_name"])[0]
        pj = os.path.join(pred_dir, f"{base}.json")
        if not os.path.exists(pj):
            continue
        data = json.load(open(pj))
        boxes  = data.get("boxes", [])
        scores = data.get("scores", [1.0]*len(boxes))
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            pred_list.append({
                "image_id": img["id"],
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })
    with open(out_json, "w") as f:
        json.dump(pred_list, f)
    print(f"[+] Wrote {len(pred_list)} detections → {out_json}")

def run_coco_eval(gt_json, pred_json):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP@0.50:0.95": coco_eval.stats[0], "mAP@0.50": coco_eval.stats[1]}

def main():
    voc_to_coco(GT_ANNOTATIONS, COCO_GT_JSON)
    preds_to_coco(PRED_JSON_DIR, COCO_GT_JSON, COCO_PRED_JSON)

    try:
        with open(COCO_PRED_JSON, "r") as f:
            pred_list = json.load(f)
    except FileNotFoundError:
        pred_list = []
    
    if not pred_list:
        print("[!] No detections found; skipping COCO evaluation.")
        coco_metrics = {"mAP@0.50:0.95": 0.0, "mAP@0.50": 0.0}
    else:
        coco_metrics = run_coco_eval(COCO_GT_JSON, COCO_PRED_JSON)
        
    print("\n>> COCO-style metrics:")
    for k,v in coco_metrics.items():
        print(f"   {k:12s}: {v:.4f}")

    # Precision / Recall / F1 @ IoU=0.5
    def iou(a,b):
        xa1,ya1,xa2,ya2 = a; xb1,yb1,xb2,yb2 = b
        xi1, yi1 = max(xa1,xb1), max(ya1,yb1)
        xi2, yi2 = min(xa2,xb2), min(ya2,yb2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        area_a = (xa2-xa1)*(ya2-ya1)
        area_b = (xb2-xb1)*(yb2-yb1)
        union = area_a + area_b - inter
        return inter/union if union>0 else 0

    # load GT
    gt_data = json.load(open(COCO_GT_JSON))
    gt = defaultdict(list)
    for ann in gt_data["annotations"]:
        x,y,w,h = ann["bbox"]
        gt[ann["image_id"]].append([x,y,x+w,y+h])

    # load detections
    preds = json.load(open(COCO_PRED_JSON))
    dt = defaultdict(list)
    for det in preds:
        x,y,w,h = det["bbox"]
        dt[det["image_id"]].append([x,y,x+w,y+h])

    TP = FP = FN = 0
    for img_id in set(gt) | set(dt):
        gts = gt.get(img_id, [])
        dts = sorted(dt.get(img_id, []), key=lambda _: -1.0)
        matched = set()
        for d in dts:
            best_i, best_iou = -1, 0
            for i, g in enumerate(gts):
                if i in matched: continue
                iou_val = iou(d, g)
                if iou_val > best_iou:
                    best_i, best_iou = i, iou_val
            if best_iou >= 0.5:
                TP += 1
                matched.add(best_i)
            else:
                FP += 1
        FN += len(gts) - len(matched)

    precision = TP / (TP+FP) if TP+FP else 0
    recall    = TP / (TP+FN) if TP+FN else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    print("\n>> Detection metrics @ IoU=0.5:")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-score  : {f1:.4f}")

if __name__ == "__main__":
    main()
