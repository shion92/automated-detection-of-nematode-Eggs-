import os
import glob
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

# ─── config ───────────────────────────────────────────────────────────────────
SPLIT          = "val"  # or "val", "test"
GT_XML_DIR     = f"dataset/{SPLIT}/annotations"
PRED_JSON_DIR  = f"Processed_Images/opencv/with_fastNIMeansDenoising 2/Predictions/{SPLIT}"
# ──────────────────────────────────────────────────────────────────────────────

def load_ground_truth(gt_dir):
    gt = {}
    for xml_path in glob.glob(os.path.join(gt_dir, "*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        fname = root.findtext("filename")
        boxes = []
        for obj in root.findall("object"):
            cls = obj.findtext("name")
            if cls != "nematode egg":
                continue
            bb = obj.find("bndbox")
            x1 = int(bb.findtext("xmin"))
            y1 = int(bb.findtext("ymin"))
            x2 = int(bb.findtext("xmax"))
            y2 = int(bb.findtext("ymax"))
            boxes.append([x1, y1, x2, y2])
        gt[fname] = boxes
    return gt

def load_predictions(pred_dir):
    preds = {}
    for pj in glob.glob(os.path.join(pred_dir, "*.json")):
        data = json.load(open(pj))
        fname = data.get("filename")
        boxes = data.get("boxes", [])
        preds[fname] = boxes
    return preds

def centroid(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def evaluate(gt, preds):
    TP = FP = FN = 0
    total_gt   = sum(len(b) for b in gt.values())
    total_pred = sum(len(b) for b in preds.values())

    for fname, gt_boxes in gt.items():
        pred_boxes = preds.get(fname, [])
        matched = set()
        # match preds by checking centroid in any unmatched GT
        for pb in pred_boxes:
            cx, cy = centroid(pb)
            hit = False
            for i, gb in enumerate(gt_boxes):
                if i in matched: 
                    continue
                x1,y1,x2,y2 = gb
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    TP += 1
                    matched.add(i)
                    hit = True
                    break
            if not hit:
                FP += 1
        # unmatched GT → FN
        FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Metric Summary for {SPLIT} set:")
    print(f"   Total GT boxes     : {total_gt}")
    print(f"   Total Pred boxes   : {total_pred}")
    print(f"   True Positives (TP): {TP}")
    print(f"   False Positives(FP): {FP}")
    print(f"   False Negatives(FN): {FN}")
    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

if __name__ == "__main__":
    gt    = load_ground_truth(GT_XML_DIR)
    preds = load_predictions(PRED_JSON_DIR)
    evaluate(gt, preds)
