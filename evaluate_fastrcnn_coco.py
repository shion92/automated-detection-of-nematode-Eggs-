import os, json, cv2, xml.etree.ElementTree as ET
import numpy as np
from glob import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ─── Configuration ─────────────────────────────────────────────────────────────
METHOD_NAME    = "faster_rcnn"
DATA_SPLITS    = ["train", "val", "test"]
GT_DIR_ROOT    = "dataset"    # split/annotations/*.xml
IMG_DIR_ROOT   = "dataset"    # split/images/*.tif
PRED_JSON_ROOT = f"Processed_Images/{METHOD_NAME}/Predictions"
OUTPUT_ROOT    = f"evaluation_outputs/{METHOD_NAME}"
IOU_THRESHOLD  = 0.5

CATEGORIES = [{"id": 1, "name": "nematode egg"}]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        if obj.findtext('name','').strip().lower() != 'nematode egg':
            continue
        bnd = obj.find('bndbox')
        boxes.append([
            int(bnd.findtext('xmin',0)),
            int(bnd.findtext('ymin',0)),
            int(bnd.findtext('xmax',0)),
            int(bnd.findtext('ymax',0))
        ])
    return boxes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    return interArea/union if union>0 else 0

def compute_coco_map(gt_json, dt_json, img_ids):
    coco_gt   = COCO(gt_json)
    coco_dt   = coco_gt.loadRes(dt_json)
    # mAP@0.5
    eval50    = COCOeval(coco_gt, coco_dt, 'bbox')
    eval50.params.imgIds = img_ids
    eval50.params.iouThrs = np.array([0.5])
    eval50.evaluate(); eval50.accumulate()
    stats50 = getattr(eval50, 'stats', [])
    map50   = float(stats50[1]) if len(stats50)>1 else 0.0
    # mAP@0.5:0.95
    evalAll  = COCOeval(coco_gt, coco_dt, 'bbox')
    evalAll.params.imgIds = img_ids
    evalAll.evaluate(); evalAll.accumulate()
    statsAll = getattr(evalAll, 'stats', [])
    mapAll   = float(statsAll[0]) if len(statsAll)>0 else 0.0
    return round(map50,4), round(mapAll,4)

for split in DATA_SPLITS:
    print(f"\n=== {split.upper()} ===")
    gt_xmls    = glob(os.path.join(GT_DIR_ROOT, split, "annotations", "*.xml"))
    pred_jsons = glob(os.path.join(PRED_JSON_ROOT, split, "*.json"))
    out_dir    = os.path.join(OUTPUT_ROOT, split)
    vis_dir    = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # prepare accumulators
    all_tp, all_fp, all_fn = 0, 0, 0
    all_ious = []
    metrics  = {}
    img_id_map = {}

    # === VOC-based loop ===
    for idx, xml_path in enumerate(gt_xmls, start=1):
        fname    = os.path.basename(xml_path).replace('.xml','.tif')
        print(fname)
        img_id_map[fname] = idx
        print(idx)

        gt_boxes  = parse_voc_xml(xml_path)
        print(gt_boxes)
        
        json_path = os.path.join(PRED_JSON_ROOT, split, f"{os.path.splitext(fname)[0]}.json")
        data      = json.load(open(json_path)) if os.path.exists(json_path) else {}
        pred_boxes= data.get("boxes", [])
        ious, matched_gt, matched_pred = [], set(), set()

        # match predictions to GT
        for i,pb in enumerate(pred_boxes):
            best_iou, best_j = 0, -1
            for j,gb in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou>best_iou:
                    best_iou, best_j = iou, j
            if best_iou>=IOU_THRESHOLD:
                matched_gt.add(best_j)
                matched_pred.add(i)
                ious.append(best_iou)

        # compute per-image metrics
        tp         = len(matched_gt)
        fp         = len(pred_boxes) - tp
        fn         = len(gt_boxes)   - tp
        precision  = tp/(tp+fp) if (tp+fp)>0 else 0
        recall     = tp/(tp+fn) if (tp+fn)>0 else 0
        f1         = 2*precision*recall/(precision+recall+1e-6)
        mean_iou   = np.mean(ious) if ious else 0

        all_tp += tp; all_fp += fp; all_fn += fn
        all_ious.extend(ious)
        metrics[fname] = {
            "TP":tp,"FP":fp,"FN":fn,
            "Precision":round(precision,4),
            "Recall":round(recall,4),
            "F1":round(f1,4),
            "Mean IoU":round(mean_iou,4)
        }

        # save visualisation
        img_path = os.path.join(IMG_DIR_ROOT, split, "images", fname)
        vis      = cv2.imread(img_path)
        for gb in gt_boxes:
            cv2.rectangle(vis, (gb[0],gb[1]),(gb[2],gb[3]), (255,0,0),2)
        for i,pb in enumerate(pred_boxes):
            col = (0,255,0) if i in matched_pred else (0,0,255)
            cv2.rectangle(vis,(pb[0],pb[1]),(pb[2],pb[3]), col,2)
        cv2.imwrite(os.path.join(vis_dir, f"eval_{fname}.png"), vis)

    # === COCO GT JSON ===
    coco_gt = {"images":[],"annotations":[],"categories":CATEGORIES}
    ann_id = 1
    for fname, idx in img_id_map.items():
        img = cv2.imread(os.path.join(IMG_DIR_ROOT, split, "images", fname))
        h,w = img.shape[:2]
        coco_gt["images"].append({"id":idx,"file_name":fname,"height":h,"width":w})
        xml_path = os.path.join(GT_DIR_ROOT, split, "annotations", fname.replace('.tif','.xml'))

        for obj in ET.parse(xml_path).getroot().findall('object'):
            if obj.findtext('name','').strip().lower()!='nematode egg': continue
            b = obj.find('bndbox')
            xmin,ymin = int(b.findtext('xmin')), int(b.findtext('ymin'))
            xmax,ymax = int(b.findtext('xmax')), int(b.findtext('ymax'))
            bw, bh = xmax-xmin, ymax-ymin
            coco_gt["annotations"].append({
                "id":ann_id, "image_id":idx, "category_id":1,
                "bbox":[xmin, ymin, bw, bh], "area":bw*bh, "iscrowd":0
            })
            ann_id+=1

    gt_json_path = os.path.join(out_dir,"coco_gt_annotations.json")
    with open(gt_json_path,'w') as f: json.dump(coco_gt,f,indent=2)

    # === COCO DET JSON ===
    coco_dt=[]
    for jp in pred_jsons:
        stem = os.path.basename(jp).replace('.json','')
        fname=stem+'.tif'
        img_id = img_id_map.get(fname)
        if img_id is None: continue
        data   = json.load(open(jp))
        boxes  = data.get("boxes",[])
        scores = data.get("scores",[1.0]*len(boxes))
        for box,score in zip(boxes,scores):
            xmin,ymin,xmax,ymax = box
            bw, bh = xmax-xmin, ymax-ymin
            coco_dt.append({
                "image_id":img_id,"category_id":1,
                "bbox":[xmin,ymin,bw,bh],"score":float(score)
            })

    dt_json_path = os.path.join(out_dir,"coco_predictions.json")
    with open(dt_json_path,'w') as f: json.dump(coco_dt,f,indent=2)

    # === Build and save summary including mAPs ===
    summary = {
        "Total Images": len(img_id_map),
        "Total TP": all_tp,
        "Total FP": all_fp,
        "Total FN": all_fn,
        "Precision": round(all_tp/(all_tp+all_fp),4) if all_tp+all_fp>0 else 0,
        "Recall":    round(all_tp/(all_tp+all_fn),4) if all_tp+all_fn>0 else 0,
        "F1":        round((2*all_tp)/(2*all_tp+all_fp+all_fn+1e-6),4),
        "Mean IoU":  round(np.mean(all_ious),4) if all_ious else 0
    }

    map50, map_all = compute_coco_map(gt_json_path, dt_json_path, list(img_id_map.values()))
    summary["mAP@0.5"]      = map50
    summary["mAP@0.5:0.95"] = map_all

    with open(os.path.join(out_dir,"metrics_imagewise.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir,"metrics_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"→ Saved split summary (+mAPs) to {out_dir}/metrics_summary.json")
