import os, json, cv2
import xml.etree.ElementTree as ET
from glob import glob

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_SPLITS     = ["train", "val", "test"]
GT_DIR_ROOT     = "dataset"                # contains train/annotations, etc.
IMAGE_DIR_ROOT  = "dataset"                # contains train/images, etc.
PRED_JSON_ROOT  = "Processed_Images/faster_rcnn/Predictions"
OUTPUT_DIR      = "evaluation_outputs/faster_rcnn/coco_format"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── COCO categories ────────────────────────────────────────────────────────────
categories = [{"id": 1, "name": "nematode egg"}]

# ─── 1) Build ground-truth COCO JSON ────────────────────────────────────────────
coco_gt = {"images": [], "annotations": [], "categories": categories}
ann_id = 1
img_id = 1

for split in DATA_SPLITS:
    xml_paths = glob(os.path.join(GT_DIR_ROOT, split, "annotations", "*.xml"))
    for xml_path in xml_paths:
        # derive filenames and read image size
        fname = os.path.basename(xml_path).replace(".xml", ".tif")
        img_path = os.path.join(IMAGE_DIR_ROOT, split, "images", fname)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # COCO image entry
        coco_gt["images"].append({
            "id": img_id,
            "file_name": fname,
            "height": h,
            "width": w
        })

        # parse VOC XML, filtering only "nematode egg"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip().lower()
            if name != "nematode egg":
                continue

            bnd = obj.find("bndbox")
            xmin = int(bnd.findtext("xmin", 0))
            ymin = int(bnd.findtext("ymin", 0))
            xmax = int(bnd.findtext("xmax", 0))
            ymax = int(bnd.findtext("ymax", 0))
            bw = xmax - xmin
            bh = ymax - ymin
            area = bw * bh

            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

# write GT JSON
with open(os.path.join(OUTPUT_DIR, "coco_gt_annotations.json"), "w") as f:
    json.dump(coco_gt, f, indent=2)
print(f"✅ Saved coco_gt_annotations.json with {len(coco_gt['images'])} images "
      f"and {len(coco_gt['annotations'])} annotations")

# ─── 2) Build predictions COCO JSON ─────────────────────────────────────────────
# build a lookup from filename → image_id
name2id = {img["file_name"]: img["id"] for img in coco_gt["images"]}

coco_dt = []
for split in DATA_SPLITS:
    # assuming each per-image JSON is named `<stem>.json`
    json_paths = glob(os.path.join(PRED_JSON_ROOT, split, "*.json"))
    for jp in json_paths:
        stem = os.path.basename(jp).replace(".json", "")
        fname = stem + ".tif"
        image_id = name2id.get(fname)
        if image_id is None:
            print(f"⚠️  Skipping {jp}: no matching image_id for {fname}")
            continue

        data = json.load(open(jp))
        boxes = data.get("boxes", [])
        scores = data.get("scores", [1.0] * len(boxes))

        # if your JSON uses a different key for confidences, adjust above
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            bw = xmax - xmin
            bh = ymax - ymin
            coco_dt.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "score": round(float(score), 4)
            })

# write predictions JSON
with open(os.path.join(OUTPUT_DIR, "coco_predictions.json"), "w") as f:
    json.dump(coco_dt, f, indent=2)
print(f"✅ Saved coco_predictions.json with {len(coco_dt)} detections")
