import os
import json
import numpy as np
from PIL import Image

# -------------------------
# Configuration
# -------------------------
SPLIT = ["train", "val", "test"]  # add "test" if needed
OUT_MASK_DIR = IMAGE_ROOT = "dataset"
TARGET_LABEL = "nematode egg"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# -------------------------
# Labelme JSON to Mask Conversion
# -------------------------
def convert_labelme_to_mask(json_path, img_shape, target_label):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if label != target_label.lower():
            continue

        points = shape.get("points", [])
        if shape.get("shape_type") == "polygon":
            points = [tuple(p) for p in points]
            from PIL import ImageDraw
            mask_img = Image.new("L", (img_shape[1], img_shape[0]), 0)
            ImageDraw.Draw(mask_img).polygon(points, outline=1, fill=1)
            mask += np.array(mask_img, dtype=np.uint8)

    mask = np.clip(mask, 0, 1) * 255
    return mask.astype(np.uint8)


# -------------------------
# Main Conversion Script
# -------------------------
for split in SPLIT:
    img_dir = os.path.join(IMAGE_ROOT, split, "images")
    mask_out_dir = os.path.join(OUT_MASK_DIR, split, "masks")
    json_dir = os.path.join(IMAGE_ROOT, split, "json")
    if not os.path.exists(img_dir):
        print(f"Warning: Skipping {split} (directory not found)")
        continue
    print(f"Processing {split}...")
    os.makedirs(mask_out_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in VALID_EXTS:
            continue

        img_path  = os.path.join(img_dir, fname)
        json_path = os.path.join(json_dir, base + ".json")

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        
        if os.path.exists(json_path):
            mask = convert_labelme_to_mask(json_path, img_np.shape, TARGET_LABEL)
        else:
            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        mask_out_path = os.path.join(mask_out_dir, base + ".png")
        Image.fromarray(mask).save(mask_out_path)
        print(f"✅ Saved mask: {mask_out_path}")

print("\nAll masks converted and saved.")
