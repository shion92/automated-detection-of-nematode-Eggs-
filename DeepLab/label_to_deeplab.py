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
            # Fix: convert points from list-of-lists to list-of-tuples
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
    if not os.path.exists(img_dir):
        print(f"⚠️ Skipping {split} (directory not found)")
        continue
    print(f"Processing {split}...")
    mask_out_dir = os.path.join(OUT_MASK_DIR, split, "masks")
    os.makedirs(mask_out_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.endswith(".json"):
            continue

        base = fname.replace(".json", "")
        img_path = os.path.join(img_dir, base + ".tif")
        json_path = os.path.join(img_dir, fname)

        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {base} (image not found)")
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        mask = convert_labelme_to_mask(json_path, img_np.shape, TARGET_LABEL)
        mask_out_path = os.path.join(mask_out_dir, base + ".png")
        Image.fromarray(mask).save(mask_out_path)
        print(f"✅ Saved mask: {mask_out_path}")

print("\nAll masks converted and saved.")
