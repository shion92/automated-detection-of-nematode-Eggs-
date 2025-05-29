# -------------------------
# Imports
# -------------------------
import json
import os
import cv2
from tqdm import tqdm

# -------------------------
# LabelMe to YOLO Conversion Function
# -------------------------
def convert_labelme_json_to_yolov8_seg(json_path, image_dir, output_label_dir):
    """
    Convert LabelMe JSON annotation to YOLO segmentation format
    
    Args:
        json_path (str): Path to the LabelMe JSON file
        image_dir (str): Directory containing the images
        output_label_dir (str): Output directory for YOLO label files
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_path = os.path.join(image_dir, data["imagePath"])
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Image not found: {image_path}")
        return
    h, w = image.shape[:2]

    label_lines = []
    for shape in data["shapes"]:
        if shape["label"].lower() != "nematode egg":
            continue

        class_id = 0  # class index
        points = shape["points"]
        norm_points = []
        for x, y in points:
            norm_x = x / w
            norm_y = y / h
            norm_points.extend([round(norm_x, 6), round(norm_y, 6)])

        line = f"{class_id} " + " ".join(map(str, norm_points))
        label_lines.append(line)

    image_base = os.path.splitext(data["imagePath"])[0]
    output_path = os.path.join(output_label_dir, f"{image_base}.txt")
    os.makedirs(output_label_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for line in label_lines:
            f.write(line + '\n')

    # print(f"✅ Converted: {json_path} to {output_path}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    annotation_dir = os.path.join(os.getcwd(), "dataset_seg", "val", "json")
    image_dir = os.path.join(os.getcwd(), "dataset_seg", "val", "images")
    output_label_dir = os.path.join(os.getcwd(), "dataset_seg", "val", "labels")

    for fname in tqdm(os.listdir(annotation_dir), desc="Converting"):
        if fname.endswith(".json"):
            json_path = os.path.join(annotation_dir, fname)
            convert_labelme_json_to_yolov8_seg(
                json_path=json_path,
                image_dir=image_dir,
                output_label_dir=output_label_dir
            )