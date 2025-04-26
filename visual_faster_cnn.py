import os
import cv2
import json
import xml.etree.ElementTree as ET

# --- Configuration ---
SPLIT = "val"
IMAGE_DIR = f"dataset/{SPLIT}/images"
PRED_DIR = f"Processed_Images/faster_rcnn/Predictions/{SPLIT}"
ANN_DIR = f"dataset/{SPLIT}/annotations"
VIS_DIR = f"Processed_Images/faster_rcnn/Predictions/{SPLIT}"
os.makedirs(VIS_DIR, exist_ok=True)

# --- IoU helper ---
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def draw_legend(image):
    cv2.rectangle(image, (10, 10), (300, 90), (255, 255, 255), -1)  # white background box
    cv2.putText(image, 'Legend:', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, 'Ground Truth (Blue)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(image, 'Faster R-CNN Prediction (Green, IoU, Conf)', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def get_ground_truth_boxes(xml_path):
    boxes = []
    if not os.path.exists(xml_path):
        return boxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != "nematode egg":
            continue
        bnd = obj.find("bndbox")
        box = [int(bnd.find("xmin").text), int(bnd.find("ymin").text),
               int(bnd.find("xmax").text), int(bnd.find("ymax").text)]
        boxes.append(box)
    return boxes

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

# --- Main Visualisation Loop ---
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')]

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    xml_path = os.path.join(ANN_DIR, img_name.replace(".tif", ".xml"))
    json_path = os.path.join(PRED_DIR, img_name.replace(".tif", ".json"))

    if not os.path.exists(json_path):
        print(f"⚠️ Skipping {img_name}: no prediction found.")
        continue

    with open(json_path, 'r') as f:
        pred_data = json.load(f)
    pred_boxes = pred_data.get("boxes", [])
    pred_scores = pred_data.get("scores", [0.0] * len(pred_boxes))  # fallback if scores not saved

    gt_boxes = get_ground_truth_boxes(xml_path)

    visual_img = draw_boxes(img_path, pred_boxes, pred_scores, gt_boxes)
    save_path = os.path.join(VIS_DIR, img_name.replace(".tif", ".jpg"))
    cv2.imwrite(save_path, visual_img)
    print(f"✅ Saved with IoU & confidence: {save_path}")
