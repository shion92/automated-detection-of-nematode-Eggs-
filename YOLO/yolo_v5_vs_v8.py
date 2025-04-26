import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random

# === Define the paths ===
yolov5_results_path = '../YOLO/nematode_yolov5s_train2/results.csv'
yolov8_results_path = '../YOLO/nematode_yolov8s_train2/results.csv'

# === Read the CSVs (validation metrics) ===
y5 = pd.read_csv(yolov5_results_path, skipinitialspace=True)
y5.columns = y5.columns.str.strip()
y8 = pd.read_csv(yolov8_results_path, skipinitialspace=True)
y8.columns = y8.columns.str.strip()

# Extract best validation metrics
yolov5_val = {
    'Precision': y5['metrics/precision'].max(),
    'Recall':    y5['metrics/recall'].max(),
    'mAP50':     y5['metrics/mAP_0.5'].max(),
    'mAP50-95':  y5['metrics/mAP_0.5:0.95'].max(),
}
yolov8_val = {
    'Precision': y8['metrics/precision(B)'].max(),
    'Recall':    y8['metrics/recall(B)'].max(),
    'mAP50':     y8['metrics/mAP50(B)'].max(),
    'mAP50-95':  y8['metrics/mAP50-95(B)'].max(),
}

# === Helper functions to compute test metrics ===
def yolo_to_box(x, y, w, h, img_w, img_h):
    # normalized center-based YOLO format → [xmin,ymin,xmax,ymax]
    xc, yc = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return [xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2]

def load_yolo_labels(label_dir, img_sizes):
    preds = {}
    for fn in os.listdir(label_dir):
        if not fn.endswith('.txt'): continue
        img_name = fn.replace('.txt', '')
        iw, ih = img_sizes[img_name]
        boxes_with_conf = []
        for line in open(os.path.join(label_dir, fn)):
            parts = line.split()
            # unpack class  coords  confidence
            cls   = float(parts[0])
            x, y, w, h = map(float, parts[1:5])
            conf  = float(parts[5]) if len(parts) >= 6 else None
            box   = yolo_to_box(x, y, w, h, iw, ih)
            boxes_with_conf.append((box, conf))
        preds[img_name] = boxes_with_conf
    return preds

def compute_iou(box1, box2):
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    a = (box1[2]-box1[0])*(box1[3]-box1[1])
    b = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (a + b - inter + 1e-6)

def eval_dataset(gt, pred, iou_thresh):
    tp = fp = fn = 0
    for img, gbs in gt.items():
        pbs = pred.get(img, [])  # each p is now either a list [xmin,ymin,xmax,ymax] or a (box,conf) tuple
        matched = set()
        for g in gbs:
            # find a pred with IoU>=thresh
            for i,p in enumerate(pbs):
                if i in matched: continue
                # unpack if p is (box, conf)
                box = p[0] if isinstance(p, (tuple, list)) and len(p)==2 and isinstance(p[0], list) else p
                if compute_iou(g, box) >= iou_thresh:
                    tp += 1; matched.add(i)
                    break
        fp += len(pbs) - len(matched)
        fn += len(gbs) - sum(1 for i in matched)
    return tp, fp, fn

def compute_test_metrics(gt, pred, iou_thresholds):
    # returns dict: {thr: (precision, recall)}
    results = {}
    for thr in iou_thresholds:
        tp, fp, fn = eval_dataset(gt, pred, thr)
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec  = tp/(tp+fn) if tp+fn>0 else 0
        results[thr] = (prec, rec)
    # mAP50 is precision@0.5, mAP50-95 is mean precision across thresholds
    precs = [results[t][0] for t in iou_thresholds]
    return {
        'Precision': results[0.5][0],
        'Recall':    results[0.5][1],
        'mAP50':     results[0.5][0],
        'mAP50-95':  np.mean(precs),
    }
    
def eval_dataset(gt, pred, iou_thresh):
    tp = fp = fn = 0
    for img, gbs in gt.items():
        pbs = pred.get(img, [])  # each p is now either a list [xmin,ymin,xmax,ymax] or a (box,conf) tuple

        matched = set()
        for g in gbs:
             # find a pred with IoU>=thresh
            for i,p in enumerate(pbs):
                if i in matched: continue
                # unpack if p is (box, conf)
                box = p[0] if isinstance(p, (tuple, list)) and len(p)==2 and isinstance(p[0], list) else p
                if compute_iou(g, box) >= iou_thresh:
                    tp += 1; matched.add(i)
                    break
        fp += len(pbs) - len(matched)
        fn += len(gbs) - sum(1 for i in matched)
    return tp, fp, fn



# === Prepare test-set ground truth and image sizes ===
test_img_dir = '../dataset/test/images'
gt_label_dir  = '../dataset/test/labels'
img_sizes = {}
gt = {}
for img_fn in os.listdir(test_img_dir):
    name, _ = os.path.splitext(img_fn)
    img = cv2.imread(os.path.join(test_img_dir, img_fn))
    h,w = img.shape[:2]
    img_sizes[name] = (w,h)
    # load GT boxes
    boxes = []
    for line in open(os.path.join(gt_label_dir, name + '.txt')):
        cls, x,y,w_,h_ = map(float, line.split())
        boxes.append(yolo_to_box(x,y,w_,h_,w,h))
    gt[name] = boxes

# === Load model predictions on test set ===
v5_pred_dir = '../YOLO/nematode_yolov5s/test/labels'
v8_pred_dir = '../YOLO/nematode_yolov8s/test/labels'
pred5 = load_yolo_labels(v5_pred_dir, img_sizes)
pred8 = load_yolo_labels(v8_pred_dir, img_sizes)

# === Compute test metrics at IoU thresholds 0.5:0.05:0.95 ===
thrs = np.arange(0.5, 1.0, 0.05)
yolov5_test = compute_test_metrics(gt, pred5, thrs)
yolov8_test = compute_test_metrics(gt, pred8, thrs)


# === Build comparison DataFrame ===
comparison_df = pd.DataFrame({
    'Metric':    ['Precision', 'Recall', 'mAP50', 'mAP50-95'],
    'YOLOv5_val': [yolov5_val[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8_val': [yolov8_val[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv5_test':[yolov5_test[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8_test':[yolov8_test[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
})

# === Save the comparison summary ===
out_dir = '../YOLO/summary'
os.makedirs(out_dir, exist_ok=True)
comparison_df.to_csv(os.path.join(out_dir,'yolov_comparison_metrics_summary.csv'), index=False)

# === Plot side-by-side comparison for validation and test ===
x = np.arange(len(comparison_df))
w = 0.2

fig, ax = plt.subplots(figsize=(12,6))
ax.bar(x - 1.5*w, comparison_df['YOLOv5_val'], w, label='v5_val')
ax.bar(x - 0.5*w, comparison_df['YOLOv8_val'], w, label='v8_val', alpha=0.7)
ax.bar(x + 0.5*w, comparison_df['YOLOv5_test'], w, label='v5_test')
ax.bar(x + 1.5*w, comparison_df['YOLOv8_test'], w, label='v8_test', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Metric'])
ax.set_ylabel('Score')
ax.set_title('YOLOv5s vs YOLOv8s: Validation & Test Metrics')
ax.legend()

for rect in ax.patches:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.01, f'{h:.2f}', 
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir,'yolov_comparison_metrics_barplot.png'))
plt.show()


# === Create overlays for ALL test images ===
out_vis = '../YOLO/summary/overlays'
os.makedirs(out_vis, exist_ok=True)

for name in gt:
    # 1. find a real image file for this sample
    img_file = None
    for ext in ('.jpg', '.jpeg', '.png', '.tif'):
        candidate = os.path.join(test_img_dir, name + ext)
        if os.path.isfile(candidate):
            img_file = candidate
            break
    if img_file is None:
        print(f'Warning: no image file found for {name}, skipping.')
        continue

    # 2. read & check
    img_bgr = cv2.imread(img_file)
    if img_bgr is None:
        print(f'Warning: cv2.imread failed for {img_file}, skipping.')
        continue
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. plot
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    ax = plt.gca()

    # 4. GT in blue
    for b in gt[name]:
        rect = plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                             fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    # 5. YOLOv5 preds in red (handle either box-only or (box,conf))
    for item in pred5.get(name, []):
        box = item[0] if isinstance(item, (list,tuple)) and len(item)==2 else item
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # 6. YOLOv8 preds in green
    for item in pred8.get(name, []):
        box = item[0] if isinstance(item, (list,tuple)) and len(item)==2 else item
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    plt.title(f'{name}: GT (blue) | v5 (red) | v8 (green)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_vis, f'{name}_overlay.png'))
    plt.close()
