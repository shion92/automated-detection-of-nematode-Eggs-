import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random

# === Define the paths ===
yolov5_results_path  = '../YOLO/nematode_yolov5s_train/results.csv'
yolov8s_results_path = '../YOLO/nematode_yolov8s_train2/results.csv'
yolov8n_results_path = '../YOLO/nematode_yolov8n_train/results.csv'

# === Read the CSVs (validation metrics) ===
y5  = pd.read_csv(yolov5_results_path,  skipinitialspace=True); y5.columns  = y5.columns.str.strip()
y8s = pd.read_csv(yolov8s_results_path, skipinitialspace=True); y8s.columns = y8s.columns.str.strip()
y8n = pd.read_csv(yolov8n_results_path, skipinitialspace=True); y8n.columns = y8n.columns.str.strip()

# Extract best validation metrics
yolov5_val = {
    'Precision': y5['metrics/precision'].max(),
    'Recall':    y5['metrics/recall'].max(),
    'mAP50':     y5['metrics/mAP_0.5'].max(),
    'mAP50-95':  y5['metrics/mAP_0.5:0.95'].max(),
}
yolov8s_val = {
    'Precision': y8s['metrics/precision(B)'].max(),
    'Recall':    y8s['metrics/recall(B)'].max(),
    'mAP50':     y8s['metrics/mAP50(B)'].max(),
    'mAP50-95':  y8s['metrics/mAP50-95(B)'].max(),
}
yolov8n_val = {
    'Precision': y8n['metrics/precision(B)'].max(),
    'Recall':    y8n['metrics/recall(B)'].max(),
    'mAP50':     y8n['metrics/mAP50(B)'].max(),
    'mAP50-95':  y8n['metrics/mAP50-95(B)'].max(),
}

# === Helper functions ===
def yolo_to_box(x, y, w, h, img_w, img_h):
    xc, yc = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return [xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2]

def load_yolo_labels(label_dir, img_sizes):
    preds = {}
    for fn in os.listdir(label_dir):
        if not fn.endswith('.txt'): continue
        name = fn[:-4]
        iw, ih = img_sizes[name]
        boxes_with_conf = []
        with open(os.path.join(label_dir, fn)) as f:
            for line in f:
                parts = line.split()
                cls = float(parts[0])
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else None
                box = yolo_to_box(x, y, w, h, iw, ih)
                boxes_with_conf.append((box, conf))
        preds[name] = boxes_with_conf
    return preds

def compute_iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a = (box1[2]-box1[0])*(box1[3]-box1[1])
    b = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (a + b - inter + 1e-6)

def eval_dataset(gt, pred, iou_thresh):
    tp = fp = fn = 0
    for img, gbs in gt.items():
        pbs = pred.get(img, [])
        matched = set()
        for g in gbs:
            for i, p in enumerate(pbs):
                if i in matched: continue
                box = p[0] if isinstance(p, (tuple,list)) and len(p)==2 else p
                if compute_iou(g, box) >= iou_thresh:
                    tp += 1
                    matched.add(i)
                    break
        fp += len(pbs) - len(matched)
        fn += len(gbs) - sum(1 for i in matched)
    return tp, fp, fn

def compute_test_metrics(gt, pred, iou_thresholds):
    results = {}
    for thr in iou_thresholds:
        tp, fp, fn = eval_dataset(gt, pred, thr)
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec = tp/(tp+fn) if tp+fn>0 else 0
        results[thr] = (prec, rec)
    precs = [results[t][0] for t in iou_thresholds]
    return {
        'Precision': results[0.5][0],
        'Recall':    results[0.5][1],
        'mAP50':     results[0.5][0],
        'mAP50-95':  np.mean(precs),
    }

# === Prepare ground truth and image sizes ===
test_img_dir = '../dataset/test/images'
gt_label_dir = '../dataset/test/labels'
img_sizes, gt = {}, {}
for fn in os.listdir(test_img_dir):
    name, _ = os.path.splitext(fn)
    img = cv2.imread(os.path.join(test_img_dir, fn))
    h, w = img.shape[:2]
    img_sizes[name] = (w, h)
    boxes = []
    with open(os.path.join(gt_label_dir, name + '.txt')) as f:
        for line in f:
            cls, x, y, w_, h_ = map(float, line.split())
            boxes.append(yolo_to_box(x, y, w_, h_, w, h))
    gt[name] = boxes

# === Load predictions ===
pred5  = load_yolo_labels('../YOLO/nematode_yolov5s/test3/labels',  img_sizes)
pred8s = load_yolo_labels('../YOLO/nematode_yolov8s/test/labels', img_sizes)
pred8n = load_yolo_labels('../YOLO/nematode_yolov8n/test/labels', img_sizes)

# === Compute test metrics ===
thrs = np.arange(0.5, 1.0, 0.05)
y5_test  = compute_test_metrics(gt, pred5,  thrs)
y8s_test = compute_test_metrics(gt, pred8s, thrs)
y8n_test = compute_test_metrics(gt, pred8n, thrs)

# === Build comparison DataFrame ===
comparison_df = pd.DataFrame({
    'Metric':        ['Precision','Recall','mAP50','mAP50-95'],
    'YOLOv5_val':    [yolov5_val[m]  for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8s_val':   [yolov8s_val[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8n_val':   [yolov8n_val[m] for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv5_test':   [y5_test[m]     for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8s_test':  [y8s_test[m]    for m in ['Precision','Recall','mAP50','mAP50-95']],
    'YOLOv8n_test':  [y8n_test[m]    for m in ['Precision','Recall','mAP50','mAP50-95']],
})
# === Save summary ===
out_dir = '../YOLO/summary'
os.makedirs(out_dir, exist_ok=True)
comparison_df.to_csv(os.path.join(out_dir,'yolov_comparison_metrics_summary.csv'), index=False)

# === Plot comparison ===
x = np.arange(len(comparison_df))
w = 0.12
positions = np.linspace(-2.5*w, 2.5*w, 6)
labels = ['v5_val','v8s_val','v8n_val','v5_test','v8s_test','v8n_test']
cols   = ['YOLOv5_val','YOLOv8s_val','YOLOv8n_val','YOLOv5_test','YOLOv8s_test','YOLOv8n_test']

fig, ax = plt.subplots(figsize=(14,6))
for i, col in enumerate(cols):
    ax.bar(x + positions[i], comparison_df[col], w, label=labels[i], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Metric'])
ax.set_ylabel('Score')
ax.set_title('YOLOv5 vs YOLOv8s vs YOLOv8n: Validation & Test Metrics')
ax.legend(ncol=2, fontsize=8)

for rect in ax.patches:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.005, f'{h:.2f}',
            ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(out_dir,'yolov_comparison_metrics_barplot.png'))
plt.show()

# === Create overlays with legend ===
out_vis = '../YOLO/summary/overlays'
os.makedirs(out_vis, exist_ok=True)

for name in gt:
    # find image file
    img_file = None
    for ext in ('.jpg','.jpeg','.png','.tif'):
        p = os.path.join(test_img_dir, name + ext)
        if os.path.isfile(p):
            img_file = p
            break
    if img_file is None:
        print(f'Warning: no image for {name}, skipping')
        continue

    img_bgr = cv2.imread(img_file)
    # draw legend
    cv2.putText(img_bgr, 'Legend:',       (20,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),   1)
    cv2.putText(img_bgr, 'GT (Blue)',     (20,50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(img_bgr, 'v5 (Red)',      (20,70),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img_bgr, 'v8s (Green)',   (20,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img_bgr, 'v8n (Yellow)',  (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(img_rgb)
    ax = plt.gca()

    # draw boxes
    for b in gt[name]:
        ax.add_patch(plt.Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, edgecolor='blue', linewidth=2))
    for item in pred5.get(name, []):
        box = item[0] if isinstance(item,(list,tuple)) and len(item)==2 else item
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1],
                                   fill=False, edgecolor='red', linewidth=2))
    for item in pred8s.get(name, []):
        box = item[0] if isinstance(item,(list,tuple)) and len(item)==2 else item
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1],
                                   fill=False, edgecolor='green', linewidth=2))
    for item in pred8n.get(name, []):
        box = item[0] if isinstance(item,(list,tuple)) and len(item)==2 else item
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1],
                                   fill=False, edgecolor='yellow', linewidth=2))

    plt.title(name)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_vis, f'{name}_overlay.png'))
    plt.close()
