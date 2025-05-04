import time
import glob
from PIL import Image
import torch
from torchvision import transforms

# fixed imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from ultralytics import YOLO

# ─── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES    = 2   # 1 class + background
FR_MODEL_PATH  = 'faster_rcnn_nematode_best.pth'
YOLO_PATHS = {
    # 'YOLOv5s': 'YOLO/nematode_yolov5s_train/weights/best.pt',
    'YOLOv8s': 'YOLO/nematode_yolov8s_train2/weights/best.pt',
    'YOLOv8n': 'YOLO/nematode_yolov8n_train/weights/best.pt'
}

# ─── Prepare test images ─────────────────────────────────────────────────────────
test_images = glob.glob('dataset/test/images/*.tif')
if not test_images:
    raise RuntimeError("No test images found in dataset/test/images")

# ─── Image transform ────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ─── Benchmark Faster R-CNN ──────────────────────────────────────────────────────
print("Loading Faster R-CNN…")
fr_model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
in_feats = fr_model.roi_heads.box_predictor.cls_score.in_features
fr_model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
fr_model.load_state_dict(torch.load(FR_MODEL_PATH, map_location=DEVICE))
fr_model.to(DEVICE).eval()

# warm-up
_ = fr_model([tf(Image.open(test_images[0]).convert('RGB')).to(DEVICE)])

times = []
for path in test_images:
    img = Image.open(path).convert('RGB')
    inp = tf(img).to(DEVICE)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = fr_model([inp])
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    times.append(time.time() - start)

avg = sum(times) / len(times)
print(f"Faster R-CNN → {1/avg:.1f} FPS, {avg*1000:.1f} ms/img")

# ─── Benchmark YOLO variants ────────────────────────────────────────────────────
for name, weights in YOLO_PATHS.items():
    print(f"\nLoading {name}…")
    yolo = YOLO(weights)
    # warm-up
    _ = yolo(test_images[0])

    times = []
    for path in test_images:
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        st = time.time()
        _ = yolo(path)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        times.append(time.time() - st)

    avg = sum(times) / len(times)
    print(f"{name} → {1/avg:.1f} FPS, {avg*1000:.1f} ms/img")
