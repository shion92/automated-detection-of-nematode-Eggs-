import time
import glob
from PIL import Image
import torch
from torchvision import transforms

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)

from ultralytics import YOLO

from pathlib import Path
from typing import List, Tuple, Dict
import colorsys, json, cv2, os

# -------------------------
# Configuration
# -------------------------
class ModelConfig:
    """Configuration for different model types"""
    def __init__(
        self,
        model_name: str,
        model_type: str,
        pred_folder: str,
        gt_folder: str,
        gt_format: str = "yolo",
        weight_path: str = "",
    ):
        self.model_name = model_name
        self.model_type = model_type  # 'yolo' or 'faster_rcnn'
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.gt_format = gt_format
        self.weight_path = weight_path  # path to .pt or .pth


# We will benchmark on CPU and, if available, on MPS (Apple GPU)
DEVICE_CPU = torch.device("cpu")
DEVICE_MPS = torch.device("mps") if torch.backends.mps.is_available() else None

# List of model configurations with their weight paths
MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        model_name="YOLOv8s-max",
        model_type="yolo",
        pred_folder="Processed_Images/YOLO/yolov8s_sgd_lr0001_max/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo",
        weight_path="model/YOLO/yolov8s_sgd_lr0001_max/weights/best.pt",
    ),
    ModelConfig(
        model_name="YOLOv8m",
        model_type="yolo",
        pred_folder="Processed_Images/YOLO/yolov8m_sgd_lr0001/test/labels",
        gt_folder="dataset/test/labels",
        gt_format="yolo",
        weight_path="model/YOLO/yolov8m_sgd_lr0001/weights/best.pt",
    ),
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.005",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.005/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc",
        weight_path="model/faster_rcnn/resnet50/faster_rcnn_resnet50_lr_0.005_best.pth",
    ),
    ModelConfig(
        model_name="Faster-RCNN-resnet50-lr0.001",
        model_type="faster_rcnn",
        pred_folder="Processed_Images/faster_rcnn_resnet50/Predictions/lr_0.001/test",
        gt_folder="dataset/test/annotations",
        gt_format="pascal_voc",
        weight_path="model/faster_rcnn/resnet50/faster_rcnn_resnet50_lr_0.01_best.pth",
    ),
]

test_images = glob.glob("dataset/val/images/*.tif")
if not test_images:
    raise RuntimeError("No test images found in dataset/val/images")

tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_faster_rcnn(model, device: torch.device, images: List[str], tf):
    model.to(device).eval()
    # Warm-up
    img0 = Image.open(images[0]).convert("RGB")
    inp0 = tf(img0).to(device)
    with torch.no_grad():
        _ = model([inp0])
    sync_device(device)

    times: List[float] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        inp = tf(img).to(device)
        sync_device(device)
        start = time.time()
        with torch.no_grad():
            _ = model([inp])
        sync_device(device)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    return avg


def benchmark_yolo(weight_path: str, device: torch.device, images: List[str]):
    # YOLO constructor does not take `device=`; we move after
    yolo = YOLO(weight_path)
    yolo.to(device)

    # Warm-up
    _ = yolo(images[0])
    sync_device(device)

    times: List[float] = []
    for path in images:
        sync_device(device)
        st = time.time()
        _ = yolo(path)
        sync_device(device)
        times.append(time.time() - st)

    avg = sum(times) / len(times)
    return avg


# Benchmark each model in MODEL_CONFIGS on CPU and (if available) MPS
for cfg in MODEL_CONFIGS:
    # CPU benchmark
    if cfg.model_type == "faster_rcnn":
        print(f"\n{cfg.model_name} on CPU:")
        # Load and prepare the model (CPU)
        fr_cpu = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_feats = fr_cpu.roi_heads.box_predictor.cls_score.in_features
        fr_cpu.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES := 2)
        fr_cpu.load_state_dict(
            torch.load(cfg.weight_path, map_location="cpu")
        )
        avg_cpu = benchmark_faster_rcnn(fr_cpu, DEVICE_CPU, test_images, tf)
        print(f"  CPU → {1/avg_cpu:.1f} FPS, {avg_cpu*1000:.1f} ms/img")

        # if DEVICE_MPS is not None:
        #     print(f"{cfg.model_name} on MPS:")
        #     fr_mps = fasterrcnn_resnet50_fpn(
        #         weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        #     )
        #     fr_mps.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
        #     fr_mps.load_state_dict(
        #         torch.load(cfg.weight_path, map_location="cpu")
        #     )
        #     avg_mps = benchmark_faster_rcnn(fr_mps, DEVICE_MPS, test_images, tf)
        #     print(f"  MPS → {1/avg_mps:.1f} FPS, {avg_mps*1000:.1f} ms/img")

    elif cfg.model_type == "yolo":
        print(f"\n{cfg.model_name} on CPU:")
        avg_cpu = benchmark_yolo(cfg.weight_path, DEVICE_CPU, test_images)
        print(f"  CPU → {1/avg_cpu:.1f} FPS, {avg_cpu*1000:.1f} ms/img")

        if DEVICE_MPS is not None:
            print(f"{cfg.model_name} on MPS:")
            avg_mps = benchmark_yolo(cfg.weight_path, DEVICE_MPS, test_images)
            print(f"  MPS → {1/avg_mps:.1f} FPS, {avg_mps*1000:.1f} ms/img")
