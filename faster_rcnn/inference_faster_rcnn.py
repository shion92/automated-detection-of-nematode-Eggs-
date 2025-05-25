
import os
import random
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights # use by fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet34_Weights


import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from tqdm import tqdm # progress bars
import logging

from faster_rcnn import predict_and_save, NUM_CLASSES, SAVE_DIR, DEVICE

# -------------------------
# Configuration
# -------------------------
backbone_name = "resnet50"  # or "resnet34"

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.makedirs("Log", exist_ok=True)
    logging.basicConfig(
        filename="Log/faster_rcnn_training_v2.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    start = time.time()
    logging.info("=== Starting Faster R-CNN inference on all saved models ===")

    # Path to your model directory
    model_dir = SAVE_DIR  # e.g., "model/resnet50"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        # Extract lr from filename if present
        if "lr_" in model_file:
            lr_str = model_file.split("lr_")[1].split("_best")[0]
            try:
                lr = float(lr_str)
            except ValueError:
                lr = lr_str  # fallback if not float
        else:
            lr = "unknown"

        # Load model
        if backbone_name == "resnet50":
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
        elif backbone_name == "resnet34":
            backbone_net = resnet_fpn_backbone('resnet34', pretrained=True)
            model = FasterRCNN(backbone_net)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'resnet50' or 'resnet34'.")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        logging.info(f"=== Running inference for model: {model_file} (lr={lr}) ===")
        for split in ["test", "val", "train"]:
            logging.info(f"Running inference on {split} set...")
            predict_and_save(model, split=split, lr=lr)

    logging.info(f"\n✅ All inference runs completed in {time.time() - start:.2f} seconds.")