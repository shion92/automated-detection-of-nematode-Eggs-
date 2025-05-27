# -------------------------
# Imports
# -------------------------
import os
import time
import logging
from ultralytics import YOLO

# -------------------------
# Configuration
# -------------------------
CONFIGS = [
    # # Default with mosaic
    # {"name": "yolov8s_default", "mosaic": 1.0},

    # # No mosaic
    # {"name": "yolov8s_default_xmosaic", "mosaic": 0},

    # No mosaic + cutout + flips https://docs.ultralytics.com/guides/yolo-data-augmentation/#auto-augment-auto_augment
    {"name": "yolov8s_default_xmosaic_cutout", "mosaic": 0, "erasing": 0.6, "fliplr": 1.0, "flipud": 0.5},

    # # SGD variants
    # {"name": "yolov8s_sgd_lr001", "optimizer": "SGD", "lr0": 0.01},
    # {"name": "yolov8s_sgd_lr0005", "optimizer": "SGD", "lr0": 0.005},
    # {"name": "yolov8s_sgd_lr0001", "optimizer": "SGD", "lr0": 0.001},
    {"name": "yolov8s_sgd_lr0001_xmosaic_cutout", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.6},
    {"name": "yolov8s_sgd_lr0001_xmosaic_cutout", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.6, "fliplr": 1.0, "flipud": 0.5},

    # # Adam variants
    # {"name": "yolov8s_adam_lr001", "optimizer": "Adam", "lr0": 0.01},
    # {"name": "yolov8s_adam_lr0005", "optimizer": "Adam", "lr0": 0.005},
    # {"name": "yolov8s_adam_lr0001", "optimizer": "Adam", "lr0": 0.001},

    # # AdamW variants
    # {"name": "yolov8s_adamw_lr001", "optimizer": "AdamW", "lr0": 0.01},
    # {"name": "yolov8s_adamw_lr0005", "optimizer": "AdamW", "lr0": 0.005},
    # {"name": "yolov8s_adamw_lr0001", "optimizer": "AdamW", "lr0": 0.001},
]

COMMON_ARGS = {
    "model": "YOLO/yolov8s.pt",
    "data": "data.yaml",
    "epochs": 200,
    "imgsz": 608,
    "batch": 16,
    "patience": 30,
    "degrees": 10,
    "translate": 0.1,
    "scale": 0.2,
    "shear": 2,
    "perspective": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "project": "model/YOLO"
}

# -------------------------
# Training Function
# -------------------------
def train_model(config: dict):
    args = COMMON_ARGS.copy()
    args.update(config)
    model = YOLO(args["model"])
    model.train(**args)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.makedirs("Log", exist_ok=True)
    logging.basicConfig(
        filename="Log/yolov8s_training.log",
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
    logging.info("=== Starting YOLOv8s training runs... ===")

    for config in CONFIGS:
        logging.info(f"Training config: {config['name']}")
        train_model(config)
        logging.info(f"Finished training: {config['name']}")

    logging.info(f"\n✅ Done! Total runtime: {time.time() - start:.2f} seconds.")
