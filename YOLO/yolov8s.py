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
    # {"name": "yolov8s_default"},

    # # No mosaic
    # {"name": "yolov8s_default_xmosaic", "mosaic": 0},

    # No mosaic + cutout + flips https://docs.ultralytics.com/guides/yolo-data-augmentation/#auto-augment-auto_augment
    # {"name": "yolov8s_default_xmosaic_cutout", "mosaic": 0, "erasing": 0.6, "fliplr": 1.0, "flipud": 0.5},

    # # SGD variants
    # {"name": "yolov8s_sgd_lr001", "optimizer": "SGD", "lr0": 0.01},
    # {"name": "yolov8s_sgd_lr0005", "optimizer": "SGD", "lr0": 0.005},
    # {"name": "yolov8s_sgd_lr0001", "optimizer": "SGD", "lr0": 0.001},  # best 
    
    # {"name": "yolov8s_sgd_lr0001_xmosaic_cutout", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.6}, # best 
    # {"name": "yolov8s_sgd_lr0001_xmosaic_cutout_3", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.6, "fliplr": 1.0, "flipud": 0.5}, # best 
    # {"name": "yolov8s_sgd_lr0001_xmosaic_cutout_4", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.5, "fliplr": 1.0, "flipud": 0.5}, # best 
    # {"name": "yolov8s_sgd_lr0001_xmosaic_cutout_degree_90", "optimizer": "SGD", "lr0": 0.001, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5}, 
    
     # SGD variants - strong mix 
    {"name": "yolov8s_sgd_lr0001_xmosaic_cutout_max", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "epochs": 300, "patience": 50}, 
    {"name": "yolov8s_sgd_lr0001_xmosaic_cutmix_eras", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5,  "epochs": 300, "patience": 50},  
    {"name": "yolov8s_sgd_lr0001_max", "optimizer": "SGD", "lr0": 0.001, "mosaic": 1, "erasing": 0.5, "fliplr": 1.0, "flipud": 0.5, "epochs": 300, "patience": 50},   
    {"name": "yolov8s_adam_lr0001_xmosaic", "optimizer": "Adam", "lr0": 0.001, "mosaic": 0, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5}, 
    
    
    
    # # Adam variants
    # {"name": "yolov8s_adam_lr001", "optimizer": "Adam", "lr0": 0.01, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 0},
    # {"name": "yolov8s_adam_lr0005", "optimizer": "Adam", "lr0": 0.005, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 0},
    # {"name": "yolov8s_adam_lr0001", "optimizer": "Adam", "lr0": 0.001, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 0}, # best 
    
    # {"name": "yolov8s_adam_lr0012", "optimizer": "Adam", "lr0": 0.01, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 90},
    # {"name": "yolov8s_adam_lr00052", "optimizer": "Adam", "lr0": 0.005, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 90},
    # {"name": "yolov8s_adam_lr00012", "optimizer": "Adam", "lr0": 0.001, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "degrees": 90}, # best

    # # AdamW variants
    # {"name": "yolov8s_adamw_lr001", "optimizer": "AdamW", "lr0": 0.01, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5},
    # {"name": "yolov8s_adamw_lr0005", "optimizer": "AdamW", "lr0": 0.005, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5},
    # {"name": "yolov8s_adamw_lr0001", "optimizer": "AdamW", "lr0": 0.001, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5}, # best
    
    
     # === YOLOv8m (larger object detection model) ===
    {"name": "yolov8m_sgd_lr0001", "model": "yolov8m.pt", "optimizer": "SGD", "lr0": 0.001, "mosaic": 1, "erasing": 0.5, "fliplr": 1.0, "flipud": 0.5, "epochs": 300, "patience": 50},
    {"name": "yolov8m_sgd_lr0001_max", "model": "yolov8m.pt", "optimizer": "SGD", "lr0": 0.001, "mosaic": 1, "erasing": 0.8, "fliplr": 1.0, "flipud": 0.5, "epochs": 300, "patience": 50},
    
    # === YOLOv8-seg for instance segmentation ===
    {"name": "yolov8s_seg_lr0001", "model": "yolov8s-seg.pt", "data": "data_seg.yaml", "task": "segment", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.0, "fliplr": 1.0, "flipud": 0.5, "epochs": 200, "patience": 30},
    {"name": "yolov8s_seg_lr0001_eras", "model": "yolov8s-seg.pt", "data": "data_seg.yaml", "task": "segment", "optimizer": "SGD", "lr0": 0.001, "mosaic": 0, "erasing": 0.5, "fliplr": 1.0, "flipud": 0.5, "epochs": 200, "patience": 30}
    
    
]

COMMON_ARGS = {
    "model": "YOLO/yolov8s.pt",
    "data": "data.yaml",
    "task": "detect",
    "epochs": 2,
    "imgsz": 608,
    "batch": 16,
    "patience": 30,
    "degrees": 90,
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

    # Return trained model weight path
    return os.path.join(args["project"], args["name"], "weights", "best.pt")
# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(weight_path: str, config_name: str,  task: str):
    model = YOLO(weight_path)
    model.val(
        data=COMMON_ARGS["data"],
        task=task,
        project=f"Processed_Images/YOLO/{config_name}",
        name="val",
        save_json=True,
        verbose=True,
        save_txt=True
    )

# -------------------------
# Prediction Function
# -------------------------
def predict_model(weight_path: str, config_name: str, task: str):
    model = YOLO(weight_path)
    model.predict(
        source="dataset/test/images",
        task=task,
        project=f"Processed_Images/YOLO/{config_name}",
        name="test",
        save_json=True,
        save_txt=True,
        save_conf=True,
        verbose=True
    )


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
        config_name = config["name"]
        task = config.get("task", "detect") 
        
        logging.info(f"Training config: {config_name}")
        weight_path = train_model(config.copy())
        logging.info(f"Finished training: {config_name}")

        logging.info(f"Evaluating model: {config_name}")
        evaluate_model(weight_path, config_name, task)

        logging.info(f"Predicting test images for: {config_name}")
        predict_model(weight_path, config_name, task)

    total_time = time.time() - start
    logging.info(f"\n✅ All runs complete in {total_time:.1f} seconds")
