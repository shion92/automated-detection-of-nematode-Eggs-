# train YOLOv8 on your annotated dataset using the Ultralytics YOLO library (which supports YOLOv5–v8 models) 
# pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=200,
    imgsz=608,
    batch=16,
    lr0=0.01,
    patience=30,
    degrees=10,
    translate=0.1,
    scale=0.2,
    shear=2,
    perspective=0.0005,
    mosaic=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=1.0,
    flipud=0.5,
    cutout=0.3,  
    project="model/YOLO",
    name="yolov8s_default_cutout"
)

