# === Train YOLOv8s on your annotated dataset (optimised settings) ===
# === Install YOLOv8 if not already installed ===
# pip install ultralytics
# === Create a data.yaml file with your dataset information ===
# Example data.yaml file:
# train: ../dataset/train/images
# val: ../dataset/val/images
# nc: 1
# names: ['nematode_egg']  # Replace with your class names

# cd /Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/YOLO

yolo task=detect mode=train model=YOLO/yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_train_default

# SGD 
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 optimizer=SGD patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_sgd_lr001
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.005 optimizer=SGD patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_sgd_lr0005
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.001 optimizer=SGD patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_sgd_lr0001

# Adam with default lr0=0.01
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 optimizer=Adam patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adam_lr001
# Adam with lr0=0.005
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.005 optimizer=Adam patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adam_lr0005
# Adam with lr0=0.001
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.001 optimizer=Adam patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adam_lr0001

# AdamW
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 optimizer=AdamW patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adamw_lr001
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.005 optimizer=AdamW patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adamw_lr0005
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.001 optimizer=AdamW patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8s_adamw_lr0001

# === View training and validation loss curves ===
# tensorboard --logdir YOLO/nematode_yolov8s_train or open YOLO/nematode_yolov8s_train/results.png

# === Evaluate the best.pt after training ===
yolo task=detect mode=val model=nematode_yolov8s_train2/weights/best.pt data=data.yaml project=../YOLO/nematode_yolov8s name=val save_json=True verbose=True


# === Predict on new test images ===
yolo task=detect mode=predict model=nematode_yolov8s_train2/weights/best.pt source=../dataset/test/images project=../YOLO/nematode_yolov8s name=test save_json=True save_txt=True save_conf=True verbose=True


