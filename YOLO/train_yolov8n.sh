# === Train YOLOv8n on your annotated dataset (optimised settings) ===
# cd /Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/YOLO

# === Install YOLOv8 if not already installed ===
# pip install ultralytics
# === Create a data.yaml file with your dataset information ===
# Example data.yaml file:
# train: ../dataset/train/images
# val: ../dataset/val/images
# nc: 1
# names: ['nematode_egg']  # Replace with your class names

# === Train YOLOv8n with default settings ===
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8n_default
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8n_default_xmosaic

# === Train YOLOv8n with different optimizers and learning rates ===

# AdamW with default lr0=0.01
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=AdamW project=model/YOLO name=yolov8n_adamw_lr001
# AdamW with lr0=0.005
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=AdamW lr0=0.005 project=model/YOLO name=yolov8n_adamw_lr0005
# AdamW with lr0=0.001
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=AdamW lr0=0.001 project=model/YOLO name=yolov8n_adamw_lr0001

# Adam with default lr0=0.01
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=Adam project=model/YOLO name=yolov8n_adam_lr001  
# Adam with lr0=0.005
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=Adam lr0=0.005 project=model/YOLO name=yolov8n_adam_lr0005
# Adam with lr0=0.001
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 translate=0.1 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 optimizer=Adam lr0=0.001 project=model/YOLO name=yolov8n_adam_lr0001
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=model/YOLO name=yolov8n

# SGD with lr0=0.01
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4  optimizer=SGD lr0=0.01 project=model/YOLO name=yolov8n_sgd_lr001

# SGD with lr0=0.005
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4  optimizer=SGD lr0=0.005 project=model/YOLO name=yolov8n_sgd_lr0005

# SGD with lr0=0.001
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 patience=30 degrees=10 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4  optimizer=SGD lr0=0.001 project=model/YOLO name=yolov8n_sgd_lr0001

# === View training and validation loss curves ===
# tensorboard --logdir YOLO/nematode_yolov8s_train or open YOLO/nematode_yolov8s_train/results.png

# === Evaluate the best.pt after training ===
yolo task=detect mode=val model=model/YOLO/yolov8n/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n name=val save_json=True verbose=True save_txt=True
# === Evaluate the best.pt after training with different optimizers ===     
yolo task=detect mode=val model=model/YOLO/yolov8n_adamw_lr001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adamw name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_adamw_lr0005/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adamw_lr0005 name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_adamw_lr0001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adamw_lr0001 name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_adam_lr0001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adam name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_adam_lr0005/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adam_lr0005 name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_adam_lr0001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_adam_lr0001 name=val save_json=True verbose=True save_txt=True

# SGD with lr0=0.01
yolo task=detect mode=val model=model/YOLO/yolov8n_sgd_lr001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_sgd_lr001 name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_sgd_lr0005/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_sgd_lr0005 name=val save_json=True verbose=True save_txt=True
yolo task=detect mode=val model=model/YOLO/yolov8n_sgd_lr0001/weights/best.pt data=data.yaml project=Processed_Images/YOLO/yolov8n_sgd_lr0001 name=val save_json=True verbose=True save_txt=True


# === Predict on new test images ===
yolo task=detect mode=predict model=model/YOLO/yolov8n/weights/best.pt source=../dataset/test/images project=Processed_Images/YOLO/yolov8n name=test save_json=True save_txt=True save_conf=True verbose=True
