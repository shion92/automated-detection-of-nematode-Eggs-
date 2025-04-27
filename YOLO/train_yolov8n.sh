# === Train YOLOv8n on your annotated dataset (optimised settings) ===
# Make sure to run this script from your YOLO folder
cd /Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/YOLO

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=YOLO name=nematode_yolov8n_train

# === Evaluate the best.pt after training ===
yolo task=detect mode=val model=nematode_yolov8n_train/weights/best.pt data=data.yaml project=nematode_yolov8n name=val save_json=True verbose=True save_txt=True

# === Predict on new test images ===
yolo task=detect mode=predict model=nematode_yolov8n_train/weights/best.pt source=../dataset/test/images project=nematode_yolov8n name=test save_json=True save_txt=True save_conf=True verbose=True
