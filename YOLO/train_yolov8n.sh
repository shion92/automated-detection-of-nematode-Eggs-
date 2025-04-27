# === Train YOLOv8s on your annotated dataset (optimised settings) ===
# === Make sure to run this script in the YOLO directory ===
cd /Users/shion/Desktop/COMP693/automated-detection-of-nematode-Eggs-/YOLO

yolo task=detect mode=train model=YOLO/yolov8s.pt data=data.yaml epochs=200 imgsz=608 batch=16 lr0=0.01 patience=30 degrees=10 translate=0.1 scale=0.2 shear=2 perspective=0.0005 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 project=YOLO name=nematode_yolov8s_train

# === Evaluate the best.pt after training ===
yolo task=detect mode=val model=nematode_yolov8s_train2/weights/best.pt data=data.yaml project=../YOLO/nematode_yolov8s name=val save_json=True verbose=True


# === Predict on new test images ===
yolo task=detect mode=predict model=nematode_yolov8s_train2/weights/best.pt source=../dataset/test/images project=../YOLO/nematode_yolov8s name=test save_json=True save_txt=True save_conf=True verbose=True


