
# ===  Train the model ===
yolo task=detect mode=train model=YOLO/YOLOv5s.pt data=YOLO/data.yaml epochs=200 imgsz=416 augment=True lr0=0.001 project=YOLO name=train
# NOTES:
# - yolov8s.pt = good lightweight model for microscope tasks
# - augment=True = helps model generalise
# - imgsz=416 = better for small objects
# - lr0=0.001 = standard; lower if model oscillates

# ===  Evaluate the model  ===
yolo task=detect mode=val model=YOLO/runs/detect/train/weights/best.pt data=data.yaml

# ===  Predict on training images (Useful for checking model overfitting) ===
# yolo task=detect mode=predict model=YOLO/runs/detect/train3/weights/best.pt source=images/train/
yolo task=detect mode=predict model=YOLO/runs/detect/train/weights/best.pt source=dataset/train/

#  ===  Predict new images  ===
yolo task=detect mode=predict model=YOLO/runs/detect/train/weights/best.pt source=dataset/test/