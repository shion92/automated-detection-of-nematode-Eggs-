import os
print(os.path.abspath("Data"))

# to train the model 
# yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=150 imgsz=416 augment=True lr0=0.001

# evaluate the model
# yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

# detect the model 
yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt source=images/train/


# predict the new image
yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt data=data.yaml source=images/image_06.tif