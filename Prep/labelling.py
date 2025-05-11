# Use LabelImg for Interactive Adjustment
# labeling images with labelImg
pip install labelImg
labelImg

# Steps to Adjust Bounding Boxes in LabelImg
# Open LabelImg.
# Click "Open Dir", and select the folder containing your images.
# Click "Change Save Dir", and select the folder containing the .txt annotations.
# Click "Open Next Image" (or manually select an image).
# The auto-detected bounding boxes should appear.
# Adjust bounding boxes manually:
# Click and drag to resize or move boxes.
# Delete incorrect boxes using the "Delete" button.
# Add new boxes using the "Create RectBox" button.
# Click "Save" after each image adjustment.
# Move to the next image and repeat.

# activate tensorboard
pip install tensorboard
# start tensorboard
# in terminal http://localhost:6006 
tensorboard --logdir=runs
