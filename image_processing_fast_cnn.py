import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define image file paths
image_files = [
    "Data/Image_34.tif",
    "Data/Image_50.tif",
    "Data/Image_54.tif",
    "Data/Image_56.tif",
    "Data/Image_62.tif"
]

# Directory to save CNN-ready processed images
os.makedirs("CNN_Ready_Images", exist_ok=True)

def preprocess_for_fastrcnn(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (600, 600))  # Resize for CNNs
    normalized = resized.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Save image as float32 (in .npy format, or convert back to uint8 if saving as image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"CNN_Ready_Images/{base_name}.png"
    cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))  # Save for visual inspection

    return normalized  # Return for feeding to your CNN model or dataset class

# Process all images
for image_file in image_files:
    preprocess_for_fastrcnn(image_file)
