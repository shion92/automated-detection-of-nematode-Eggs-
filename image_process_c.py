import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define image file paths
image_files = [
    "images/train/Image_01.tif",
    "images/train/Image_02.tif",
    "images/train/Image_04.tif",
    "images/train/Image_16.tif",
    "images/train/Image_17.tif",
    "images/train/Image_18.tif",
    "images/train/Image_41.tif",
    "images/train/Image_43.tif",
    "images/train/Image_46.tif",
    "images/train/Image_49.tif",
    "images/train/Image_55.tif",
    "images/train/Image_60.tif",
    "images/train/Image_67.tif",
    "images/train/Image_68.tif",

    
]

# Create directory to save outputs
os.makedirs("Processed_Images", exist_ok=True)

def optimized_process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Reduce background noise with Gaussian Blur
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 2: Apply global thresholding
    _, thresholded = cv2.threshold(blurred_gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Apply morphological opening
    morph_kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # Step 4: Edge detection
    edges = cv2.Canny(thresholded, 50, 100)

    # Step 5: Morphological closing
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Step 6: Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Filter contours
    filtered_contours = []
    min_area = 600  # Lowered to include smaller eggs
    max_area = 12000
    aspect_ratio_threshold = 1.2  # Slightly relaxed to allow more rounded eggs
    solidity_threshold = 0.4  # Lowered to allow more natural irregularities

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse
            aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if aspect_ratio > aspect_ratio_threshold and solidity > solidity_threshold:
                filtered_contours.append(cnt)

    # Step 8: Draw final contours
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Step 9: Save processed image for YOLO or other use
    filename = os.path.basename(image_path)
    output_path = os.path.join("Processed_Images", f"processed_{filename}")
    cv2.imwrite(output_path, output_image)

   
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("1. Grayscale Image")

    axes[0, 1].imshow(blurred_gray, cmap="gray")
    axes[0, 1].set_title("2. Gaussian Blurred")

    axes[0, 2].imshow(thresholded, cmap="gray")
    axes[0, 2].set_title("3. Thresholded + Morph Open")

    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("4. Canny Edges")

    axes[1, 1].imshow(edges_closed, cmap="gray")
    axes[1, 1].set_title("5. Smoothed Edges (Morph Close)")

    contour_image = np.zeros_like(gray)
    cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
    axes[1, 2].imshow(contour_image, cmap="gray")
    axes[1, 2].set_title("6. Contour Detection")

    filtered_contour_image = np.zeros_like(gray)
    cv2.drawContours(filtered_contour_image, filtered_contours, -1, 255, thickness=1)
    axes[2, 0].imshow(filtered_contour_image, cmap="gray")
    axes[2, 0].set_title("7. Filtered Contours")

    axes[2, 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title(f"8. Final Detection: {len(filtered_contours)} Eggs")

    plt.tight_layout()
    plt.show()

# Process each image and output results
for image_file in image_files:
    optimized_process_image(image_file)