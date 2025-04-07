import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Setup paths ===
image_folder = "images/train/"
processed_dir = "Processed_Images"
steps_dir = os.path.join(processed_dir, "Steps")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(steps_dir, exist_ok=True)

# Gather .tif images from folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith('.tif')]

def optimized_process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Apply Non-Local Means Denoising
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Gaussian Blur
    blurred_gray = cv2.GaussianBlur(denoised_gray, (7, 7), 0)

    # Step 3: Global thresholding
    _, thresholded = cv2.threshold(blurred_gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Morphological opening
    morph_kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # Step 5: Canny edge detection
    edges = cv2.Canny(thresholded, 50, 100)

    # Step 6: Morphological closing to smooth the edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Step 7: Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Filter contours
    filtered_contours = []
    min_area = 600
    max_area = 12000
    aspect_ratio_threshold = 1.2
    solidity_threshold = 0.4

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

    # Step 9: Draw filtered contours
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Step 10: Save final processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(processed_dir, f"processed_{filename}")
    cv2.imwrite(output_path, output_image)

    # Step 11: Visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("1. Grayscale Image")

    axes[0, 1].imshow(denoised_gray, cmap="gray")
    axes[0, 1].set_title("2. Denoised (fastNlMeans)")

    axes[0, 2].imshow(blurred_gray, cmap="gray")
    axes[0, 2].set_title("3. Gaussian Blurred")

    axes[1, 0].imshow(thresholded, cmap="gray")
    axes[1, 0].set_title("4. Thresholded + Morph Open")

    axes[1, 1].imshow(edges, cmap="gray")
    axes[1, 1].set_title("5. Canny Edges")

    axes[1, 2].imshow(edges_closed, cmap="gray")
    axes[1, 2].set_title("6. Smoothed Edges (Morph Close)")

    contour_image = np.zeros_like(gray)
    cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
    axes[2, 0].imshow(contour_image, cmap="gray")
    axes[2, 0].set_title("7. Contour Detection")

    filtered_contour_image = np.zeros_like(gray)
    cv2.drawContours(filtered_contour_image, filtered_contours, -1, 255, thickness=1)
    axes[2, 1].imshow(filtered_contour_image, cmap="gray")
    axes[2, 1].set_title("8. Filtered Contours")

    axes[2, 2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title(f"9. Final Detection: {len(filtered_contours)} Eggs")

    plt.tight_layout()
    plt.show()

# Process all images
for image_file in image_files:
    optimized_process_image(image_file)
