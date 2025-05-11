import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# === Setup paths ===
split = "train"
image_folder = f"dataset/{split}/images"
processed_dir = "Processed_Images/with_fastNIMeansDenoising/"
steps_dir = os.path.join(processed_dir, "Steps")
pred_json_dir = os.path.join(processed_dir, "Predictions", split)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(steps_dir, exist_ok=True)
os.makedirs(pred_json_dir, exist_ok=True)

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
    _, thresholded_raw = cv2.threshold(blurred_gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Morphological opening (refined)
    morph_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded_raw, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # Step 4.5: Remove small debris using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_egg_area = 600
    cleaned = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_egg_area:
            cleaned[labels == i] = 255
    thresholded = cleaned

    # Step 5: Canny edge detection
    edges = cv2.Canny(thresholded, 50, 100)

    # Step 6: Morphological closing to smooth the edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Step 7: Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Filter contours based on oval geometry
    filtered_contours = []
    predicted_boxes = []
    min_area = 600
    max_area = 12000
    min_aspect_ratio = 1.3
    max_aspect_ratio = 2.5
    solidity_threshold = 0.85

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

            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and solidity >= solidity_threshold:
                filtered_contours.append(cnt)
                x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                predicted_boxes.append([x_box, y_box, x_box + w_box, y_box + h_box])

    # Step 9: Draw filtered contours and overlay file name
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    filename = os.path.basename(image_path)
    cv2.putText(
        output_image,
        f"{filename}",
        org=(10, output_image.shape[0] - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    # Save prediction boxes as JSON
    json_path = os.path.join(pred_json_dir, f"{os.path.splitext(filename)[0]}.json")
    with open(json_path, "w") as jf:
        json.dump({"filename": filename, "boxes": predicted_boxes}, jf, indent=2)

    # Step 10: Save final processed image
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

    axes[1, 0].imshow(thresholded_raw, cmap="gray")
    axes[1, 0].set_title("4. Raw Thresholded")

    axes[1, 1].imshow(opened, cmap="gray")
    axes[1, 1].set_title("5. Morph Open (3x3)")

    axes[1, 2].imshow(thresholded, cmap="gray")
    axes[1, 2].set_title("6. Post-clean (by area)")

    axes[2, 0].imshow(edges_closed, cmap="gray")
    axes[2, 0].set_title("7. Edges + Morph Close")

    filtered_contour_image = np.zeros_like(gray)
    cv2.drawContours(filtered_contour_image, filtered_contours, -1, 255, thickness=1)
    axes[2, 1].imshow(filtered_contour_image, cmap="gray")
    axes[2, 1].set_title("8. Filtered Contours")

    axes[2, 2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title(f"9. Final Detection: {len(filtered_contours)} Eggs ({filename})")
    
    plt.tight_layout()
    step_image_path = os.path.join(steps_dir, f"steps_{filename}.png")
    plt.savefig(step_image_path, dpi=150, bbox_inches='tight')
    plt.close()

# Process all images
for image_file in image_files:
    optimized_process_image(image_file)
