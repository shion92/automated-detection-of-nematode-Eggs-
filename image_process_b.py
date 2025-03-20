import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the microscope image
image_path = "Data/Image_05.tif"  
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Remove extreme dark high-contrast objects (e.g., frames, bubbles, and stripes)
_, dark_contrast_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 2: Apply morphological opening to refine dark object removal
kernel = np.ones((6, 6), np.uint8)
dark_contrast_mask = cv2.morphologyEx(dark_contrast_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 3: Apply mask to remove dark objects while keeping fine structures
filtered_gray = cv2.bitwise_and(gray, gray, mask=dark_contrast_mask)

# Step 4: Apply CLAHE to enhance contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(filtered_gray)

# Step 5: Apply Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 10
)

# Step 6: Apply Canny edge detection to enhance contour visibility
edges = cv2.Canny(enhanced_gray, 80, 200)

# Step 7: Combine adaptive thresholding and edge detection results
combined_thresh = cv2.bitwise_or(adaptive_thresh, edges)

# Step 8: Perform morphological operations to refine detected regions
cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=4)

# Step 9: Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 10: Filter contours based on size, aspect ratio, and solidity
filtered_contours = []
min_area = 900
max_area = 12000
aspect_ratio_threshold = 1.3
solidity_threshold = 0.90

for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area <= area <= max_area and len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (major_axis, minor_axis), angle = ellipse
        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

        # Compute convex hull solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if aspect_ratio > aspect_ratio_threshold and solidity > solidity_threshold:
            filtered_contours.append(cnt)

# Step 11: Draw filtered contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Step 12: Display all processing steps
fig, axes = plt.subplots(3, 4, figsize=(15, 12))

axes[0, 0].imshow(gray, cmap="gray")
axes[0, 0].set_title("1. Grayscale Image")

axes[0, 1].imshow(dark_contrast_mask, cmap="gray")
axes[0, 1].set_title("2. Dark Object Removal")

axes[0, 2].imshow(filtered_gray, cmap="gray")
axes[0, 2].set_title("3. Filtered Gray Image")

axes[0, 3].imshow(enhanced_gray, cmap="gray")
axes[0, 3].set_title("4. CLAHE Enhanced")

axes[1, 0].imshow(adaptive_thresh, cmap="gray")
axes[1, 0].set_title("5. Adaptive Thresholding")

axes[1, 1].imshow(edges, cmap="gray")
axes[1, 1].set_title("6. Canny Edge Detection")

axes[1, 2].imshow(combined_thresh, cmap="gray")
axes[1, 2].set_title("7. Combined Thresholding + Edges")

axes[1, 3].imshow(cleaned, cmap="gray")
axes[1, 3].set_title("8. Morphological Processing")

# Convert contours to binary for visualization
contour_image = np.zeros_like(gray)
cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
axes[2, 0].imshow(contour_image, cmap="gray")
axes[2, 0].set_title("9. Contour Detection")

filtered_contour_image = np.zeros_like(gray)
cv2.drawContours(filtered_contour_image, filtered_contours, -1, 255, thickness=1)
axes[2, 1].imshow(filtered_contour_image, cmap="gray")
axes[2, 1].set_title("10. Filtered Contours")

axes[2, 2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title(f"11. Final Detection: {len(filtered_contours)} Eggs")

plt.tight_layout()
plt.show()
