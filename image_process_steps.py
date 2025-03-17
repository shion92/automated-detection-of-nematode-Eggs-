import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the microscope image
image_path = "Data/Image_05.tif"  # Adjust the path if needed
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 5))
plt.subplot(3, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")

# Remove extreme dark high-contrast objects (e.g., frames, bubbles, and stripes) with refined Otsu-based thresholding
_, dark_contrast_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply morphological opening with an increased iteration count to further refine dark object removal
kernel = np.ones((5, 5), np.uint8)  # Increased kernel size for better dark region removal
dark_contrast_mask = cv2.morphologyEx(dark_contrast_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Apply mask to remove dark objects while keeping fine structures
filtered_gray = cv2.bitwise_and(gray, gray, mask=dark_contrast_mask)
plt.subplot(3, 3, 2)
plt.imshow(filtered_gray, cmap="gray")
plt.title("Filtered Dark-Contrast Objects")

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Adjusted clip limit for improved contrast enhancement
enhanced_gray = clahe.apply(filtered_gray)
plt.subplot(3, 3, 3)
plt.imshow(enhanced_gray, cmap="gray")
plt.title("CLAHE Enhanced")

# Apply Adaptive Thresholding with modified parameters for better contrast separation
adaptive_thresh = cv2.adaptiveThreshold(
    enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 10  # Adjusted block size and C value
)
plt.subplot(3, 3, 4)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Thresholding")

# Apply Canny edge detection to enhance contour visibility
edges = cv2.Canny(enhanced_gray, 80, 200)  # Adjusted thresholds for better edge detection
plt.subplot(3, 3, 5)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")

# Combine adaptive thresholding and edge detection results
combined_thresh = cv2.bitwise_or(adaptive_thresh, edges)  # Retains detailed edges while reducing noise further
plt.subplot(3, 3, 6)
plt.imshow(combined_thresh, cmap="gray")
plt.title("Combined Threshold")

# Perform morphological operations to refine the detected regions
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)  # Increased iterations for better noise suppression
plt.subplot(3, 3, 7)
plt.imshow(cleaned, cmap="gray")
plt.title("Morph Close")

cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=4)  # Increased iterations for stronger artifact removal
plt.subplot(3, 3, 8)
plt.imshow(cleaned, cmap="gray")
plt.title("Morph Open")

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering criteria
filtered_contours = []
min_area = 700  # Increased minimum area to ignore more small debris
max_area = 12000  # Ensuring full egg detection while filtering large artifacts
aspect_ratio_threshold = 1.3  # Adjusted to capture elongated eggs
solidity_threshold = 0.90  # Increased to remove irregular and hollow debris

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        continue

    if len(cnt) >= 5:  # At least 5 points needed to fit an ellipse
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (major_axis, minor_axis), angle = ellipse
        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

        # Compute convex hull solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if aspect_ratio > aspect_ratio_threshold and solidity > solidity_threshold:
            filtered_contours.append(cnt)

# Draw only filtered contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Display final detection
num_eggs = len(filtered_contours)
plt.subplot(3, 3, 9)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Nematode Eggs: {num_eggs}")

plt.tight_layout()
plt.show()
