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

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased contrast enhancement
enhanced_gray = clahe.apply(gray)
plt.subplot(3, 3, 2)
plt.imshow(enhanced_gray, cmap="gray")
plt.title("CLAHE Enhanced")

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 2)  # Tuned to reduce noise while preserving edges
plt.subplot(3, 3, 3)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blur")

# Apply Adaptive Thresholding with modified parameters
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 8  # Increased C for better separation
)
plt.subplot(3, 3, 4)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Thresholding")

# Apply Canny edge detection to enhance contour visibility
edges = cv2.Canny(blurred, 100, 250)  # Increased lower threshold to reduce false edges
plt.subplot(3, 3, 5)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")

# Combine adaptive thresholding and edge detection results
combined_thresh = cv2.bitwise_and(adaptive_thresh, edges)  # Using AND to reduce noise further
plt.subplot(3, 3, 6)
plt.imshow(combined_thresh, cmap="gray")
plt.title("Combined Threshold")

# Perform morphological operations to refine the detected regions
kernel = np.ones((5, 5), np.uint8)  # Slightly larger kernel for stronger processing
cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)  # Further noise suppression
plt.subplot(3, 3, 7)
plt.imshow(cleaned, cmap="gray")
plt.title("Morph Close")

cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=3)  # Stronger artifact removal
plt.subplot(3, 3, 8)
plt.imshow(cleaned, cmap="gray")
plt.title("Morph Open")

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering criteria
filtered_contours = []
min_area = 500  # Increased minimum area to ignore small debris
max_area = 10000  # Adjusted to capture full eggs without noise
aspect_ratio_threshold = 1.5  # Increased to filter out non-elongated noise
solidity_threshold = 0.85  # Increased to ensure only well-defined objects are detected

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
