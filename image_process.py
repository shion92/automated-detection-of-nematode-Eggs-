import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the microscope image
image_path = "Data/Image_05.tif"  # Adjust the path if needed
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_gray = clahe.apply(gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Apply global thresholding first (removes unnecessary background variation)
_, global_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Apply adaptive thresholding for fine details
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
)

# Apply Canny edge detection to enhance contour visibility
edges = cv2.Canny(blurred, 75, 200)  # Increased thresholds to reduce noise artifacts

# Combine both thresholding results (to balance detail and noise removal)
combined_thresh = cv2.bitwise_and(global_thresh, adaptive_thresh)
# combined_thresh = cv2.bitwise_or(adaptive_thresh, edges)

# Perform morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering criteria
filtered_contours = []
min_area = 200  # Ignore small objects
max_area = 5000  # Ignore very large objects (e.g., frame artifacts)
aspect_ratio_threshold = 1.2  # Must be elongated 
solidity_threshold = 0.85  # Filter out hollow bubbles and irregular debris

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        continue  # Skip objects too small or too large

    if len(cnt) >= 5:  # At least 5 points needed to fit an ellipse
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (major_axis, minor_axis), angle = ellipse  # Get ellipse parameters
        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

        # Compute convex hull solidity (to eliminate bubbles and irregular debris)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if aspect_ratio > aspect_ratio_threshold and solidity > solidity_threshold:
            filtered_contours.append(cnt)

# Draw only filtered contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Display results
num_eggs = len(filtered_contours)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap="gray")
plt.title("Grayscale Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Nematode Eggs: {num_eggs}")

plt.show()