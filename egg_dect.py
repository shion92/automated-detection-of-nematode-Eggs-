import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the microscope image
image_path = "Data/Image_04.tif"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Perform morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(
    thresh, cv2.MORPH_CLOSE, kernel, iterations=2
)  
cleaned = cv2.morphologyEx(
    cleaned, cv2.MORPH_OPEN, kernel, iterations=1
)  

# Find contours of possible eggs
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on circularity and size
filtered_contours = []
min_area = 150  # Minimum area to filter small objects
circularity_threshold = 0.8  

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue  # Avoid division by zero

    circularity = 4 * np.pi * (area / (perimeter**2))  

    if area > min_area and circularity > circularity_threshold:  
        filtered_contours.append(cnt)

# Draw filtered contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Display results
num_eggs = len(filtered_contours)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Eggs: {num_eggs}")

plt.show()
