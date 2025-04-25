import cv2
import os

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, "Data", "Images")  
yolo_dir = os.path.join(current_dir, "Data", "YOLO_Annotations")  

# Define class names and colors for visualization
CLASS_MAPPING = {
    0: "nematode_egg",
    1: "water_bubble",
    2: "debris",
    3: "green_grid"
}

CLASS_COLORS = {
    0: (0, 255, 0),  # Green for eggs
    1: (255, 0, 0),  # Blue for water bubbles
    2: (0, 0, 255),  # Red for debris
    3: (255, 255, 0)  # Cyan for grids
}

# Get list of images
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".tif"))]

# Function to visualize YOLO annotations
def visualize_yolo_annotations(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Get corresponding YOLO annotation file
    annotation_file = os.path.join(yolo_dir, os.path.basename(image_path).replace(".tif", ".txt"))

    if not os.path.exists(annotation_file):
        print(f"⚠️ No annotation found for {image_path}")
        return

    # Read YOLO annotations
    with open(annotation_file, "r") as f:
        lines = f.readlines()

    # Draw bounding boxes
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, norm_w, norm_h = map(float, data[1:])

        # Convert normalized values to absolute coordinates
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        w = int(norm_w * width)
        h = int(norm_h * height)

        xmin = int(x_center - w / 2)
        ymin = int(y_center - h / 2)
        xmax = int(x_center + w / 2)
        ymax = int(y_center + h / 2)

        # Get color and label
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default white if unknown
        label = CLASS_MAPPING.get(class_id, "unknown")

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image
    cv2.imshow("YOLO Annotations", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize all images
for img_file in image_files:
    visualize_yolo_annotations(os.path.join(image_dir, img_file))
