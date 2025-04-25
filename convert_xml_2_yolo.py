import os
import xml.etree.ElementTree as ET

current_dir = os.getcwd()
xml_dir = os.path.join(current_dir, "dataset", "test", "annotations")  
yolo_dir = os.path.join(current_dir, "dataset", "test", "yolo_annotations")
os.makedirs(yolo_dir, exist_ok=True)

# Define class mappings 
CLASS_MAPPING = {
    "nematode egg": 0,
    # "water bubble": 1,  # No longer using these classes
    # "debris": 2,        # No longer using these classes
    # "green grids": 3    # No longer using these classes
}

# Print absolute paths
print(f"Searching for XML annotations in: {xml_dir}")
print(f"YOLO annotations will be saved in: {yolo_dir}")

# Check if XML directory exists and contains files
if not os.path.exists(xml_dir):
    print(f"ERROR: XML annotation folder not found: {xml_dir}")
    exit(1)
else:
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if len(xml_files) == 0:
        print(f"ERROR: No XML files found in {xml_dir}")
        exit(1)
    else:
        print(f"Found {len(xml_files)} XML files in {xml_dir}")

# Convert Pascal VOC .xml to YOLO .txt
def convert_voc_to_yolo(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Debugging: Print root tag to check format
        print(f"Processing: {xml_file} | Root Tag: {root.tag}")

        # Get image size
        size = root.find("size")
        if size is None:
            print(f"ERROR: Missing <size> tag in {xml_file}")
            return
        
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        # YOLO annotation file
        annotation_file = os.path.join(yolo_dir, os.path.basename(xml_file).replace(".xml", ".txt"))

        with open(annotation_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text.lower()

                # Check if the object class is in our predefined mapping
                if class_name not in CLASS_MAPPING:
                    print(f" WARNING: Unknown class '{class_name}' in {xml_file}. Skipping!")  # NEW: Skip non-nematode objects
                    continue
                
                class_id = CLASS_MAPPING[class_name]  # Assign the correct class ID

                # Bounding box
                bbox = obj.find("bndbox")
                if bbox is None:
                    print(f"ERROR: Missing <bndbox> in {xml_file}")
                    continue
                
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # Normalize for YOLO format (values between 0 and 1)
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                norm_width = (xmax - xmin) / img_width
                norm_height = (ymax - ymin) / img_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

        print(f"Successfully converted: {xml_file} → {annotation_file}")

    except Exception as e:
        print(f"ERROR processing {xml_file}: {str(e)}")

# Process all XML files
xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(".xml")]

for xml_file in xml_files:
    convert_voc_to_yolo(xml_file)

print(f"YOLO annotations saved in: {yolo_dir}")
