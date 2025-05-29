# -------------------------
# Imports
# -------------------------
import os
import shutil
import random

# -------------------------
# Configuration
# -------------------------
IMAGE_DIR = os.path.join("Raw_Data")
LABEL_DIR = os.path.join("Raw_Data")
OUTPUT_BASE = os.path.join("dataset")

N_TOTAL = 113
N_TRAIN = 79
N_VAL = 17
N_TEST = N_TOTAL - N_TRAIN - N_VAL

# -------------------------
# Collect and Shuffle Files
# -------------------------
xml_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith(".xml")]
if len(xml_files) != N_TOTAL:
    # Also check for images without corresponding .xml and create empty .xml for them
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")]
    xml_stems = set(os.path.splitext(os.path.basename(f))[0] for f in xml_files)
    missing_xml = []
    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        if stem not in xml_stems:
            missing_xml.append(stem)

    if len(xml_files) != N_TOTAL:
        print(f"Expected {N_TOTAL} samples, found {len(xml_files)} .xml files and {len(missing_xml)} missing .xml files.")
        proceed = input(f"Do you want to create {len(missing_xml)} empty .xml files for {missing_xml} and continue? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting without creating empty .xml files.")
            exit(1)

    for stem in missing_xml:
        empty_xml_path = os.path.join(LABEL_DIR, stem + ".xml")
        with open(empty_xml_path, "w") as f:
            f.write("<annotation></annotation>")
        print(f"Created empty annotation file for {stem}.xml as it is empty.")
        xml_files.append(empty_xml_path)

random.shuffle(xml_files)

train_xml = xml_files[:N_TRAIN]
val_xml = xml_files[N_TRAIN:N_TRAIN + N_VAL]
test_xml = xml_files[N_TRAIN + N_VAL:]

# -------------------------
# Utility Functions
# -------------------------
def copy_files(xml_list, subset):
    image_out = os.path.join(OUTPUT_BASE, subset, "images")
    label_out = os.path.join(OUTPUT_BASE, subset, "annotations")
    json_out = os.path.join(OUTPUT_BASE, subset, "json")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)
    os.makedirs(json_out, exist_ok=True)

    for xml_path in xml_list:
        # Copy annotation file
        shutil.copy(xml_path, os.path.join(label_out, os.path.basename(xml_path)))

        # Copy corresponding image file
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        image_name = stem + ".tif"
        image_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(image_out, image_name))
        else:
            print(f"Warning: Image not found for {stem}")

        # -------------------------
        # Copy corresponding .json file (for DeepLab training)
        # -------------------------
        json_name = stem + ".json"
        json_path = os.path.join(IMAGE_DIR, json_name)
        if os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(json_out, json_name))
        else:
            print(f"Warning: JSON not found for {stem}")

# -------------------------
# Main Split and Copy
# -------------------------
copy_files(train_xml, "train")
copy_files(val_xml, "val")
copy_files(test_xml, "test")

print(f"Random split images by Train/Val/Test:")
print(f"Train: {len(train_xml)} | Val: {len(val_xml)} | Test: {len(test_xml)}")
print(f"Split completed successfully. .xml and .json files copied to {OUTPUT_BASE}.")
# -------------------------