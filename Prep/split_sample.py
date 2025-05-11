import os
import shutil
import random
from pathlib import Path

# Paths
image_dir = Path("Data")
label_dir = Path("labels/train")
output_base = Path("dataset")

n_total = 58
n_train = 40
n_val = 9
n_test = n_total - n_train - n_val 

# Collect and shuffle .xml files
xml_files = list(label_dir.glob("*.xml"))
if len(xml_files) != n_total:
    raise ValueError(f"Expected 58 samples, found {len(xml_files)}. Please check the dataset.")

random.shuffle(xml_files)

train_xml = xml_files[:n_train]
val_xml = xml_files[n_train:n_train + n_val]
test_xml = xml_files[n_train + n_val:]

def copy_files(xml_list, subset):
    image_out = output_base / subset / "images"
    label_out = output_base / subset / "annotations"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for xml_path in xml_list:
        shutil.copy(xml_path, label_out / xml_path.name)

        image_name = xml_path.stem + ".tif"
        image_path = image_dir / image_name
        if image_path.exists():
            shutil.copy(image_path, image_out / image_name)
        else:
            print(f"Warning: Image not found for {xml_path.stem}")

# Copy files into train/val/test
copy_files(train_xml, "train")
copy_files(val_xml, "val")
copy_files(test_xml, "test")

print(f"Train: {len(train_xml)} | Val: {len(val_xml)} | Test: {len(test_xml)}")
