
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from glob import glob
import sys


# -------------------------
# Configuration
# -------------------------
lr_list = [
    0.01, 
    0.001, 
    0.0001, 
    0.0005, 
    0.0008
    ]
PRED_OUTPUT_DIR = "Processed_Images/deeplab/Predictions"
OUTPUT_FILE_EXTENSION = ".json"  
BASE_MODEL_DIR = os.path.join("model", "deeplab")
IMG_SIZE = 512
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------
# Inference & Save Predictions
# -------------------------
class EggSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        self.images = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base = os.path.splitext(img_name)[0]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = None
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, base + ".png")
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask = (mask > 0).astype(np.float32) 

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].unsqueeze(0).float()
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.
            if mask is not None:
                mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.

        return image, mask, img_name

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])


def infer_collate(batch):
    images, _, names = zip(*batch)
    # stack images but leave names as a list
    images = torch.stack(images, dim=0)
    return images, names

def load_best_model(lr):
    model_out_dir = os.path.join(BASE_MODEL_DIR, f"lr_{lr}")
    model_path = os.path.join(model_out_dir, f"deeplabv3plus_lr_{lr}_best.pth")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Cannot find model at {model_path}")
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model
def predict_and_save(model, lr, split="test"):
    img_dir = os.path.join("dataset", split, "images")
    loader = DataLoader(EggSegmentationDataset(img_dir, None, get_val_transform()),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=infer_collate
                )
    out_dir = os.path.join(PRED_OUTPUT_DIR, f"lr_{lr}", split)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for image, names in tqdm(loader, desc=f"[{lr}] Inference on {split}"):
            image = image.to(DEVICE)
            output = model(image)
            pred_mask = torch.sigmoid(output).squeeze(1)
            for mask_arr, fname in zip(pred_mask.cpu().numpy(), names):
                raw_mask = mask_arr.astype(float)
                out_path = os.path.join(
                    out_dir,
                    os.path.splitext(fname)[0] + OUTPUT_FILE_EXTENSION
                )
                with open(out_path, 'w') as f:
                    json.dump({"mask": raw_mask.tolist()}, f, indent=2)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    start = time.time()
    # print("\n=== Starting training... ===")
    # train_model()

    for lr in lr_list:
        print(f"\n=== Running predictions for learning rate {lr}... ===")
        model = load_best_model(lr=lr)
        for split in ["test", "val", "train"]:
            print(f"\n Running inference on {split} set...")
            predict_and_save(model, lr, split=split)
        
    print(f"\n✅ Done! Total runtime: {time.time() - start:.2f} seconds.")


