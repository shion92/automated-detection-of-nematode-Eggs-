# -------------------------
# Imports
# -------------------------
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

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MASK_DIR = "dataset/val/masks"
VAL_MASK_DIR = "dataset/val/masks"
PRED_OUTPUT_DIR = "Processed_Images/deeplab/Predictions"
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 512
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# Dataset
# -------------------------
class EggSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0).float()
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.

        return image, mask, os.path.basename(img_path)

# -------------------------
# Albumentations Transformations
# -------------------------
def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])

# -------------------------
# DataLoaders
# -------------------------
def get_loader(img_dir, mask_dir, transform):
    dataset = EggSegmentationDataset(images_dir=img_dir, masks_dir=mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Loss Function
# -------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# -------------------------
# Model
# -------------------------
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model.to(DEVICE)

# -------------------------
# Training Function
# -------------------------
def train_model():
    train_loader = get_loader(os.path.join(DATA_DIR, "images"), MASK_DIR, get_train_transform())
    val_loader = get_loader(os.path.join(VAL_DIR, "images"), VAL_MASK_DIR, get_val_transform())

    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Avg Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "model/deeplab/deeplabv3plus_egg_segmentation_best.pth")
            print("\t✅ New best model saved.")

# -------------------------
# Inference & Save Predictions
# -------------------------
def predict_and_save(split="test"):
    img_dir = os.path.join("dataset", split, "images")
    mask_dir = os.path.join("masks", split)
    loader = get_loader(img_dir, mask_dir, get_val_transform())
    os.makedirs(os.path.join(PRED_OUTPUT_DIR, split), exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, _, names in tqdm(loader, desc=f"Predicting {split}"):
            for image, name in zip(images, names):
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                out_path = os.path.join(PRED_OUTPUT_DIR, split, name.replace(".tif", ".json"))
                pred_json = {
                    "mask": pred_mask.tolist()
                }
                with open(out_path, 'w') as f:
                    json.dump(pred_json, f, indent=2)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    start = time.time()
    print("\n=== Starting training... ===")
    train_model()
    
    print("\n=== Running predictions... ===")
    for split in ["test", "val", "train"]:
        print(f"\n Running inference on {split} set...")
        predict_and_save(split=split)
        
    print(f"\n✅ Done! Total runtime: {time.time() - start:.2f} seconds.")
