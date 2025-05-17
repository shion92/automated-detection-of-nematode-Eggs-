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
from glob import glob

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MASK_DIR = os.path.join(DATA_DIR, "masks")
VAL_MASK_DIR = os.path.join(VAL_DIR, "masks")
PRED_OUTPUT_DIR = "Processed_Images/deeplab/Predictions"
OUTPUT_FILE_EXTENSION = ".json"  # Configurable output file extension
MODEL_OUT_DIR = "model/deeplab"
BATCH_SIZE = 2
NUM_EPOCHS = 100
lr_list = [1e-2, 1e-3, 1e-4]
IMG_SIZE = 512
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# Dataset
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
def get_loader(img_dir, mask_dir=None, transform=None, batch_size=BATCH_SIZE, shuffle=True):
    dataset = EggSegmentationDataset(images_dir=img_dir, masks_dir=mask_dir, transform=transform)
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle,
                      num_workers=2,
                      pin_memory=True)

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
        avg_pred = inputs.sum() / inputs.numel()
        print("Average output probability:", avg_pred.item())   
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
print("Model parameters are on:", next(model.parameters()).device)

# -------------------------
# Training Function
# -------------------------
def train_model(lr: float):
    train_loader = get_loader(os.path.join(DATA_DIR, "images"), MASK_DIR, get_train_transform(), batch_size=BATCH_SIZE,
        shuffle=True)
    val_loader = get_loader(os.path.join(VAL_DIR, "images"), VAL_MASK_DIR, get_val_transform(), batch_size=BATCH_SIZE,
        shuffle=False)

    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    # Per-LR output directories
    run_id = f"lr_{lr}"
    metrics_dir = os.path.join("evaluation", "deeplab", run_id)
    model_out_dir = os.path.join(MODEL_OUT_DIR, run_id)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(model_out_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0

        for images, masks, _ in tqdm(train_loader, desc=f"lr = {run_id} Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"lr = {run_id} Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"lr = {run_id} Epoch {epoch}: Avg Val Loss = {avg_val_loss:.4f}")
        
        # Save metrics
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "train_loss_history.json"), "w") as tf:
            json.dump(train_losses, tf, indent=2)
        with open(os.path.join(metrics_dir, "val_loss_history.json"), "w") as vf:
            json.dump(val_losses, vf, indent=2)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_out_dir = os.path.join(model_out_dir, f"deeplabv3plus_{run_id}_best.pth")
            torch.save(model.state_dict(), model_out_dir)
            print("\t✅ New best model saved (lr = {run_id}).")

# -------------------------
# Inference & Save Predictions
# -------------------------

def infer_collate(batch):
    images, _, names = zip(*batch)
    # stack images but leave names as a list
    images = torch.stack(images, dim=0)
    return images, names

def load_best_model(lr):
    model_out_dir = os.path.join(MODEL_OUT_DIR, lr)
    model_path = f"{model_out_dir}/deeplabv3plus_{lr}_best.pth"
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict_and_save(split="test"):
    img_dir = os.path.join("dataset", split, "images")
    loader = DataLoader(EggSegmentationDataset(img_dir, None, get_val_transform()),
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=infer_collate
            )
    out_dir = os.path.join(PRED_OUTPUT_DIR, split)
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    model.eval()
    with torch.no_grad():
            for image, names in tqdm(loader, desc=f"Inference on {split}"):
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
        print(f"\n=== Training with learning rate = {lr} ===")
        train_model(lr)
        print(f"\n✅ lr = {lr} training complete.")
    
    print(f"\n✅ All runs complete in {time.time() - start:.1f}s")
    
    model = load_best_model(lr=1e-4)
    print("\n=== Running predictions... ===")
    for split in ["test", "val", "train"]:
        print(f"\n Running inference on {split} set...")
        predict_and_save(split=split)
        
    print(f"\n✅ Done! Total runtime: {time.time() - start:.2f} seconds.")
