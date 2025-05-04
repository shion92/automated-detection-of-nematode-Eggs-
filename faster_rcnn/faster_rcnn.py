import os
import random
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from tqdm import tqdm # progress bars

# -------------------------
# Configuration
# -------------------------
# DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
NUM_CLASSES = 2  # 1 class (nematode egg) + background
CLASS_NAMES = ["__background__", "nematode egg"]
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"
PRED_OUTPUT_DIR = "Processed_Images/faster_rcnn/Predictions"
num_epochs = 20

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# -------------------------
#  Albumentations pipeline
# -------------------------
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.2), p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# -------------------------
# Dataset
# -------------------------
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    def __getitem__(self, idx):
        # load image and annotation
        image_id = self.images[idx]
        img_path = os.path.join(self.image_dir, image_id)
        ann_path = os.path.join(self.annotation_dir, image_id.replace('.tif', '.xml'))
        
        # Parse boxes and labels as Python lists
        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name != "nematode egg":
                continue
            bnd = obj.find("bndbox")
            box = [int(bnd.find("xmin").text), int(bnd.find("ymin").text),
                int(bnd.find("xmax").text), int(bnd.find("ymax").text)]
            
            # # Reject invalid boxes
            # if box[2] <= box[0] or box[3] <= box[1]:
            #     print(f"⚠️ Invalid box in {image_id}: {box}")
            #     continue
            
            boxes.append(box)
            labels.append(1) # “nematode egg” → class 1

        if len(boxes) == 0:
            print(f"⚠️ Skipping image with no valid boxes: {image_id}")
            return None  # Skip invalid sample
        
        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32).view(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64).view(-1)
        
        # Make sure we have shape [N,4], even if N==1
        if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
        
        target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
            }
        
        # Apply transformations

        # Check Albumentations vs F.to_tensor
        if isinstance(self.transforms, Compose):
            # Albumentations expects numpy arrays
            img_np = np.array(img)
            # format bboxes for Albumentations
            bboxes = [box for box in target['boxes'].tolist()]
            labels = target['labels'].tolist()
            
            augmented = self.transforms(image=img_np, bboxes=bboxes, labels=labels)
            img = augmented['image']
            # rebuild target from augmented bboxes
            target['boxes']  = torch.tensor(augmented['bboxes'], dtype=torch.float32).view(-1, 4)
            target['labels'] = torch.tensor(augmented['labels'], dtype=torch.int64).view(-1)
            
            target['area']    = (target['boxes'][:,3] - target['boxes'][:,1]) * (target['boxes'][:,2] - target['boxes'][:,0])
            target['iscrowd']= torch.zeros((len(target['labels']),), dtype=torch.int64)

        elif callable(self.transforms):
            img = self.transforms(img)
        else:
            # no transform
            img = F.to_tensor(img)
        
        return img, target, image_id


    def __len__(self):
        return len(self.images)

# -------------------------
# Loaders
# -------------------------
def get_loader(root, batch_size=2, transforms=F.to_tensor):
    dataset = VOCDataset(
        image_dir=os.path.join(root, "images"),
        annotation_dir=os.path.join(root, "annotations"),
        transforms=transforms
    )
    
    # Modified collate function to skip None
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return tuple(zip(*batch)) if batch else ([], [], [])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# -------------------------
# Training Loop
# -------------------------
def train_model(num_epochs):
    set_seed()

    train_loader = get_loader(TRAIN_DIR, transforms=get_train_transforms()) # apply Albumentations on train only
    val_loader = get_loader(VAL_DIR) # uses default F.to_tensor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(DEVICE)

    print(f"=== Using device: {DEVICE} ")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    torch.autograd.set_detect_anomaly(True)
    best_val_loss = float('inf')
    
    print(f"===Training on {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples. ===")
    print(f"Optimizer: {optimizer.__class__.__name__} ===")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            if not images:   # <- empty list
                continue
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            for t in targets:
                print(f"Boxes: {t['boxes'].shape}, Labels: {t['labels']}, Areas: {t['area']}")
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"⚠️ Non-finite loss at batch {batch_idx}: {losses.item()}")
                continue

            optimizer.zero_grad()
            if not torch.isfinite(losses):
                print(f"⚠️ Non-finite loss detected: {losses.item()} — skipping batch")
                continue
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}, Training Loss: {losses.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}")

        # --- Validation loop ---
        # Force model to compute loss, even though we call it "validation"
        model.train()   
        
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets, _) in enumerate(val_loader):
            # ── skip empty validation batches ───────────────────────
                if len(images) == 0:
                    continue

                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"   Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "faster_rcnn_nematode_best.pth")
            print(f"   ✅ New best model saved! (val loss: {best_val_loss:.4f})")

    return model


# -------------------------
# Inference & Save Predictions
# -------------------------
def predict_and_save(model, split="test"):
    os.makedirs(os.path.join(PRED_OUTPUT_DIR, split), exist_ok=True)
    loader = get_loader(os.path.join("dataset", split), batch_size=1)
    model.eval()

    with torch.no_grad():
        for imgs, _, names in tqdm(loader, desc=f"Inference on {split} set"):
            skipped = 0
            if not imgs:  # Empty batch
                print("⚠️ Skipping batch with no valid images.")
                skipped += 1
                continue
            
            img = imgs[0].to(DEVICE)
            output = model([img])[0]

            boxes = output['boxes'].cpu().numpy().tolist()
            scores = output['scores'].cpu().numpy().tolist()

            # Keep predictions with confidence >= 0.5
            keep = [i for i, s in enumerate(scores) if s >= 0.5]
            filtered_boxes = [boxes[i] for i in keep]
            filtered_scores = [scores[i] for i in keep]

            # Save both boxes and their confidence scores
            pred_json = {
                "boxes": [[int(x) for x in box] for box in filtered_boxes],
                "scores": [round(float(s), 4) for s in filtered_scores]
            }

            out_path = os.path.join(PRED_OUTPUT_DIR, split, names[0].replace(".tif", ".json"))
            with open(out_path, 'w') as f:
                json.dump(pred_json, f, indent=2)
                
            print(f"✅ Done predicting {split} set. Skipped {skipped} images.")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    start_time = time.time()

    print("=== Starting training... ===")
    model = train_model(num_epochs=num_epochs)

    for split in ["test", "val", "train"]:
        print(f"\n Running inference on {split} set...")
        predict_and_save(model, split=split)

    total_time = time.time() - start_time
    print(f"\n✅ Done! Total runtime: {total_time:.2f} seconds.")

