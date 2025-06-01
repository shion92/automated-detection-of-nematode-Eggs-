# -------------------------
# Imports
# -------------------------
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
LR_LIST = ["lr_0.01", "lr_0.001", "lr_0.0001", "lr_0.005", "lr_0.0005"]

MODEL = "faster_rcnn"  # deeplab
EVAL_DIR = f"evaluation/{MODEL}/resnet34"

# -------------------------
# 1) Plot train loss curves for each LR
# -------------------------
plt.figure(figsize=(10, 6))
for lr_id in LR_LIST:
    eval_dir = os.path.join(EVAL_DIR, lr_id)
    train_loss_path = os.path.join(eval_dir, "train_loss_history.json")
    if not os.path.exists(train_loss_path):
        print(f"Warning: Missing train loss file for {lr_id}, skipping.")
        continue
    with open(train_loss_path, "r") as f:
        train_loss = np.array(json.load(f))
    epochs = np.arange(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label=f"Train {lr_id}", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss Curves ({MODEL})")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 2) Plot val loss curves for each LR
# -------------------------
plt.figure(figsize=(10, 6))
for lr_id in LR_LIST:
    eval_dir = os.path.join(EVAL_DIR, lr_id)
    val_loss_path = os.path.join(eval_dir, "val_loss_history.json")
    if not os.path.exists(val_loss_path):
        print(f"Warning: Missing val loss file for {lr_id}, skipping.")
        continue
    with open(val_loss_path, "r") as f:
        val_loss = np.array(json.load(f))
    epochs = np.arange(1, len(val_loss) + 1)
    plt.plot(epochs, val_loss, label=f"Val {lr_id}", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Validation Loss Curves ({MODEL})")
plt.legend()
plt.tight_layout()
plt.show()