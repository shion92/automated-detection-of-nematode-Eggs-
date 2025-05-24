import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# -------------------------
# Configuration
# -------------------------
SPLIT    = "test"   # change to "val" if needed
EVAL_DIR = f"evaluation/deeplab/{SPLIT}"
pr_path  = os.path.join(EVAL_DIR, "pr_data.json")

# -------------------------
# 1) Load PR data
# -------------------------
with open(pr_path, "r") as f:
    pr = json.load(f)

precisions = np.array(pr["precisions"])
recalls    = np.array(pr["recalls"])
thresholds = np.array(pr["thresholds"])
auc_pr     = float(pr["auc_pr"])

# -------------------------
# 2) Compute F1 at each threshold
#    note: len(precisions) = len(thresholds) + 1
# -------------------------
f1_scores = (2 * precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-8)

# -------------------------
# 3) Find best threshold by max F1
# -------------------------
best_idx       = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1        = f1_scores[best_idx]
best_prec      = precisions[best_idx+1]
best_rec       = recalls[best_idx+1]

print(f"Best threshold: {best_threshold:.3f}")
print(f"Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, F1: {best_f1:.3f}")

# -------------------------
# 4) Plot Precision–Recall curve
# -------------------------
plt.figure()
plt.plot(recalls, precisions, label=f"AUC = {auc_pr:.3f}")
plt.scatter([best_rec], [best_prec], color="red", label=f"Best F1 @ {best_threshold:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 5) Plot F1 vs Threshold
# -------------------------
plt.figure()
plt.plot(thresholds, f1_scores, marker="o")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best thr = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold")
plt.legend()
plt.tight_layout()
plt.show()
