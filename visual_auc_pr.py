import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# 1) Load your saved PR data
with open("pr_data.json", "r") as f:
    data = json.load(f)
all_scores = np.array(data["scores"])
all_labels = np.array(data["labels"])
auc_pr      = data["auc_pr"]

# 2) Compute precision and recall at all thresholds
precisions, recalls, _ = precision_recall_curve(all_labels, all_scores)

# 3) Plot the PR curve
plt.figure()
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve (AUC = {auc_pr:.3f})")
plt.show()
