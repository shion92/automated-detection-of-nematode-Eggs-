import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3, grid_size=7):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        # Backbone: Feature extractor (e.g., simplified CNN for demonstration)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes)),
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detector(features)
        return predictions.view(-1, self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes)

# Instantiate the model
model = YOLO(num_classes=20)
print(model)