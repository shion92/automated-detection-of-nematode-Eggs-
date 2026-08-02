# Automated Detection of Nematode Eggs

This repository contains the main scripts and documentation for an automated deep-learning system that detects nematode eggs in microscopic images. The project was developed with Lincoln University (Christchurch, New Zealand) and the Department of Agricultural Sciences to investigate a faster, repeatable computer-vision workflow for livestock-health research.

The code supports training, fine-tuning, evaluation, inference, and visualisation for YOLOv8, Faster R-CNN, DeepLabV3+, and YOLOv8 segmentation models.

![Nematode egg detection](image.png)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![GitHub Repo stars](https://img.shields.io/github/stars/shion92/automated-detection-of-nematode-Eggs-)](https://github.com/shion92/automated-detection-of-nematode-Eggs-/stargazers)
[![Issues](https://img.shields.io/github/issues/shion92/automated-detection-of-nematode-Eggs-)](https://github.com/shion92/automated-detection-of-nematode-Eggs-/issues)

## Recorded results

The table below reproduces the deployment-candidate rows from Table 1, “Performance Metrics for Test Set,” in the final project report. Values are preserved at the report's three-decimal precision and are also available in [`docs/results/reported-test-model-comparison.csv`](docs/results/reported-test-model-comparison.csv).

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s-max | **1.000** | **1.000** | **1.000** | **1.000** | 0.537 |
| YOLOv8m | 0.947 | **1.000** | 0.973 | **1.000** | 0.567 |
| YOLOv8m-max | 0.900 | **1.000** | 0.947 | **1.000** | 0.592 |
| Faster R-CNN, ResNet50, lr=0.005 | 0.947 | **1.000** | 0.973 | **1.000** | **0.676** |
| Faster R-CNN, ResNet50, lr=0.001 | **1.000** | **1.000** | **1.000** | **1.000** | 0.663 |

These figures describe one small held-out test set from this project. Separate saved JSON/CSV files may contain validation runs or later evaluation runs and therefore may not reproduce Table 1 exactly. The results should not be treated as estimates of performance on other laboratories, imaging systems, or nematode species.

## Prediction examples

Each image is an actual saved inference comparison. From left to right, the panels show YOLOv8s-max, YOLOv8m, Faster R-CNN ResNet50 (`lr=0.005`), and Faster R-CNN ResNet50 (`lr=0.001`). Bounding boxes and confidence scores are model outputs.

**Example 1 - low-contrast egg**

![Four-model prediction comparison for image 01](docs/predictions/image-01-model-comparison.jpg)

**Example 2 - dark, partially occluded egg**

![Four-model prediction comparison for image 62](docs/predictions/image-62-model-comparison.jpg)

**Example 3 - egg near microscope artefacts**

![Four-model prediction comparison for image 77](docs/predictions/image-77-model-comparison.jpg)

**Example 4 - overexposed field**

![Four-model prediction comparison for image 113](docs/predictions/image-113-model-comparison.jpg)

## Repository structure

```text
automated-detection-of-nematode-Eggs-/
├── DeepLab/                # DeepLabV3+ training, inference, and evaluation
├── faster_rcnn/            # Faster R-CNN training, inference, and evaluation
├── YOLO/                   # YOLOv8 training, inference, and evaluation
├── Helper/                 # Dataset conversion, comparison, and utility scripts
├── docs/
│   ├── predictions/        # README-ready examples from saved model outputs
│   └── results/            # Reported metric values used in this README
├── data.yaml               # YOLO detection dataset configuration
├── data_seg.yaml           # YOLO segmentation dataset configuration
└── requirements.txt        # Python dependencies
```

The public repository does not include the raw microscope dataset, annotations, model weights, training logs, or complete evaluation outputs. To request access to research data, contact the project author at the address below.

## Main components

### Data preparation

- Scripts in `Helper/` split raw data into training, validation, and test sets.
- Conversion utilities transform Pascal VOC XML or LabelMe JSON annotations into the formats required by YOLO, Faster R-CNN, and DeepLabV3+.
- The expected local dataset folders include correctly named `annotations`, `images`, `json`, `labels`, and `masks` directories.

### Model training and tuning

- **Faster R-CNN:** `faster_rcnn/faster_rcnn.py`
- **YOLOv8:** `YOLO/yolo_training.py`
- **DeepLabV3+:** `DeepLab/deeplab_training.py`

The scripts expose experiment-specific values such as dataset paths, model paths, learning rates, batch sizes, and output directories. Review these values before running an experiment.

### Inference and evaluation

- Faster R-CNN inference: `faster_rcnn/inference_faster_rcnn.py`
- YOLO inference: `YOLO/inference_visual_yolo_prediction.py`
- DeepLabV3+ inference: `DeepLab/inference_deeplab_model.py`
- Cross-model comparison: `Helper/universal_comparison.py`

Evaluation scripts calculate metrics including precision, recall, F1, IoU, and COCO-style mAP. TensorBoard is used for experiment tracking where configured.

## Installation

Python 3.11.9 was used for the project.

```bash
git clone https://github.com/shion92/automated-detection-of-nematode-Eggs-.git
cd automated-detection-of-nematode-Eggs-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before training or inference, update the dataset and model paths in the relevant script and YAML configuration. See [`Helper/label_instruction.md`](Helper/label_instruction.md) for annotation guidance.

## Limitations

- The available dataset contains 113 microscope images, so the reported metrics have high uncertainty and may not generalise to new laboratories, microscopes, sample preparation methods, egg types, or imaging conditions.
- Extreme cases, including unfamiliar egg shapes, severe blur, and heavily overlapping eggs, have not been evaluated comprehensively.
- Detection and segmentation models use different output representations and evaluation procedures; their metrics should not be compared as if they were identical tasks.
- Training and fine-tuning are computationally expensive. Runtime and memory requirements depend on the model, image resolution, hardware, and software environment.
- The repository provides research scripts rather than a production application. It does not yet include a user interface, deployment service, automated data-ingestion workflow, or clinical/diagnostic validation.

## Contact

For questions or contributions, contact Jacquelin Ruan at [shionshine@gmail.com](mailto:shionshine@gmail.com), or open an issue or pull request.
├── README.md              # Project documentation (this file)
└── requirements.txt       # Python dependencies
```


---

## Main Components

### 1. **Data Preparation for Training **
- Script in `Helper/` to split raw data into train/val/test sets 
- Convert images to formats that suitable for training (e.g., Pascal VOC XML to YOLO readable .txt, etc.).

### 2. **Model Training & Tuning**
- **Faster R-CNN:**  
  - `Faster_rcnn/faster_rcnn.py` for training and fine-tuning with different learning rates and backbones.
- **YOLOv8:**  
  - `YOLO/yolo_training.py` for training YOLOv8 models with various hyperparameters and optimisers.
- **DeepLabV3+:**  
  - `DeepLab/deeplab_training.py` for segmentation training.

### 3. **Inference & Prediction**
- `inference_faster_rcnn.py` for example, is a script for running inference on trained models and saving predictions. It is built to run inference separately if needed, which usually is already integrated into the training pipeline. 
- Outputs are typically saved in `/Processed_Images`

### 4. **Evaluation & Visualization**
- `DeepLab/evaluate_visual_deeplab.py`, `faster_rcnn/evaluate_visual_faster_cnn.py` etc for evaluating predictions (precision, recall, F1, mAP, PR curves). Outputs are typically saved in `evaluation`.
- For YOLO, TensorBoard was used to compare different YOLO model variants. See more https://www.tensorflow.org/tensorboard/get_started 


---
## How to Use
**Install Dependencies**

Ensure Python 3.11.9 is installed, then install the required packages:

```bash
   pip install -r requirements.txt
```

---
## Debug notes

- Adjust paths and hyperparameters as needed for your experiments.
- Refers to Helper/label_instruction.md if you have any questions around labeling. 

---

## Contact
For questions or contributions, please contact Jacquelin Ruan at shionshine@gmail.com or open an issue or pull request.

---
