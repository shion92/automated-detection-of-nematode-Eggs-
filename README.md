# Automated Detection of Nematode Eggs

This repository contains main scripts, and documentation for the automated detection of nematode eggs using deep learning. 

The project supports training, fine-tuning, evaluation, prediction, and visualisation for Faster R-CNN, YOLOv8s and DeepLabV3+ models.

---

## Folder Structure 

This Git repository does not include image, CSV, or other data files. To access these, please send a request to shionshine@gmail.com. 

```
automated-detection-of-nematode-Eggs-/
├── Data/                  # Raw microscopic images and annotation files
├── dataset/               # Processed datasets (train/val/test splits, YOLO labels, DeepLab masks etc.)
│   ├── test
│   │   ├── annotaions     # .xml files for bounding box annotations
│   │   ├── images         # .tif image files
│   │   ├── json           # .json files. Faster RCNN training requires.
│   │   ├── labels         # .txt files. YOLO training requires.
│   │   └── masks          # .png files. DeepLab training requires. 
│   ├── train
│   │   └── ...
│   ├── val
│   │   └── ...
├── DeepLab/               # DeepLabV3+ segmentation scripts and configs
│   ├── deeplab_training.py # Training or fine-tuning 
│   ├── evaluate_visual_deeplab.py # Perform evaluation + visualisation 
│   ├── inference_deeplab_model.py # Run prediction separately if needed (it's usually integrated with training).
│   └── ...
├── Faster_rcnn/           # Faster R-CNN detection scripts and configs
│   ├── faster_rcnn.py
│   ├── evaluate_visual_faster_cnn.py
│   ├── inference_faster_rcnn.py
│   └── ...
├── YOLO/                  # YOLOv8 training scripts
│   ├── yolo_training.py
│   ├── evaluate_YOLO.py
│   ├── visual_yolo_prediction.py 
│   └── ...
├── Processed_Images/      # Model predictions and visualisations
├── model/                 # Saved model weights and checkpoints
├── Prep/                  # Data preparation and splitting scripts
│   ├── split_prep_sample.py
│   └── ...
├── evaluation/            # Evaluation metrics, running logs and Tensorboard logs
├── README.md              # Project documentation (this file)
└── requirements.txt       # Python dependencies
```


---

## Main Components

### 1. **Data Preparation**
- Scripts in `Prep/` split raw data into train/val/test sets and convert annotation formats (Pascal VOC XML to YOLO, etc.).
- Example: `split_prep_sample.py` for dataset splitting.

### 2. **Model Training & Tuning**
- **Faster R-CNN:**  
  - `faster_rcnn/faster_rcnn.py` for training and fine-tuning with different learning rates and optimizers.
- **YOLOv8:**  
  - `YOLO/train_yolov8n.sh` for training YOLOv8 models with various hyperparameters and optimizers.
- **DeepLabV3+:**  
  - `DeepLab/deeplab_training.py` for semantic segmentation training.

### 3. **Inference & Prediction**
- `faster_rcnn/inference_faster_rcnn.py` and similar scripts for running inference on trained models and saving predictions.

### 4. **Evaluation & Visualization**
- `DeepLab/evaluate_visual_deeplab.py`, `faster_rcnn/evaluate_visual_faster_cnn.py` for evaluating predictions (precision, recall, F1, mAP, PR curves).
- `DeepLab/visual_auc.py` and similar scripts for plotting metrics and visualizing results.

---

## How to Use

1. **Prepare Data:**  
   Place raw images and annotation files in the `Data/` folder. Use scripts in `Prep/` to split and convert data as needed.

2. **Train Models:**  
   - For Faster R-CNN:  
     ```bash
     python faster_rcnn/faster_rcnn.py
     ```
   - For YOLOv8:  
     ```bash
     bash YOLO/train_yolov8n.sh
     ```
   - For DeepLabV3+:  
     ```bash
     python DeepLab/deeplab_training.py
     ```

3. **Run Inference:**  
   Use the provided inference scripts in each model folder to generate predictions.

4. **Evaluate & Visualize:**  
   Use evaluation scripts to compute metrics and visualize results. Outputs are saved in [Processed_Images](http://_vscodecontentref_/2) and [evaluation](http://_vscodecontentref_/3).

---

## Requirements

- Python 3.8+
- PyTorch, torchvision
- Ultralytics YOLOv8
- segmentation_models_pytorch
- albumentations
- scikit-learn
- matplotlib
- tqdm
- PIL
- (See [requirements.txt](http://_vscodecontentref_/4) for full list)

---

## Notes

- All scripts are modular and can be run independently.
- Adjust paths and hyperparameters as needed for your experiments.
- ....

---

## Contact
For questions or contributions, please contact Jacquelin Ruan at shionshine@gmail.com or open an issue or pull request.

---
