# Ear Infection Classification (U-Net Based)

This repository focuses on classifying middle ear infections (abnormal vs. normal) using otoscopic images. 

We adapt the original U-Net architecture (2015), which was primarily designed for semantic segmentation, by attaching a Classification Head to output a binary diagnosis. This allows us to leverage U-Net's powerful feature extraction capabilities for disease detection.

## Quick Links
- [Kaggle Tympanic Membrane Dataset](https://www.kaggle.com/)
- [Pytorch-UNet (Original Source)](https://github.com/milesial/Pytorch-UNet)

## Repository Structure

```text
ear_infection_unet/
├── data/
│   ├── dataset.py        # PyTorch Dataset for loading images
│   └── transforms.py     # Preprocessing (CLAHE, Letterbox resizing)
├── models/
│   ├── unet_2015/        # Modified 2015 U-Net with Classification Head
│   │   ├── unet_model.py
│   │   └── unet_parts.py
│   └── (future models)   # Directory ready for U-Net variants (Attention, ResUNet, etc.)
├── train.py              # Training script (CrossEntropyLoss)
├── evaluate.py           # Evaluation script (Accuracy, Sensitivity, Specificity)
└── README.md
```

## Setup & Installation

1. Install PyTorch and required dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

2. Structure your dataset directory as follows:
```text
dataset_path/
├── normal/
└── abnormal/
```

## Training

To train the modified U-Net model on your dataset, run:
```bash
python train.py --data_dir "C:\Users\phamm\Desktop\ear canca\Kaggle_956\eardrumDs" --epochs 50 --batch_size 8
```
