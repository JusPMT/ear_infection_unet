import os
import numpy as np
from torchvision import datasets
import torchvision.transforms.v2 as T
import torch

def get_transforms(img_size=256, is_train=True):
    """
    Standard preprocessing for medical classification.
    Uses torchvision.transforms.v2 for modern, faster augmentations.
    """
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_dataset(data_dir, img_size=256, val_split=0.2, batch_size=8):
    """
    Loads dataset from a directory where subfolders represent classes (e.g. Normal, Abnormal).
    Automatically splits into Train and Validation sets.
    """
    # Create the full dataset with training transforms first
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(img_size=img_size, is_train=True)
    )
    
    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducible splits
    )
    
    # Overwrite the transform for the validation subset to remove augmentations
    # Note: random_split wraps the dataset in a Subset object.
    val_dataset.dataset.transform = get_transforms(img_size=img_size, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Calculate class weights from training dataset for Weighted Loss
    train_indices = train_dataset.indices
    targets = [full_dataset.targets[i] for i in train_indices]
    
    class_counts = np.bincount(targets, minlength=len(full_dataset.classes))
    total_samples = len(targets)
    n_classes = len(full_dataset.classes)
    
    # Formula: weight = total_samples / (n_classes * count)
    class_weights = total_samples / (n_classes * (class_counts + 1e-6))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return train_loader, val_loader, full_dataset.classes, class_weights
