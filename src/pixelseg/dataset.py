"""
Dataset class for segmentation training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Optional, Tuple, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation.
    
    Args:
        image_dir: Directory containing RGB images
        mask_dir: Directory containing .npy mask files
        class_mapping: Dictionary mapping class names to IDs
        transform: Albumentations transform pipeline
        img_size: Target image size (height, width) or None for original size
    """
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 class_mapping: dict,
                 transform: Optional[Callable] = None,
                 img_size: Optional[Tuple[int, int]] = None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_mapping = class_mapping
        self.num_classes = max(class_mapping.values()) + 1  # +1 for background (0)
        self.transform = transform
        self.img_size = img_size
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.images)} images in {image_dir}")
        print(f"Number of classes: {self.num_classes}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = os.path.splitext(img_name)[0] + '.npy'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.load(mask_path)
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Resize if needed
        if self.img_size is not None:
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # print(type(mask))
        # # Convert mask to long tensor
        # mask = torch.from_numpy(mask).long()
        
        return image, mask.long(), img_name


def get_training_transforms(img_size=(720, 1080)):
    """
    Get training augmentation transforms.
    
    Args:
        img_size: Target size (height, width)
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_validation_transforms(img_size=(720, 1080)):
    """
    Get validation transforms (no augmentation).
    
    Args:
        img_size: Target size (height, width)
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(train_image_dir: str,
                      train_mask_dir: str,
                      val_image_dir: str,
                      val_mask_dir: str,
                      class_mapping: dict,
                      batch_size: int = 4,
                      img_size: Tuple[int, int] = (720, 1080),
                      num_workers: int = 4):
    """
    Create training and validation dataloaders.
    
    Args:
        train_image_dir: Training images directory
        train_mask_dir: Training masks directory
        val_image_dir: Validation images directory
        val_mask_dir: Validation masks directory
        class_mapping: Dictionary mapping class names to IDs
        batch_size: Batch size
        img_size: Target image size (height, width)
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SegmentationDataset(
        train_image_dir,
        train_mask_dir,
        class_mapping,
        transform=get_training_transforms(img_size)
    )
    
    val_dataset = SegmentationDataset(
        val_image_dir,
        val_mask_dir,
        class_mapping,
        transform=get_validation_transforms(img_size)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
