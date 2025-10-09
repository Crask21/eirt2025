"""
Semantic Segmentation Module

This module provides semantic segmentation functionality using DeepLabV3+
for generating pixel-wise class predictions from RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from typing import Union, List, Optional, Tuple
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class SemanticSegmentationModel:
    """
    Semantic segmentation model using DeepLabV3+ with ResNet50 backbone.
    
    This model performs pixel-wise classification to generate semantic
    segmentation masks that will be used for painting LiDAR points.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
        model_name: str = "deeplabv3_resnet50"
    ):
        """
        Initialize the semantic segmentation model.
        
        Args:
            num_classes: Number of output classes (including background)
            device: Device to run the model on
            pretrained: Whether to use pretrained weights
            model_name: Name of the segmentation model architecture
        """
        self.num_classes = num_classes
        self.device = device
        self.model_name = model_name
        
        # Initialize model
        self.model = self._create_model(pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = self._get_transform()
        
        # Default KITTI class mapping
        self.class_names = ["background", "car", "pedestrian", "cyclist"]
        
        logger.info(f"Initialized {model_name} with {num_classes} classes on {device}")
    
    def _create_model(self, pretrained: bool) -> nn.Module:
        """Create the segmentation model."""
        if self.model_name == "deeplabv3_resnet50":
            model = deeplabv3_resnet50(pretrained=pretrained)
            
            # Modify classifier for custom number of classes
            if self.num_classes != 21:  # COCO has 21 classes by default
                model.classifier[-1] = nn.Conv2d(
                    model.classifier[-1].in_channels,
                    self.num_classes,
                    kernel_size=1
                )
                
            # Also modify auxiliary classifier if it exists
            if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
                model.aux_classifier[-1] = nn.Conv2d(
                    model.aux_classifier[-1].in_channels,
                    self.num_classes,
                    kernel_size=1
                )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.ToPILImage() if not isinstance(transforms.ToPILImage(), type) else lambda x: x,
            transforms.Resize((512, 1024)),  # KITTI-like aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(
        self,
        image: Union[np.ndarray, str, Image.Image],
        return_probabilities: bool = True,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate semantic segmentation for an input image.
        
        Args:
            image: Input image as numpy array (H, W, 3), file path, or PIL Image
            return_probabilities: If True, return class probabilities; if False, return logits
            temperature: Temperature scaling for softmax (higher = softer predictions)
            
        Returns:
            Segmentation scores/probabilities of shape (H, W, num_classes)
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        original_shape = image.shape[:2]  # (H, W)
        
        # Preprocess for model
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed (OpenCV convention)
            if np.max(image) <= 1.0:
                image = (image * 255).astype(np.uint8)
        else:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs['out']  # DeepLab outputs dictionary
            else:
                logits = outputs
                
        # Apply temperature scaling and softmax
        logits = logits / temperature
        
        if return_probabilities:
            probs = F.softmax(logits, dim=1)
            result = probs.squeeze(0).cpu().numpy()  # (C, H, W)
        else:
            result = logits.squeeze(0).cpu().numpy()  # (C, H, W)
        
        # Transpose to (H, W, C) and resize to original image size
        result = result.transpose(1, 2, 0)  # (H, W, C)
        
        # Resize back to original image size
        result_resized = np.zeros((original_shape[0], original_shape[1], self.num_classes))
        for c in range(self.num_classes):
            result_resized[:, :, c] = np.array(
                Image.fromarray(result[:, :, c]).resize(
                    (original_shape[1], original_shape[0]), 
                    Image.BILINEAR
                )
            )
        
        return result_resized.astype(np.float32)
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
        batch_size: int = 4,
        return_probabilities: bool = True
    ) -> List[np.ndarray]:
        """
        Process multiple images in batches.
        
        Args:
            images: List of images to process
            batch_size: Number of images to process at once
            return_probabilities: If True, return class probabilities
            
        Returns:
            List of segmentation results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = [
                self.predict(img, return_probabilities) for img in batch_images
            ]
            results.extend(batch_results)
            
        return results
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create a visualization overlay of segmentation on original image.
        
        Args:
            image: Original image (H, W, 3)
            segmentation: Segmentation scores (H, W, C)
            alpha: Transparency of overlay
            
        Returns:
            Overlaid visualization image
        """
        # Get predicted class for each pixel
        predicted_classes = np.argmax(segmentation, axis=2)
        
        # Create color map
        colors = np.array([
            [0, 0, 0],        # background - black
            [255, 0, 0],      # car - red  
            [0, 255, 0],      # pedestrian - green
            [0, 0, 255],      # cyclist - blue
        ])
        
        # Extend color map if we have more classes
        while len(colors) < self.num_classes:
            colors = np.vstack([colors, np.random.randint(0, 256, (1, 3))])
        
        # Create colored segmentation mask
        colored_mask = colors[predicted_classes]
        
        # Blend with original image
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        blended = (1 - alpha) * image + alpha * colored_mask
        return blended.astype(np.uint8)
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")
    
    def save_model(self, checkpoint_path: str):
        """
        Save model weights to a checkpoint file.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, checkpoint_path)
        
        logger.info(f"Saved model to {checkpoint_path}")
    
    def get_class_predictions(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Get predicted class labels from segmentation scores.
        
        Args:
            segmentation: Segmentation scores (H, W, C)
            
        Returns:
            Predicted class labels (H, W)
        """
        return np.argmax(segmentation, axis=2)
    
    def get_confidence_mask(
        self,
        segmentation: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate confidence mask based on maximum class probability.
        
        Args:
            segmentation: Segmentation scores (H, W, C)
            threshold: Minimum confidence threshold
            
        Returns:
            Boolean confidence mask (H, W)
        """
        max_probs = np.max(segmentation, axis=2)
        return max_probs >= threshold