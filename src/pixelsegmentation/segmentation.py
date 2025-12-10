"""
Semantic Segmentation using DeepLabV3+
"""

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional, Tuple


class PixelSegmenter:
    """
    Pixel-wise semantic segmentation using DeepLabV3+ (ResNet101 backbone).
    
    Supports COCO dataset classes (21 classes including background).
    """
    
    # PASCAL VOC class names (DeepLabV3 default)
    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, device: Optional[str] = None, weights: str = 'DEFAULT'):
        """
        Initialize the segmentation model.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            weights: Pretrained weights to use ('DEFAULT' or 'COCO_WITH_VOC_LABELS_V1')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pretrained DeepLabV3+ model
        if weights == 'DEFAULT':
            weights_enum = DeepLabV3_ResNet101_Weights.DEFAULT
        else:
            weights_enum = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            
        self.model = deeplabv3_resnet101(weights=weights_enum)
        self.model.to(self.device)
        self.model.eval()
        
        # Get preprocessing transforms from weights
        self.preprocess = weights_enum.transforms()
        
        print(f"DeepLabV3+ model loaded on {self.device}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply preprocessing
        input_tensor = self.preprocess(image).unsqueeze(0)
        return input_tensor.to(self.device)
    
    def segment(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Segmentation mask as numpy array with class indices
        """
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get class predictions
        segmentation_mask = output.argmax(0).cpu().numpy()
        
        return segmentation_mask
    
    def segment_with_confidence(self, image: Union[np.ndarray, Image.Image, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform semantic segmentation and return confidence scores.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Tuple of (segmentation_mask, confidence_scores)
        """
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=0)
        
        # Get class predictions and confidence
        confidence, segmentation_mask = probabilities.max(0)
        
        return segmentation_mask.cpu().numpy(), confidence.cpu().numpy()
    
    def colorize_mask(self, mask: np.ndarray, palette: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert segmentation mask to color image.
        
        Args:
            mask: Segmentation mask with class indices
            palette: Optional color palette (num_classes x 3). If None, uses default PASCAL VOC colors
            
        Returns:
            RGB color image
        """
        if palette is None:
            palette = self._create_pascal_palette()
        
        color_mask = palette[mask]
        return color_mask.astype(np.uint8)
    
    def _create_pascal_palette(self) -> np.ndarray:
        """Create PASCAL VOC color palette."""
        palette = np.zeros((256, 3), dtype=np.uint8)
        
        # PASCAL VOC colors
        colors = [
            [0, 0, 0],       # background
            [128, 0, 0],     # aeroplane
            [0, 128, 0],     # bicycle
            [128, 128, 0],   # bird
            [0, 0, 128],     # boat
            [128, 0, 128],   # bottle
            [0, 128, 128],   # bus
            [128, 128, 128], # car
            [64, 0, 0],      # cat
            [192, 0, 0],     # chair
            [64, 128, 0],    # cow
            [192, 128, 0],   # diningtable
            [64, 0, 128],    # dog
            [192, 0, 128],   # horse
            [64, 128, 128],  # motorbike
            [192, 128, 128], # person
            [0, 64, 0],      # pottedplant
            [128, 64, 0],    # sheep
            [0, 192, 0],     # sofa
            [128, 192, 0],   # train
            [0, 64, 128],    # tvmonitor
        ]
        
        palette[:len(colors)] = colors
        return palette
    
    def segment_and_visualize(self, image: Union[np.ndarray, Image.Image, str], 
                             alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image and create visualization overlay.
        
        Args:
            image: Input image
            alpha: Transparency of overlay (0-1)
            
        Returns:
            Tuple of (original_image, segmentation_mask, overlay_image)
        """
        # Load and convert image
        if isinstance(image, str):
            orig_image = cv2.imread(image)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            orig_image = np.array(image)
        else:
            orig_image = image.copy()
        
        # Perform segmentation
        mask = self.segment(image)
        
        # Resize mask to match original image size
        if mask.shape != orig_image.shape[:2]:
            mask = cv2.resize(mask, (orig_image.shape[1], orig_image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        color_mask = self.colorize_mask(mask)
        
        # Create overlay
        overlay = cv2.addWeighted(orig_image, 1 - alpha, color_mask, alpha, 0)
        
        return orig_image, mask, overlay
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID."""
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"Unknown class {class_id}"
    
    def extract_class_mask(self, mask: np.ndarray, class_id: int) -> np.ndarray:
        """
        Extract binary mask for a specific class.
        
        Args:
            mask: Segmentation mask
            class_id: Class ID to extract
            
        Returns:
            Binary mask (0 or 255)
        """
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        return binary_mask
    
    def get_class_statistics(self, mask: np.ndarray) -> dict:
        """
        Get statistics about classes present in the mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with class statistics
        """
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        stats = {}
        for class_id, count in zip(unique, counts):
            stats[self.get_class_name(class_id)] = {
                'class_id': int(class_id),
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        return stats
