"""
Predictor for inference on new images
"""

import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union, Optional, Tuple


class SegmentationPredictor:
    """
    Predictor for semantic segmentation inference.
    
    Args:
        model: Trained segmentation model
        class_mapping: Dictionary mapping class names to IDs
        img_size: Input image size (height, width)
        device: Device to run inference on
    """
    def __init__(self, 
                 model,
                 class_mapping: dict,
                 img_size: Tuple[int, int] = (720, 1080),
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_mapping = class_mapping
        self.img_size = img_size
        self.num_classes = max(class_mapping.values()) + 1
        
        # Create reverse mapping (ID -> name)
        self.id_to_class = {v: k for k, v in class_mapping.items()}
        self.id_to_class[0] = 'background'
        
        # Transform for inference
        self.transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"Predictor initialized on {device}")
        
    def preprocess_image(self, image: Union[np.ndarray, str]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Tuple of (preprocessed tensor, original image)
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR if opencv image
            if image.dtype == np.uint8 and np.max(image) > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original = image.copy()
        
        # Apply transform
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0)
        
        return tensor.to(self.device), original
    
    def predict(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Segmentation mask with class IDs
        """
        tensor, original = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(tensor)
            prediction = output.argmax(dim=1)[0]
        
        # Convert to numpy
        mask = prediction.cpu().numpy()
        
        # Resize to original size
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def predict_with_confidence(self, image: Union[np.ndarray, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation mask with confidence scores.
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Tuple of (mask, confidence_map)
        """
        tensor, original = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)
        
        # Convert to numpy
        mask = prediction.cpu().numpy()
        confidence_map = confidence.cpu().numpy()
        
        # Resize to original size
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
            confidence_map = cv2.resize(confidence_map, (original.shape[1], original.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        
        return mask, confidence_map
    
    def colorize_mask(self, mask: np.ndarray, alpha: float = 0.5, 
                     image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create colored visualization of segmentation mask.
        
        Args:
            mask: Segmentation mask
            alpha: Transparency for overlay (0-1)
            image: Optional original image for overlay
            
        Returns:
            Colored mask or overlay image
        """
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Color palette
        colors = [
            [0, 0, 0],       # background - black
            [255, 0, 0],     # class 1 - red
            [0, 255, 0],     # class 2 - green
            [0, 0, 255],     # class 3 - blue
            [255, 255, 0],   # class 4 - yellow
            [255, 0, 255],   # class 5 - magenta
            [0, 255, 255],   # class 6 - cyan
            [128, 0, 0],     # class 7 - maroon
            [0, 128, 0],     # class 8 - dark green
            [0, 0, 128],     # class 9 - navy
        ]
        
        for cls in range(min(self.num_classes, len(colors))):
            color_mask[mask == cls] = colors[cls]
        
        if image is not None:
            # Create overlay
            return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        else:
            return color_mask
    
    def visualize_prediction(self, image: Union[np.ndarray, str], 
                           alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize prediction with overlay.
        
        Args:
            image: Input image
            alpha: Transparency for overlay
            
        Returns:
            Tuple of (original_image, mask, overlay)
        """
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            original = image.copy()
        
        mask = self.predict(image)
        overlay = self.colorize_mask(mask, alpha=alpha, image=original)
        
        return original, mask, overlay
    
    def get_class_masks(self, mask: np.ndarray) -> dict:
        """
        Extract binary masks for each class.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary mapping class names to binary masks
        """
        class_masks = {}
        
        for class_id, class_name in self.id_to_class.items():
            binary_mask = (mask == class_id).astype(np.uint8) * 255
            class_masks[class_name] = binary_mask
        
        return class_masks
    
    def get_statistics(self, mask: np.ndarray) -> dict:
        """
        Get statistics about segmentation.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with class statistics
        """
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        stats = {}
        for class_id, count in zip(unique, counts):
            class_name = self.id_to_class.get(class_id, f"class_{class_id}")
            stats[class_name] = {
                'class_id': int(class_id),
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        return stats


def load_model_for_inference(checkpoint_path: str, 
                             model_class,
                             num_classes: int,
                             class_mapping: dict,
                             img_size: Tuple[int, int] = (720, 1080),
                             device: str = 'cuda',
                             **model_kwargs) -> SegmentationPredictor:
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to instantiate
        num_classes: Number of classes
        class_mapping: Dictionary mapping class names to IDs
        img_size: Input image size
        device: Device to run on
        **model_kwargs: Additional arguments for model
        
    Returns:
        SegmentationPredictor instance
    """
    # Create model
    model = model_class(num_classes=num_classes, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    # Create predictor
    predictor = SegmentationPredictor(model, class_mapping, img_size, device)
    
    return predictor
