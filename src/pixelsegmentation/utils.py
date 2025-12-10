"""
Utility functions for pixel segmentation
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def resize_with_aspect_ratio(image: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size for the longer edge
        
    Returns:
        Tuple of (resized_image, original_size)
    """
    h, w = image.shape[:2]
    original_size = (h, w)
    
    if max(h, w) > target_size:
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, original_size
    
    return image, original_size


def batch_segment(segmenter, image_paths: List[str], output_dir: Optional[str] = None) -> List[np.ndarray]:
    """
    Perform batch segmentation on multiple images.
    
    Args:
        segmenter: PixelSegmenter instance
        image_paths: List of image file paths
        output_dir: Optional directory to save results
        
    Returns:
        List of segmentation masks
    """
    import os
    from PIL import Image
    
    masks = []
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Segment image
        mask = segmenter.segment(img_path)
        masks.append(mask)
        
        # Save if output directory is specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save mask as numpy array
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(output_dir, f"{basename}_mask.npy")
            np.save(mask_path, mask)
            
            # Save colored visualization
            color_mask = segmenter.colorize_mask(mask)
            color_path = os.path.join(output_dir, f"{basename}_color.png")
            Image.fromarray(color_mask).save(color_path)
    
    return masks


def apply_morphological_operations(mask: np.ndarray, operation: str = 'close', 
                                   kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological operations to clean up segmentation mask.
    
    Args:
        mask: Binary or multi-class segmentation mask
        operation: 'open', 'close', 'dilate', 'erode'
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilate':
        return cv2.dilate(mask, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(mask, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def mask_to_contours(mask: np.ndarray, class_id: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract contours from segmentation mask.
    
    Args:
        mask: Segmentation mask
        class_id: Optional class ID to extract contours for
        
    Returns:
        List of contours
    """
    if class_id is not None:
        binary_mask = (mask == class_id).astype(np.uint8) * 255
    else:
        binary_mask = mask.astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def create_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0),
                  alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay of mask on image with specified color.
    
    Args:
        image: Original image
        mask: Binary mask
        color: RGB color for overlay
        alpha: Transparency (0-1)
        
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # Create colored mask
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    
    return result
