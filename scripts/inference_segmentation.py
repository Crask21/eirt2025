"""
Inference script for trained segmentation model
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.pixelseg.model import create_model
from src.pixelseg.predictor import load_model_for_inference


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    class_mapping = config['class_mapping']
    num_classes = max(class_mapping.values()) + 1
    architecture = config.get('architecture', 'unet')
    img_size = (config['img_height'], config['img_width'])
    
    print(f"Loading model: {architecture}")
    print(f"Classes: {class_mapping}")
    print(f"Image size: {img_size}")
    
    # Load model
    model_class = lambda **kwargs: create_model(num_classes, architecture=architecture, **kwargs)
    predictor = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        model_class=lambda num_classes, **kwargs: create_model(num_classes, architecture=architecture, **kwargs),
        num_classes=num_classes,
        class_mapping=class_mapping,
        img_size=img_size
    )
    
    # Run inference
    print(f"\nProcessing image: {args.image}")
    original, mask, overlay = predictor.visualize_prediction(args.image, alpha=0.5)
    
    # Get statistics
    stats = predictor.get_statistics(mask)
    print("\nSegmentation statistics:")
    for class_name, info in stats.items():
        print(f"  {class_name}: {info['percentage']:.2f}% ({info['pixel_count']} pixels)")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(args.image).stem
        
        # Save mask
        mask_path = output_dir / f"{base_name}_mask.npy"
        np.save(mask_path, mask)
        print(f"\nMask saved to: {mask_path}")
        
        # Save visualization
        overlay_path = output_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Overlay saved to: {overlay_path}")
        
        # Save colored mask
        color_mask = predictor.colorize_mask(mask)
        color_path = output_dir / f"{base_name}_color.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        print(f"Color mask saved to: {color_path}")
    
    # Visualize
    if args.visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(predictor.colorize_mask(mask))
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
