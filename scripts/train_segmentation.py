"""
Training script for pixel segmentation model
"""

import os
import json
import torch
import argparse
from pathlib import Path

from src.pixelseg import SegmentationModel, SegmentationTrainer
from src.pixelseg.dataset import create_dataloaders
from src.pixelseg.model import create_model


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--train_images', type=str, required=True, help='Training images directory')
    parser.add_argument('--train_masks', type=str, required=True, help='Training masks directory')
    parser.add_argument('--val_images', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_masks', type=str, required=True, help='Validation masks directory')
    parser.add_argument('--class_mapping', type=str, required=True, help='Path to class_id.json')
    parser.add_argument('--output_dir', type=str, default='outputs/segmentation', help='Output directory')
    parser.add_argument('--architecture', type=str, default='unet', choices=['unet', 'mobilenet', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--img_height', type=int, default=720, help='Image height')
    parser.add_argument('--img_width', type=int, default=1080, help='Image width')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = max(class_mapping.values()) + 1  # +1 for background
    
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Training images: {args.train_images}")
    print(f"Training masks: {args.train_masks}")
    print(f"Validation images: {args.val_images}")
    print(f"Validation masks: {args.val_masks}")
    print(f"Class mapping: {class_mapping}")
    print(f"Number of classes: {num_classes}")
    print(f"Architecture: {args.architecture}")
    print(f"Image size: {args.img_height}x{args.img_width}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_image_dir=args.train_images,
        train_mask_dir=args.train_masks,
        val_image_dir=args.val_images,
        val_mask_dir=args.val_masks,
        class_mapping=class_mapping,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating {args.architecture} model...")
    model = create_model(num_classes, architecture=args.architecture)
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=args.device,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Save config
    config = vars(args)
    config['class_mapping'] = class_mapping
    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train
    trainer.train(num_epochs=args.epochs, save_interval=5)


if __name__ == '__main__':
    main()
