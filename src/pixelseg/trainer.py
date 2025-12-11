"""
Training loop for segmentation model
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path


class SegmentationTrainer:
    """
    Trainer class for semantic segmentation.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        device: Device to train on
        learning_rate: Initial learning rate
        output_dir: Directory to save checkpoints and logs
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 num_classes,
                 device='cuda',
                 learning_rate=1e-4,
                 output_dir='outputs/segmentation'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function - CrossEntropyLoss with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Training state
        self.epoch = 0
        self.best_miou = 0.0
        
        print(f"Trainer initialized on {device}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]")
        
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_pixels / total_pixels
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_pixels / total_pixels
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        # IoU computation
        intersection = torch.zeros(self.num_classes).to(self.device)
        union = torch.zeros(self.num_classes).to(self.device)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]")
        
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Statistics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()
                
                # IoU per class
                for cls in range(self.num_classes):
                    pred_mask = (predictions == cls)
                    true_mask = (masks == cls)
                    
                    intersection[cls] += (pred_mask & true_mask).sum().item()
                    union[cls] += (pred_mask | true_mask).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_pixels / total_pixels
        
        # Compute mIoU
        iou_per_class = intersection / (union + 1e-10)
        miou = iou_per_class.mean().item()
        
        return avg_loss, accuracy, miou, iou_per_class.cpu().numpy()
    
    def train(self, num_epochs, save_interval=5):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_miou, iou_per_class = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_miou)
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)
            self.writer.add_scalar('mIoU/val', val_miou, self.epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.epoch)
            
            print(f"\nEpoch {self.epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val mIoU: {val_miou:.4f}")
            print(f"IoU per class: {iou_per_class}")
            
            # Save checkpoint
            if self.epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint('best_model.pth')
                print(f"✓ New best model saved with mIoU: {val_miou:.4f}")
        
        print("\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        self.writer.close()
        
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_miou = checkpoint['best_miou']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, best mIoU: {self.best_miou:.4f}")


def calculate_class_weights(train_loader, num_classes, device='cuda'):
    """
    Calculate class weights based on inverse frequency.
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to put weights on
        
    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)
    
    print("Calculating class weights...")
    for _, masks, _ in tqdm(train_loader):
        for cls in range(num_classes):
            class_counts[cls] += (masks == cls).sum().item()
    
    # Inverse frequency
    total = class_counts.sum()
    class_weights = total / (class_counts + 1e-10)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return class_weights.to(device)
