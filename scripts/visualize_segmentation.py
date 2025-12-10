"""
Visualize segmentation predictions on validation dataset
"""

import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pixelseg.model import create_model
from src.pixelseg.predictor import SegmentationPredictor
from src.pixelseg.dataset import SegmentationDataset, get_validation_transforms


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load trained model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_mapping = config['class_mapping']
    num_classes = max(class_mapping.values()) + 1
    architecture = config.get('architecture', 'unet')
    img_size = (config['img_height'], config['img_width'])
    
    print(f"Loading model: {architecture}")
    print(f"Classes: {class_mapping}")
    print(f"Image size: {img_size}")
    
    # Create model
    model = create_model(num_classes, architecture=architecture)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    # Create predictor
    predictor = SegmentationPredictor(model, class_mapping, img_size, device)
    
    return predictor, config


def create_color_palette(num_classes):
    """Create color palette for visualization"""
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
    
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    palette[:len(colors)] = colors[:num_classes]
    
    return palette


def visualize_sample(image, ground_truth, prediction, class_mapping, palette, save_path=None):
    """Visualize a single sample with ground truth and prediction"""
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # Convert CHW to HWC
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Denormalize
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # Convert masks to tensors if needed
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    # Create colored masks
    gt_colored = palette[ground_truth]
    pred_colored = palette[prediction]
    
    # Create overlays
    overlay_gt = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, gt_colored, 0.4, 0)
    overlay_pred = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, pred_colored, 0.4, 0)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlay_gt)
    axes[0, 2].set_title('GT Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_colored)
    axes[1, 1].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend
    legend_elements = []
    id_to_class = {v: k for k, v in class_mapping.items()}
    id_to_class[0] = 'background'
    
    for class_id, color in enumerate(palette):
        if class_id < len(id_to_class):
            class_name = id_to_class.get(class_id, f'class_{class_id}')
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color/255, label=class_name))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), 
              bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def calculate_metrics(ground_truth, prediction, num_classes):
    """Calculate segmentation metrics"""
    ground_truth = ground_truth.to(device='cpu')
    prediction = prediction.to(device='cpu')
    print(ground_truth.device)
    print(prediction.device)
    # Pixel accuracy
    correct = (prediction == ground_truth).sum()
    total = ground_truth.numel() if torch.is_tensor(ground_truth) else ground_truth.size
    accuracy = correct / total if torch.is_tensor(correct) else correct.item() / total
    
    # IoU per class
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = (prediction == cls)
        true_mask = (ground_truth == cls)
        
        intersection[cls] = (pred_mask & true_mask).sum()
        union[cls] = (pred_mask | true_mask).sum()
    
    iou_per_class = intersection / (union + 1e-10)
    miou = iou_per_class.mean()
    
    return accuracy, miou, iou_per_class


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize segmentation predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--val_images', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_masks', type=str, required=True, help='Validation masks directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SEGMENTATION VISUALIZATION")
    print("="*80)
    
    # Load model
    predictor, config = load_model(args.checkpoint, args.config, args.device)
    
    class_mapping = config['class_mapping']
    num_classes = max(class_mapping.values()) + 1
    img_size = (config['img_height'], config['img_width'])
    
    # Create dataset
    print(f"\nLoading validation dataset...")
    transform = get_validation_transforms(img_size)
    dataset = SegmentationDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        class_mapping=class_mapping,
        transform=transform
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create color palette
    palette = create_color_palette(num_classes)
    
    # Create save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {save_dir}")
    
    # Process samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nProcessing {num_samples} samples...")
    
    all_metrics = {
        'accuracy': [],
        'miou': [],
        'iou_per_class': []
    }
    
    # Set interactive mode
    if args.interactive:
        plt.ion()
    
    for idx in tqdm(range(num_samples)):
        # Load sample
        image, ground_truth, img_name = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(args.device)
            output = predictor.model(image_batch)
            prediction = output.argmax(dim=1)[0]
        
        # Calculate metrics
        accuracy, miou, iou_per_class = calculate_metrics(
            ground_truth, prediction, num_classes
        )
        
        all_metrics['accuracy'].append(accuracy)
        all_metrics['miou'].append(miou)
        all_metrics['iou_per_class'].append(iou_per_class)
        
        print(f"\n{img_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  mIoU: {miou:.4f}")
        print(f"  IoU per class: {iou_per_class}")
        
        # Visualize
        save_path = None
        if args.save_dir:
            save_path = save_dir / f"sample_{idx:03d}_{Path(img_name).stem}.png"
        
        visualize_sample(
            image, ground_truth, prediction, 
            class_mapping, palette, save_path
        )
        
        if not args.interactive and not args.save_dir:
            # Close plot if not interactive and not saving
            plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Accuracy: {np.mean(all_metrics['accuracy']):.4f} ± {np.std(all_metrics['accuracy']):.4f}")
    print(f"Average mIoU: {np.mean(all_metrics['miou']):.4f} ± {np.std(all_metrics['miou']):.4f}")
    
    avg_iou_per_class = np.mean(all_metrics['iou_per_class'], axis=0)
    print(f"\nAverage IoU per class:")
    
    id_to_class = {v: k for k, v in class_mapping.items()}
    id_to_class[0] = 'background'
    
    for class_id, iou in enumerate(avg_iou_per_class):
        class_name = id_to_class.get(class_id, f'class_{class_id}')
        print(f"  {class_name}: {iou:.4f}")
    
    if args.interactive:
        plt.ioff()
        plt.show()
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
