"""
Segmentation Visualization Script

This script iterates through the first 100 images from the KITTI dataset
and displays the original image alongside the segmentation mask with legends.
Use keyboard arrows or Enter to navigate between images.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pointpainting import (
    SemanticSegmentationModel,
    KITTIDataLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationVisualizer:
    """Interactive visualizer for segmentation results."""
    
    def __init__(self, kitti_root: str, max_samples: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            kitti_root: Root directory of KITTI dataset
            max_samples: Maximum number of samples to visualize
        """
        self.kitti_root = kitti_root
        self.max_samples = max_samples
        self.current_index = 0
        
        # Initialize data loader
        logger.info("Loading KITTI dataset...")
        self.data_loader = KITTIDataLoader(kitti_root, split="training")
        
        # Get sample IDs (limit to max_samples)
        self.sample_ids = self.data_loader.sample_ids[:max_samples]
        logger.info(f"Loaded {len(self.sample_ids)} samples")
        
        # Initialize segmentation model with 21 COCO classes (use pretrained properly)
        logger.info("Loading segmentation model...")
        self.seg_model = SemanticSegmentationModel(
            num_classes=21,  # Use original COCO classes
            device="cuda" if __import__('torch').cuda.is_available() else "cpu"
        )
        
        # COCO class names (Pascal VOC + COCO subset used by DeepLabV3)
        self.coco_class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # KITTI class names and mapping from COCO
        self.class_names = ["Background", "Car", "Pedestrian", "Cyclist"]
        
        # Map COCO classes to KITTI classes
        # COCO class 7 = car, 15 = person, 2 = bicycle (for cyclist)
        self.coco_to_kitti = {
            0: 0,   # background -> background
            7: 1,   # car -> car
            6: 1,   # bus -> car
            19: 1,  # train -> car
            15: 2,  # person -> pedestrian
            2: 3,   # bicycle -> cyclist
            14: 3,  # motorbike -> cyclist
        }
        
        # Class colors for visualization
        self.class_colors = [
            [0, 0, 0],        # Background - black
            [255, 0, 0],      # Car - red
            [0, 255, 0],      # Pedestrian - green
            [0, 0, 255]       # Cyclist - blue
        ]
        
        # Setup figure
        self.setup_figure()
        
    def setup_figure(self):
        """Setup matplotlib figure and axes."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.fig.suptitle(f'KITTI Segmentation Visualization (Sample 0/{len(self.sample_ids)})', 
                         fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            Patch(facecolor=np.array(color)/255.0, label=name)
            for name, color in zip(self.class_names, self.class_colors)
        ]
        self.fig.legend(handles=legend_elements, loc='upper center', 
                       ncol=4, bbox_to_anchor=(0.5, 0.95))
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
    def segmentation_to_color(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Convert segmentation probabilities to colored image.
        
        Args:
            segmentation: Segmentation scores of shape (H, W, num_classes) with 21 COCO classes
            
        Returns:
            Colored segmentation image (H, W, 3) with KITTI classes
        """
        # Get COCO class predictions (argmax over 21 classes)
        coco_class_pred = np.argmax(segmentation, axis=2)  # (H, W)
        
        # Map COCO classes to KITTI classes
        h, w = coco_class_pred.shape
        kitti_class_pred = np.zeros((h, w), dtype=np.uint8)
        
        for coco_class, kitti_class in self.coco_to_kitti.items():
            mask = coco_class_pred == coco_class
            kitti_class_pred[mask] = kitti_class
        
        # Create colored image using KITTI classes
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.class_colors):
            mask = kitti_class_pred == class_id
            colored[mask] = color
            
        return colored
    
    def get_class_statistics(self, segmentation: np.ndarray) -> dict:
        """
        Compute class distribution statistics.
        
        Args:
            segmentation: Segmentation scores of shape (H, W, num_classes) with 21 COCO classes
            
        Returns:
            Dictionary with KITTI class statistics
        """
        # Get COCO class predictions
        coco_class_pred = np.argmax(segmentation, axis=2)
        
        # Map to KITTI classes
        h, w = coco_class_pred.shape
        kitti_class_pred = np.zeros((h, w), dtype=np.uint8)
        
        for coco_class, kitti_class in self.coco_to_kitti.items():
            mask = coco_class_pred == coco_class
            kitti_class_pred[mask] = kitti_class
        
        # Compute statistics for KITTI classes
        total_pixels = kitti_class_pred.size
        
        stats = {}
        for class_id, class_name in enumerate(self.class_names):
            count = np.sum(kitti_class_pred == class_id)
            percentage = (count / total_pixels) * 100
            stats[class_name] = {
                'count': count,
                'percentage': percentage
            }
        
        return stats
    
    def visualize_sample(self, sample_id: str):
        """
        Visualize a single sample with its segmentation.
        
        Args:
            sample_id: Sample ID to visualize
        """
        # Load image
        logger.info(f"Processing sample {sample_id}...")
        sample_data = self.data_loader.load_sample(sample_id)
        image = sample_data["image"]
        
        # Generate segmentation
        logger.info("Generating segmentation...")
        segmentation = self.seg_model.predict(
            image, 
            return_probabilities=True
        )
        
        # Debug: Print segmentation statistics
        logger.info(f"Segmentation shape: {segmentation.shape}")
        logger.info(f"Segmentation range: [{segmentation.min():.4f}, {segmentation.max():.4f}]")
        class_pred = np.argmax(segmentation, axis=2)
        unique_classes, counts = np.unique(class_pred, return_counts=True)
        logger.info(f"Unique predicted classes: {unique_classes}")
        logger.info(f"Class counts: {counts}")
        
        # Check probability values for a sample of pixels
        sample_probs = segmentation[segmentation.shape[0]//2, segmentation.shape[1]//2, :]
        logger.info(f"Sample pixel probabilities: {sample_probs}")
        
        # Convert segmentation to colored image
        seg_colored = self.segmentation_to_color(segmentation)
        
        # Get statistics
        stats = self.get_class_statistics(segmentation)
        
        # Display original image
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax1.set_title(f'Original Image - Sample {sample_id}', fontweight='bold')
        self.ax1.axis('off')
        
        # Display segmentation
        self.ax2.clear()
        self.ax2.imshow(seg_colored)
        
        # Add statistics to title
        stats_text = " | ".join([
            f"{name}: {stats[name]['percentage']:.1f}%" 
            for name in self.class_names if stats[name]['percentage'] > 0.1
        ])
        self.ax2.set_title(f'Segmentation - {stats_text}', fontweight='bold')
        self.ax2.axis('off')
        
        # Update figure title
        self.fig.suptitle(
            f'KITTI Segmentation Visualization - Sample {self.current_index + 1}/{len(self.sample_ids)} (ID: {sample_id})\n'
            f'Press → or Enter for next, ← for previous, Q to quit',
            fontsize=12, fontweight='bold'
        )
        
        # Log statistics
        logger.info("Class distribution:")
        for name, stat in stats.items():
            if stat['percentage'] > 0.1:
                logger.info(f"  {name}: {stat['count']} pixels ({stat['percentage']:.2f}%)")
        
        plt.draw()
    
    def on_key_press(self, event):
        """
        Handle keyboard events for navigation.
        
        Args:
            event: Keyboard event
        """
        if event.key in ['right', 'enter']:
            # Next sample
            self.current_index = (self.current_index + 1) % len(self.sample_ids)
            self.visualize_sample(self.sample_ids[self.current_index])
            
        elif event.key == 'left':
            # Previous sample
            self.current_index = (self.current_index - 1) % len(self.sample_ids)
            self.visualize_sample(self.sample_ids[self.current_index])
            
        elif event.key in ['q', 'escape']:
            # Quit
            logger.info("Closing visualizer...")
            plt.close(self.fig)
            
        elif event.key == 'home':
            # Go to first sample
            self.current_index = 0
            self.visualize_sample(self.sample_ids[self.current_index])
            
        elif event.key == 'end':
            # Go to last sample
            self.current_index = len(self.sample_ids) - 1
            self.visualize_sample(self.sample_ids[self.current_index])
    
    def run(self):
        """Start the interactive visualization."""
        logger.info("Starting interactive visualization...")
        logger.info("Controls:")
        logger.info("  → or Enter: Next sample")
        logger.info("  ←: Previous sample")
        logger.info("  Home: First sample")
        logger.info("  End: Last sample")
        logger.info("  Q or Escape: Quit")
        
        # Show first sample
        self.visualize_sample(self.sample_ids[self.current_index])
        plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize KITTI segmentation results interactively"
    )
    parser.add_argument(
        "--kitti-root",
        type=str,
        default="./data/kitti",
        help="Root directory of KITTI dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to visualize (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Verify KITTI directory exists
    kitti_path = Path(args.kitti_root)
    if not kitti_path.exists():
        logger.error(f"KITTI directory not found: {kitti_path}")
        logger.error("Please provide the correct path using --kitti-root")
        sys.exit(1)
    
    # Create visualizer and run
    visualizer = SegmentationVisualizer(
        kitti_root=args.kitti_root,
        max_samples=args.max_samples
    )
    visualizer.run()


if __name__ == "__main__":
    main()
