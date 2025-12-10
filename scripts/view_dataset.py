"""
Interactive viewer for cycling through dataset images with masks
Click 'Next' button or press 'n' to go to next image
Press 'q' to quit
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DatasetViewer:
    def __init__(self, image_dir, mask_dir, class_mapping):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.class_mapping = class_mapping
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.current_idx = 0
        self.num_images = len(self.images)
        
        # Create color palette
        self.palette = self.create_color_palette()
        
        # Create reverse mapping for legend
        self.id_to_class = {v: k for k, v in class_mapping.items()}
        self.id_to_class[0] = 'background'
        
        print(f"Loaded {self.num_images} images")
        print(f"Classes: {self.id_to_class}")
        
        # Setup figure
        self.setup_figure()
        
    def create_color_palette(self):
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
        
        num_classes = max(self.class_mapping.values()) + 1
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        palette[:len(colors)] = colors[:num_classes]
        
        return palette
    
    def load_image_and_mask(self, idx):
        """Load image and mask at given index"""
        img_name = self.images[idx]
        
        # Load image
        img_path = self.image_dir / img_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = Path(img_name).stem + '.npy'
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            mask = np.load(mask_path)
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
        else:
            print(f"Warning: Mask not found for {img_name}")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        return image, mask, img_name
    
    def get_statistics(self, mask):
        """Get mask statistics"""
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        stats = []
        for class_id, count in zip(unique, counts):
            class_name = self.id_to_class.get(class_id, f"class_{class_id}")
            percentage = count / total_pixels * 100
            stats.append(f"{class_name}: {percentage:.1f}%")
        
        return ", ".join(stats)
    
    def setup_figure(self):
        """Setup matplotlib figure with button"""
        self.fig = plt.figure(figsize=(16, 8))
        
        # Create grid for subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                                   left=0.05, right=0.95, top=0.92, bottom=0.12)
        
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_mask = self.fig.add_subplot(gs[0, 1])
        self.ax_overlay = self.fig.add_subplot(gs[0, 2])
        
        # Bottom row - same layout
        self.ax_img2 = self.fig.add_subplot(gs[1, 0])
        self.ax_mask2 = self.fig.add_subplot(gs[1, 1])
        self.ax_overlay2 = self.fig.add_subplot(gs[1, 2])
        
        # Create buttons
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.05])
        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.05])
        
        self.btn_next = Button(ax_next, 'Next (n)')
        self.btn_next.on_clicked(self.next_image)
        
        self.btn_prev = Button(ax_prev, 'Previous (p)')
        self.btn_prev.on_clicked(self.prev_image)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial display
        self.update_display()
    
    def update_display(self):
        """Update the display with current image"""
        # Load image and mask
        image, mask, img_name = self.load_image_and_mask(self.current_idx)
        
        # Create colored mask
        color_mask = self.palette[mask]
        
        # Create overlay with different alpha values
        overlay_light = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
        overlay_heavy = cv2.addWeighted(image, 0.4, color_mask, 0.6, 0)
        
        # Get statistics
        stats = self.get_statistics(mask)
        
        # Clear axes
        for ax in [self.ax_img, self.ax_mask, self.ax_overlay, 
                   self.ax_img2, self.ax_mask2, self.ax_overlay2]:
            ax.clear()
            ax.axis('off')
        
        # Top row - lighter overlay
        self.ax_img.imshow(image)
        self.ax_img.set_title('Original Image', fontsize=12, fontweight='bold')
        
        self.ax_mask.imshow(color_mask)
        self.ax_mask.set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        
        self.ax_overlay.imshow(overlay_light)
        self.ax_overlay.set_title('Overlay (30% mask)', fontsize=12, fontweight='bold')
        
        # Bottom row - heavier overlay
        self.ax_img2.imshow(image)
        self.ax_img2.set_title('Original Image', fontsize=12, fontweight='bold')
        
        self.ax_mask2.imshow(color_mask)
        self.ax_mask2.set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        
        self.ax_overlay2.imshow(overlay_heavy)
        self.ax_overlay2.set_title('Overlay (60% mask)', fontsize=12, fontweight='bold')
        
        # Update title
        self.fig.suptitle(
            f'Image {self.current_idx + 1}/{self.num_images}: {img_name}\n{stats}',
            fontsize=14, fontweight='bold'
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for class_id, color in enumerate(self.palette):
            if class_id in self.id_to_class:
                class_name = self.id_to_class[class_id]
                legend_elements.append(Patch(facecolor=color/255, label=class_name))
        
        self.fig.legend(handles=legend_elements, loc='lower center', 
                       ncol=len(legend_elements), bbox_to_anchor=(0.35, 0.02), 
                       fontsize=10, frameon=True)
        
        # Refresh canvas
        self.fig.canvas.draw_idle()
    
    def next_image(self, event=None):
        """Go to next image"""
        self.current_idx = (self.current_idx + 1) % self.num_images
        self.update_display()
    
    def prev_image(self, event=None):
        """Go to previous image"""
        self.current_idx = (self.current_idx - 1) % self.num_images
        self.update_display()
    
    def jump_next_100(self):
        """Jump forward 100 images"""
        self.current_idx = (self.current_idx + 100) % self.num_images
        self.update_display()
    
    def jump_prev_100(self):
        """Jump backward 100 images"""
        self.current_idx = (self.current_idx - 100) % self.num_images
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'n' or event.key == 'right':
            self.next_image()
        elif event.key == 'p' or event.key == 'left':
            self.prev_image()
        elif event.key == 'N':
            self.jump_next_100()
        elif event.key == 'P':
            self.jump_prev_100()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def show(self):
        """Show the viewer"""
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive dataset viewer')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--masks', type=str, required=True, help='Masks directory (.npy files)')
    parser.add_argument('--class_mapping', type=str, required=True, help='Path to class_id.json')
    
    args = parser.parse_args()
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    print("="*80)
    print("INTERACTIVE DATASET VIEWER")
    print("="*80)
    print("Controls:")
    print("  - Click 'Next' button or press 'n' / Right arrow: Next image")
    print("  - Click 'Previous' button or press 'p' / Left arrow: Previous image")
    print("  - Press 'q': Quit")
    print("="*80)
    
    # Create viewer
    viewer = DatasetViewer(args.images, args.masks, class_mapping)
    
    # Show
    viewer.show()


if __name__ == '__main__':
    main()
