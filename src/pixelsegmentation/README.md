# Pixel Segmentation Module

Semantic segmentation using DeepLabV3+ with ResNet101 backbone.

## Features

- **Pre-trained DeepLabV3+**: Uses PyTorch's pretrained model on PASCAL VOC dataset
- **21 Classes**: Supports background + 20 object classes
- **Easy-to-use API**: Simple interface for segmentation tasks
- **Visualization**: Built-in colorization and overlay functions
- **Batch Processing**: Process multiple images efficiently
- **Confidence Scores**: Get segmentation confidence for each pixel

## Installation

Install required dependencies:

```bash
pip install torch torchvision opencv-python pillow numpy scipy
```

## Quick Start

```python
from src.pixelsegmentation import PixelSegmenter

# Initialize segmenter
segmenter = PixelSegmenter()

# Segment an image
mask = segmenter.segment("path/to/image.jpg")

# Segment with visualization
original, mask, overlay = segmenter.segment_and_visualize("path/to/image.jpg", alpha=0.5)

# Get class statistics
stats = segmenter.get_class_statistics(mask)
print(stats)
```

## Classes

The model supports 21 classes from PASCAL VOC:

0. background
1. aeroplane
2. bicycle
3. bird
4. boat
5. bottle
6. bus
7. car
8. cat
9. chair
10. cow
11. diningtable
12. dog
13. horse
14. motorbike
15. person
16. pottedplant
17. sheep
18. sofa
19. train
20. tvmonitor

## API Reference

### PixelSegmenter

**`__init__(device=None, weights='DEFAULT')`**
- `device`: 'cuda', 'cpu', or None for auto-detect
- `weights`: 'DEFAULT' or 'COCO_WITH_VOC_LABELS_V1'

**`segment(image)`**
- Returns segmentation mask as numpy array with class indices

**`segment_with_confidence(image)`**
- Returns (mask, confidence_scores)

**`segment_and_visualize(image, alpha=0.5)`**
- Returns (original_image, mask, overlay)

**`colorize_mask(mask, palette=None)`**
- Convert mask to RGB color image

**`extract_class_mask(mask, class_id)`**
- Extract binary mask for specific class

**`get_class_statistics(mask)`**
- Get pixel counts and percentages for each class

## Examples

### Basic Segmentation

```python
from src.pixelsegmentation import PixelSegmenter
import matplotlib.pyplot as plt

segmenter = PixelSegmenter()

# Segment image
mask = segmenter.segment("image.jpg")

# Visualize
color_mask = segmenter.colorize_mask(mask)
plt.imshow(color_mask)
plt.show()
```

### Extract Specific Class

```python
# Extract all persons from image
person_mask = segmenter.extract_class_mask(mask, class_id=15)

# Save binary mask
import cv2
cv2.imwrite("person_mask.png", person_mask)
```

### Batch Processing

```python
from src.pixelsegmentation.utils import batch_segment
import glob

image_paths = glob.glob("images/*.jpg")
masks = batch_segment(segmenter, image_paths, output_dir="output")
```

### Get Statistics

```python
stats = segmenter.get_class_statistics(mask)
for class_name, info in stats.items():
    print(f"{class_name}: {info['percentage']:.2f}% ({info['pixel_count']} pixels)")
```

## Utilities

The `utils.py` module provides additional helper functions:

- `resize_with_aspect_ratio()`: Resize maintaining aspect ratio
- `batch_segment()`: Process multiple images
- `apply_morphological_operations()`: Clean up masks
- `mask_to_contours()`: Extract contours from masks
- `create_overlay()`: Custom color overlays

## Performance

- GPU (CUDA): ~30-50 FPS for 512x512 images
- CPU: ~2-5 FPS for 512x512 images

For faster inference, use GPU and consider resizing large images.
