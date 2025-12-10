# Pixel Segmentation Module

Custom semantic segmentation for specific object classes with training pipeline.

## Overview

This module provides:
- Custom U-Net, MobileNet, and ResNet-based segmentation models
- Training pipeline with data augmentation
- Inference and visualization tools
- Support for custom class mappings

## Installation

```bash
pip install torch torchvision opencv-python pillow numpy albumentations tensorboard tqdm
```

## Dataset Structure

```
dataset/
├── train/
│   ├── images/         # RGB images (.png, .jpg)
│   └── masks/          # Segmentation masks (.npy)
└── val/
    ├── images/
    └── masks/
```

Mask files should be `.npy` arrays of shape `(H, W)` where each pixel value corresponds to a class ID.

## Class Mapping

Create `class_id.json`:
```json
{
    "chair": 2,
    "table": 4,
    "person": 6,
    "sofa": 8
}
```

Background class (0) is automatically included.

## Training

### Command Line

```bash
python scripts/train_segmentation.py \
    --train_images "path/to/train/images" \
    --train_masks "path/to/train/masks" \
    --val_images "path/to/val/images" \
    --val_masks "path/to/val/masks" \
    --class_mapping "path/to/class_id.json" \
    --output_dir "outputs/segmentation" \
    --architecture unet \
    --img_height 720 \
    --img_width 1080 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4
```

### Python API

```python
import json
from src.pixelseg import SegmentationModel, SegmentationTrainer, create_dataloaders
from src.pixelseg.model import create_model

# Load class mapping
with open('class_id.json', 'r') as f:
    class_mapping = json.load(f)

num_classes = max(class_mapping.values()) + 1

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_image_dir='path/to/train/images',
    train_mask_dir='path/to/train/masks',
    val_image_dir='path/to/val/images',
    val_mask_dir='path/to/val/masks',
    class_mapping=class_mapping,
    batch_size=4,
    img_size=(720, 1080)
)

# Create model
model = create_model(num_classes, architecture='unet')

# Train
trainer = SegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=num_classes,
    device='cuda',
    learning_rate=1e-4,
    output_dir='outputs/segmentation'
)

trainer.train(num_epochs=100)
```

## Inference

### Command Line

```bash
python scripts/inference_segmentation.py \
    --checkpoint "outputs/segmentation/best_model.pth" \
    --config "outputs/segmentation/config.json" \
    --image "path/to/image.jpg" \
    --output "outputs/predictions" \
    --visualize
```

### Python API

```python
from src.pixelseg.predictor import load_model_for_inference
from src.pixelseg.model import create_model

# Load model
predictor = load_model_for_inference(
    checkpoint_path='outputs/segmentation/best_model.pth',
    model_class=lambda num_classes, **kwargs: create_model(num_classes, 'unet', **kwargs),
    num_classes=5,
    class_mapping=class_mapping,
    img_size=(720, 1080)
)

# Predict
mask = predictor.predict('image.jpg')

# Visualize
original, mask, overlay = predictor.visualize_prediction('image.jpg')

# Get statistics
stats = predictor.get_statistics(mask)
```

## Model Architectures

### U-Net (Default)
- Custom U-Net architecture
- Good balance of speed and accuracy
- ~31M parameters

```python
model = create_model(num_classes, architecture='unet', base_channels=64)
```

### MobileNet
- Lightweight, faster inference
- Good for deployment
- ~3-5M parameters

```python
model = create_model(num_classes, architecture='mobilenet')
```

### ResNet (DeepLabV3)
- Most accurate
- Slower inference
- ~40M+ parameters

```python
model = create_model(num_classes, architecture='resnet', backbone='resnet50')
```

## Data Augmentation

Training automatically applies:
- Random horizontal flip
- Random brightness/contrast
- Gaussian noise
- Gaussian blur
- Color jitter

Validation uses only resize and normalization.

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/segmentation/logs
```

Metrics logged:
- Training/validation loss
- Pixel accuracy
- Mean IoU (mIoU)
- Per-class IoU
- Learning rate

## Output Files

Training creates:
- `best_model.pth` - Best model by mIoU
- `checkpoint_epoch_N.pth` - Periodic checkpoints
- `config.json` - Training configuration
- `logs/` - TensorBoard logs

## Tips

1. **Image Size**: Start with smaller size (e.g., 360x540) for faster training, then fine-tune on full resolution
2. **Batch Size**: Reduce if out of memory (try 2 or 1)
3. **Learning Rate**: Default 1e-4 works well, reduce if training unstable
4. **Architecture**: Start with `unet`, use `mobilenet` for speed, `resnet` for accuracy
5. **Epochs**: 50-100 epochs usually sufficient

## Example Workflow

```bash
# 1. Prepare dataset
# Ensure images in .png/.jpg and masks in .npy format

# 2. Create class mapping
echo '{"chair": 2, "table": 4, "person": 6, "sofa": 8}' > class_id.json

# 3. Train model
python scripts/train_segmentation.py \
    --train_images "G:/datasets/eirt_output/batch03/rgb" \
    --train_masks "G:/datasets/eirt_output/batch03/mask" \
    --val_images "G:/datasets/eirt_output/batch04/rgb" \
    --val_masks "G:/datasets/eirt_output/batch04/mask" \
    --class_mapping "class_id.json" \
    --epochs 50 \
    --batch_size 4

# 4. Run inference
python scripts/inference_segmentation.py \
    --checkpoint "outputs/segmentation/best_model.pth" \
    --config "outputs/segmentation/config.json" \
    --image "test_image.jpg" \
    --output "predictions" \
    --visualize
```
