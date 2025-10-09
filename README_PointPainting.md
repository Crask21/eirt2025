# PointPainting Implementation for KITTI Dataset

This repository contains an independent implementation of **PointPainting** - a sequential fusion method for 3D object detection that combines RGB camera data with LiDAR point clouds.

## Overview

PointPainting enhances 3D object detection by "painting" LiDAR point clouds with semantic information from RGB images. The method consists of three main stages:

1. **Semantic Segmentation**: Generate pixel-wise class predictions from RGB images
2. **Point Projection**: Project 3D LiDAR points onto the 2D image plane  
3. **Point Painting**: Augment each 3D point with semantic class scores

The painted point cloud can then be used with any LiDAR-based 3D detector.

## Features

- ✅ **Clean Implementation**: Independent codebase without external dependencies issues
- ✅ **KITTI Ready**: Direct support for KITTI dataset format
- ✅ **Modular Design**: Easy to extend and customize components
- ✅ **Indoor Adaptable**: Framework ready for indoor environment extension
- ✅ **Visualization Tools**: Comprehensive 2D and 3D visualization utilities
- ✅ **Batch Processing**: Support for processing multiple samples efficiently

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eirt2025
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Prepare KITTI Dataset

Download the KITTI Object Detection dataset and extract it:

```python
# If you have KITTI zip files, extract them first
python scripts/pointpainting_demo.py --kitti_archives "D:/datasets/kitti_archives" --kitti_root "D:/datasets/kitti"
```

Expected directory structure:
```
D:/datasets/kitti/
├── training/
│   ├── velodyne/     # Point cloud files (.bin)
│   ├── image_2/      # Left camera images (.png)
│   ├── calib/        # Calibration files (.txt)
│   └── label_2/      # Object labels (.txt)
└── testing/
    ├── velodyne/
    ├── image_2/
    └── calib/
```

### 2. Run Demo Script

```bash
# Single sample demo
python scripts/pointpainting_demo.py --kitti_root "D:/datasets/kitti" --sample_id "000000"

# Batch processing
python scripts/pointpainting_demo.py --kitti_root "D:/datasets/kitti" --batch_process --max_samples 5
```

### 3. Interactive Jupyter Notebook

Launch the interactive notebook for detailed exploration:

```bash
jupyter notebook notebooks/pointpainting_demo.ipynb
```

## Usage Examples

### Basic PointPainting Pipeline

```python
from pointpainting import PointPainter, KITTIDataLoader

# Initialize components
data_loader = KITTIDataLoader("D:/datasets/kitti", split="training")
painter = PointPainter()

# Load sample
sample_data = data_loader.load_sample("000000")

# Run PointPainting
results = painter.paint_pointcloud(
    point_cloud=sample_data["point_cloud"],
    image=sample_data["image"],
    camera_matrix=sample_data["camera_matrix"],
    transform_matrix=sample_data["transform_matrix"]
)

painted_points = results["painted_points"]
print(f"Original: {sample_data['point_cloud'].shape} -> Painted: {painted_points.shape}")
```

### Custom Segmentation Model

```python
from pointpainting import SemanticSegmentationModel, PointPainter

# Initialize custom segmentation model
seg_model = SemanticSegmentationModel(
    num_classes=10,  # Custom number of classes
    device="cuda"
)

# Use with PointPainter
painter = PointPainter(
    segmentation_model=seg_model,
    class_names=["background", "car", "truck", "bus", "person", ...]
)
```

### Batch Processing

```python
from pointpainting import PointPainter, KITTIDataLoader

data_loader = KITTIDataLoader("D:/datasets/kitti")
painter = PointPainter()

# Process multiple samples
sample_ids = ["000000", "000001", "000002"]
for sample_id in sample_ids:
    sample_data = data_loader.load_sample(sample_id)
    results = painter.paint_pointcloud(**sample_data)
    
    # Save results
    save_painted_pointcloud(
        results["painted_points"], 
        f"outputs/painted_{sample_id}.npy"
    )
```

## Architecture

### Core Components

- **`PointPainter`**: Main orchestrator class that manages the complete pipeline
- **`SemanticSegmentationModel`**: Handles RGB image segmentation using DeepLabV3+  
- **`PointCloudProjector`**: Projects 3D points to 2D image coordinates
- **`KITTIDataLoader`**: KITTI dataset loading utilities
- **`utils`**: Visualization, I/O, and evaluation utilities

### Key Features

#### Modular Design
Each component can be used independently or customized:

```python
# Custom projector settings
projector = PointCloudProjector(
    filter_behind_camera=True,
    min_distance=1.0,
    max_distance=80.0
)

# Custom segmentation model
seg_model = SemanticSegmentationModel(
    model_name="deeplabv3_resnet50",
    num_classes=4,
    pretrained=True
)
```

#### Comprehensive Visualization

```python
from pointpainting.utils import visualize_results, visualize_3d_pointcloud

# 2D visualization with multiple views
visualize_results(
    image=image,
    point_cloud=point_cloud, 
    painted_points=painted_points,
    projection_result=projection_result
)

# Interactive 3D visualization
visualize_3d_pointcloud(painted_points, class_names)
```

## Extending to Indoor Environments

The implementation is designed to be easily extended for indoor scenarios:

### 1. Custom Dataset Loader

```python
class IndoorDataLoader:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        
    def load_sample(self, sample_id):
        return {
            "point_cloud": self.load_point_cloud(sample_id),
            "image": self.load_rgb_image(sample_id),
            "camera_matrix": self.load_camera_params(sample_id),
            "transform_matrix": self.load_lidar_camera_transform(sample_id)
        }
```

### 2. Indoor Class Definitions

```python
# Define indoor-specific classes
indoor_classes = [
    "background", "wall", "floor", "ceiling",
    "chair", "table", "desk", "bookshelf",
    "door", "window", "person", "robot"
]

# Categorize by motion characteristics
static_classes = ["wall", "floor", "ceiling", "door", "window"]
semi_static_classes = ["chair", "table", "desk", "bookshelf"] 
dynamic_classes = ["person", "robot"]
```

### 3. Temporal Analysis for Motion Detection

```python
class TemporalPointPainter(PointPainter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_buffer = []
        
    def classify_motion_type(self, painted_points_sequence):
        # Analyze point movement over time
        # Classify as static, semi-static, or dynamic
        pass
```

## Performance Considerations

### Model Selection
- **Real-time**: Use lightweight segmentation models (MobileNet backbone)
- **Accuracy**: Use ResNet/ResNeXt backbones with DeepLabV3+
- **Memory**: Adjust input resolution and batch size

### Optimization Tips
- Filter points by distance to reduce computation
- Use GPU acceleration for segmentation
- Cache segmentation results for temporal sequences
- Downsample point clouds for faster processing

## Results Format

The painted point cloud has the following structure:

```
Original point cloud: (N, 4) [x, y, z, intensity]
Painted point cloud:  (N, 4+C) [x, y, z, intensity, class_0, class_1, ..., class_C-1]
```

Where:
- `x, y, z`: 3D coordinates in LiDAR frame
- `intensity`: LiDAR intensity measurement  
- `class_i`: Semantic probability for class i

## Evaluation Metrics

The implementation provides several evaluation utilities:

- **Class Distribution**: Point-wise class statistics
- **Projection Statistics**: Valid projection ratios and distances
- **Confidence Analysis**: Prediction confidence distributions
- **Visualization**: Comprehensive result visualization

## Future Extensions

### 1. Integration with 3D Detectors
- PointPillars integration
- VoxelNet support
- PointRCNN compatibility

### 2. Indoor Environment Support
- Dynamic object tracking
- Static vs semi-static classification
- Multi-sensor fusion (RGB-D + LiDAR)

### 3. Real-time Optimization
- Model quantization
- Pipeline parallelization
- Memory optimization

## References

1. **PointPainting: Sequential Fusion for 3D Object Detection** (Vora et al., CVPR 2020)
2. **KITTI Object Detection Dataset** (Geiger et al., CVPR 2012)
3. **DeepLab: Semantic Image Segmentation** (Chen et al., TPAMI 2018)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{eirt_pointpainting2025,
  title={PointPainting Implementation for KITTI Dataset},
  author={EIRT Team},
  year={2025},
  url={https://github.com/your-repo/eirt2025}
}
```