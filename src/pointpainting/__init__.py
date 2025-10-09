"""
PointPainting: Independent Implementation for KITTI Dataset

This package implements the PointPainting method for 3D object detection by fusing
RGB camera data with LiDAR point clouds. The implementation is designed to work
with the KITTI dataset and can be extended for indoor environments.

Key components:
- Semantic Segmentation: DeepLabV3+ for pixel-wise semantic segmentation
- Point Cloud Projection: Projects 3D LiDAR points into 2D image space
- Point Painting: Augments point clouds with semantic information
- Data Loaders: KITTI dataset handling utilities

References:
- PointPainting: Sequential Fusion for 3D Object Detection (Vora et al., 2020)
"""

from .core import PointPainter
from .segmentation import SemanticSegmentationModel
from .projection import PointCloudProjector
from .data_loader import KITTIDataLoader
from .utils import (
    load_point_cloud,
    load_image,
    save_painted_pointcloud,
    visualize_results,
    visualize_3d_pointcloud,
    compute_metrics
)

__version__ = "0.1.0"
__author__ = "EIRT Team"

__all__ = [
    "PointPainter",
    "SemanticSegmentationModel", 
    "PointCloudProjector",
    "KITTIDataLoader",
    "load_point_cloud",
    "load_image", 
    "save_painted_pointcloud",
    "visualize_results",
    "visualize_3d_pointcloud",
    "compute_metrics"
]