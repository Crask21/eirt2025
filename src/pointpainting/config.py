"""
Configuration file for PointPainting implementation.

This file contains all configurable parameters for the PointPainting pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class KITTIConfig:
    """Configuration for KITTI dataset."""
    
    # Dataset paths
    kitti_root: str = "D:/datasets/kitti"
    kitti_archives_dir: Optional[str] = None  # Directory with .zip files
    split: str = "training"  # "training" or "testing"
    
    # Archive filenames (if extracting from zip files)
    archive_files: List[str] = None
    
    def __post_init__(self):
        if self.archive_files is None:
            self.archive_files = [
                "data_object_calib.zip",
                "data_object_image_2.zip", 
                "data_object_image_3.zip",
                "data_object_label_2.zip",
                "data_object_velodyne.zip"
            ]


@dataclass
class SegmentationConfig:
    """Configuration for semantic segmentation model."""
    
    model_name: str = "deeplabv3_resnet50"
    num_classes: int = 4
    pretrained: bool = True
    device: str = "auto"  # "auto", "cuda", "cpu"
    input_size: tuple = (512, 1024)  # (height, width)
    
    # Class names for KITTI
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["background", "car", "pedestrian", "cyclist"]


@dataclass
class ProjectionConfig:
    """Configuration for point cloud projection."""
    
    filter_behind_camera: bool = True
    min_distance: float = 1.0
    max_distance: float = 80.0
    
    # For creating default camera matrix if not available
    default_focal_length: float = 721.5377  # KITTI P2 camera


@dataclass
class PointPaintingConfig:
    """Configuration for PointPainting pipeline."""
    
    # Sub-configurations
    kitti: KITTIConfig = None
    segmentation: SegmentationConfig = None
    projection: ProjectionConfig = None
    
    # Output configuration
    output_dir: str = "./outputs"
    save_painted_clouds: bool = True
    save_visualizations: bool = True
    save_metrics: bool = True
    
    # Processing configuration
    batch_size: int = 1
    max_samples: Optional[int] = None  # None = process all samples
    
    # Visualization options
    visualize_3d: bool = False
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        if self.kitti is None:
            self.kitti = KITTIConfig()
        if self.segmentation is None:
            self.segmentation = SegmentationConfig()
        if self.projection is None:
            self.projection = ProjectionConfig()


@dataclass
class IndoorConfig:
    """Configuration for indoor environment extension."""
    
    # Indoor-specific class names
    class_names: List[str] = None
    
    # Motion classification
    static_classes: List[str] = None
    semi_static_classes: List[str] = None
    dynamic_classes: List[str] = None
    
    # Temporal analysis
    temporal_window_size: int = 10
    motion_threshold: float = 0.1  # meters
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                "background", "wall", "floor", "ceiling",
                "chair", "table", "desk", "bookshelf",
                "door", "window", "person", "robot",
                "box", "bin", "cart", "pallet"
            ]
        
        if self.static_classes is None:
            self.static_classes = ["wall", "floor", "ceiling", "door", "window"]
            
        if self.semi_static_classes is None:
            self.semi_static_classes = [
                "chair", "table", "desk", "bookshelf", 
                "box", "bin", "cart", "pallet"
            ]
            
        if self.dynamic_classes is None:
            self.dynamic_classes = ["person", "robot"]


# Default configurations
def get_default_config() -> PointPaintingConfig:
    """Get default configuration for KITTI dataset."""
    return PointPaintingConfig()


def get_indoor_config() -> PointPaintingConfig:
    """Get configuration adapted for indoor environments."""
    config = PointPaintingConfig()
    
    # Override with indoor-specific settings
    indoor_config = IndoorConfig()
    config.segmentation.class_names = indoor_config.class_names
    config.segmentation.num_classes = len(indoor_config.class_names)
    
    # Adjust projection for shorter indoor ranges
    config.projection.max_distance = 20.0
    config.projection.min_distance = 0.3
    
    return config


def load_config_from_dict(config_dict: dict) -> PointPaintingConfig:
    """Load configuration from dictionary."""
    # Extract sub-configurations
    kitti_config = KITTIConfig(**config_dict.get("kitti", {}))
    seg_config = SegmentationConfig(**config_dict.get("segmentation", {}))
    proj_config = ProjectionConfig(**config_dict.get("projection", {}))
    
    # Create main config
    main_config_dict = {k: v for k, v in config_dict.items() 
                       if k not in ["kitti", "segmentation", "projection"]}
    
    config = PointPaintingConfig(
        kitti=kitti_config,
        segmentation=seg_config,
        projection=proj_config,
        **main_config_dict
    )
    
    return config


def save_config_to_dict(config: PointPaintingConfig) -> dict:
    """Save configuration to dictionary."""
    from dataclasses import asdict
    return asdict(config)


# Example configurations
KITTI_FAST_CONFIG = {
    "kitti": {
        "kitti_root": "D:/datasets/kitti",
        "split": "training"
    },
    "segmentation": {
        "model_name": "deeplabv3_resnet50",
        "num_classes": 4,
        "device": "cuda",
        "input_size": (256, 512)  # Smaller for faster processing
    },
    "projection": {
        "max_distance": 50.0  # Shorter range for speed
    },
    "max_samples": 10,
    "visualize_3d": False
}


KITTI_ACCURATE_CONFIG = {
    "kitti": {
        "kitti_root": "D:/datasets/kitti",
        "split": "training"
    },
    "segmentation": {
        "model_name": "deeplabv3_resnet50",
        "num_classes": 4,
        "device": "cuda",
        "input_size": (512, 1024)  # Full resolution
    },
    "projection": {
        "max_distance": 80.0
    },
    "save_painted_clouds": True,
    "save_visualizations": True,
    "visualize_3d": False
}


INDOOR_CONFIG = {
    "segmentation": {
        "num_classes": 16,
        "class_names": [
            "background", "wall", "floor", "ceiling",
            "chair", "table", "desk", "bookshelf",
            "door", "window", "person", "robot",
            "box", "bin", "cart", "pallet"
        ]
    },
    "projection": {
        "max_distance": 20.0,
        "min_distance": 0.3
    }
}