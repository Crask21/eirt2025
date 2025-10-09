"""
KITTI Dataset Loader

This module provides utilities for loading and processing KITTI dataset files
including point clouds, images, calibration data, and labels.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import glob

logger = logging.getLogger(__name__)


class KITTIDataLoader:
    """
    KITTI dataset loader for PointPainting experiments.
    
    Handles loading of:
    - Point clouds (.bin files)  
    - RGB images (.png files)
    - Calibration data (calib files)
    - Object labels (label files)
    """
    
    def __init__(self, kitti_root: str, split: str = "training"):
        """
        Initialize KITTI data loader.
        
        Args:
            kitti_root: Root directory of KITTI dataset
            split: Data split to use ("training", "testing")
        """
        self.kitti_root = Path(kitti_root)
        self.split = split
        
        # Define directory structure
        self.split_dir = self.kitti_root / split
        self.velodyne_dir = self.split_dir / "velodyne"
        self.image_dir = self.split_dir / "image_2"  # Left color camera
        self.calib_dir = self.split_dir / "calib"
        self.label_dir = self.split_dir / "label_2"
        
        # Verify directories exist
        self._verify_structure()
        
        # Get available sample IDs
        self.sample_ids = self._get_sample_ids()
        
        logger.info(f"Initialized KITTI loader for {split} split with {len(self.sample_ids)} samples")
    
    def _verify_structure(self):
        """Verify that required directories exist."""
        required_dirs = [self.split_dir, self.velodyne_dir, self.image_dir, self.calib_dir]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
                
        # Label directory is optional for testing split
        if self.split == "training" and not self.label_dir.exists():
            logger.warning(f"Label directory not found: {self.label_dir}")
    
    def _get_sample_ids(self) -> List[str]:
        """Get list of available sample IDs."""
        # Use velodyne files to determine available samples
        velodyne_files = list(self.velodyne_dir.glob("*.bin"))
        sample_ids = [f.stem for f in velodyne_files]
        sample_ids.sort()
        return sample_ids
    
    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.sample_ids)
    
    def load_velodyne(self, sample_id: str) -> np.ndarray:
        """
        Load LiDAR point cloud data.
        
        Args:
            sample_id: Sample identifier (e.g., "000000")
            
        Returns:
            Point cloud array (N, 4) with [x, y, z, intensity]
        """
        velodyne_file = self.velodyne_dir / f"{sample_id}.bin"
        
        if not velodyne_file.exists():
            raise FileNotFoundError(f"Velodyne file not found: {velodyne_file}")
        
        # Load binary point cloud data
        points = np.fromfile(str(velodyne_file), dtype=np.float32).reshape(-1, 4)
        
        logger.debug(f"Loaded velodyne data for {sample_id}: {points.shape}")
        return points
    
    def load_image(self, sample_id: str, camera: str = "image_2") -> np.ndarray:
        """
        Load camera image.
        
        Args:
            sample_id: Sample identifier
            camera: Camera name ("image_2" for left color, "image_3" for right color)
            
        Returns:
            Image array (H, W, 3) in RGB format
        """
        if camera == "image_2":
            image_dir = self.image_dir
        elif camera == "image_3":
            image_dir = self.split_dir / "image_3"
        else:
            raise ValueError(f"Unsupported camera: {camera}")
        
        image_file = image_dir / f"{sample_id}.png"
        
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
        
        # Load image using PIL to ensure RGB format
        from PIL import Image
        image = Image.open(image_file).convert('RGB')
        image_array = np.array(image)
        
        logger.debug(f"Loaded image for {sample_id}: {image_array.shape}")
        return image_array
    
    def load_calibration(self, sample_id: str) -> Dict[str, np.ndarray]:
        """
        Load calibration data.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Dictionary containing calibration matrices:
            - P0, P1, P2, P3: Projection matrices for cameras 0-3
            - R0_rect: Rectification matrix
            - Tr_velo_to_cam: Transformation from velodyne to camera
            - Tr_imu_to_velo: Transformation from IMU to velodyne
        """
        calib_file = self.calib_dir / f"{sample_id}.txt"
        
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        
        calib_data = {}
        
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                    
                key = parts[0].strip()
                values = np.array([float(x) for x in parts[1].strip().split()])
                
                # Reshape matrices based on key
                if key.startswith('P'):
                    # Projection matrices are 3x4
                    calib_data[key] = values.reshape(3, 4)
                elif key in ['R0_rect']:
                    # Rectification matrix is 3x3, but stored as 9 values
                    calib_data[key] = values.reshape(3, 3)
                elif key in ['Tr_velo_to_cam', 'Tr_imu_to_velo']:
                    # Transformation matrices are 3x4, convert to 4x4
                    T = np.eye(4)
                    T[:3, :] = values.reshape(3, 4)
                    calib_data[key] = T
                else:
                    calib_data[key] = values
        
        logger.debug(f"Loaded calibration for {sample_id}: {list(calib_data.keys())}")
        return calib_data
    
    def load_labels(self, sample_id: str) -> List[Dict]:
        """
        Load object labels.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            List of object dictionaries with fields:
            - type: Object class name
            - truncated: Truncation level
            - occluded: Occlusion level  
            - alpha: Observation angle
            - bbox: 2D bounding box [x1, y1, x2, y2]
            - dimensions: 3D object dimensions [height, width, length]
            - location: 3D object location [x, y, z] in camera coordinates
            - rotation_y: Rotation around Y-axis in camera coordinates
        """
        if not self.label_dir.exists():
            logger.warning("Label directory not available")
            return []
            
        label_file = self.label_dir / f"{sample_id}.txt"
        
        if not label_file.exists():
            logger.warning(f"Label file not found: {label_file}")
            return []
        
        objects = []
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                    'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                    'rotation_y': float(parts[14])
                }
                
                objects.append(obj)
        
        logger.debug(f"Loaded {len(objects)} objects for {sample_id}")
        return objects
    
    def get_camera_matrix(self, sample_id: str, camera: str = "P2") -> np.ndarray:
        """
        Get camera projection matrix.
        
        Args:
            sample_id: Sample identifier
            camera: Camera matrix key (P0, P1, P2, P3)
            
        Returns:
            Camera projection matrix (3, 4)
        """
        calib = self.load_calibration(sample_id)
        return calib[camera]
    
    def get_velo_to_cam_transform(self, sample_id: str) -> np.ndarray:
        """
        Get transformation matrix from velodyne to camera coordinates.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Transformation matrix (4, 4)
        """
        calib = self.load_calibration(sample_id)
        
        # KITTI provides Tr_velo_to_cam (3x4) and R0_rect (3x3)
        # Complete transformation: P2 * R0_rect * Tr_velo_to_cam
        
        Tr_velo_to_cam = calib['Tr_velo_to_cam']  # 4x4
        R0_rect = calib['R0_rect']  # 3x3
        
        # Convert R0_rect to 4x4
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect
        
        # Combined transformation
        transform = R0_rect_4x4 @ Tr_velo_to_cam
        
        return transform
    
    def load_sample(self, sample_id: str) -> Dict[str, Union[np.ndarray, List, Dict]]:
        """
        Load complete sample data including point cloud, image, calibration, and labels.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Dictionary containing all sample data
        """
        sample_data = {
            'sample_id': sample_id,
            'point_cloud': self.load_velodyne(sample_id),
            'image': self.load_image(sample_id),
            'calibration': self.load_calibration(sample_id),
            'camera_matrix': self.get_camera_matrix(sample_id),
            'transform_matrix': self.get_velo_to_cam_transform(sample_id),
        }
        
        # Add labels if available
        try:
            sample_data['labels'] = self.load_labels(sample_id)
        except Exception as e:
            logger.warning(f"Could not load labels for {sample_id}: {e}")
            sample_data['labels'] = []
        
        return sample_data
    
    def load_batch(self, sample_ids: List[str]) -> List[Dict]:
        """
        Load multiple samples in batch.
        
        Args:
            sample_ids: List of sample identifiers
            
        Returns:
            List of sample data dictionaries
        """
        batch_data = []
        
        for sample_id in sample_ids:
            try:
                sample_data = self.load_sample(sample_id)
                batch_data.append(sample_data)
            except Exception as e:
                logger.error(f"Error loading sample {sample_id}: {e}")
                continue
        
        return batch_data
    
    def get_split_indices(self, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Split sample IDs into training and validation sets.
        
        Args:
            train_ratio: Ratio of samples to use for training
            
        Returns:
            Tuple of (train_ids, val_ids)
        """
        np.random.seed(42)  # For reproducible splits
        shuffled_ids = np.random.permutation(self.sample_ids)
        
        split_idx = int(len(shuffled_ids) * train_ratio)
        train_ids = shuffled_ids[:split_idx].tolist()
        val_ids = shuffled_ids[split_idx:].tolist()
        
        return train_ids, val_ids
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Analyze class distribution in the dataset.
        
        Returns:
            Dictionary mapping class names to occurrence counts
        """
        class_counts = {}
        
        for sample_id in self.sample_ids:
            try:
                labels = self.load_labels(sample_id)
                for obj in labels:
                    class_name = obj['type']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception as e:
                logger.warning(f"Error loading labels for {sample_id}: {e}")
                continue
        
        return class_counts
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.sample_ids)
    
    def __getitem__(self, index: int) -> Dict:
        """Get sample by index."""
        sample_id = self.sample_ids[index]
        return self.load_sample(sample_id)
    
    def __iter__(self):
        """Make the loader iterable."""
        for sample_id in self.sample_ids:
            yield self.load_sample(sample_id)