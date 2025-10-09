"""
Core PointPainting Implementation

This module contains the main PointPainter class that orchestrates the entire
PointPainting pipeline: semantic segmentation, point cloud projection, and
point painting (fusion).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, List
import logging

from .segmentation import SemanticSegmentationModel
from .projection import PointCloudProjector
from .utils import load_point_cloud, load_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointPainter:
    """
    Main PointPainting class that handles the complete pipeline.
    
    The PointPainting process:
    1. Semantic Segmentation: Generate pixel-wise class predictions from RGB image
    2. Point Projection: Project 3D LiDAR points onto the 2D image plane  
    3. Point Painting: Augment each 3D point with semantic class scores
    """
    
    def __init__(
        self,
        segmentation_model: Optional[SemanticSegmentationModel] = None,
        projector: Optional[PointCloudProjector] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize PointPainter.
        
        Args:
            segmentation_model: Pre-initialized segmentation model
            projector: Pre-initialized point cloud projector  
            device: Device to run computations on
            class_names: List of class names for segmentation
        """
        self.device = device
        self.class_names = class_names or self._get_default_kitti_classes()
        
        # Initialize components
        self.segmentation_model = segmentation_model or SemanticSegmentationModel(
            num_classes=len(self.class_names),
            device=self.device
        )
        
        self.projector = projector or PointCloudProjector()
        
        logger.info(f"PointPainter initialized with {len(self.class_names)} classes on {self.device}")
    
    def _get_default_kitti_classes(self) -> List[str]:
        """Get default KITTI class names for object detection."""
        return ["background", "car", "pedestrian", "cyclist"]
    
    def paint_pointcloud(
        self,
        point_cloud: Union[np.ndarray, str],
        image: Union[np.ndarray, str], 
        camera_matrix: np.ndarray,
        transform_matrix: np.ndarray,
        return_painted_points: bool = True,
        return_segmentation: bool = False
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Execute the complete PointPainting pipeline.
        
        Args:
            point_cloud: Point cloud data (N, 4) with [x, y, z, intensity] or path to .bin file
            image: RGB image (H, W, 3) or path to image file
            camera_matrix: Camera intrinsic matrix (3, 4)
            transform_matrix: Transformation matrix from LiDAR to camera coordinates (4, 4)
            return_painted_points: Whether to return the painted point cloud
            return_segmentation: Whether to return segmentation results
            
        Returns:
            Dictionary containing:
            - painted_points: Augmented point cloud with semantic features (N, 4+C)
            - segmentation_scores: Per-pixel class scores (H, W, C) [optional]
            - projected_points: 2D pixel coordinates of valid points (M, 2) [optional]
            - valid_mask: Boolean mask of points visible in image (N,) [optional]
        """
        # Load inputs if they are file paths
        if isinstance(point_cloud, str):
            point_cloud = load_point_cloud(point_cloud)
        if isinstance(image, str):
            image = load_image(image)
            
        logger.info(f"Processing point cloud with {point_cloud.shape[0]} points and image of shape {image.shape}")
        
        # Step 1: Semantic Segmentation
        segmentation_scores = self.segmentation_model.predict(image)
        logger.info(f"Generated segmentation with shape {segmentation_scores.shape}")
        
        # Step 2: Point Cloud Projection
        projection_results = self.projector.project_points(
            point_cloud[:, :3],  # Only xyz coordinates
            camera_matrix,
            transform_matrix,
            image.shape[:2]  # (height, width)
        )
        
        projected_points = projection_results["projected_points"]
        valid_mask = projection_results["valid_mask"]
        
        logger.info(f"Projected {valid_mask.sum()} valid points out of {len(point_cloud)}")
        
        # Step 3: Point Painting (Fusion)
        painted_points = self._paint_points(
            point_cloud,
            segmentation_scores,
            projected_points,
            valid_mask
        )
        
        # Prepare results
        results = {"painted_points": painted_points}
        
        if return_segmentation:
            results["segmentation_scores"] = segmentation_scores
            results["projected_points"] = projected_points
            results["valid_mask"] = valid_mask
            
        return results
    
    def _paint_points(
        self,
        point_cloud: np.ndarray,
        segmentation_scores: np.ndarray,
        projected_points: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Paint point cloud with semantic information.
        
        Args:
            point_cloud: Original point cloud (N, 4) with [x, y, z, intensity]
            segmentation_scores: Pixel-wise class scores (H, W, C)
            projected_points: 2D coordinates of projected points (M, 2)
            valid_mask: Boolean mask indicating valid projections (N,)
            
        Returns:
            painted_points: Augmented point cloud (N, 4+C) with semantic features
        """
        num_points, point_dim = point_cloud.shape
        num_classes = segmentation_scores.shape[2]
        
        # Initialize painted point cloud with original features + semantic scores
        painted_points = np.zeros((num_points, point_dim + num_classes), dtype=np.float32)
        
        # Copy original point features
        painted_points[:, :point_dim] = point_cloud
        
        # For points not visible in image, set background class (index 0) to 1.0
        painted_points[:, point_dim] = 1.0  # Background class
        
        # For valid projected points, extract semantic scores
        if valid_mask.sum() > 0:
            valid_projected = projected_points[valid_mask]
            
            # Extract semantic scores for valid points
            pixel_y = np.clip(valid_projected[:, 1].astype(int), 0, segmentation_scores.shape[0] - 1)
            pixel_x = np.clip(valid_projected[:, 0].astype(int), 0, segmentation_scores.shape[1] - 1)
            
            # Get semantic scores for projected pixels
            semantic_features = segmentation_scores[pixel_y, pixel_x]  # (M, C)
            
            # Assign semantic scores to valid points
            painted_points[valid_mask, point_dim:] = semantic_features
        
        logger.info(f"Painted point cloud shape: {painted_points.shape}")
        return painted_points
    
    def batch_paint_pointclouds(
        self,
        point_clouds: List[Union[np.ndarray, str]],
        images: List[Union[np.ndarray, str]],
        camera_matrices: List[np.ndarray],
        transform_matrices: List[np.ndarray],
        batch_size: int = 8
    ) -> List[Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Process multiple point clouds and images in batches.
        
        Args:
            point_clouds: List of point clouds or paths
            images: List of images or paths  
            camera_matrices: List of camera intrinsic matrices
            transform_matrices: List of transformation matrices
            batch_size: Number of samples to process at once
            
        Returns:
            List of painted point cloud results
        """
        results = []
        num_samples = len(point_clouds)
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            logger.info(f"Processing batch {i//batch_size + 1}: samples {i}-{batch_end-1}")
            
            batch_results = []
            for j in range(i, batch_end):
                result = self.paint_pointcloud(
                    point_clouds[j],
                    images[j], 
                    camera_matrices[j],
                    transform_matrices[j]
                )
                batch_results.append(result)
                
            results.extend(batch_results)
            
        return results
    
    def get_class_statistics(self, painted_points: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics about class distribution in painted point cloud.
        
        Args:
            painted_points: Painted point cloud (N, 4+C)
            
        Returns:
            Dictionary with class statistics
        """
        # Semantic features start after the 4 original point features
        semantic_features = painted_points[:, 4:]
        
        # Get predicted class for each point (highest score)
        predicted_classes = np.argmax(semantic_features, axis=1)
        
        # Compute class distribution
        stats = {}
        total_points = len(painted_points)
        
        for i, class_name in enumerate(self.class_names):
            count = np.sum(predicted_classes == i)
            percentage = (count / total_points) * 100
            stats[class_name] = {
                "count": int(count),
                "percentage": percentage
            }
            
        return stats