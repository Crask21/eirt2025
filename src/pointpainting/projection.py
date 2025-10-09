"""
Point Cloud Projection Module

This module handles the projection of 3D LiDAR points into 2D image coordinates
using camera intrinsic and extrinsic parameters.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PointCloudProjector:
    """
    Projects 3D LiDAR points into 2D image coordinates.
    
    This class handles the geometric transformation from the LiDAR coordinate
    system to the camera coordinate system, followed by projection onto the
    image plane using camera intrinsic parameters.
    """
    
    def __init__(self, filter_behind_camera: bool = True):
        """
        Initialize the point cloud projector.
        
        Args:
            filter_behind_camera: Whether to filter out points behind the camera
        """
        self.filter_behind_camera = filter_behind_camera
        
    def project_points(
        self,
        points_3d: np.ndarray,
        camera_matrix: np.ndarray,
        transform_matrix: np.ndarray,
        image_shape: Tuple[int, int],
        min_distance: float = 0.1,
        max_distance: float = 80.0
    ) -> Dict[str, np.ndarray]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in LiDAR coordinates (N, 3) [x, y, z]
            camera_matrix: Camera intrinsic matrix (3, 4) or (3, 3)
            transform_matrix: Transformation matrix LiDAR->Camera (4, 4)
            image_shape: Target image shape (height, width)
            min_distance: Minimum distance threshold for valid points
            max_distance: Maximum distance threshold for valid points
            
        Returns:
            Dictionary containing:
            - projected_points: 2D pixel coordinates (M, 2) [u, v]
            - valid_mask: Boolean mask indicating valid projections (N,)
            - distances: Distance of each point from camera (N,)
            - camera_points: 3D points in camera coordinates (N, 3)
        """
        logger.debug(f"Projecting {len(points_3d)} points to image shape {image_shape}")
        
        # Convert to homogeneous coordinates
        points_3d_homo = self._to_homogeneous(points_3d)
        
        # Transform from LiDAR to camera coordinates
        camera_points_homo = transform_matrix @ points_3d_homo.T  # (4, N)
        camera_points = camera_points_homo[:3].T  # (N, 3)
        
        # Calculate distances from camera
        distances = np.linalg.norm(camera_points, axis=1)
        
        # Initial validity mask
        valid_mask = np.ones(len(points_3d), dtype=bool)
        
        # Filter by distance
        distance_mask = (distances >= min_distance) & (distances <= max_distance)
        valid_mask &= distance_mask
        
        # Filter points behind camera (negative Z in camera coordinates)
        if self.filter_behind_camera:
            front_mask = camera_points[:, 2] > 0
            valid_mask &= front_mask
        
        # Project to image coordinates
        projected_points = np.zeros((len(points_3d), 2), dtype=np.float32)
        
        if valid_mask.sum() > 0:
            valid_camera_points = camera_points[valid_mask]
            
            # Handle different camera matrix formats
            if camera_matrix.shape == (3, 4):
                # Full projection matrix
                homo_coords = self._to_homogeneous(valid_camera_points)
                image_coords_homo = camera_matrix @ homo_coords.T  # (3, M)
            elif camera_matrix.shape == (3, 3):
                # Intrinsic matrix only
                image_coords_homo = camera_matrix @ valid_camera_points.T  # (3, M)
            else:
                raise ValueError(f"Invalid camera matrix shape: {camera_matrix.shape}")
            
            # Convert from homogeneous to 2D coordinates
            image_coords_homo = image_coords_homo.T  # (M, 3)
            
            # Avoid division by zero
            valid_depth = image_coords_homo[:, 2] > 1e-8
            if valid_depth.sum() > 0:
                image_coords = image_coords_homo[valid_depth, :2] / image_coords_homo[valid_depth, 2:3]
                
                # Update valid mask to reflect depth filtering
                valid_indices = np.where(valid_mask)[0]
                invalid_depth_indices = valid_indices[~valid_depth]
                valid_mask[invalid_depth_indices] = False
                
                # Set projected coordinates for valid points
                final_valid_indices = valid_indices[valid_depth]
                projected_points[final_valid_indices] = image_coords
        
        # Filter points outside image boundaries
        image_mask = self._filter_image_bounds(
            projected_points[valid_mask], 
            image_shape
        )
        
        # Update valid mask with image boundary filtering
        valid_indices = np.where(valid_mask)[0]
        invalid_image_indices = valid_indices[~image_mask]
        valid_mask[invalid_image_indices] = False
        
        logger.debug(f"Projected {valid_mask.sum()} valid points out of {len(points_3d)}")
        
        return {
            "projected_points": projected_points,
            "valid_mask": valid_mask,
            "distances": distances,
            "camera_points": camera_points
        }
    
    def _to_homogeneous(self, points: np.ndarray) -> np.ndarray:
        """Convert 3D points to homogeneous coordinates."""
        return np.hstack([points, np.ones((len(points), 1))])
    
    def _filter_image_bounds(
        self,
        projected_points: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Filter points that fall outside image boundaries.
        
        Args:
            projected_points: 2D pixel coordinates (M, 2)
            image_shape: Image shape (height, width)
            
        Returns:
            Boolean mask indicating points within image bounds
        """
        if len(projected_points) == 0:
            return np.array([], dtype=bool)
            
        height, width = image_shape
        
        u_coords = projected_points[:, 0]
        v_coords = projected_points[:, 1]
        
        # Check bounds with small margin for numerical stability
        margin = 0.5
        u_valid = (u_coords >= margin) & (u_coords < width - margin)
        v_valid = (v_coords >= margin) & (v_coords < height - margin)
        
        return u_valid & v_valid
    
    def project_points_batch(
        self,
        points_3d_list: list,
        camera_matrices: list,
        transform_matrices: list,
        image_shapes: list,
        **kwargs
    ) -> list:
        """
        Project multiple point clouds in batch.
        
        Args:
            points_3d_list: List of 3D point arrays
            camera_matrices: List of camera matrices
            transform_matrices: List of transformation matrices  
            image_shapes: List of image shapes
            **kwargs: Additional arguments for project_points
            
        Returns:
            List of projection results
        """
        results = []
        
        for i, (points, cam_mat, trans_mat, img_shape) in enumerate(
            zip(points_3d_list, camera_matrices, transform_matrices, image_shapes)
        ):
            logger.debug(f"Processing batch item {i+1}/{len(points_3d_list)}")
            result = self.project_points(
                points, cam_mat, trans_mat, img_shape, **kwargs
            )
            results.append(result)
            
        return results
    
    def get_projection_statistics(
        self,
        projection_result: Dict[str, np.ndarray]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate statistics about projection results.
        
        Args:
            projection_result: Result from project_points method
            
        Returns:
            Dictionary with projection statistics
        """
        valid_mask = projection_result["valid_mask"]
        distances = projection_result["distances"]
        
        total_points = len(valid_mask)
        valid_points = valid_mask.sum()
        
        stats = {
            "total_points": total_points,
            "valid_points": int(valid_points),
            "valid_percentage": (valid_points / total_points) * 100 if total_points > 0 else 0,
            "mean_distance": float(np.mean(distances[valid_mask])) if valid_points > 0 else 0,
            "min_distance": float(np.min(distances[valid_mask])) if valid_points > 0 else 0,
            "max_distance": float(np.max(distances[valid_mask])) if valid_points > 0 else 0,
        }
        
        return stats
    
    def visualize_projection(
        self,
        image: np.ndarray,
        projected_points: np.ndarray,
        valid_mask: np.ndarray,
        point_colors: Optional[np.ndarray] = None,
        point_size: int = 2
    ) -> np.ndarray:
        """
        Visualize projected points overlaid on image.
        
        Args:
            image: Original image (H, W, 3)
            projected_points: 2D pixel coordinates (N, 2)
            valid_mask: Boolean mask for valid points (N,)
            point_colors: RGB colors for each point (N, 3) [optional]
            point_size: Size of points to draw
            
        Returns:
            Image with projected points overlaid
        """
        import cv2
        
        # Copy image to avoid modifying original
        vis_image = image.copy()
        
        if valid_mask.sum() == 0:
            return vis_image
            
        valid_projected = projected_points[valid_mask]
        
        # Default colors if not provided
        if point_colors is None:
            point_colors = np.full((valid_mask.sum(), 3), [0, 255, 0], dtype=np.uint8)  # Green
        else:
            # Check if point_colors is already filtered or needs filtering
            if len(point_colors) == len(projected_points):
                # point_colors has same length as all points, needs filtering
                point_colors = point_colors[valid_mask]
            elif len(point_colors) == valid_mask.sum():
                # point_colors is already filtered to valid points only
                pass  # Use as is
            else:
                raise ValueError(f"point_colors length ({len(point_colors)}) doesn't match "
                               f"total points ({len(projected_points)}) or valid points ({valid_mask.sum()})")
        
        # Draw points
        for i, (u, v) in enumerate(valid_projected):
            center = (int(round(u)), int(round(v)))
            color = tuple(map(int, point_colors[i]))
            cv2.circle(vis_image, center, point_size, color, -1)
        
        return vis_image
    
    @staticmethod
    def create_kitti_camera_matrix(focal_length: float, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a basic camera matrix for KITTI-like setup.
        
        Args:
            focal_length: Camera focal length in pixels
            image_shape: Image shape (height, width)
            
        Returns:
            Camera intrinsic matrix (3, 3)
        """
        height, width = image_shape
        
        # Assume principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    @staticmethod
    def create_lidar_to_camera_transform(
        translation: np.ndarray,
        rotation: np.ndarray
    ) -> np.ndarray:
        """
        Create transformation matrix from LiDAR to camera coordinates.
        
        Args:
            translation: Translation vector (3,) [x, y, z]
            rotation: Rotation matrix (3, 3) or Euler angles (3,)
            
        Returns:
            Transformation matrix (4, 4)
        """
        T = np.eye(4, dtype=np.float32)
        
        # Handle rotation input
        if rotation.shape == (3, 3):
            T[:3, :3] = rotation
        elif rotation.shape == (3,):
            # Convert Euler angles to rotation matrix
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler('xyz', rotation)
            T[:3, :3] = rot.as_matrix()
        else:
            raise ValueError(f"Invalid rotation shape: {rotation.shape}")
        
        T[:3, 3] = translation
        
        return T