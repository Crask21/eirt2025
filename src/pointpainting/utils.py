"""
Utility Functions for PointPainting

This module provides various utility functions for loading, saving, and
visualizing point clouds and images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import open3d as o3d
from typing import Union, Optional, List, Dict, Tuple
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def load_point_cloud(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load point cloud from file.
    
    Args:
        file_path: Path to point cloud file (.bin, .pcd, .ply)
        
    Returns:
        Point cloud array (N, 4) with [x, y, z, intensity]
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.bin':
        # KITTI binary format
        points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
    elif file_path.suffix in ['.pcd', '.ply']:
        # Open3D supported formats
        pcd = o3d.io.read_point_cloud(str(file_path))
        points_xyz = np.asarray(pcd.points)
        
        # Add intensity channel if not present
        if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
            intensity = np.mean(colors, axis=1)  # Convert RGB to intensity
        else:
            intensity = np.ones(len(points_xyz))  # Default intensity
            
        points = np.hstack([points_xyz, intensity.reshape(-1, 1)])
    else:
        raise ValueError(f"Unsupported point cloud format: {file_path.suffix}")
    
    logger.debug(f"Loaded point cloud with {len(points)} points from {file_path}")
    return points


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Image array (H, W, 3) in RGB format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Load using PIL to ensure RGB format
    image = Image.open(file_path).convert('RGB')
    image_array = np.array(image)
    
    logger.debug(f"Loaded image with shape {image_array.shape} from {file_path}")
    return image_array


def save_painted_pointcloud(
    painted_points: np.ndarray,
    output_path: Union[str, Path],
    format: str = "bin"
) -> None:
    """
    Save painted point cloud to file.
    
    Args:
        painted_points: Painted point cloud (N, 4+C)
        output_path: Output file path
        format: Output format ("bin", "pcd", "ply", "npy")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "bin":
        # Save as binary format (KITTI style)
        painted_points.astype(np.float32).tofile(str(output_path))
    elif format == "npy":
        # Save as NumPy array
        np.save(str(output_path), painted_points)
    elif format in ["pcd", "ply"]:
        # Save using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(painted_points[:, :3])
        
        # Use semantic scores as colors
        if painted_points.shape[1] > 4:
            semantic_scores = painted_points[:, 4:]
            # Convert class predictions to colors
            predicted_classes = np.argmax(semantic_scores, axis=1)
            colors = get_class_colors(predicted_classes)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        if format == "pcd":
            o3d.io.write_point_cloud(str(output_path), pcd)
        else:  # ply
            o3d.io.write_point_cloud(str(output_path), pcd)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved painted point cloud to {output_path}")


def get_class_colors(class_indices: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Generate colors for different classes.
    
    Args:
        class_indices: Array of class indices (N,)
        num_classes: Total number of classes
        
    Returns:
        RGB colors array (N, 3)
    """
    # Default KITTI colors
    default_colors = np.array([
        [128, 128, 128],  # background - gray
        [255, 0, 0],      # car - red
        [0, 255, 0],      # pedestrian - green  
        [0, 0, 255],      # cyclist - blue
    ])
    
    # Extend with random colors if needed
    while len(default_colors) < num_classes:
        random_color = np.random.randint(0, 256, (1, 3))
        default_colors = np.vstack([default_colors, random_color])
    
    # Map class indices to colors
    colors = default_colors[class_indices]
    return colors.astype(np.uint8)


def visualize_results(
    image: np.ndarray,
    point_cloud: np.ndarray,
    painted_points: np.ndarray,
    projection_result: Dict,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create comprehensive visualization of PointPainting results.
    
    Args:
        image: Original RGB image (H, W, 3)
        point_cloud: Original point cloud (N, 4)
        painted_points: Painted point cloud (N, 4+C) 
        projection_result: Result from point projection
        class_names: List of class names
        save_path: Path to save visualization
    """
    if class_names is None:
        class_names = ["background", "car", "pedestrian", "cyclist"]
    
    # Extract projection data
    projected_points = projection_result["projected_points"]
    valid_mask = projection_result["valid_mask"]
    
    # Extract semantic scores and predictions
    semantic_scores = painted_points[:, 4:]
    predicted_classes = np.argmax(semantic_scores, axis=1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original image
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # 2. Projected points on image
    ax2 = plt.subplot(2, 3, 2)
    plt.imshow(image)
    if valid_mask.sum() > 0:
        valid_projected = projected_points[valid_mask]
        colors = get_class_colors(predicted_classes[valid_mask])
        plt.scatter(valid_projected[:, 0], valid_projected[:, 1], 
                   c=colors/255.0, s=1, alpha=0.7)
    plt.title("Projected Points (Colored by Class)")
    plt.axis('off')
    
    # 3. Point cloud top view (original)
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], 
               c=point_cloud[:, 3], s=0.5, cmap='viridis')
    plt.title("Original Point Cloud (Top View)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Intensity")
    plt.axis('equal')
    
    # 4. Point cloud top view (painted)
    ax4 = plt.subplot(2, 3, 4)
    colors = get_class_colors(predicted_classes)
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], 
               c=colors/255.0, s=0.5)
    plt.title("Painted Point Cloud (Top View)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    
    # 5. Class distribution
    ax5 = plt.subplot(2, 3, 5)
    class_counts = [np.sum(predicted_classes == i) for i in range(len(class_names))]
    bars = plt.bar(range(len(class_names)), class_counts)
    
    # Color bars according to class colors
    bar_colors = get_class_colors(np.arange(len(class_names)))
    for bar, color in zip(bars, bar_colors):
        bar.set_color(color/255.0)
    
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel("Number of Points")
    plt.title("Class Distribution")
    
    # 6. Distance vs Class scatter
    ax6 = plt.subplot(2, 3, 6)
    distances = projection_result["distances"]
    colors = get_class_colors(predicted_classes)
    
    for i, class_name in enumerate(class_names):
        mask = predicted_classes == i
        if mask.sum() > 0:
            plt.scatter(distances[mask], np.full(mask.sum(), i),
                       c=[colors[mask][0]/255.0], s=0.5, alpha=0.6, label=class_name)
    
    plt.xlabel("Distance (m)")
    plt.ylabel("Class")
    plt.yticks(range(len(class_names)), class_names)
    plt.title("Distance vs Class Distribution")
    plt.legend()
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_pointcloud(
    painted_points: np.ndarray,
    class_names: Optional[List[str]] = None,
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
) -> None:
    """
    Visualize painted point cloud in 3D using Open3D.
    
    Args:
        painted_points: Painted point cloud (N, 4+C)
        class_names: List of class names
        point_size: Size of points in visualization
        background_color: Background color (r, g, b) in [0, 1]
    """
    if class_names is None:
        class_names = ["background", "car", "pedestrian", "cyclist"]
    
    # Extract 3D coordinates and semantic predictions
    xyz = painted_points[:, :3]
    semantic_scores = painted_points[:, 4:]
    predicted_classes = np.argmax(semantic_scores, axis=1)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Assign colors based on classes
    colors = get_class_colors(predicted_classes, len(class_names))
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Painted Point Cloud", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Configure visualization options
    opt = vis.get_render_option()
    opt.background_color = np.array(background_color)
    opt.point_size = point_size
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def compute_metrics(
    painted_points: np.ndarray,
    ground_truth_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute evaluation metrics for painted point cloud.
    
    Args:
        painted_points: Painted point cloud (N, 4+C)
        ground_truth_labels: Ground truth class labels (N,) [optional]
        class_names: List of class names
        
    Returns:
        Dictionary containing computed metrics
    """
    if class_names is None:
        class_names = ["background", "car", "pedestrian", "cyclist"]
    
    # Extract predictions
    semantic_scores = painted_points[:, 4:]
    predicted_classes = np.argmax(semantic_scores, axis=1)
    prediction_confidence = np.max(semantic_scores, axis=1)
    
    # Basic statistics
    metrics = {
        "total_points": len(painted_points),
        "num_classes": len(class_names),
        "mean_confidence": float(np.mean(prediction_confidence)),
        "std_confidence": float(np.std(prediction_confidence)),
        "class_distribution": {}
    }
    
    # Class distribution
    for i, class_name in enumerate(class_names):
        count = np.sum(predicted_classes == i)
        metrics["class_distribution"][class_name] = {
            "count": int(count),
            "percentage": float(count / len(painted_points) * 100)
        }
    
    # If ground truth is available, compute accuracy metrics
    if ground_truth_labels is not None:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(ground_truth_labels, predicted_classes)
        report = classification_report(
            ground_truth_labels, 
            predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        conf_matrix = confusion_matrix(ground_truth_labels, predicted_classes)
        
        metrics.update({
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        })
    
    return metrics


def filter_points_by_distance(
    point_cloud: np.ndarray,
    min_distance: float = 0.0,
    max_distance: float = 80.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter point cloud by distance from origin.
    
    Args:
        point_cloud: Point cloud array (N, D)
        min_distance: Minimum distance threshold
        max_distance: Maximum distance threshold
        
    Returns:
        Tuple of (filtered_points, valid_mask)
    """
    distances = np.linalg.norm(point_cloud[:, :3], axis=1)
    valid_mask = (distances >= min_distance) & (distances <= max_distance)
    
    filtered_points = point_cloud[valid_mask]
    
    logger.debug(f"Filtered {len(filtered_points)} points from {len(point_cloud)} "
                f"(distance range: {min_distance}-{max_distance}m)")
    
    return filtered_points, valid_mask


def crop_points_to_fov(
    point_cloud: np.ndarray,
    fov_horizontal: float = 90.0,
    fov_vertical: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop point cloud to field of view.
    
    Args:
        point_cloud: Point cloud array (N, D)
        fov_horizontal: Horizontal field of view in degrees
        fov_vertical: Vertical field of view in degrees [optional]
        
    Returns:
        Tuple of (cropped_points, valid_mask)
    """
    xyz = point_cloud[:, :3]
    
    # Horizontal FOV filtering
    angles_h = np.arctan2(xyz[:, 1], xyz[:, 0]) * 180 / np.pi
    h_mask = np.abs(angles_h) <= fov_horizontal / 2
    
    # Vertical FOV filtering (if specified)
    if fov_vertical is not None:
        distances_xy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        angles_v = np.arctan2(xyz[:, 2], distances_xy) * 180 / np.pi
        v_mask = np.abs(angles_v) <= fov_vertical / 2
        valid_mask = h_mask & v_mask
    else:
        valid_mask = h_mask
    
    cropped_points = point_cloud[valid_mask]
    
    logger.debug(f"Cropped to {len(cropped_points)} points from {len(point_cloud)} "
                f"(FOV: {fov_horizontal}°H x {fov_vertical}°V)")
    
    return cropped_points, valid_mask


def downsample_points(
    point_cloud: np.ndarray,
    voxel_size: float = 0.1,
    method: str = "voxel"
) -> np.ndarray:
    """
    Downsample point cloud to reduce computational load.
    
    Args:
        point_cloud: Point cloud array (N, D)
        voxel_size: Voxel size for downsampling
        method: Downsampling method ("voxel", "random", "uniform")
        
    Returns:
        Downsampled point cloud
    """
    if method == "voxel":
        # Use Open3D for voxel downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Add other attributes as colors/normals if needed
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        downsampled_xyz = np.asarray(downsampled_pcd.points)
        
        # For now, just return xyz coordinates
        # TODO: Properly handle other point attributes after downsampling
        if point_cloud.shape[1] > 3:
            # Create dummy attributes for additional dimensions
            additional_dims = point_cloud.shape[1] - 3
            dummy_attrs = np.ones((len(downsampled_xyz), additional_dims))
            downsampled_points = np.hstack([downsampled_xyz, dummy_attrs])
        else:
            downsampled_points = downsampled_xyz
            
    elif method == "random":
        # Random sampling
        n_target = int(len(point_cloud) * 0.5)  # Keep 50% of points
        indices = np.random.choice(len(point_cloud), n_target, replace=False)
        downsampled_points = point_cloud[indices]
        
    elif method == "uniform":
        # Uniform sampling
        step = max(1, len(point_cloud) // 10000)  # Target ~10k points
        downsampled_points = point_cloud[::step]
        
    else:
        raise ValueError(f"Unsupported downsampling method: {method}")
    
    logger.debug(f"Downsampled from {len(point_cloud)} to {len(downsampled_points)} points")
    
    return downsampled_points