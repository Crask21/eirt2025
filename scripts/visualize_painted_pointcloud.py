"""
Interactive 3D Painted Point Cloud Visualizer

This script provides an interactive 3D visualization of painted point clouds
with color-coded semantic classes using Open3D.
Supports panning, rotating, and zooming with mouse controls.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import open3d as o3d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaintedPointCloudViewer:
    """Interactive 3D viewer for painted point clouds using Open3D."""
    
    def __init__(self, painted_points: np.ndarray, class_names: list = None):
        """
        Initialize the viewer.
        
        Args:
            painted_points: Painted point cloud (N, 4+C) with [x, y, z, intensity, class_scores...]
            class_names: List of class names
        """
        self.painted_points = painted_points
        self.class_names = class_names or ["Background", "Car", "Pedestrian", "Cyclist"]
        
        # Extract point features
        self.xyz = painted_points[:, :3]
        self.intensity = painted_points[:, 3]
        
        # Extract class predictions
        self.class_scores = painted_points[:, 4:]
        self.class_ids = np.argmax(self.class_scores, axis=1)
        self.confidence = np.max(self.class_scores, axis=1)
        
        # Define class colors (RGB in [0, 1] range)
        self.class_colors = np.array([
            [0.2, 0.2, 0.2],      # Background - dark gray
            [1.0, 0.0, 0.0],      # Car - red
            [0.0, 1.0, 0.0],      # Pedestrian - green
            [0.0, 0.0, 1.0]       # Cyclist - blue
        ])
        
        # View settings
        self.view_mode = 'class'  # 'class', 'intensity', or 'confidence'
        self.distance_filter = 80.0  # Maximum distance to show
        
        # Create Open3D point cloud
        self.pcd = o3d.geometry.PointCloud()
        
        # Statistics
        self.compute_statistics()
        
    def compute_statistics(self):
        """Compute class distribution statistics."""
        self.stats = {}
        total_points = len(self.painted_points)
        
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.class_ids == i)
            percentage = (count / total_points) * 100
            avg_confidence = np.mean(self.confidence[self.class_ids == i]) if count > 0 else 0
            
            self.stats[class_name] = {
                'count': count,
                'percentage': percentage,
                'avg_confidence': avg_confidence
            }
            
        logger.info("\n" + "="*60)
        logger.info("Point Cloud Statistics:")
        logger.info("="*60)
        for class_name, stat in self.stats.items():
            logger.info(f"  {class_name:12s}: {stat['count']:7d} points ({stat['percentage']:5.1f}%) "
                       f"| Avg confidence: {stat['avg_confidence']:.3f}")
        logger.info("="*60)
    
    def get_point_colors(self, mode='class'):
        """
        Get colors for points based on visualization mode.
        
        Args:
            mode: 'class', 'intensity', or 'confidence'
            
        Returns:
            RGB colors array (N, 3)
        """
        if mode == 'class':
            # Color by class
            colors = self.class_colors[self.class_ids]
            
        elif mode == 'intensity':
            # Color by intensity (grayscale)
            intensity_norm = (self.intensity - self.intensity.min()) / (self.intensity.max() - self.intensity.min() + 1e-8)
            colors = np.stack([intensity_norm] * 3, axis=1)
            
        elif mode == 'confidence':
            # Color by confidence (colormap from red to green)
            colors = np.zeros((len(self.confidence), 3))
            colors[:, 0] = 1 - self.confidence  # Red channel (high when confidence is low)
            colors[:, 1] = self.confidence      # Green channel (high when confidence is high)
            colors[:, 2] = 0.0                   # Blue channel
            
        else:
            colors = np.ones((len(self.xyz), 3)) * 0.5  # Gray
            
        return colors
    
    def filter_points(self, max_distance=None):
        """
        Filter points by distance from origin.
        
        Args:
            max_distance: Maximum distance to show
            
        Returns:
            Boolean mask for points to display
        """
        if max_distance is None:
            max_distance = self.distance_filter
            
        distances = np.linalg.norm(self.xyz, axis=1)
        mask = distances <= max_distance
        return mask
    
    def update_point_cloud(self, mode=None):
        """
        Update the Open3D point cloud with current settings.
        
        Args:
            mode: Visualization mode (if None, uses current mode)
        """
        if mode is not None:
            self.view_mode = mode
            
        # Filter points
        mask = self.filter_points()
        xyz_filtered = self.xyz[mask]
        
        # Get colors
        colors = self.get_point_colors(self.view_mode)
        colors_filtered = colors[mask]
        
        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
        self.pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
        
        logger.info(f"Updated point cloud: {len(xyz_filtered)}/{len(self.xyz)} points | Mode: {self.view_mode}")
    
    def create_coordinate_frame(self, size=2.0):
        """
        Create a coordinate frame for reference.
        
        Args:
            size: Size of the coordinate frame
            
        Returns:
            Open3D coordinate frame geometry
        """
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    
    def create_ground_plane(self, size=50.0, z=-1.5):
        """
        Create a ground plane mesh.
        
        Args:
            size: Size of the ground plane
            z: Z-coordinate of the ground plane
            
        Returns:
            Open3D mesh geometry
        """
        # Create a simple ground plane
        vertices = np.array([
            [-size, -size, z],
            [size, -size, z],
            [size, size, z],
            [-size, size, z]
        ])
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.3, 0.3, 0.3])  # Gray
        mesh.compute_vertex_normals()
        
        return mesh
    
    def show(self, show_ground=True, show_coordinate_frame=True):
        """
        Display the interactive visualization using Open3D.
        
        Args:
            show_ground: Whether to show ground plane
            show_coordinate_frame: Whether to show coordinate frame
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Open3D Interactive Viewer")
        logger.info("="*60)
        logger.info("Mouse Controls:")
        logger.info("  Left Button + Drag    : Rotate view")
        logger.info("  Right Button + Drag   : Pan view")
        logger.info("  Scroll Wheel          : Zoom in/out")
        logger.info("  Ctrl + Left Button    : Pan view")
        logger.info("\nKeyboard Controls:")
        logger.info("  C : Toggle between visualization modes (class/intensity/confidence)")
        logger.info("  G : Toggle ground plane")
        logger.info("  F : Toggle coordinate frame")
        logger.info("  + : Increase distance filter")
        logger.info("  - : Decrease distance filter")
        logger.info("  R : Reset view")
        logger.info("  H : Print help")
        logger.info("  Q or ESC : Quit")
        logger.info("="*60)
        
        # Initialize point cloud
        self.update_point_cloud()
        
        # Create visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Painted Point Cloud Viewer - {self.view_mode.capitalize()} Mode", 
                         width=1280, height=720)
        
        # Add geometries
        vis.add_geometry(self.pcd)
        
        # Add coordinate frame
        self.coord_frame = None
        if show_coordinate_frame:
            self.coord_frame = self.create_coordinate_frame(size=3.0)
            vis.add_geometry(self.coord_frame)
        
        # Add ground plane
        self.ground_plane = None
        if show_ground:
            self.ground_plane = self.create_ground_plane(size=50.0, z=-1.5)
            vis.add_geometry(self.ground_plane)
        
        # Register key callbacks
        def toggle_mode(vis):
            modes = ['class', 'intensity', 'confidence']
            current_idx = modes.index(self.view_mode)
            new_mode = modes[(current_idx + 1) % len(modes)]
            self.update_point_cloud(new_mode)
            vis.update_geometry(self.pcd)
            vis.update_renderer()
            return False
        
        def toggle_ground(vis):
            if self.ground_plane is not None:
                vis.remove_geometry(self.ground_plane, reset_bounding_box=False)
                self.ground_plane = None
                logger.info("Ground plane hidden")
            else:
                self.ground_plane = self.create_ground_plane(size=50.0, z=-1.5)
                vis.add_geometry(self.ground_plane, reset_bounding_box=False)
                logger.info("Ground plane shown")
            vis.update_renderer()
            return False
        
        def toggle_frame(vis):
            if self.coord_frame is not None:
                vis.remove_geometry(self.coord_frame, reset_bounding_box=False)
                self.coord_frame = None
                logger.info("Coordinate frame hidden")
            else:
                self.coord_frame = self.create_coordinate_frame(size=3.0)
                vis.add_geometry(self.coord_frame, reset_bounding_box=False)
                logger.info("Coordinate frame shown")
            vis.update_renderer()
            return False
        
        def increase_distance(vis):
            self.distance_filter = min(200, self.distance_filter + 10)
            logger.info(f"Distance filter: {self.distance_filter}m")
            self.update_point_cloud()
            vis.update_geometry(self.pcd)
            vis.update_renderer()
            return False
        
        def decrease_distance(vis):
            self.distance_filter = max(10, self.distance_filter - 10)
            logger.info(f"Distance filter: {self.distance_filter}m")
            self.update_point_cloud()
            vis.update_geometry(self.pcd)
            vis.update_renderer()
            return False
        
        def reset_view(vis):
            vis.reset_view_point(True)
            logger.info("View reset")
            return False
        
        def print_help(vis):
            logger.info("\n" + "="*60)
            logger.info("Help - Keyboard Controls:")
            logger.info("="*60)
            logger.info("  C : Toggle mode (class/intensity/confidence)")
            logger.info("  G : Toggle ground plane")
            logger.info("  F : Toggle coordinate frame")
            logger.info("  + : Increase distance filter")
            logger.info("  - : Decrease distance filter")
            logger.info("  R : Reset view")
            logger.info("  H : Print this help")
            logger.info("  Q or ESC : Quit")
            logger.info("="*60)
            return False
        
        # Register callbacks
        vis.register_key_callback(ord("C"), toggle_mode)
        vis.register_key_callback(ord("G"), toggle_ground)
        vis.register_key_callback(ord("F"), toggle_frame)
        vis.register_key_callback(ord("+"), increase_distance)
        vis.register_key_callback(ord("="), increase_distance)  # For keyboards without numpad
        vis.register_key_callback(ord("-"), decrease_distance)
        vis.register_key_callback(ord("R"), reset_view)
        vis.register_key_callback(ord("H"), print_help)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        render_option.show_coordinate_frame = False  # We have our own
        
        # Run visualization
        vis.run()
        vis.destroy_window()
        logger.info("Viewer closed")


def load_painted_pointcloud(file_path: str) -> np.ndarray:
    """
    Load a painted point cloud from file.
    
    Args:
        file_path: Path to .npy or .bin file
        
    Returns:
        Painted point cloud array
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        painted_points = np.load(file_path)
    elif file_path.suffix == '.bin':
        # Load binary format
        painted_points = np.fromfile(file_path, dtype=np.float32)
        # Reshape based on expected dimensions (need to know num_features)
        # Assuming 4 (xyz + intensity) + 4 (class scores) = 8 features
        num_features = 8
        painted_points = painted_points.reshape(-1, num_features)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded painted point cloud from {file_path}")
    logger.info(f"Shape: {painted_points.shape}")
    
    return painted_points


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Interactive 3D Painted Point Cloud Visualizer (Open3D)"
    )
    parser.add_argument(
        "pointcloud_file",
        type=str,
        help="Path to painted point cloud file (.npy or .bin)"
    )
    parser.add_argument(
        "--class-names",
        nargs='+',
        default=["Background", "Car", "Pedestrian", "Cyclist"],
        help="Class names (default: Background Car Pedestrian Cyclist)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['class', 'intensity', 'confidence'],
        default='class',
        help="Initial visualization mode (default: class)"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=80.0,
        help="Initial distance filter in meters (default: 80.0)"
    )
    parser.add_argument(
        "--no-ground",
        action="store_true",
        help="Don't show ground plane"
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Don't show coordinate frame"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.pointcloud_file).exists():
        logger.error(f"File not found: {args.pointcloud_file}")
        sys.exit(1)
    
    # Load painted point cloud
    try:
        painted_points = load_painted_pointcloud(args.pointcloud_file)
    except Exception as e:
        logger.error(f"Error loading point cloud: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create viewer
    viewer = PaintedPointCloudViewer(
        painted_points=painted_points,
        class_names=args.class_names
    )
    
    # Set initial parameters
    viewer.view_mode = args.mode
    viewer.distance_filter = args.distance
    
    # Show viewer
    viewer.show(
        show_ground=not args.no_ground,
        show_coordinate_frame=not args.no_frame
    )


if __name__ == "__main__":
    main()
