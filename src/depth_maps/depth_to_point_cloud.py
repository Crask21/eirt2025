import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


def load_depth_map(depth_path):
    """
    Load a grayscale depth map image.

    Args:
        depth_path: Path to the depth map image

    Returns:
        numpy array of the depth map
    """
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        raise ValueError(f"Could not load depth map from {depth_path}")
    return depth_img


def depth_map_to_point_cloud(depth_map, rgb_image=None, invert_depth=True,
                             max_depth=10.0, fx=525.0, fy=525.0,
                             cx=None, cy=None):
    """
    Convert a depth map to a 3D point cloud.

    Args:
        depth_map: 2D numpy array (grayscale image) where darker = closer, brighter = farther
        rgb_image: Optional RGB image for coloring the point cloud
        invert_depth: If True, invert the depth values (darker pixels = closer)
        max_depth: Maximum depth value in meters/units
        fx, fy: Focal lengths (intrinsic camera parameters)
        cx, cy: Principal point (center of image). If None, uses image center

    Returns:
        open3d.geometry.PointCloud object
    """
    height, width = depth_map.shape

    # Set principal point to image center if not provided
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Normalize depth map to [0, 1] range
    depth_normalized = depth_map.astype(np.float32) / 255.0

    # Invert if darker means closer (which is your case)
    if invert_depth:
        depth_normalized = 1.0 - depth_normalized

    # Scale to actual depth values
    depth_values = depth_normalized * max_depth

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D coordinates using pinhole camera model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    z = depth_values
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack coordinates and reshape to (N, 3)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter out points with zero or very small depth
    valid_mask = points[:, 2] > 0.01
    points = points[valid_mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors if RGB image is provided
    if rgb_image is not None:
        if len(rgb_image.shape) == 2:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
        elif rgb_image.shape[2] == 4:
            # Convert RGBA to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
        else:
            # Convert BGR to RGB (OpenCV loads as BGR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        colors = rgb_image.reshape(-1, 3)[valid_mask] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use depth-based coloring (blue = close, red = far)
        colors = np.zeros((len(points), 3))
        depth_normalized_filtered = depth_normalized.flatten()[valid_mask]
        colors[:, 0] = depth_normalized_filtered  # Red channel
        colors[:, 2] = 1.0 - depth_normalized_filtered  # Blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_point_cloud(pcd, window_name="Point Cloud Viewer"):
    """
    Visualize a point cloud using Open3D.

    Args:
        pcd: open3d.geometry.PointCloud object
        window_name: Name of the visualization window
    """
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    # Set view options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def load_and_visualize(depth_path, rgb_path=None, **kwargs):
    """
    Load a depth map (and optionally RGB image) and visualize as point cloud.

    Args:
        depth_path: Path to depth map image
        rgb_path: Optional path to RGB image
        **kwargs: Additional arguments for depth_map_to_point_cloud
    """
    # Load depth map
    depth_map = load_depth_map(depth_path)

    # Load RGB image if provided
    rgb_image = None
    if rgb_path is not None:
        rgb_image = cv2.imread(str(rgb_path))
        if rgb_image is None:
            print(f"Warning: Could not load RGB image from {rgb_path}")

    # Convert to point cloud
    pcd = depth_map_to_point_cloud(depth_map, rgb_image, **kwargs)

    # Visualize
    visualize_point_cloud(
        pcd, window_name=f"Point Cloud - {Path(depth_path).name}")

    return pcd


def batch_visualize(depth_dir, rgb_dir=None, pattern="*.png", **kwargs):
    """
    Visualize multiple depth maps from a directory.

    Args:
        depth_dir: Directory containing depth map images
        rgb_dir: Optional directory containing RGB images
        pattern: File pattern to match (default: "*.png")
        **kwargs: Additional arguments for depth_map_to_point_cloud
    """
    depth_dir = Path(depth_dir)
    depth_files = sorted(depth_dir.glob(pattern))

    if not depth_files:
        print(f"No depth maps found in {depth_dir} with pattern {pattern}")
        return

    print(f"Found {len(depth_files)} depth maps")

    for depth_path in depth_files:
        print(f"\nProcessing: {depth_path.name}")

        # Find corresponding RGB image if rgb_dir is provided
        rgb_path = None
        if rgb_dir is not None:
            rgb_path = Path(rgb_dir) / depth_path.name
            if not rgb_path.exists():
                print(f"Warning: No matching RGB image found at {rgb_path}")
                rgb_path = None

        # Load and visualize
        load_and_visualize(depth_path, rgb_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    depth_dir = Path("data/batch01/depth")
    rgb_dir = Path("data/batch01/rgb")

    # Visualize all depth maps in the directory
    batch_visualize(depth_dir, rgb_dir, max_depth=10.0, invert_depth=True)
