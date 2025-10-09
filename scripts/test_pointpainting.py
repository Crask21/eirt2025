"""
Test script to verify PointPainting implementation works correctly.

This script tests the core components without requiring the full KITTI dataset.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pointpainting import (
    PointPainter,
    SemanticSegmentationModel,
    PointCloudProjector
)


def test_segmentation_model():
    """Test semantic segmentation model."""
    print("Testing SemanticSegmentationModel...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    
    # Initialize model
    model = SemanticSegmentationModel(num_classes=4, device="cpu")
    
    # Run prediction
    segmentation = model.predict(dummy_image)
    
    print(f"✓ Input image shape: {dummy_image.shape}")
    print(f"✓ Segmentation shape: {segmentation.shape}")
    print(f"✓ Score range: [{segmentation.min():.3f}, {segmentation.max():.3f}]")
    
    assert segmentation.shape == (375, 1242, 4)
    assert 0 <= segmentation.min() <= segmentation.max() <= 1
    print("✓ SemanticSegmentationModel test passed!\n")


def test_point_projector():
    """Test point cloud projector."""
    print("Testing PointCloudProjector...")
    
    # Create dummy point cloud
    points_3d = np.random.randn(1000, 3) * 10
    points_3d[:, 2] += 20  # Ensure points are in front of camera
    
    # Create dummy camera matrix and transform
    camera_matrix = np.array([
        [721.5377, 0, 609.5593],
        [0, 721.5377, 172.8540],
        [0, 0, 1]
    ], dtype=np.float32)
    
    transform_matrix = np.eye(4, dtype=np.float32)
    
    # Initialize projector
    projector = PointCloudProjector()
    
    # Run projection
    results = projector.project_points(
        points_3d=points_3d,
        camera_matrix=camera_matrix,
        transform_matrix=transform_matrix,
        image_shape=(375, 1242)
    )
    
    projected_points = results["projected_points"]
    valid_mask = results["valid_mask"]
    
    print(f"✓ Input points: {len(points_3d)}")
    print(f"✓ Valid projections: {valid_mask.sum()}")
    print(f"✓ Projection ratio: {valid_mask.sum() / len(points_3d) * 100:.1f}%")
    
    assert projected_points.shape == (len(points_3d), 2)
    assert valid_mask.shape == (len(points_3d),)
    print("✓ PointCloudProjector test passed!\n")


def test_point_painter():
    """Test complete PointPainter pipeline."""
    print("Testing PointPainter...")
    
    # Create dummy data
    point_cloud = np.random.randn(1000, 4)
    point_cloud[:, 2] += 20  # Ensure points are in front
    point_cloud[:, 3] = np.random.uniform(0, 1, 1000)  # Intensity
    
    image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    
    camera_matrix = np.array([
        [721.5377, 0, 609.5593],
        [0, 721.5377, 172.8540],
        [0, 0, 1]
    ], dtype=np.float32)
    
    transform_matrix = np.eye(4, dtype=np.float32)
    
    # Initialize PointPainter
    painter = PointPainter(device="cpu")
    
    # Run pipeline
    results = painter.paint_pointcloud(
        point_cloud=point_cloud,
        image=image,
        camera_matrix=camera_matrix,
        transform_matrix=transform_matrix
    )
    
    painted_points = results["painted_points"]
    
    print(f"✓ Original point cloud: {point_cloud.shape}")
    print(f"✓ Painted point cloud: {painted_points.shape}")
    print(f"✓ Added semantic features: {painted_points.shape[1] - point_cloud.shape[1]}")
    
    # Check that painted points have additional semantic features
    assert painted_points.shape[0] == point_cloud.shape[0]
    assert painted_points.shape[1] == point_cloud.shape[1] + 4  # 4 classes
    
    # Check that original features are preserved
    np.testing.assert_array_almost_equal(
        painted_points[:, :4], point_cloud, decimal=5
    )
    
    # Check semantic features
    semantic_features = painted_points[:, 4:]
    assert semantic_features.shape[1] == 4
    assert np.all(semantic_features >= 0)
    assert np.all(semantic_features <= 1)
    
    print("✓ PointPainter test passed!\n")


def test_class_statistics():
    """Test class statistics computation."""
    print("Testing class statistics...")
    
    # Create painted point cloud with known class distribution
    n_points = 1000
    painted_points = np.random.randn(n_points, 8)  # 4 original + 4 semantic
    
    # Create specific class distribution
    semantic_features = np.zeros((n_points, 4))
    semantic_features[:500, 0] = 1.0    # 500 background
    semantic_features[500:700, 1] = 1.0  # 200 car
    semantic_features[700:900, 2] = 1.0  # 200 pedestrian
    semantic_features[900:, 3] = 1.0     # 100 cyclist
    
    painted_points[:, 4:] = semantic_features
    
    # Test statistics
    painter = PointPainter(device="cpu")
    stats = painter.get_class_statistics(painted_points)
    
    expected_counts = [500, 200, 200, 100]
    class_names = ["background", "car", "pedestrian", "cyclist"]
    
    print("Class distribution:")
    for i, class_name in enumerate(class_names):
        count = stats[class_name]["count"]
        percentage = stats[class_name]["percentage"]
        expected = expected_counts[i]
        
        print(f"  {class_name}: {count} points ({percentage:.1f}%)")
        assert count == expected
    
    print("✓ Class statistics test passed!\n")


def main():
    """Run all tests."""
    print("Running PointPainting implementation tests...\n")
    
    try:
        test_segmentation_model()
        test_point_projector()
        test_point_painter()
        test_class_statistics()
        
        print("🎉 All tests passed! PointPainting implementation is working correctly.")
        print("\nNext steps:")
        print("1. Extract your KITTI dataset to D:/datasets/kitti")
        print("2. Run: python scripts/pointpainting_demo.py")
        print("3. Or use the Jupyter notebook: notebooks/pointpainting_demo.ipynb")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()