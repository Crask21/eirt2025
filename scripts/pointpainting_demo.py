"""
PointPainting Demo Script for KITTI Dataset

This script demonstrates how to use the PointPainting implementation
to process KITTI dataset samples and visualize results.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pointpainting import (
    PointPainter,
    SemanticSegmentationModel,
    PointCloudProjector,
    KITTIDataLoader,
    visualize_results,
    visualize_3d_pointcloud,
    compute_metrics,
    save_painted_pointcloud
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_kitti_dataset(kitti_archive_dir: str, output_dir: str):
    """
    Extract KITTI dataset archives to output directory.
    
    Args:
        kitti_archive_dir: Directory containing KITTI .zip files
        output_dir: Directory to extract files to
    """
    import zipfile
    
    kitti_archive_dir = Path(kitti_archive_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected KITTI files
    expected_files = [
        "data_object_calib.zip",
        "data_object_image_2.zip", 
        "data_object_image_3.zip",
        "data_object_label_2.zip",
        "data_object_velodyne.zip"
    ]
    
    for zip_file in expected_files:
        zip_path = kitti_archive_dir / zip_file
        
        if not zip_path.exists():
            logger.warning(f"Archive not found: {zip_path}")
            continue
            
        logger.info(f"Extracting {zip_file}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to training directory
            zip_ref.extractall(output_dir)
            
        logger.info(f"Extracted {zip_file}")
    
    # Reorganize structure if needed
    training_dir = output_dir / "training"
    if not training_dir.exists():
        # Check if files were extracted to a subdirectory
        for subdir in output_dir.iterdir():
            if subdir.is_dir() and (subdir / "training").exists():
                # Move training directory to expected location
                (subdir / "training").rename(training_dir)
                break
    
    logger.info(f"KITTI dataset extracted to {output_dir}")


def demo_pointpainting_pipeline(
    kitti_root: str,
    sample_id: str = "000000",
    output_dir: str = "./outputs",
    visualize_3d: bool = False,
    save_results: bool = True
):
    """
    Demonstrate the complete PointPainting pipeline on a KITTI sample.
    
    Args:
        kitti_root: Root directory of KITTI dataset
        sample_id: Sample ID to process
        output_dir: Directory to save outputs
        visualize_3d: Whether to show 3D visualization
        save_results: Whether to save painted point cloud
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running PointPainting demo on KITTI sample {sample_id}")
    
    # Initialize components
    logger.info("Initializing PointPainting components...")
    
    # Data loader
    data_loader = KITTIDataLoader(kitti_root, split="training")
    logger.info(f"Loaded KITTI dataset with {len(data_loader)} samples")
    
    # Segmentation model (using pretrained DeepLabV3+)
    segmentation_model = SemanticSegmentationModel(
        num_classes=4,  # background, car, pedestrian, cyclist
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )
    
    # Point cloud projector
    projector = PointCloudProjector()
    
    # Main PointPainter
    painter = PointPainter(
        segmentation_model=segmentation_model,
        projector=projector,
        class_names=["background", "car", "pedestrian", "cyclist"]
    )
    
    # Load sample data
    logger.info(f"Loading sample {sample_id}...")
    try:
        sample_data = data_loader.load_sample(sample_id)
    except Exception as e:
        logger.error(f"Error loading sample {sample_id}: {e}")
        return
    
    point_cloud = sample_data["point_cloud"]
    image = sample_data["image"]
    camera_matrix = sample_data["camera_matrix"]
    transform_matrix = sample_data["transform_matrix"]
    
    logger.info(f"Sample loaded - Point cloud: {point_cloud.shape}, Image: {image.shape}")
    
    # Run PointPainting pipeline
    logger.info("Running PointPainting pipeline...")
    
    try:
        painting_results = painter.paint_pointcloud(
            point_cloud=point_cloud,
            image=image,
            camera_matrix=camera_matrix,
            transform_matrix=transform_matrix,
            return_painted_points=True,
            return_segmentation=True
        )
    except Exception as e:
        logger.error(f"Error in PointPainting pipeline: {e}")
        return
    
    painted_points = painting_results["painted_points"]
    segmentation_scores = painting_results["segmentation_scores"]
    projected_points = painting_results["projected_points"]
    valid_mask = painting_results["valid_mask"]
    
    logger.info(f"PointPainting completed - Painted points: {painted_points.shape}")
    
    # Compute statistics
    class_stats = painter.get_class_statistics(painted_points)
    logger.info("Class distribution:")
    for class_name, stats in class_stats.items():
        logger.info(f"  {class_name}: {stats['count']} points ({stats['percentage']:.1f}%)")
    
    # Create projection results dict for visualization
    projection_result = {
        "projected_points": projected_points,
        "valid_mask": valid_mask,
        "distances": np.linalg.norm(point_cloud[:, :3], axis=1)
    }
    
    # Visualize results
    logger.info("Creating visualizations...")
    
    # 2D visualization
    vis_path = output_dir / f"pointpainting_demo_{sample_id}.png"
    visualize_results(
        image=image,
        point_cloud=point_cloud,
        painted_points=painted_points,
        projection_result=projection_result,
        class_names=painter.class_names,
        save_path=vis_path
    )
    
    # 3D visualization (optional)
    if visualize_3d:
        logger.info("Showing 3D visualization...")
        visualize_3d_pointcloud(
            painted_points=painted_points,
            class_names=painter.class_names
        )
    
    # Save painted point cloud
    if save_results:
        painted_pc_path = output_dir / f"painted_pointcloud_{sample_id}.bin"
        save_painted_pointcloud(
            painted_points=painted_points,
            output_path=painted_pc_path,
            format="bin"
        )
        
        # Also save as numpy array for easy loading
        painted_npy_path = output_dir / f"painted_pointcloud_{sample_id}.npy"
        save_painted_pointcloud(
            painted_points=painted_points,
            output_path=painted_npy_path,
            format="npy"
        )
    
    # Compute and save metrics
    metrics = compute_metrics(
        painted_points=painted_points,
        class_names=painter.class_names
    )
    
    metrics_path = output_dir / f"metrics_{sample_id}.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"PointPainting Metrics for Sample {sample_id}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total points: {metrics['total_points']}\n")
        f.write(f"Number of classes: {metrics['num_classes']}\n")
        f.write(f"Mean confidence: {metrics['mean_confidence']:.3f}\n")
        f.write(f"Std confidence: {metrics['std_confidence']:.3f}\n\n")
        f.write("Class Distribution:\n")
        for class_name, stats in metrics["class_distribution"].items():
            f.write(f"  {class_name}: {stats['count']} points ({stats['percentage']:.1f}%)\n")
    
    logger.info(f"Demo completed! Results saved to {output_dir}")
    

def batch_process_samples(
    kitti_root: str,
    sample_ids: list,
    output_dir: str = "./outputs/batch",
    max_samples: int = 10
):
    """
    Process multiple KITTI samples in batch.
    
    Args:
        kitti_root: Root directory of KITTI dataset
        sample_ids: List of sample IDs to process
        output_dir: Directory to save batch outputs
        max_samples: Maximum number of samples to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch processing of {min(len(sample_ids), max_samples)} samples")
    
    # Initialize PointPainter
    painter = PointPainter()
    data_loader = KITTIDataLoader(kitti_root, split="training")
    
    # Process samples
    processed_count = 0
    for i, sample_id in enumerate(sample_ids[:max_samples]):
        try:
            logger.info(f"Processing sample {i+1}/{min(len(sample_ids), max_samples)}: {sample_id}")
            
            # Load sample
            sample_data = data_loader.load_sample(sample_id)
            
            # Run PointPainting
            painting_results = painter.paint_pointcloud(
                point_cloud=sample_data["point_cloud"],
                image=sample_data["image"],
                camera_matrix=sample_data["camera_matrix"],
                transform_matrix=sample_data["transform_matrix"]
            )
            
            # Save results
            painted_points = painting_results["painted_points"]
            output_path = output_dir / f"painted_{sample_id}.npy"
            save_painted_pointcloud(painted_points, output_path, format="npy")
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            continue
    
    logger.info(f"Batch processing completed: {processed_count}/{max_samples} samples processed")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="PointPainting Demo for KITTI Dataset")
    
    parser.add_argument(
        "--kitti_root", 
        type=str,
        default="F:/datasets/kitti",
        help="Root directory of KITTI dataset"
    )
    parser.add_argument(
        "--kitti_archives",
        type=str,
        help="Directory containing KITTI .zip archives (will extract if provided)"
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="000100",
        help="Sample ID to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize_3d",
        action="store_true",
        help="Show 3D visualization"
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        help="Process multiple samples in batch"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples for batch processing"
    )
    
    args = parser.parse_args()
    
    # Extract KITTI dataset if archives provided
    if args.kitti_archives:
        logger.info("Extracting KITTI dataset...")
        extract_kitti_dataset(args.kitti_archives, args.kitti_root)
    
    # Check if KITTI dataset exists
    kitti_path = Path(args.kitti_root)
    if not kitti_path.exists():
        logger.error(f"KITTI dataset not found at {kitti_path}")
        logger.info("Please provide the correct path or use --kitti_archives to extract")
        return
    
    try:
        if args.batch_process:
            # Get available sample IDs
            data_loader = KITTIDataLoader(args.kitti_root, split="training")
            sample_ids = data_loader.sample_ids
            
            batch_process_samples(
                kitti_root=args.kitti_root,
                sample_ids=sample_ids,
                output_dir=args.output_dir + "/batch",
                max_samples=args.max_samples
            )
        else:
            # Single sample demo
            demo_pointpainting_pipeline(
                kitti_root=args.kitti_root,
                sample_id=args.sample_id,
                output_dir=args.output_dir,
                visualize_3d=args.visualize_3d,
                save_results=True
            )
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()