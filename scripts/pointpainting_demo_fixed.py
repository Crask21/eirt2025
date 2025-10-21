"""
PointPainting Demo Script for KITTI Dataset (Fixed Version)

This script demonstrates how to use the PointPainting implementation
to process KITTI dataset samples and visualize results.
Uses proper COCO-to-KITTI class mapping for accurate segmentation.
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


class COCOtoKITTIMapper:
    """
    Maps COCO class predictions (21 classes) to KITTI classes (4 classes).
    This allows using pretrained DeepLabV3 models without losing pretrained weights.
    """
    
    def __init__(self):
        """Initialize the COCO to KITTI class mapper."""
        # COCO class names (Pascal VOC + COCO subset used by DeepLabV3)
        self.coco_class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # KITTI class names
        self.kitti_class_names = ["background", "car", "pedestrian", "cyclist"]
        
        # Map COCO classes to KITTI classes
        # COCO indices: 0=background, 2=bicycle, 6=bus, 7=car, 14=motorbike, 15=person, 19=train
        self.coco_to_kitti = {
            0: 0,   # background -> background
            7: 1,   # car -> car
            6: 1,   # bus -> car
            19: 1,  # train -> car
            15: 2,  # person -> pedestrian
            2: 3,   # bicycle -> cyclist
            14: 3,  # motorbike -> cyclist
        }
    
    def map_segmentation(self, coco_segmentation: np.ndarray) -> np.ndarray:
        """
        Map COCO segmentation (21 classes) to KITTI segmentation (4 classes).
        
        Args:
            coco_segmentation: Segmentation scores of shape (H, W, 21)
            
        Returns:
            KITTI segmentation scores of shape (H, W, 4)
        """
        h, w, num_coco_classes = coco_segmentation.shape
        
        # Get COCO class predictions
        coco_class_pred = np.argmax(coco_segmentation, axis=2)  # (H, W)
        
        # Create KITTI segmentation with probability accumulation
        kitti_segmentation = np.zeros((h, w, 4), dtype=np.float32)
        
        # Accumulate probabilities from COCO classes to KITTI classes
        for coco_class, kitti_class in self.coco_to_kitti.items():
            if coco_class < num_coco_classes:
                kitti_segmentation[:, :, kitti_class] += coco_segmentation[:, :, coco_class]
        
        # Normalize to ensure probabilities sum to ~1
        prob_sum = np.sum(kitti_segmentation, axis=2, keepdims=True)
        prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)  # Avoid division by zero
        kitti_segmentation = kitti_segmentation / prob_sum
        
        return kitti_segmentation
    
    def map_painted_points_classes(self, painted_points: np.ndarray, coco_segmentation: np.ndarray) -> np.ndarray:
        """
        Map painted point cloud classes from COCO to KITTI.
        
        Args:
            painted_points: Painted points array (N, 7) [x, y, z, intensity, class_id, confidence, original_idx]
            coco_segmentation: Original COCO segmentation used for painting
            
        Returns:
            Updated painted points with KITTI class IDs
        """
        # This is a simplified version - in practice, you'd need to re-paint with KITTI segmentation
        # For now, we'll just map the class IDs
        painted_points_copy = painted_points.copy()
        
        for i in range(len(painted_points_copy)):
            coco_class = int(painted_points_copy[i, 4])
            if coco_class in self.coco_to_kitti:
                painted_points_copy[i, 4] = self.coco_to_kitti[coco_class]
            else:
                painted_points_copy[i, 4] = 0  # Map to background if not in mapping
        
        return painted_points_copy


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
    Uses proper COCO-to-KITTI class mapping.
    
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
    
    # Segmentation model (using pretrained DeepLabV3+ with 21 COCO classes)
    segmentation_model = SemanticSegmentationModel(
        num_classes=21,  # Use original COCO classes to keep pretrained weights
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )
    
    # Initialize COCO-to-KITTI mapper
    class_mapper = COCOtoKITTIMapper()
    
    # Point cloud projector
    projector = PointCloudProjector()
    
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
    
    # Run segmentation with COCO model
    logger.info("Running semantic segmentation (COCO classes)...")
    coco_segmentation = segmentation_model.predict(
        image=image,
        return_probabilities=True
    )
    logger.info(f"COCO segmentation shape: {coco_segmentation.shape}")
    
    # Map COCO segmentation to KITTI classes
    logger.info("Mapping COCO classes to KITTI classes...")
    kitti_segmentation = class_mapper.map_segmentation(coco_segmentation)
    logger.info(f"KITTI segmentation shape: {kitti_segmentation.shape}")
    
    # Log KITTI class distribution in segmentation
    kitti_class_pred = np.argmax(kitti_segmentation, axis=2)
    for i, class_name in enumerate(class_mapper.kitti_class_names):
        count = np.sum(kitti_class_pred == i)
        percentage = (count / kitti_class_pred.size) * 100
        if percentage > 0.1:
            logger.info(f"  Segmentation - {class_name}: {percentage:.2f}%")
    
    # Create a temporary PointPainter with KITTI classes
    painter = PointPainter(
        segmentation_model=segmentation_model,
        projector=projector,
        class_names=class_mapper.kitti_class_names
    )
    
    # Paint point cloud using KITTI segmentation directly
    logger.info("Painting point cloud with KITTI classes...")
    
    # Project points to image
    projection_result = projector.project_points(
        points_3d=point_cloud[:, :3],  # Only xyz coordinates
        camera_matrix=camera_matrix,
        transform_matrix=transform_matrix,
        image_shape=image.shape[:2]  # (height, width)
    )
    
    projected_points = projection_result["projected_points"]
    valid_mask = projection_result["valid_mask"]
    
    # Paint points with KITTI segmentation
    painted_points = painter._paint_points(
        point_cloud=point_cloud,
        segmentation_scores=kitti_segmentation,
        projected_points=projected_points,
        valid_mask=valid_mask
    )
    
    logger.info(f"PointPainting completed - Painted points: {painted_points.shape}")
    
    # Compute statistics
    class_stats = painter.get_class_statistics(painted_points)
    logger.info("Point cloud class distribution:")
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
    Process multiple KITTI samples in batch using proper COCO-to-KITTI mapping.
    
    Args:
        kitti_root: Root directory of KITTI dataset
        sample_ids: List of sample IDs to process
        output_dir: Directory to save batch outputs
        max_samples: Maximum number of samples to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch processing of {min(len(sample_ids), max_samples)} samples")
    
    # Initialize components with COCO classes
    segmentation_model = SemanticSegmentationModel(
        num_classes=21,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )
    
    projector = PointCloudProjector()
    class_mapper = COCOtoKITTIMapper()
    data_loader = KITTIDataLoader(kitti_root, split="training")
    
    painter = PointPainter(
        segmentation_model=segmentation_model,
        projector=projector,
        class_names=class_mapper.kitti_class_names
    )
    
    # Process samples
    processed_count = 0
    for i, sample_id in enumerate(sample_ids[:max_samples]):
        try:
            logger.info(f"Processing sample {i+1}/{min(len(sample_ids), max_samples)}: {sample_id}")
            
            # Load sample
            sample_data = data_loader.load_sample(sample_id)
            
            # Run segmentation with COCO model
            coco_segmentation = segmentation_model.predict(
                image=sample_data["image"],
                return_probabilities=True
            )
            
            # Map to KITTI classes
            kitti_segmentation = class_mapper.map_segmentation(coco_segmentation)
            
            # Project and paint
            projection_result = projector.project_points(
                points_3d=sample_data["point_cloud"][:, :3],
                camera_matrix=sample_data["camera_matrix"],
                transform_matrix=sample_data["transform_matrix"],
                image_shape=sample_data["image"].shape[:2]
            )
            
            painted_points = painter._paint_points(
                point_cloud=sample_data["point_cloud"],
                segmentation_scores=kitti_segmentation,
                projected_points=projection_result["projected_points"],
                valid_mask=projection_result["valid_mask"]
            )
            
            # Save results
            output_path = output_dir / f"painted_{sample_id}.npy"
            save_painted_pointcloud(painted_points, output_path, format="npy")
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"Batch processing completed: {processed_count}/{max_samples} samples processed")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="PointPainting Demo for KITTI Dataset (Fixed)")
    
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
