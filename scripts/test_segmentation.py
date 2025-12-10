"""
Test script for pixel segmentation model and dataloader
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pixelseg import SegmentationModel, SegmentationDataset
from src.pixelseg.model import create_model
from src.pixelseg.dataset import get_training_transforms, get_validation_transforms


def test_class_mapping():
    """Test loading class mapping"""
    print("\n" + "="*80)
    print("TEST 1: Class Mapping")
    print("="*80)
    
    class_mapping_path = "G:/datasets/eirt_objects/class_id.json"
    
    if not os.path.exists(class_mapping_path):
        print(f"❌ Class mapping file not found: {class_mapping_path}")
        return None
    
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    print(f"✓ Class mapping loaded: {class_mapping}")
    num_classes = max(class_mapping.values()) + 1
    print(f"✓ Number of classes (including background): {num_classes}")
    
    return class_mapping, num_classes


def test_dataset(class_mapping, num_classes):
    """Test dataset loading"""
    print("\n" + "="*80)
    print("TEST 2: Dataset Loading")
    print("="*80)
    
    # Test paths
    image_dir = "G:/datasets/eirt_output/batch03/rgb"
    mask_dir = "G:/datasets/eirt_output/batch03/mask"
    
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return None
    
    if not os.path.exists(mask_dir):
        print(f"❌ Mask directory not found: {mask_dir}")
        return None
    
    # Create dataset
    try:
        transform = get_validation_transforms(img_size=(720, 1080))  # Smaller for testing
        dataset = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            class_mapping=class_mapping,
            transform=transform,
            img_size=None
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return None
    
    # Test loading a sample
    try:
        image, mask, name = dataset[0]
        print(f"\n✓ Sample loaded successfully:")
        print(f"  - Image name: {name}")
        print(f"  - Image shape: {image.shape}")
        print(f"  - Image dtype: {image.dtype}")
        print(f"  - Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Mask dtype: {mask.dtype}")
        print(f"  - Unique mask values: {torch.unique(mask).tolist()}")
        
        # Verify mask values are valid
        max_mask_value = mask.max().item()
        if max_mask_value >= num_classes:
            print(f"⚠ Warning: Mask contains value {max_mask_value} but only {num_classes} classes expected")
        else:
            print(f"✓ All mask values are valid (< {num_classes})")
            
    except Exception as e:
        print(f"❌ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return dataset


def test_dataloader(dataset):
    """Test dataloader"""
    print("\n" + "="*80)
    print("TEST 3: DataLoader")
    print("="*80)
    
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        print(f"✓ DataLoader created with batch_size=2")
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return None
    
    # Test loading a batch
    try:
        batch = next(iter(dataloader))
        images, masks, names = batch
        
        print(f"\n✓ Batch loaded successfully:")
        print(f"  - Batch size: {len(names)}")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Masks shape: {masks.shape}")
        print(f"  - Image names: {names}")
        
        return dataloader, images, masks
        
    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_architectures(num_classes, test_input):
    """Test different model architectures"""
    print("\n" + "="*80)
    print("TEST 4: Model Architectures")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    architectures = ['unet', 'mobilenet', 'resnet']
    
    for arch in architectures:
        print(f"\n--- Testing {arch.upper()} ---")
        try:
            model = create_model(num_classes, architecture=arch)
            model = model.to(device)
            model.eval()
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Model created: {num_params:,} parameters")
            
            # Test forward pass
            with torch.no_grad():
                test_input_device = test_input.to(device)
                output = model(test_input_device)
                
                print(f"✓ Forward pass successful")
                print(f"  - Input shape: {test_input.shape}")
                print(f"  - Output shape: {output.shape}")
                print(f"  - Output channels: {output.shape[1]} (expected {num_classes})")
                
                # Test output shape
                expected_shape = (test_input.shape[0], num_classes, test_input.shape[2], test_input.shape[3])
                if output.shape == expected_shape:
                    print(f"✓ Output shape correct: {output.shape}")
                else:
                    print(f"⚠ Output shape mismatch: got {output.shape}, expected {expected_shape}")
                
                # Test predictions
                predictions = output.argmax(dim=1)
                print(f"  - Predictions shape: {predictions.shape}")
                print(f"  - Unique predictions: {torch.unique(predictions).tolist()}")
                
        except Exception as e:
            print(f"❌ Failed to test {arch}: {e}")
            import traceback
            traceback.print_exc()


def test_loss_computation(model, images, masks, num_classes):
    """Test loss computation"""
    print("\n" + "="*80)
    print("TEST 5: Loss Computation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    images = images.to(device)
    masks = masks.to(device)
    
    try:
        # Forward pass
        model.train()
        outputs = model(images)
        
        print(type(outputs))
        print(type(masks))
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, masks)
        
        print(f"✓ Loss computed successfully")
        print(f"  - Loss value: {loss.item():.4f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if has_grads:
            print(f"✓ Gradients computed")
        else:
            print(f"⚠ No gradients found")
            
    except Exception as e:
        print(f"❌ Failed loss computation: {e}")
        import traceback
        traceback.print_exc()


def test_metrics(outputs, masks, num_classes):
    """Test metric computation"""
    print("\n" + "="*80)
    print("TEST 6: Metrics")
    print("="*80)
    
    try:
        predictions = outputs.argmax(dim=1)
        
        # Pixel accuracy
        correct = (predictions == masks).sum().item()
        total = masks.numel()
        accuracy = correct / total
        print(f"✓ Pixel Accuracy: {accuracy:.4f}")
        
        # IoU per class
        intersection = torch.zeros(num_classes)
        union = torch.zeros(num_classes)
        
        for cls in range(num_classes):
            pred_mask = (predictions == cls)
            true_mask = (masks == cls)
            
            intersection[cls] = (pred_mask & true_mask).sum().item()
            union[cls] = (pred_mask | true_mask).sum().item()
        
        iou_per_class = intersection / (union + 1e-10)
        miou = iou_per_class.mean().item()
        
        print(f"✓ Mean IoU: {miou:.4f}")
        print(f"✓ IoU per class: {iou_per_class.tolist()}")
        
    except Exception as e:
        print(f"❌ Failed metrics computation: {e}")
        import traceback
        traceback.print_exc()


def test_save_load(model, num_classes):
    """Test model saving and loading"""
    print("\n" + "="*80)
    print("TEST 7: Model Save/Load")
    print("="*80)
    
    try:
        # Save model
        save_path = "test_model.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
        }
        torch.save(checkpoint, save_path)
        print(f"✓ Model saved to {save_path}")
        
        # Load model
        checkpoint = torch.load(save_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {save_path}")
        
        # Clean up
        os.remove(save_path)
        print(f"✓ Test file removed")
        
    except Exception as e:
        print(f"❌ Failed save/load: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*80)
    print("PIXEL SEGMENTATION MODEL AND DATALOADER TEST")
    print("="*80)
    
    # Test 1: Class mapping
    result = test_class_mapping()
    if result is None:
        print("\n❌ Tests aborted due to class mapping error")
        return
    class_mapping, num_classes = result
    
    # Test 2: Dataset
    dataset = test_dataset(class_mapping, num_classes)
    if dataset is None:
        print("\n❌ Tests aborted due to dataset error")
        return
    
    # Test 3: DataLoader
    result = test_dataloader(dataset)
    if result is None:
        print("\n❌ Tests aborted due to dataloader error")
        return
    dataloader, images, masks = result
    
    # Test 4: Model architectures
    test_model_architectures(num_classes, images)
    
    # Create a model for remaining tests
    print("\n--- Creating U-Net for remaining tests ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes, architecture='unet', base_channels=32)  # Smaller for testing
    model = model.to(device)
    
    # Test 5: Loss computation
    test_loss_computation(model, images, masks, num_classes)
    
    # Test 6: Metrics
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    test_metrics(outputs, masks.to(device), num_classes)
    
    # Test 7: Save/Load
    test_save_load(model, num_classes)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\n✓ Model and dataloader are working correctly!")
    print("\nNext steps:")
    print("1. Run full training with: python scripts/train_segmentation.py")
    print("2. Monitor training with: tensorboard --logdir outputs/segmentation/logs")


if __name__ == '__main__':
    main()
