"""
Segmentation Model Architecture
U-Net based architecture for pixel-wise segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with two conv layers"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block with conv and pooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and conv"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class SegmentationModel(nn.Module):
    """
    U-Net style segmentation model for pixel-wise classification.
    
    Args:
        num_classes: Number of output classes (including background)
        in_channels: Number of input channels (3 for RGB)
        base_channels: Base number of channels (default: 64)
    """
    def __init__(self, num_classes, in_channels=3, base_channels=64):
        super(SegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)
        
        # Final output
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final output
        x = self.final(x)
        
        return x


class SegmentationModelMobileNet(nn.Module):
    """
    Lightweight segmentation model using MobileNetV2 backbone.
    Better for faster training and inference.
    """
    def __init__(self, num_classes, in_channels=3):
        super(SegmentationModelMobileNet, self).__init__()
        
        from torchvision.models import mobilenet_v2
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        
        self.num_classes = num_classes
        
        # Use pretrained MobileNetV2 as backbone
        mobilenet = mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = mobilenet.features
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Encoder
        features = self.backbone(x)
        
        # Decoder
        x = self.decoder(features)
        
        # Classification
        x = self.classifier(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x


class SegmentationModelResNet(nn.Module):
    """
    Segmentation model using ResNet backbone with DeepLabV3 head.
    More accurate but slower than MobileNet.
    """
    def __init__(self, num_classes, in_channels=3, backbone='resnet50'):
        super(SegmentationModelResNet, self).__init__()
        
        from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        
        self.num_classes = num_classes
        
        # Load pretrained model
        if backbone == 'resnet50':
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        
        # Replace classifier head with custom number of classes
        model.classifier = DeepLabHead(2048, num_classes)
        model.aux_classifier = None  # Remove auxiliary classifier
        
        self.model = model
        
    def forward(self, x):
        return self.model(x)['out']


def create_model(num_classes, architecture='unet', in_channels=3, **kwargs):
    """
    Factory function to create segmentation models.
    
    Args:
        num_classes: Number of output classes
        architecture: 'unet', 'mobilenet', or 'resnet'
        in_channels: Number of input channels
        **kwargs: Additional arguments for specific architectures
        
    Returns:
        Segmentation model
    """
    if architecture == 'unet':
        return SegmentationModel(num_classes, in_channels, **kwargs)
    elif architecture == 'mobilenet':
        return SegmentationModelMobileNet(num_classes, in_channels)
    elif architecture == 'resnet':
        return SegmentationModelResNet(num_classes, in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
