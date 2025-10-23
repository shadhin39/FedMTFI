import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class SimpleCNN(nn.Module):
    """A lightweight CNN with feature extraction for Cluster 0."""

    def __init__(self, num_classes: int = 10, width: int = 64, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.pool = nn.MaxPool2d(2, 2)

        # Use adaptive pooling to handle arbitrary image sizes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        feat = self.features(x)
        pooled = self.gap(feat)
        logits = self.head(pooled)
        return logits, feat


class ResNetLike(nn.Module):
    """ResNet-like architecture with residual connections for Cluster 1."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Sequential processing without problematic residual connections
        x = self.res_block1(x)  # 32x32 -> 64x14x14 (with stride=2)
        x = F.relu(x)
        
        x = self.res_block2(x)  # 64x14x14 -> 128x7x7 (with stride=2)
        x = F.relu(x)
        
        x = self.res_block3(x)  # 128x7x7 -> 256x4x4 (with stride=2)
        x = F.relu(x)
        
        return x
    
    def forward(self, x):
        feat = self.features(x)
        pooled = self.gap(feat)
        logits = self.head(pooled)
        return logits, feat


class MobileNetLike(nn.Module):
    """MobileNet-like architecture with depthwise separable convolutions for Cluster 2."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable convolution blocks
        self.dw_blocks = nn.ModuleList([
            self._make_dw_block(32, 64, stride=1),
            self._make_dw_block(64, 128, stride=2),
            self._make_dw_block(128, 128, stride=1),
            self._make_dw_block(128, 256, stride=2),
            self._make_dw_block(256, 256, stride=1),
            self._make_dw_block(256, 512, stride=2),
        ])
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def _make_dw_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.dw_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        feat = self.features(x)
        pooled = self.gap(feat)
        logits = self.head(pooled)
        return logits, feat


class ResNet18Like(nn.Module):
    """ResNet-18 inspired architecture for Cluster 3."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks with proper residual connections
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block may have stride > 1 for downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks have stride = 1
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        logits = self.head(pooled)
        return logits, feat


class BasicBlock(nn.Module):
    """Basic ResNet block with proper residual connections."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class StudentCNN(nn.Module):
    """Simple student model for knowledge distillation."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    
    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
    
    def forward(self, x):
        feat = self.features(x)
        pooled = self.gap(feat)
        logits = self.head(pooled)
        return logits, feat


def build_cluster_model(cluster_id: int, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
    """Build model based on cluster ID with adaptive input channels."""
    if cluster_id == 0:
        return SimpleCNN(num_classes=num_classes, width=64, in_channels=in_channels, image_size=image_size)
    elif cluster_id == 1:
        return ResNetLike(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    elif cluster_id == 2:
        return MobileNetLike(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    elif cluster_id == 3:
        return ResNet18Like(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    else:
        raise ValueError(f"Invalid cluster_id: {cluster_id}. Must be 0, 1, 2, or 3.")


def build_student(num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
    """Build student model for knowledge distillation with adaptive input channels."""
    return StudentCNN(num_classes=num_classes, in_channels=in_channels, image_size=image_size)


def build_adaptive_model(cluster_id: int, dataset_name: str, num_classes: int = 10, image_size: int = 28):
    """Build model with adaptive input channels based on dataset."""
    # Determine input channels based on dataset
    if dataset_name in ["MNIST", "FashionMNIST"]:
        in_channels = 1  # Grayscale
    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        in_channels = 3  # RGB
    else:
        in_channels = 1  # Default to grayscale
    
    return build_cluster_model(cluster_id, num_classes, in_channels, image_size)


def build_adaptive_student(dataset_name: str, num_classes: int = 10, image_size: int = 28):
    """Build student model with adaptive input channels based on dataset."""
    # Determine input channels based on dataset
    if dataset_name in ["MNIST", "FashionMNIST"]:
        in_channels = 1  # Grayscale
    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        in_channels = 3  # RGB
    else:
        in_channels = 1  # Default to grayscale
    
    return build_student(num_classes, in_channels, image_size)


# Legacy functions for backward compatibility
def build_teacher(num_classes: int = 10, in_channels: int = 1, image_size: int = 28, use_pretrained: bool = False, cluster_id: int = 0):
    """Legacy function - now redirects to build_cluster_model."""
    return build_cluster_model(cluster_id, num_classes, in_channels, image_size)