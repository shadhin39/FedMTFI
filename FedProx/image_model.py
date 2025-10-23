import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for FMNIST and CIFAR-10 classification."""
    
    def __init__(self, args, name, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.num_classes = num_classes
        
        # Determine input channels based on dataset
        if hasattr(args, 'dataset') and args.dataset == 'FMNIST':
            in_channels = 1  # Grayscale
            # For FMNIST (28x28): after 3 pooling -> 3x3
            fc_input_size = 128 * 3 * 3  # 1152
        else:  # CIFAR-10
            in_channels = 3  # RGB
            # For CIFAR-10 (32x32): after 3 pooling -> 4x4
            fc_input_size = 128 * 4 * 4  # 2048
            
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers with correct input size
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # For FMNIST: 28x28 -> 14x14, For CIFAR: 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # For FMNIST: 14x14 -> 7x7, For CIFAR: 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # For FMNIST: 7x7 -> 3x3, For CIFAR: 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper CNN."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCNN(nn.Module):
    """ResNet-style CNN for better performance on image classification."""
    
    def __init__(self, args, name, num_classes=10):
        super(ResNetCNN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.num_classes = num_classes
        
        # Determine input channels based on dataset
        if hasattr(args, 'dataset') and args.dataset == 'FMNIST':
            in_channels = 1  # Grayscale
        else:  # CIFAR-10
            in_channels = 3  # RGB
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_model(args, name, model_type='simple'):
    """Factory function to create models."""
    num_classes = 10  # Both FMNIST and CIFAR-10 have 10 classes
    
    if model_type == 'simple':
        return SimpleCNN(args, name, num_classes)
    elif model_type == 'resnet':
        return ResNetCNN(args, name, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")