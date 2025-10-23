#!/usr/bin/env python3
"""
Test script to verify dataset-specific parameter configurations for CIFAR-10 and FashionMNIST.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CFG
from server import ClusterServer
from models import build_adaptive_student

def test_dataset_configs():
    """Test that dataset-specific configurations are properly loaded and used."""
    
    print("Testing dataset-specific parameter configurations...")
    print("=" * 60)
    
    # Test 1: Verify config loading
    print("1. Testing configuration loading:")
    
    for dataset_name in ["FashionMNIST", "CIFAR10"]:
        config = CFG.dataset_configs.get(dataset_name)
        print(f"\n{dataset_name} configuration:")
        print(f"  - Input channels: {config['in_channels']}")
        print(f"  - Image size: {config['image_size']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Learning rate: {config['lr_server']}")
        print(f"  - Distillation epochs: {config['distill_epochs']}")
        print(f"  - Temperature: {config['temperature']}")
        print(f"  - Lambda KD: {config['lambda_kd']}")
        print(f"  - Lambda Feature: {config['lambda_feat']}")
        print(f"  - Lambda CE: {config['lambda_ce']}")
        print(f"  - Normalization: {config['normalization']}")
    
    print("\n" + "=" * 60)
    
    # Test 2: Verify model creation with different input channels
    print("2. Testing model creation with dataset-specific parameters:")
    
    device = torch.device("cpu")
    
    for dataset_name in ["FashionMNIST", "CIFAR10"]:
        config = CFG.dataset_configs[dataset_name]
        
        # Create student model with dataset-specific parameters
        student_model = build_adaptive_student(
            dataset_name=dataset_name,
            num_classes=CFG.num_classes,
            image_size=config['image_size']
        ).to(device)
        
        print(f"\n{dataset_name} Student Model:")
        print(f"  - Expected input channels: {config['in_channels']}")
        
        # Test with sample input
        sample_input = torch.randn(1, config['in_channels'], config['image_size'], config['image_size'])
        
        try:
            with torch.no_grad():
                output = student_model(sample_input)
                if isinstance(output, (tuple, list)):
                    logits, _ = output
                else:
                    logits = output
                print(f"  - Model output shape: {logits.shape}")
                print(f"  - ✓ Model accepts {config['in_channels']}-channel input successfully")
        except Exception as e:
            print(f"  - ✗ Error with {config['in_channels']}-channel input: {e}")
    
    print("\n" + "=" * 60)
    
    # Test 3: Verify ClusterServer initialization with different datasets
    print("3. Testing ClusterServer with dataset-specific parameters:")
    
    for dataset_name in ["FashionMNIST", "CIFAR10"]:
        config = CFG.dataset_configs[dataset_name]
        
        try:
            server = ClusterServer(
                device=device,
                num_clusters=2,  # Use fewer clusters for testing
                num_classes=CFG.num_classes,
                in_channels=config['in_channels'],
                image_size=config['image_size'],
                dataset_name=dataset_name
            )
            
            print(f"\n{dataset_name} ClusterServer:")
            print(f"  - ✓ Successfully initialized with {config['in_channels']} input channels")
            
            # Test sample input
            sample_input = torch.randn(2, config['in_channels'], config['image_size'], config['image_size'])
            sample_labels = torch.randint(0, CFG.num_classes, (2,))
            
            with torch.no_grad():
                student_output = server.student(sample_input)
                if isinstance(student_output, (tuple, list)):
                    student_logits, _ = student_output
                else:
                    student_logits = student_output
                
                print(f"  - Student model output shape: {student_logits.shape}")
                print(f"  - ✓ Student model processes {config['in_channels']}-channel input correctly")
                
        except Exception as e:
            print(f"\n{dataset_name} ClusterServer:")
            print(f"  - ✗ Error initializing server: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset configuration testing completed!")

if __name__ == "__main__":
    test_dataset_configs()