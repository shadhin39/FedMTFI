"""
Test script to verify that the student model can handle both FashionMNIST and CIFAR-10 
with unified 3-channel input processing.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import CFG
from server import ClusterServer
from models import build_adaptive_student

def test_unified_channel_handling():
    """Test that the student model can handle both datasets with 3-channel inputs."""
    print("Testing unified 3-channel handling for both FashionMNIST and CIFAR-10...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Verify dataset configurations
    print("\n1. Testing dataset configurations...")
    fmnist_config = CFG.dataset_configs["FashionMNIST"]
    cifar10_config = CFG.dataset_configs["CIFAR10"]
    
    print(f"FashionMNIST config: {fmnist_config}")
    print(f"CIFAR-10 config: {cifar10_config}")
    
    # Test 2: Create transforms that convert FashionMNIST to 3 channels
    print("\n2. Testing transforms...")
    
    # Transform for FashionMNIST (grayscale to 3-channel)
    fmnist_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Transform for CIFAR-10 (already 3-channel)
    cifar10_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load small samples for testing
    fmnist_dataset = datasets.FashionMNIST(
        root=CFG.data_dir, train=False, download=True, transform=fmnist_transform
    )
    cifar10_dataset = datasets.CIFAR10(
        root=CFG.data_dir, train=False, download=True, transform=cifar10_transform
    )
    
    fmnist_loader = DataLoader(fmnist_dataset, batch_size=2, shuffle=False)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=2, shuffle=False)
    
    # Get sample batches
    fmnist_batch = next(iter(fmnist_loader))
    cifar10_batch = next(iter(cifar10_loader))
    
    print(f"FashionMNIST batch shape: {fmnist_batch[0].shape}")
    print(f"CIFAR-10 batch shape: {cifar10_batch[0].shape}")
    
    # Test 3: Create student model with 3 channels for unified handling
    print("\n3. Testing student model creation...")
    
    # Use build_student directly with 3 channels for unified handling
    from models import build_student
    student_model = build_student(
        num_classes=CFG.num_classes,
        in_channels=3,  # Use 3 channels for unified handling
        image_size=CFG.image_size
    ).to(device)
    
    print(f"Student model created with 3 channels for unified handling")
    print(f"First layer: {list(student_model.children())[0]}")
    
    # Test 4: Test student model with both datasets
    print("\n4. Testing student model inference...")
    
    student_model.eval()
    with torch.no_grad():
        # Test with FashionMNIST (converted to 3 channels)
        fmnist_input = fmnist_batch[0].to(device)
        fmnist_logits, fmnist_features = student_model(fmnist_input)
        print(f"FashionMNIST input shape: {fmnist_input.shape}")
        print(f"FashionMNIST logits shape: {fmnist_logits.shape}")
        print(f"FashionMNIST features shape: {fmnist_features.shape}")
        
        # Test with CIFAR-10 (native 3 channels)
        cifar10_input = cifar10_batch[0].to(device)
        cifar10_logits, cifar10_features = student_model(cifar10_input)
        print(f"CIFAR-10 input shape: {cifar10_input.shape}")
        print(f"CIFAR-10 logits shape: {cifar10_logits.shape}")
        print(f"CIFAR-10 features shape: {cifar10_features.shape}")
    
    # Test 5: Test ClusterServer initialization with CIFAR-10
    print("\n5. Testing ClusterServer with CIFAR-10 configuration...")
    
    server = ClusterServer(
        device=device,
        num_clusters=CFG.num_clusters,
        num_classes=CFG.num_classes,
        in_channels=3,  # 3 channels for unified handling
        image_size=CFG.image_size,
        dataset_name="CIFAR10"
    )
    
    print(f"ClusterServer initialized successfully")
    print(f"Student model input channels: 3")
    
    # Test server's student model with both datasets
    server.student.eval()
    with torch.no_grad():
        server_fmnist_logits, server_fmnist_features = server.student(fmnist_input)
        server_cifar10_logits, server_cifar10_features = server.student(cifar10_input)
        
        print(f"Server student - FashionMNIST logits: {server_fmnist_logits.shape}")
        print(f"Server student - FashionMNIST features: {server_fmnist_features.shape}")
        print(f"Server student - CIFAR-10 logits: {server_cifar10_logits.shape}")
        print(f"Server student - CIFAR-10 features: {server_cifar10_features.shape}")
    
    print("\n✅ All tests passed! The student model can handle both datasets with unified 3-channel processing.")
    print("✅ FashionMNIST is converted from 1-channel to 3-channel format")
    print("✅ CIFAR-10 uses native 3-channel format")
    print("✅ Both datasets work seamlessly with the same student model architecture")

if __name__ == "__main__":
    test_unified_channel_handling()