#!/usr/bin/env python3
"""
Test refined FedAvg implementation on FashionMNIST dataset.
Tests the improved algorithm with alpha=0.5 and 10 rounds.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from fedavg_server import FedAvgServer
from fedavg_client import FedAvgClient
from fedavg_config import FedAvgConfig
import numpy as np
from collections import defaultdict
import time

def create_fashionmnist_model():
    """Create a CNN model for FashionMNIST."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

def load_fashionmnist_data():
    """Load FashionMNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset

def create_non_iid_split(dataset, num_clients, alpha=0.5):
    """Create non-IID data split using Dirichlet distribution."""
    num_classes = 10
    num_samples = len(dataset)
    
    # Get labels
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    
    # Create Dirichlet distribution for each client
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        # Split class indices among clients
        class_splits = np.split(class_indices, proportions)
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices.tolist())
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def test_refined_fedavg_fashionmnist():
    """Test refined FedAvg on FashionMNIST with 5 clients for quick validation."""
    print("Testing Refined FedAvg on FashionMNIST...")
    print("=" * 60)
    
    # Configuration for quick test
    config = FedAvgConfig()
    config.num_clients = 5
    config.clients_per_round = 3
    config.num_rounds = 3  # Quick test with 3 rounds
    config.local_epochs = 2
    config.alpha = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading FashionMNIST dataset...")
    train_dataset, test_dataset = load_fashionmnist_data()
    
    # Create non-IID split
    print(f"Creating non-IID split with alpha={config.alpha}...")
    client_indices = create_non_iid_split(train_dataset, config.num_clients, config.alpha)
    
    # Create server
    model = create_fashionmnist_model()
    server = FedAvgServer(model, device)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Create clients
    clients = []
    for client_id in range(config.num_clients):
        # Create client dataset
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        
        # Create client
        client_model = create_fashionmnist_model()
        client = FedAvgClient(client_id, client_model, client_loader, device)
        clients.append(client)
        
        print(f"Client {client_id}: {len(client_indices[client_id])} samples")
    
    print(f"\nStarting refined FedAvg training for {config.num_rounds} rounds...")
    
    # Training loop
    for round_num in range(config.num_rounds):
        print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")
        
        # Select clients for this round
        selected_clients = np.random.choice(
            range(config.num_clients), 
            config.clients_per_round, 
            replace=False
        )
        
        print(f"Selected clients: {selected_clients.tolist()}")
        
        # Distribute global model to selected clients
        global_params = server.get_global_model_parameters()
        
        client_models = []
        client_weights = []
        
        # Local training
        for client_id in selected_clients:
            client = clients[client_id]
            
            # Set global model parameters
            client.set_model_parameters(global_params)
            
            # Local training
            client.train_local(epochs=config.local_epochs, round_num=round_num + 1)
            
            # Get updated model parameters
            client_models.append(client.get_model_parameters())
            client_weights.append(client.get_num_samples())
        
        # Server aggregation
        print(f"Aggregating {len(client_models)} client models...")
        server.update_global_model(client_models, client_weights)
        
        # Evaluate global model
        metrics = server.evaluate_global_model(test_loader, "FashionMNIST")
        print(f"Round {round_num + 1} - Global Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("Refined FedAvg FashionMNIST test completed successfully!")
    
    # Final evaluation
    final_metrics = server.evaluate_global_model(test_loader, "FashionMNIST")
    print(f"Final Results - Accuracy: {final_metrics['accuracy']:.4f}, Loss: {final_metrics['loss']:.4f}")
    
    return final_metrics['accuracy'] > 0.3  # Basic accuracy threshold for quick test

def main():
    """Run the refined FedAvg test on FashionMNIST."""
    print("REFINED FEDAVG FASHIONMNIST TEST")
    print("=" * 60)
    
    try:
        success = test_refined_fedavg_fashionmnist()
        
        if success:
            print("\n✓ Refined FedAvg FashionMNIST test PASSED!")
            return True
        else:
            print("\n✗ Refined FedAvg FashionMNIST test FAILED!")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)