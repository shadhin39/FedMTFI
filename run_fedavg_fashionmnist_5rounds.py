#!/usr/bin/env python3
"""
Run FedAvg experiment on FashionMNIST dataset with 5 rounds.
Uses the refined FedAvg implementation with updated configuration.
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
import pandas as pd
import os
from datetime import datetime

def create_fashionmnist_model():
    """Create an improved CNN model for FashionMNIST."""
    class ImprovedFashionMNISTCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(ImprovedFashionMNISTCNN, self).__init__()
            # Improved architecture with batch normalization
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            
            # Calculate the correct input size for fc1
            # FashionMNIST: 28x28 -> 14x14 -> 7x7 -> 3x3 -> 1x1 (after 4 pooling operations)
            self.fc1 = nn.Linear(512 * 1 * 1, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
            x = self.dropout1(x)
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
            x = self.dropout1(x)
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3
            x = self.dropout1(x)
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))  # 3x3 -> 1x1
            
            x = x.view(x.size(0), -1)  # Flatten: batch_size x (512*1*1)
            x = self.dropout2(torch.relu(self.fc1(x)))
            x = self.dropout1(torch.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    return ImprovedFashionMNISTCNN()

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

def run_fedavg_fashionmnist():
    """Run FedAvg experiment on FashionMNIST with 5 rounds."""
    print("=" * 80)
    print("FEDAVG FASHIONMNIST EXPERIMENT - 5 ROUNDS")
    print("=" * 80)
    
    # Load configuration
    config = FedAvgConfig()
    config.local_epochs = 3  # Reduced for better convergence
    config.learning_rate = 0.01  # Improved learning rate
    print(f"Configuration:")
    print(f"  - Clients: {config.num_clients}")
    print(f"  - Clients per round: {config.clients_per_round}")
    print(f"  - Rounds: {config.num_rounds}")
    print(f"  - Local epochs: {config.local_epochs}")
    print(f"  - Alpha (non-IID): {config.alpha}")
    print(f"  - Device: {config.device}")
    
    device = torch.device(config.device)
    
    # Load data
    print("\nLoading FashionMNIST dataset...")
    train_dataset, test_dataset = load_fashionmnist_data()
    
    # Create non-IID split
    print(f"Creating non-IID split with alpha={config.alpha}...")
    client_indices = create_non_iid_split(train_dataset, config.num_clients, config.alpha)
    
    # Create server
    model = create_fashionmnist_model()
    server = FedAvgServer(model, device)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    
    # Create clients
    clients = []
    for client_id in range(config.num_clients):
        # Create client dataset
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(client_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Create client
        client_model = create_fashionmnist_model()
        client = FedAvgClient(client_id, client_model, client_loader, device)
        clients.append(client)
    
    print(f"\nCreated {len(clients)} clients")
    print(f"Sample distribution: {[len(client_indices[i]) for i in range(min(5, len(clients)))]}")
    
    # Results storage
    results = []
    
    print(f"\nStarting FedAvg training for {config.num_rounds} rounds...")
    print("=" * 80)
    
    # Initial evaluation
    initial_metrics = server.evaluate_global_model(test_loader, "FashionMNIST")
    print(f"Initial Global Model - Accuracy: {initial_metrics['accuracy']:.4f}, Loss: {initial_metrics['loss']:.4f}")
    
    # Training loop
    for round_num in range(config.num_rounds):
        print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")
        
        # Select clients for this round
        selected_clients = np.random.choice(
            range(config.num_clients), 
            config.clients_per_round, 
            replace=False
        )
        
        print(f"Selected clients: {selected_clients[:10].tolist()}{'...' if len(selected_clients) > 10 else ''}")
        
        # Distribute global model to selected clients
        global_params = server.get_global_model_parameters()
        
        client_models = []
        client_weights = []
        client_metrics = []
        
        # Local training
        for client_id in selected_clients:
            client = clients[client_id]
            
            # Set global model parameters
            client.set_model_parameters(global_params)
            
            # Local training
            metrics = client.train_local(epochs=config.local_epochs, round_num=round_num + 1)
            
            # Get updated model parameters
            client_models.append(client.get_model_parameters())
            client_weights.append(client.get_num_samples())
            client_metrics.append(metrics)
        
        # Server aggregation
        print(f"Aggregating {len(client_models)} client models...")
        server.update_global_model(client_models, client_weights)
        
        # Evaluate global model
        global_metrics = server.evaluate_global_model(test_loader, "FashionMNIST")
        
        # Log round metrics
        server.log_round_metrics(round_num + 1, client_metrics, global_metrics)
        
        # Store results
        results.append({
            'round': round_num + 1,
            'global_accuracy': global_metrics['accuracy'],
            'global_loss': global_metrics['loss'],
            'avg_client_accuracy': np.mean([m['accuracy'] for m in client_metrics]),
            'avg_client_loss': np.mean([m['loss'] for m in client_metrics]),
            'num_clients': len(client_metrics),
            'total_samples': sum(client_weights)
        })
        
        print(f"Round {round_num + 1} - Global Accuracy: {global_metrics['accuracy']:.4f}, Loss: {global_metrics['loss']:.4f}")
    
    print("\n" + "=" * 80)
    print("FEDAVG FASHIONMNIST EXPERIMENT COMPLETED!")
    print("=" * 80)
    
    # Final results
    final_metrics = server.evaluate_global_model(test_loader, "FashionMNIST")
    print(f"Final Results:")
    print(f"  - Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - Loss: {final_metrics['loss']:.4f}")
    print(f"  - Improvement: {final_metrics['accuracy'] - initial_metrics['accuracy']:.4f}")
    
    # Save results
    results_dir = "fedavg_experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    df = pd.DataFrame(results)
    metrics_file = os.path.join(results_dir, "fedavg_fashionmnist_5rounds_metrics.csv")
    df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save model
    model_file = os.path.join(results_dir, "fedavg_fashionmnist_5rounds_model.pth")
    server.save_global_model(model_file)
    print(f"Model saved to: {model_file}")
    
    return final_metrics

def main():
    """Main function to run the experiment."""
    try:
        print(f"Starting FedAvg FashionMNIST experiment at {datetime.now()}")
        final_metrics = run_fedavg_fashionmnist()
        print(f"\nExperiment completed successfully!")
        print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)