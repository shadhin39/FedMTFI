"""
FedAvg CIFAR-10 Experiment with 5 Rounds
This script runs the regular FedAvg algorithm on CIFAR-10 dataset for 5 rounds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from datetime import datetime
import os
import sys

# Import FedAvg components
from fedavg_server import FedAvgServer
from fedavg_client import FedAvgClient
from fedavg_config import FedAvgConfig

def create_cifar10_model():
    """Create an improved ResNet model for CIFAR-10."""
    
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out
    
    class ImprovedResNet(nn.Module):
        def __init__(self, num_classes=10):
            super(ImprovedResNet, self).__init__()
            self.in_channels = 32
            
            # Improved initial convolution
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            
            # More layers for better feature extraction
            self.layer1 = self._make_layer(BasicBlock, 32, 3, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 64, 3, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 128, 3, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 256, 3, stride=2)
            
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(256, num_classes)
        
        def _make_layer(self, block, out_channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)
            return out
    
    return ImprovedResNet()

def create_non_iid_split(dataset, num_clients, alpha=0.5):
    """Create non-IID data split using Dirichlet distribution."""
    num_classes = 10
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
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
            client_indices[client_id].extend(indices)
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def run_fedavg_cifar10():
    """Run FedAvg experiment on CIFAR-10."""
    # Configuration
    config = FedAvgConfig()
    config.local_epochs = 3  # Reduced for better convergence
    config.learning_rate = 0.01  # Improved learning rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Configuration:")
    print(f"  - Clients: {config.num_clients}")
    print(f"  - Clients per round: {config.clients_per_round}")
    print(f"  - Rounds: {config.num_rounds}")
    print(f"  - Local epochs: {config.local_epochs}")
    print(f"  - Alpha (non-IID): 0.5")
    print(f"  - Device: {device}")
    print()
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create non-IID split
    print("Creating non-IID split with alpha=0.5...")
    client_indices = create_non_iid_split(train_dataset, config.num_clients, alpha=0.5)
    
    # Create model
    model = create_cifar10_model()
    
    # Initialize server
    server = FedAvgServer(model, device)
    
    # Create clients
    clients = []
    for client_id in range(config.num_clients):
        client_dataset = Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        
        client = FedAvgClient(
            client_id=client_id,
            model=create_cifar10_model(),
            train_loader=client_loader,
            device=device
        )
        clients.append(client)
    
    print(f"\nCreated {len(clients)} clients")
    
    # Sample distribution check
    sample_counts = [len(indices) for indices in client_indices[:5]]
    print(f"Sample distribution: {sample_counts}")
    print()
    
    # Initial evaluation
    print("Starting FedAvg training for 5 rounds...")
    print("=" * 80)
    print()
    
    initial_metrics = server.evaluate_global_model(test_loader, "CIFAR-10")
    print(f"Initial - Loss: {initial_metrics['loss']:.4f}, Accuracy: {initial_metrics['accuracy']:.4f}")
    
    # Training rounds
    for round_num in range(1, config.num_rounds + 1):
        print(f"\n--- Round {round_num}/{config.num_rounds} ---")
        
        # Select clients for this round
        selected_clients = np.random.choice(clients, config.clients_per_round, replace=False)
        
        # Send global model to selected clients
        global_params = server.get_global_model_parameters()
        client_updates = []
        
        for client in selected_clients:
            client.set_model_parameters(global_params)
            client_metrics = client.train_local(
                epochs=config.local_epochs,
                lr=config.learning_rate,
                round_num=round_num
            )
            client_params = client.get_model_parameters()
            num_samples = len(client.train_loader.dataset)
            client_updates.append((client_params, num_samples))
        
        # Extract client models and sample counts from updates
        client_models = [update[0] for update in client_updates]
        client_sample_counts = [update[1] for update in client_updates]
        
        # Aggregate client updates
        server.aggregate_models(client_models, client_sample_counts)
        
        # Evaluate global model
        global_metrics = server.evaluate_global_model(test_loader, "CIFAR-10")
        
        # Collect client metrics for logging
        client_metrics = []
        for client in selected_clients:
            client_metrics.append({
                'loss': 0.0,  # Placeholder - would need to track during training
                'accuracy': 0.0,  # Placeholder - would need to track during training
                'num_samples': len(client.train_loader.dataset)
            })
        
        server.log_round_metrics(round_num, client_metrics, global_metrics)
        
        print(f"Round {round_num} - Global Accuracy: {global_metrics['accuracy']:.4f}, Loss: {global_metrics['loss']:.4f}")
    
    # Final evaluation
    final_metrics = server.evaluate_global_model(test_loader, "CIFAR-10")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Initial Accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Accuracy Improvement: {final_metrics['accuracy'] - initial_metrics['accuracy']:.4f}")
    print(f"Initial Loss: {initial_metrics['loss']:.4f}")
    print(f"Final Loss: {final_metrics['loss']:.4f}")
    print(f"Loss Reduction: {initial_metrics['loss'] - final_metrics['loss']:.4f}")
    
    # Save results
    results_dir = "./fedavg_experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(results_dir, "fedavg_cifar10_model.pth")
    server.save_global_model(model_path)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "fedavg_cifar10_metrics.csv")
    server.save_metrics(metrics_path)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    return final_metrics

def main():
    """Main function to run the experiment."""
    print(f"Starting FedAvg CIFAR-10 experiment at {datetime.now()}")
    print("=" * 80)
    print("FEDAVG CIFAR-10 EXPERIMENT - 5 ROUNDS")
    print("=" * 80)
    
    try:
        final_metrics = run_fedavg_cifar10()
        print(f"\nExperiment completed successfully!")
        print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final loss: {final_metrics['loss']:.4f}")
        
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()