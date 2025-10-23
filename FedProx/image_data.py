# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23
@Author: Assistant
@File: image_data.py
@Description: Data loading utilities for FMNIST and CIFAR-10 datasets in FedProx
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict


def get_transforms(dataset_name):
    """Get appropriate transforms for the dataset."""
    if dataset_name == 'FMNIST':
        # FashionMNIST transforms
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST statistics
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    
    elif dataset_name == 'CIFAR10':
        # CIFAR-10 transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform_train, transform_test


def load_dataset(dataset_name, data_dir='./data'):
    """Load the specified dataset."""
    transform_train, transform_test = get_transforms(dataset_name)
    
    if dataset_name == 'FMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    
    elif dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def create_non_iid_split(dataset, num_clients, alpha=0.5):
    """
    Create non-IID data split using Dirichlet distribution.
    
    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        List of client data indices
    """
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Create Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Distribute indices according to proportions
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + int(proportions[client_id] * len(indices))
            if client_id == num_clients - 1:  # Last client gets remaining
                end_idx = len(indices)
            
            client_indices[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def get_client_data_loaders(dataset_name, client_id, client_indices, batch_size, data_dir='./data'):
    """
    Get data loaders for a specific client.
    
    Args:
        dataset_name: Name of the dataset ('FMNIST' or 'CIFAR10')
        client_id: ID of the client
        client_indices: List of data indices for all clients
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
    
    Returns:
        train_loader, test_loader
    """
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)
    
    # Create client's training subset
    client_train_indices = client_indices[client_id]
    client_train_dataset = Subset(train_dataset, client_train_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        client_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Use full test set for evaluation (or could be split as well)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        drop_last=False  # Keep all test samples
    )
    
    return train_loader, test_loader


def get_validation_split(train_loader, val_ratio=0.1):
    """
    Split training data into train and validation sets.
    
    Args:
        train_loader: Original training data loader
        val_ratio: Ratio of data to use for validation
    
    Returns:
        new_train_loader, val_loader
    """
    dataset = train_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Shuffle and split
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1 - val_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create new datasets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Create new data loaders
    new_train_loader = DataLoader(
        train_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=False  # Keep all validation samples
    )
    
    return new_train_loader, val_loader


# Dataset class names for reference
DATASET_CLASSES = {
    'FMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
}