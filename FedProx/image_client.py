# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23
@Author: Assistant
@File: image_client.py
@Description: FedProx client implementation for image classification datasets
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from image_data import get_client_data_loaders, get_validation_split


def get_val_loss_and_accuracy(args, model, val_loader):
    """Calculate validation loss and accuracy."""
    model.eval()
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = loss_function(output, target)
            val_loss.append(loss.item())
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
    
    val_accuracy = accuracy_score(all_labels, all_preds)
    return np.mean(val_loss), val_accuracy


def train_client(args, model, server_model, client_id, client_indices):
    """
    Train a client model using FedProx algorithm.
    
    Args:
        args: Training arguments
        model: Client model
        server_model: Global server model (for proximal term)
        client_id: ID of the client
        client_indices: Data indices for all clients
    
    Returns:
        Trained client model
    """
    model.train()
    
    # Get client's data loaders
    train_loader, _ = get_client_data_loaders(
        args.dataset, client_id, client_indices, args.B, args.data_dir
    )
    
    # Split training data into train/validation
    train_loader, val_loader = get_validation_split(train_loader, val_ratio=0.1)
    
    # Store dataset size for aggregation
    model.len = len(train_loader.dataset)
    
    # Create a copy of the global model for proximal term
    global_model = copy.deepcopy(server_model)
    for param in global_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Training loop
    loss_function = nn.CrossEntropyLoss().to(args.device)
    best_model = None
    best_val_accuracy = 0.0
    min_epochs = 5
    
    print(f'Training client {client_id} with {model.len} samples...')
    
    for epoch in tqdm(range(args.E), desc=f"Client {client_id}"):
        model.train()
        train_loss = []
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Standard cross-entropy loss
            ce_loss = loss_function(output, target)
            
            # Compute proximal term (FedProx regularization)
            proximal_term = 0.0
            for w, w_global in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_global).norm(2) ** 2
            
            # Total loss with proximal regularization
            total_loss = ce_loss + (args.mu / 2) * proximal_term
            
            total_loss.backward()
            optimizer.step()
            
            train_loss.append(total_loss.item())
            
            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        scheduler.step()
        
        # Validation
        val_loss, val_accuracy = get_val_loss_and_accuracy(args, model, val_loader)
        train_accuracy = correct / total
        
        # Save best model based on validation accuracy
        if epoch >= min_epochs and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        
        if args.verbose:
            print(f'Client {client_id} Epoch {epoch:03d}: '
                  f'Train Loss: {np.mean(train_loss):.6f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'Val Acc: {val_accuracy:.4f}')
    
    # Return best model if found, otherwise return the last model
    if best_model is not None:
        return best_model
    else:
        return model


def test_client(args, model, client_id=None, test_loader=None):
    """
    Test a client model.
    
    Args:
        args: Testing arguments
        model: Model to test
        client_id: ID of the client (if testing specific client data)
        test_loader: Test data loader (if provided, use this instead of creating new one)
    
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    
    # Get test data loader if not provided
    if test_loader is None:
        from image_data import load_dataset
        _, test_dataset = load_dataset(args.dataset, args.data_dir)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.B, 
            shuffle=False,
            num_workers=2
        )
    
    loss_function = nn.CrossEntropyLoss().to(args.device)
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += loss_function(output, target).item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Store for detailed metrics
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    # Print results
    client_str = f"Client {client_id}" if client_id is not None else "Global Model"
    print(f'{client_str} Test Results:')
    print(f'  Average Loss: {test_loss:.6f}')
    print(f'  Accuracy: {accuracy:.4f} ({correct}/{total})')
    
    # Detailed classification report (optional, can be verbose)
    if args.verbose and hasattr(args, 'show_classification_report') and args.show_classification_report:
        from image_data import DATASET_CLASSES
        class_names = DATASET_CLASSES.get(args.dataset, None)
        print(f'\nClassification Report for {client_str}:')
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels
    }