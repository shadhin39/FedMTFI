# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23
@Author: Assistant
@File: image_args.py
@Description: Argument parser for FedProx image classification experiments
"""
import argparse
import torch


def args_parser():
    """Parse command line arguments for FedProx image experiments."""
    parser = argparse.ArgumentParser(description='FedProx for Image Classification')

    # Federated Learning Parameters
    parser.add_argument('--E', type=int, default=5, 
                       help='number of local epochs')
    parser.add_argument('--r', type=int, default=20, 
                       help='number of communication rounds')
    parser.add_argument('--K', type=int, default=100, 
                       help='number of total clients')
    parser.add_argument('--C', type=float, default=0.1, 
                       help='sampling rate (fraction of clients per round)')
    parser.add_argument('--B', type=int, default=32, 
                       help='local batch size')
    
    # FedProx Specific Parameters
    parser.add_argument('--mu', type=float, default=0.01, 
                       help='proximal term constant (FedProx regularization)')
    
    # Optimization Parameters
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                       choices=['sgd', 'adam'], help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, 
                       help='step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, 
                       help='learning rate decay factor')
    
    # Dataset Parameters
    parser.add_argument('--dataset', type=str, default='FMNIST', 
                       choices=['FMNIST', 'CIFAR10'], 
                       help='dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='directory to store datasets')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Dirichlet concentration parameter for non-IID split')
    
    # Model Parameters
    parser.add_argument('--model_type', type=str, default='simple', 
                       choices=['simple', 'resnet'], 
                       help='type of model architecture')
    
    # Device and System Parameters
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to use for training')
    parser.add_argument('--seed', type=int, default=42, 
                       help='random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='number of workers for data loading')
    
    # Evaluation Parameters
    parser.add_argument('--eval_every', type=int, default=5, 
                       help='evaluate global model every N rounds')
    parser.add_argument('--eval_clients', action='store_true', 
                       help='evaluate individual client models')
    parser.add_argument('--verbose', action='store_true', 
                       help='verbose output during training')
    parser.add_argument('--show_classification_report', action='store_true', 
                       help='show detailed classification report')
    
    # Saving Parameters
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='save training results and model')
    parser.add_argument('--results_dir', type=str, default='./fedprox_results', 
                       help='directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device(args.device)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    return args


def get_default_args(dataset='FMNIST'):
    """Get default arguments for quick testing."""
    import sys
    
    # Temporarily modify sys.argv to avoid conflicts
    original_argv = sys.argv
    sys.argv = ['image_args.py']  # Minimal argv
    
    try:
        args = args_parser()
        
        # Set dataset-specific defaults
        args.dataset = dataset
        
        if dataset == 'FMNIST':
            args.lr = 0.01
            args.E = 5
            args.r = 20
            args.mu = 0.01
        elif dataset == 'CIFAR10':
            args.lr = 0.01
            args.E = 5
            args.r = 20
            args.mu = 0.01
        
        # Quick test settings
        args.K = 10  # Fewer clients for testing
        args.C = 0.5  # Higher sampling rate
        args.eval_every = 2
        args.verbose = True
        
        return args
        
    finally:
        # Restore original argv
        sys.argv = original_argv


def print_args(args):
    """Print all arguments in a formatted way."""
    print("="*60)
    print("FEDPROX IMAGE CLASSIFICATION CONFIGURATION")
    print("="*60)
    
    print(f"Dataset: {args.dataset}")
    print(f"Model Type: {args.model_type}")
    print(f"Device: {args.device}")
    print()
    
    print("Federated Learning Settings:")
    print(f"  Total Clients (K): {args.K}")
    print(f"  Clients per Round: {int(args.C * args.K)} ({args.C:.1%})")
    print(f"  Communication Rounds: {args.r}")
    print(f"  Local Epochs: {args.E}")
    print(f"  Batch Size: {args.B}")
    print()
    
    print("FedProx Settings:")
    print(f"  Proximal Term (μ): {args.mu}")
    print(f"  Non-IID Alpha: {args.alpha}")
    print()
    
    print("Optimization Settings:")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  LR Scheduler Step: {args.step_size}")
    print(f"  LR Decay Factor: {args.gamma}")
    print()
    
    print("Evaluation Settings:")
    print(f"  Evaluate Every: {args.eval_every} rounds")
    print(f"  Evaluate Clients: {args.eval_clients}")
    print(f"  Verbose: {args.verbose}")
    print()
    
    print("="*60)