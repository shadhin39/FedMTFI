"""
Project configuration for FedMTFI (Feature-Importance Optimized Multi-Teacher KD).

This file centralizes hyperparameters and settings used across the project.
"""

from dataclasses import dataclass


class CFG:
    # Dataset configuration
    dataset = "MNIST"  # Private dataset for client training
    eval_datasets = ["FashionMNIST", "CIFAR10"] 
    data_dir = "./data"
    
    # Federated learning configuration (interactive input)
    @staticmethod
    def get_federated_config():
        """Get federated learning configuration from user input."""
        print("\n=== Federated Learning Configuration ===")
        
        try:
            num_clients = int(input("Enter number of clients (default: 100): ") or "100")
        except ValueError:
            print("Invalid input. Using default: 100")
            num_clients = 100
            
        try:
            num_clusters = int(input("Enter number of clusters (default: 4): ") or "4")
        except ValueError:
            print("Invalid input. Using default: 4")
            num_clusters = 4
            
        try:
            clients_per_round = int(input("Enter clients per round (default: 20): ") or "20")
        except ValueError:
            print("Invalid input. Using default: 20")
            clients_per_round = 20
            
        try:
            rounds = int(input("Enter number of rounds (default: 30): ") or "30")
        except ValueError:
            print("Invalid input. Using default: 30")
            rounds = 30
            
        try:
            local_epochs = int(input("Enter local epochs (default: 5): ") or "5")
        except ValueError:
            print("Invalid input. Using default: 5")
            local_epochs = 5
        
        print("\n=== Multi-Teacher Knowledge Distillation Configuration ===")
        
        try:
            fmnist_distill_epochs = int(input("Enter FashionMNIST multi-teacher distillation epochs (default: 5): ") or "5")
        except ValueError:
            print("Invalid input. Using default: 5")
            fmnist_distill_epochs = 5
            
        try:
            cifar10_distill_epochs = int(input("Enter CIFAR-10 multi-teacher distillation epochs (default: 10): ") or "10")
        except ValueError:
            print("Invalid input. Using default: 10")
            cifar10_distill_epochs = 10
        
        print("\n=== Final Student Model Training Configuration ===")
        
        try:
            fmnist_student_epochs = int(input("Enter FashionMNIST final student training epochs (default: 15): ") or "15")
        except ValueError:
            print("Invalid input. Using default: 15")
            fmnist_student_epochs = 15
            
        try:
            cifar10_student_epochs = int(input("Enter CIFAR-10 final student training epochs (default: 20): ") or "20")
        except ValueError:
            print("Invalid input. Using default: 20")
            cifar10_student_epochs = 20
            
        print(f"\nConfiguration set:")
        print(f"  - Number of clients: {num_clients}")
        print(f"  - Number of clusters: {num_clusters}")
        print(f"  - Clients per round: {clients_per_round}")
        print(f"  - Rounds: {rounds}")
        print(f"  - Local epochs: {local_epochs}")
        print(f"  - FashionMNIST distillation epochs: {fmnist_distill_epochs}")
        print(f"  - CIFAR-10 distillation epochs: {cifar10_distill_epochs}")
        print(f"  - FashionMNIST student epochs: {fmnist_student_epochs}")
        print(f"  - CIFAR-10 student epochs: {cifar10_student_epochs}")
        
        return num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs
    
    @staticmethod
    def set_federated_config(num_clients=None, num_clusters=None, clients_per_round=None, rounds=None, local_epochs=None, 
                           fmnist_distill_epochs=None, cifar10_distill_epochs=None, fmnist_student_epochs=None, cifar10_student_epochs=None):
        """Set federated learning configuration programmatically."""
        if num_clients is not None:
            CFG.num_clients = num_clients
        if num_clusters is not None:
            CFG.num_clusters = num_clusters
        if clients_per_round is not None:
            CFG.clients_per_round = clients_per_round
        if rounds is not None:
            CFG.rounds = rounds
        if local_epochs is not None:
            CFG.local_epochs = local_epochs
        
        # Update dataset-specific distillation epochs
        if fmnist_distill_epochs is not None:
            CFG.dataset_configs["FashionMNIST"]["distill_epochs"] = fmnist_distill_epochs
        if cifar10_distill_epochs is not None:
            CFG.dataset_configs["CIFAR10"]["distill_epochs"] = cifar10_distill_epochs
        
        # Update student training epochs
        if fmnist_student_epochs is not None:
            CFG.fmnist_student_epochs = fmnist_student_epochs
        if cifar10_student_epochs is not None:
            CFG.cifar10_student_epochs = cifar10_student_epochs
            
        print(f"Configuration updated:")
        print(f"  - Number of clients: {CFG.num_clients}")
        print(f"  - Number of clusters: {CFG.num_clusters}")
        print(f"  - Clients per round: {CFG.clients_per_round}")
        print(f"  - Rounds: {CFG.rounds}")
        print(f"  - Local epochs: {CFG.local_epochs}")
        print(f"  - FashionMNIST distillation epochs: {CFG.dataset_configs['FashionMNIST']['distill_epochs']}")
        print(f"  - CIFAR-10 distillation epochs: {CFG.dataset_configs['CIFAR10']['distill_epochs']}")
        print(f"  - FashionMNIST student epochs: {getattr(CFG, 'fmnist_student_epochs', 15)}")
        print(f"  - CIFAR-10 student epochs: {getattr(CFG, 'cifar10_student_epochs', 20)}")
    
    # Default values (used when imported as module)
    num_clients = 100
    num_clusters = 4  # Number of clusters for cluster-based FL
    clients_per_round = int(num_clients*0.1)  # Clients need to participate randomly
    rounds = 30
    local_epochs = 5
    
    # Model configuration
    num_classes = 10
    in_channels = 1
    image_size = 28
    
    # Dataset-specific parameter configurations
    dataset_configs = {
        "FashionMNIST": {
            "in_channels": 1,
            "image_size": 28,
            "batch_size": 64,
            "lr_server": 0.001,
            "distill_epochs": 5,
            "temperature": 3.0,
            "lambda_kd": 0.6,
            "lambda_feat": 0.3,
            "lambda_ce": 0.1,
            "normalization": {"mean": (0.5,), "std": (0.5,)}
        },
        "CIFAR10": {
            "in_channels": 3,
            "image_size": 32,
            "batch_size": 128,
            "lr_server": 0.0005,
            "distill_epochs": 10,
            "temperature": 5.0,
            "lambda_kd": 0.5,
            "lambda_feat": 0.4,
            "lambda_ce": 0.1,
            "normalization": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        }
    }
    
    # Training configuration
    batch_size = 64
    lr_client = 0.01
    lr_server = 0.001
    weight_decay = 1e-4
    
    # Knowledge distillation configuration
    distill_epochs = 3  # Epochs for server-side distillation
    temperature = 4.0
    alpha = 0.7  # Weight for KD loss vs CE loss
    
    # Multi-teacher distillation loss weights
    lambda_kd = 0.5    # Knowledge distillation loss weight
    lambda_feat = 0.3  # Feature alignment loss weight
    lambda_ce = 0.2    # Cross-entropy loss weight
    
    # Non-IID configuration
    alpha_dirichlet = 0.5  # Controls non-IID distribution
    
    # Device configuration
    device = "cuda" if __name__ == "__main__" else "cpu"
    
    # Cluster-specific# Cluster and student training epochs
    cluster_distill_epochs = 3  # Epochs for cluster model training
    student_distill_epochs = 20  # Epochs for student training with cluster teachers
    
    # Dataset-specific student training epochs (for final student model)
    fmnist_student_epochs = 15  # Final student training epochs for FashionMNIST
    cifar10_student_epochs = 20  # Final student training epochs for CIFAR-10
    
    # Evaluation configuration
    eval_batch_size = 256
    
    # Feature importance configuration
    use_captum = False  # Whether to use Captum for feature importance