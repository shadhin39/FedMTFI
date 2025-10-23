import sys
import torch
import torch.nn as nn
import random
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CFG
from client import Client
from server import ClusterServer
from models import build_adaptive_model
from non_iid_distributor import NonIIDDataDistributor
from metrics_logger import MetricsLogger


def assign_clients_to_clusters(num_clients: int, num_clusters: int):
    """Assign clients to clusters in a round-robin fashion."""
    assignments = {}
    for client_id in range(num_clients):
        cluster_id = client_id % num_clusters
        assignments[client_id] = cluster_id
    return assignments


def select_random_clients(clients, clients_per_round: int):
    """
    Randomly select a subset of clients for the current round.
    
    Args:
        clients: List of all available clients
        clients_per_round: Number of clients to select
    
    Returns:
        List of randomly selected clients
    """
    if clients_per_round >= len(clients):
        return clients
    
    return random.sample(clients, clients_per_round)


def load_datasets():
    """Load and prepare datasets for training and evaluation."""
    # Transform for grayscale datasets (MNIST, FashionMNIST) - convert to 3 channels for compatibility
    grayscale_to_rgb_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3-channel normalization
    ])
    
    # Transform for RGB datasets (CIFAR-10)
    rgb_transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    ])
    
    datasets_dict = {}
    
    # Load FashionMNIST (converted to 3 channels for compatibility with student model)
    fashion_train = datasets.FashionMNIST(
        root=CFG.data_dir, train=True, download=True, transform=grayscale_to_rgb_transform
    )
    fashion_test = datasets.FashionMNIST(
        root=CFG.data_dir, train=False, download=True, transform=grayscale_to_rgb_transform
    )
    datasets_dict["FashionMNIST"] = {"train": fashion_train, "test": fashion_test}
    
    # Load MNIST (converted to 3 channels for compatibility with student model)
    mnist_train = datasets.MNIST(
        root=CFG.data_dir, train=True, download=True, transform=grayscale_to_rgb_transform
    )
    mnist_test = datasets.MNIST(
        root=CFG.data_dir, train=False, download=True, transform=grayscale_to_rgb_transform
    )
    datasets_dict["MNIST"] = {"train": mnist_train, "test": mnist_test}
    
    # Load CIFAR-10
    cifar10_train = datasets.CIFAR10(
        root=CFG.data_dir, train=True, download=True, transform=rgb_transform
    )
    cifar10_test = datasets.CIFAR10(
        root=CFG.data_dir, train=False, download=True, transform=rgb_transform
    )
    datasets_dict["CIFAR10"] = {"train": cifar10_train, "test": cifar10_test}
    
    return datasets_dict


def main():
    # Interactive configuration setup
    print("=" * 60)
    print("FedMTFI - Federated Multi-Task Feature Integration")
    print("=" * 60)
    print("Configure your federated learning experiment:")
    print()
    
    # Get interactive configuration
    num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs = CFG.get_federated_config()
    
    # Update configuration with user input
    CFG.set_federated_config(
        num_clients=num_clients,
        num_clusters=num_clusters,
        clients_per_round=clients_per_round,
        rounds=rounds,
        local_epochs=local_epochs,
        fmnist_distill_epochs=fmnist_distill_epochs,
        cifar10_distill_epochs=cifar10_distill_epochs,
        fmnist_student_epochs=fmnist_student_epochs,
        cifar10_student_epochs=cifar10_student_epochs
    )
    
    print("=" * 60)
    print(f"[FedMTFI] Starting cluster-based federated learning on {CFG.device}")
    print(f"[FedMTFI] Configuration: {CFG.num_clients} clients, {CFG.num_clusters} clusters, {CFG.rounds} rounds, {CFG.local_epochs} local epochs")
    
    # Start overall training timer
    overall_start_time = time.time()
    
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    print(f"[FedMTFI] Using device: {device}")
    
    # Load datasets
    print("[FedMTFI] Loading datasets...")
    datasets_dict = load_datasets()
    
    # MNIST as private data for client training
    private_dataset = datasets_dict["MNIST"]
    
    # Create public datasets for knowledge distillation (both FashionMNIST and CIFAR-10)
    public_datasets = {
        "FashionMNIST": datasets_dict["FashionMNIST"],
        "CIFAR10": datasets_dict["CIFAR10"]
    }
    
    # Create loaders for both public datasets
    public_loaders = {}
    for dataset_name, dataset in public_datasets.items():
        config = CFG.dataset_configs[dataset_name]
        public_loaders[dataset_name] = DataLoader(
            dataset["train"], 
            batch_size=config["batch_size"], 
            shuffle=True
        )
        print(f"[FedMTFI] Created public dataset ({dataset_name}) with {len(dataset['train'])} samples for knowledge distillation")
    
    # Use FashionMNIST as the primary public dataset for initial training
    public_loader = public_loaders["FashionMNIST"]
    
    # Distribute private data (MNIST) to clients (non-IID)
    print("[FedMTFI] Distributing private data (MNIST) to clients (non-IID)...")
    distributor = NonIIDDataDistributor(
        dataset=private_dataset["train"],
        num_clients=CFG.num_clients,
        num_classes=10
    )
    client_datasets, client_preferences = distributor.bias_based_distribution(primary_bias=0.8)
    
    # Assign clients to clusters
    cluster_assignments = assign_clients_to_clusters(CFG.num_clients, CFG.num_clusters)
    print(f"[FedMTFI] Client-cluster assignments: {cluster_assignments}")
    
    # Initialize cluster server with CIFAR10 for student model (3 channels) to handle both datasets
    server = ClusterServer(
        device=device,
        num_clusters=CFG.num_clusters,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels,
        image_size=CFG.image_size,
        dataset_name="CIFAR10"  # Use CIFAR10 for student model to handle 3-channel inputs
    )
    
    # Build clients with cluster-specific models
    print("[FedMTFI] Building clients...")
    clients = []
    for client_id in range(CFG.num_clients):
        cluster_id = cluster_assignments[client_id]
        
        # Create client with cluster-specific model using unified 3-channel architecture
        model = build_adaptive_model(cluster_id, "CIFAR10", CFG.num_classes, CFG.image_size)
        client_loader = DataLoader(
            client_datasets[client_id], 
            batch_size=CFG.batch_size, 
            shuffle=True
        )
        
        client = Client(
            cid=client_id,
            model=model,
            train_loader=client_loader,
            device=device,
            cluster_id=cluster_id
        )
        clients.append(client)
        print(f"[FedMTFI] Client {client_id} -> Cluster {cluster_id} (model: {type(model).__name__})")
    
    # Initialize metrics logger
    logger = MetricsLogger()
    
    print(f"[FedMTFI] Starting federated learning for {CFG.rounds} rounds...")
    
    # Federated learning rounds
    for round_num in range(1, CFG.rounds + 1):
        print(f"\n[FedMTFI] ===== Round {round_num}/{CFG.rounds} =====")
        
        # Start round timer
        round_start_time = time.time()
        
        # Select random clients for this round
        selected_clients = select_random_clients(clients, CFG.clients_per_round)
        print(f"[FedMTFI] Selected {len(selected_clients)} clients for round {round_num}: {[c.id for c in selected_clients]}")
        
        # Client local training
        print(f"[FedMTFI] Local training for {CFG.local_epochs} epochs...")
        client_signals = []
        
        for client in selected_clients:
            try:
                print(f"[FedMTFI] Training client {client.id} (cluster {client.cluster_id})...")
                
                # Local training with detailed logging
                local_stats, training_time = client.train_local(epochs=CFG.local_epochs, round_num=round_num, logger=logger)
                print(f"[FedMTFI] Client {client.id} local training completed successfully")
                
                # Generate signals for server
                print(f"[FedMTFI] Generating signals for client {client.id}...")
                signal = client.produce_signals(public_loader, current_round=round_num)
                signal["cluster_id"] = client.cluster_id
                client_signals.append(signal)
                print(f"[FedMTFI] Client {client.id} signal generation completed successfully")
                
                # Log client metrics
                logger.add_local(local_stats)
                
            except Exception as e:
                print(f"[ERROR] Client {client.id} failed with error: {str(e)}")
                print(f"[ERROR] Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # Continue with next client instead of crashing
                continue
            
            # Log global aggregation metrics for this client's cluster
            cluster_clients = [c for c in selected_clients if c.cluster_id == client.cluster_id]
            if len(cluster_clients) > 0:
                avg_loss = sum([stat["loss"] for stat in local_stats]) / len(local_stats)
                avg_acc = sum([stat["acc"] for stat in local_stats]) / len(local_stats)
                logger.log_global_aggregation(
                    round_num=round_num,
                    cluster_id=client.cluster_id,
                    num_clients=len(cluster_clients),
                    avg_loss=avg_loss,
                    avg_accuracy=avg_acc
                )
        
        # Server-side cluster model training
        print(f"[FedMTFI] Training cluster models...")
        cluster_stats = server.train_cluster_models(
            client_signals, 
            public_loader, 
            epochs=CFG.cluster_distill_epochs,
            current_round=round_num,
            logger=logger
        )
        
        # Log cluster training metrics
        logger.add_distill(cluster_stats)
        
        # Update client models with cluster models (federated averaging within clusters)
        print(f"[FedMTFI] Updating selected client models...")
        for client in selected_clients:
            cluster_id = client.cluster_id
            cluster_model = server.cluster_models[cluster_id]
            
            # Copy cluster model state to client
            client.model.load_state_dict(cluster_model.state_dict())
        
        print(f"[FedMTFI] Round {round_num} completed")
        
        # Log round timing
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        logger.log_round_time(round_num, round_duration)
        print(f"[FedMTFI] Round {round_num} took {round_duration:.2f} seconds")
    
    # Post-hoc Knowledge Distillation: Train cluster models on both public datasets, then use as teachers
    print(f"\n[FedMTFI] ===== Post-hoc Knowledge Distillation =====")
    
    # Train cluster models and student on both public datasets
    for dataset_name in ["FashionMNIST", "CIFAR10"]:
        print(f"\n[FedMTFI] Step 1: Training cluster models on {dataset_name}...")
        
        # Get the appropriate loader and test loader for this dataset
        train_loader = public_loaders[dataset_name]
        test_loader = DataLoader(
            public_datasets[dataset_name]["test"],
            batch_size=CFG.eval_batch_size,
            shuffle=False
        )
        
        # Train each cluster model on the current public dataset
        cluster_training_stats = server.train_cluster_models_on_public_dataset(
            train_loader,
            dataset_name=dataset_name,
            epochs=None,  # Use dataset-specific epochs from config
            logger=logger
        )
        
        # Log cluster training metrics
        for stat in cluster_training_stats:
            logger.log_server_metrics(stat)
        
        # Evaluate trained cluster models on the current dataset
        print(f"[FedMTFI] Evaluating cluster models after {dataset_name} training...")
        for cluster_id in range(CFG.num_clusters):
            cluster_acc = server.evaluate_cluster_model(cluster_id, test_loader)
            print(f"[FedMTFI] Cluster {cluster_id} accuracy on {dataset_name} after training: {cluster_acc*100:.2f}%")
        
        print(f"[FedMTFI] Step 2: Multi-teacher knowledge distillation using trained cluster models on {dataset_name}...")
        
        # Now use the trained cluster models as teachers for multi-teacher knowledge distillation
        student_stats = server.train_student_with_teachers(
            train_loader,
            dataset_name=dataset_name,
            epochs=None,  # Use dataset-specific epochs from config
            current_round=-1,  # Special round number for post-federated training
            logger=logger,
            client_signals=None  # No client signals needed for post-hoc distillation
        )
        
        # Log student training metrics
        for stat in student_stats:
            logger.log_server_metrics(stat)
        
        # Evaluate student model after post-hoc knowledge distillation on current dataset
        print(f"[FedMTFI] Evaluating student model after post-hoc distillation on {dataset_name}...")
        student_acc = server.evaluate_student(test_loader)
        print(f"[FedMTFI] Student accuracy after post-hoc distillation on {dataset_name}: {student_acc*100:.2f}%")
        
        # Log evaluation metrics
        eval_record = {
            "round": -1,  # Special round number for post-federated training
            "dataset": dataset_name.lower(),
            "model": "student_posthoc",
            "accuracy": student_acc
        }
        logger.log_evaluation_metrics(eval_record)
    
    # Also evaluate on original test datasets (but don't print duplicate for FashionMNIST)
    for dataset_name in CFG.eval_datasets:
        if dataset_name == "FashionMNIST":
            continue  # Skip duplicate evaluation for FashionMNIST
            
        test_loader = DataLoader(
            datasets_dict[dataset_name]["test"],
            batch_size=CFG.eval_batch_size,
            shuffle=False
        )
        
        student_acc_orig = server.evaluate_student(test_loader)
        print(f"[FedMTFI] Student accuracy after post-hoc distillation on {dataset_name}: {student_acc_orig*100:.2f}%")
        
        # Log evaluation metrics
        eval_record = {
            "round": -1,  # Special round number for post-federated training
            "dataset": dataset_name,
            "model": "student_posthoc",
            "accuracy": student_acc_orig
        }
        logger.log_evaluation_metrics(eval_record)
    
    # Global student training on public dataset (removed - not needed for post-hoc approach)
    # The post-hoc knowledge distillation above is the final refinement step
    
    # Save metrics
    logger.save_excel("federated_learning_results.xlsx")
    print(f"[FedMTFI] Results saved to federated_learning_results.xlsx")
    
    # Create plots from the saved metrics
    print("[FedMTFI] Creating comprehensive plots...")
    from plotting_utils import create_plots_from_excel
    try:
        create_plots_from_excel("federated_learning_results.xlsx", "plots")
        print("[FedMTFI] All plots created successfully in 'plots' folder!")
    except Exception as e:
        print(f"[FedMTFI] Error creating plots: {e}")
    
    # Log overall training time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    logger.log_overall_training_time(overall_duration)
    print(f"[FedMTFI] Total training time: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes)")
    
    print(f"[FedMTFI] Cluster-based federated learning with post-hoc knowledge distillation completed!")


if __name__ == "__main__":
    main()