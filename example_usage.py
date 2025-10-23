#!/usr/bin/env python3
"""
Example Usage of Interactive Configuration

This script demonstrates different ways to use the interactive configuration:
1. Using default values
2. Setting values programmatically
3. Getting values interactively (commented out to avoid blocking)
"""

from config import CFG

def example_default_usage():
    """Example 1: Using default configuration values."""
    print("=== Example 1: Default Configuration ===")
    print(f"Number of clients: {CFG.num_clients}")
    print(f"Number of clusters: {CFG.num_clusters}")
    print(f"Clients per round: {CFG.clients_per_round}")
    print(f"Rounds: {CFG.rounds}")
    print(f"Local epochs: {CFG.local_epochs}")
    print()

def example_programmatic_usage():
    """Example 2: Setting configuration programmatically."""
    print("=== Example 2: Programmatic Configuration ===")
    
    # Set new values programmatically
    CFG.set_federated_config(
        num_clients=50,
        num_clusters=3,
        clients_per_round=15,
        rounds=25,
        local_epochs=3,
        fmnist_distill_epochs=8,
        cifar10_distill_epochs=12,
        fmnist_student_epochs=18,
        cifar10_student_epochs=25
    )
    print()

def example_interactive_usage():
    """Example 3: Interactive configuration (commented out)."""
    print("=== Example 3: Interactive Configuration ===")
    print("To use interactive configuration, uncomment the following lines:")
    print("# num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs = CFG.get_federated_config()")
    print("# CFG.set_federated_config(num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs)")
    print()
    
    # Uncomment these lines to use interactive input:
    # num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs = CFG.get_federated_config()
    # CFG.set_federated_config(num_clients, num_clusters, clients_per_round, rounds, local_epochs, fmnist_distill_epochs, cifar10_distill_epochs, fmnist_student_epochs, cifar10_student_epochs)

def main():
    """Run all examples."""
    print("FedMTFI Configuration Examples")
    print("=" * 50)
    
    # Example 1: Default values
    example_default_usage()
    
    # Example 2: Programmatic setting
    example_programmatic_usage()
    
    # Example 3: Interactive (commented)
    example_interactive_usage()
    
    print("=" * 50)
    print("Configuration methods available:")
    print("1. CFG.get_federated_config() - Interactive input")
    print("2. CFG.set_federated_config() - Programmatic setting")
    print("3. Direct access: CFG.num_clients, CFG.num_clusters, etc.")

if __name__ == "__main__":
    main()