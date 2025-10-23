"""
Run FedAvg experiment on CIFAR-10 with alpha=0.5, 10 rounds
"""

from fedavg_config import FEDAVG_CFG
from fedavg_main import run_fedavg

def main():
    """Run FedAvg experiment on CIFAR-10."""
    
    # Configure FedAvg settings
    FEDAVG_CFG.set_fedavg_config(
        num_clients=100,
        clients_per_round=30,
        num_rounds=10,
        local_epochs=5,
        learning_rate=0.01,
        alpha=0.5,  # Non-IID parameter
        batch_size=32,
        eval_every_n_rounds=2,
        save_metrics=True,
        save_model=True,
        results_dir="./fedavg_experiment_results"
    )
    
    print("=" * 60)
    print("FedAvg Experiment on CIFAR-10: Alpha=0.5, 10 Rounds")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Clients: {FEDAVG_CFG.num_clients}")
    print(f"  - Clients per round: {FEDAVG_CFG.clients_per_round}")
    print(f"  - Rounds: {FEDAVG_CFG.num_rounds}")
    print(f"  - Local epochs: {FEDAVG_CFG.local_epochs}")
    print(f"  - Alpha (non-IID): {FEDAVG_CFG.alpha}")
    print(f"  - Learning rate: {FEDAVG_CFG.learning_rate}")
    print()
    
    try:
        print("Running FedAvg on CIFAR-10...")
        final_metrics, round_metrics = run_fedavg("CIFAR10")
        
        print(f"\nCIFAR-10 Results:")
        print(f"  Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Final Loss: {final_metrics['loss']:.4f}")
        print(f"  Total Rounds: {len(round_metrics)}")
        print(f"  Results saved in: {FEDAVG_CFG.results_dir}")
        
    except Exception as e:
        print(f"Error running CIFAR-10: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("CIFAR-10 experiment completed!")
    print(f"Results saved in: {FEDAVG_CFG.results_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()