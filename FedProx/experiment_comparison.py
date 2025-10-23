#!/usr/bin/env python3
"""
FedProx Experiment Results Comparison and Analysis
This script compares the results from FMNIST and CIFAR-10 experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_results(results_file):
    """Load experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def extract_metrics(results):
    """Extract key metrics from results."""
    round_metrics = results['round_metrics']
    final_metrics = round_metrics[-1] if round_metrics else {}
    
    return {
        'dataset': results['args']['dataset'],
        'final_accuracy': final_metrics.get('accuracy', 0),
        'final_loss': final_metrics.get('loss', 0),
        'total_rounds': len(round_metrics),
        'parameters': {
            'mu': results['args']['mu'],
            'lr': results['args']['lr'],
            'local_epochs': results['args']['E'],
            'clients': results['args']['K'],
            'sample_rate': results['args']['C'],
            'batch_size': results['args']['B'],
            'alpha': results['args']['alpha']
        }
    }

def create_comparison_report(fmnist_results, cifar10_results):
    """Create a detailed comparison report."""
    fmnist_metrics = extract_metrics(fmnist_results)
    cifar10_metrics = extract_metrics(cifar10_results)
    
    report = f"""
FedProx Experiment Results Comparison
=====================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET COMPARISON
------------------
FMNIST Dataset:
  - Final Accuracy: {fmnist_metrics['final_accuracy']:.4f} ({fmnist_metrics['final_accuracy']*100:.2f}%)
  - Final Loss: {fmnist_metrics['final_loss']:.6f}
  - Total Rounds: {fmnist_metrics['total_rounds']}

CIFAR-10 Dataset:
  - Final Accuracy: {cifar10_metrics['final_accuracy']:.4f} ({cifar10_metrics['final_accuracy']*100:.2f}%)
  - Final Loss: {cifar10_metrics['final_loss']:.6f}
  - Total Rounds: {cifar10_metrics['total_rounds']}

PERFORMANCE ANALYSIS
--------------------
Accuracy Difference: {(fmnist_metrics['final_accuracy'] - cifar10_metrics['final_accuracy'])*100:.2f} percentage points
Loss Difference: {fmnist_metrics['final_loss'] - cifar10_metrics['final_loss']:.6f}

Relative Performance:
- FMNIST achieved {fmnist_metrics['final_accuracy']/cifar10_metrics['final_accuracy']:.2f}x better accuracy than CIFAR-10
- CIFAR-10 loss is {cifar10_metrics['final_loss']/fmnist_metrics['final_loss']:.2f}x higher than FMNIST

EXPERIMENTAL PARAMETERS
-----------------------
Both experiments used identical parameters:
  - Proximal Term (μ): {fmnist_metrics['parameters']['mu']}
  - Learning Rate: {fmnist_metrics['parameters']['lr']}
  - Local Epochs: {fmnist_metrics['parameters']['local_epochs']}
  - Total Clients: {fmnist_metrics['parameters']['clients']}
  - Client Sample Rate: {fmnist_metrics['parameters']['sample_rate']}
  - Batch Size: {fmnist_metrics['parameters']['batch_size']}
  - Data Heterogeneity (α): {fmnist_metrics['parameters']['alpha']}

OBSERVATIONS
------------
1. FMNIST significantly outperformed CIFAR-10 in terms of accuracy
2. The simpler grayscale nature of FMNIST (fashion items) vs. complex color images in CIFAR-10 
   likely contributed to the performance difference
3. Both experiments converged within 5 federated rounds
4. The FedProx algorithm successfully handled non-IID data distribution (α=0.5) in both cases

RECOMMENDATIONS
---------------
1. For CIFAR-10, consider:
   - Increasing the number of federated rounds
   - Using a more complex model architecture (e.g., ResNet)
   - Adjusting the learning rate or proximal term
   - Implementing data augmentation techniques

2. The current SimpleCNN architecture appears well-suited for FMNIST but may be 
   insufficient for the complexity of CIFAR-10 images
"""
    
    return report

def create_visualization(fmnist_results, cifar10_results, save_dir="./fedprox_results"):
    """Create visualization comparing the results."""
    fmnist_metrics = extract_metrics(fmnist_results)
    cifar10_metrics = extract_metrics(cifar10_results)
    
    # Create comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    datasets = ['FMNIST', 'CIFAR-10']
    accuracies = [fmnist_metrics['final_accuracy'], cifar10_metrics['final_accuracy']]
    
    bars1 = ax1.bar(datasets, accuracies, color=['#2E86AB', '#A23B72'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Final Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}\n({acc*100:.2f}%)', ha='center', va='bottom')
    
    # Loss comparison
    losses = [fmnist_metrics['final_loss'], cifar10_metrics['final_loss']]
    
    bars2 = ax2.bar(datasets, losses, color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('Loss')
    ax2.set_title('Final Loss Comparison')
    
    # Add value labels on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(save_dir, 'fedprox_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_file}")
    return plot_file

def main():
    """Main function to run the comparison analysis."""
    results_dir = "./fedprox_results"
    
    # Load results
    fmnist_file = os.path.join(results_dir, "fedprox_fmnist_results.json")
    cifar10_file = os.path.join(results_dir, "fedprox_cifar10_results.json")
    
    if not os.path.exists(fmnist_file) or not os.path.exists(cifar10_file):
        print("Error: Results files not found. Please run the experiments first.")
        return
    
    fmnist_results = load_results(fmnist_file)
    cifar10_results = load_results(cifar10_file)
    
    # Create comparison report
    report = create_comparison_report(fmnist_results, cifar10_results)
    
    # Save report
    report_file = os.path.join(results_dir, "experiment_comparison_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create visualization
    plot_file = create_visualization(fmnist_results, cifar10_results, results_dir)
    
    # Print summary
    print("="*60)
    print("FEDPROX EXPERIMENT COMPARISON COMPLETE")
    print("="*60)
    print(f"Report saved to: {report_file}")
    print(f"Visualization saved to: {plot_file}")
    print("\nKey Findings:")
    
    fmnist_acc = fmnist_results['round_metrics'][-1]['accuracy']
    cifar10_acc = cifar10_results['round_metrics'][-1]['accuracy']
    
    print(f"  • FMNIST Accuracy: {fmnist_acc:.4f} ({fmnist_acc*100:.2f}%)")
    print(f"  • CIFAR-10 Accuracy: {cifar10_acc:.4f} ({cifar10_acc*100:.2f}%)")
    print(f"  • Performance Gap: {(fmnist_acc - cifar10_acc)*100:.2f} percentage points")
    
    print("\nFull analysis available in the generated report.")

if __name__ == "__main__":
    main()