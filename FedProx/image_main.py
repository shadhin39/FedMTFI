# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23
@Author: Assistant
@File: image_main.py
@Description: Main script for running FedProx on image classification datasets
"""
import os
import time
from datetime import datetime

from image_args import args_parser, print_args
from image_server import FedProxImageServer


def run_fedprox_experiment(dataset_name=None):
    """
    Run FedProx experiment on specified dataset.
    
    Args:
        dataset_name: 'FMNIST' or 'CIFAR10'. If None, uses args.dataset
    
    Returns:
        Dictionary with experiment results
    """
    # Parse arguments
    args = args_parser()
    
    # Override dataset if specified
    if dataset_name:
        args.dataset = dataset_name
    
    # Print configuration
    print_args(args)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Initialize FedProx server
        print(f"Initializing FedProx server for {args.dataset}...")
        fedprox_server = FedProxImageServer(args)
        
        # Run federated training
        print(f"Starting federated training...")
        global_model = fedprox_server.server_train()
        
        # Final evaluation
        final_metrics = fedprox_server.global_test()
        
        # Save results
        if args.save_results:
            results_file, model_file = fedprox_server.save_results(args.results_dir)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Dataset: {args.dataset}")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final Loss: {final_metrics['loss']:.6f}")
        
        if args.save_results:
            print(f"Results saved to: {args.results_dir}")
        
        return {
            'success': True,
            'dataset': args.dataset,
            'final_metrics': final_metrics,
            'total_time': total_time,
            'args': vars(args)
        }
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'dataset': args.dataset if 'args' in locals() else dataset_name,
            'error': str(e),
            'total_time': time.time() - start_time if 'start_time' in locals() else 0
        }


def run_both_datasets():
    """Run FedProx experiments on both FMNIST and CIFAR-10."""
    print("="*80)
    print("RUNNING FEDPROX ON BOTH FMNIST AND CIFAR-10")
    print("="*80)
    
    datasets = ['FMNIST', 'CIFAR10']
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*20} STARTING {dataset} EXPERIMENT {'='*20}")
        
        result = run_fedprox_experiment(dataset)
        results[dataset] = result
        
        if result['success']:
            print(f"✅ {dataset} experiment completed successfully!")
            print(f"   Accuracy: {result['final_metrics']['accuracy']:.4f}")
            print(f"   Time: {result['total_time']:.2f}s")
        else:
            print(f"❌ {dataset} experiment failed!")
            print(f"   Error: {result['error']}")
        
        print(f"{'='*20} {dataset} EXPERIMENT FINISHED {'='*20}")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL EXPERIMENT SUMMARY")
    print("="*80)
    
    for dataset, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{dataset}: {status}")
        if result['success']:
            print(f"  Accuracy: {result['final_metrics']['accuracy']:.4f}")
            print(f"  Loss: {result['final_metrics']['loss']:.6f}")
            print(f"  Time: {result['total_time']:.2f}s")
        else:
            print(f"  Error: {result['error']}")
    
    return results


def main():
    """Main function."""
    print("FedProx Image Classification Experiments")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if specific dataset is requested via command line
    import sys
    if len(sys.argv) > 1 and '--dataset' in sys.argv:
        # Single dataset experiment
        result = run_fedprox_experiment()
        return result['success']
    else:
        # Interactive mode - ask user what to run
        print("\nSelect experiment to run:")
        print("1. FashionMNIST only")
        print("2. CIFAR-10 only") 
        print("3. Both datasets")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                result = run_fedprox_experiment('FMNIST')
                return result['success']
            elif choice == '2':
                result = run_fedprox_experiment('CIFAR10')
                return result['success']
            elif choice == '3':
                results = run_both_datasets()
                return all(r['success'] for r in results.values())
            elif choice == '4':
                print("Exiting...")
                return True
            else:
                print("Invalid choice. Please run again.")
                return False
                
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user.")
            return False
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)