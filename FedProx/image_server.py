# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23
@Author: Assistant
@File: image_server.py
@Description: FedProx server implementation for image classification datasets
"""
import copy
import random
import numpy as np
import torch
from tqdm import tqdm
import os
import json
from datetime import datetime

from image_model import create_model
from image_client import train_client, test_client
from image_data import create_non_iid_split, load_dataset


class FedProxImageServer:
    """FedProx server for image classification datasets."""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # Create global model
        self.global_model = create_model(args, 'global_server', args.model_type).to(self.device)
        
        # Create client models
        self.client_models = []
        for i in range(self.args.K):
            client_model = copy.deepcopy(self.global_model)
            client_model.name = f'client_{i}'
            self.client_models.append(client_model)
        
        # Load dataset and create non-IID split
        print(f"Loading {args.dataset} dataset...")
        train_dataset, test_dataset = load_dataset(args.dataset, args.data_dir)
        self.test_dataset = test_dataset
        
        print(f"Creating non-IID data split with alpha={args.alpha}...")
        self.client_indices = create_non_iid_split(train_dataset, args.K, args.alpha)
        
        # Print data distribution statistics
        self._print_data_distribution(train_dataset)
        
        # Initialize metrics tracking
        self.round_metrics = []
        self.client_metrics = []
        
    def _print_data_distribution(self, dataset):
        """Print statistics about data distribution across clients."""
        print("\nData Distribution Statistics:")
        print(f"Total training samples: {len(dataset)}")
        
        # Calculate samples per client
        samples_per_client = [len(indices) for indices in self.client_indices]
        print(f"Samples per client - Min: {min(samples_per_client)}, "
              f"Max: {max(samples_per_client)}, "
              f"Mean: {np.mean(samples_per_client):.1f}, "
              f"Std: {np.std(samples_per_client):.1f}")
        
        # Calculate class distribution for first few clients (for brevity)
        print("\nClass distribution for first 5 clients:")
        labels = np.array(dataset.targets)
        for client_id in range(min(5, self.args.K)):
            client_labels = labels[self.client_indices[client_id]]
            unique, counts = np.unique(client_labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"Client {client_id}: {class_dist}")
    
    def server_train(self):
        """Main federated training loop."""
        print(f"\nStarting FedProx training for {self.args.r} rounds...")
        print(f"Configuration: {self.args.K} clients, {int(self.args.C * self.args.K)} clients per round")
        
        for round_num in tqdm(range(self.args.r), desc="Federated Rounds"):
            print(f'\n--- Round {round_num + 1}/{self.args.r} ---')
            
            # Client sampling
            m = max(int(self.args.C * self.args.K), 1)
            selected_clients = random.sample(range(self.args.K), m)
            print(f"Selected clients: {selected_clients}")
            
            # Dispatch global model to selected clients
            self._dispatch_model(selected_clients)
            
            # Client local training
            self._client_update(selected_clients, round_num)
            
            # Server aggregation
            self._aggregate_models(selected_clients)
            
            # Evaluation
            if (round_num + 1) % self.args.eval_every == 0 or round_num == self.args.r - 1:
                metrics = self._evaluate_global_model(round_num + 1)
                self.round_metrics.append(metrics)
                
                print(f"Round {round_num + 1} Global Test Results:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Loss: {metrics['loss']:.6f}")
        
        return self.global_model
    
    def _dispatch_model(self, selected_clients):
        """Send global model parameters to selected clients."""
        for client_id in selected_clients:
            for global_param, client_param in zip(
                self.global_model.parameters(), 
                self.client_models[client_id].parameters()
            ):
                client_param.data = global_param.data.clone()
    
    def _client_update(self, selected_clients, round_num):
        """Perform local training on selected clients."""
        print("Starting client local training...")
        
        for client_id in selected_clients:
            print(f"Training client {client_id}...")
            
            # Train client model
            trained_model = train_client(
                self.args, 
                self.client_models[client_id], 
                self.global_model,
                client_id, 
                self.client_indices
            )
            
            # Update client model
            self.client_models[client_id] = trained_model
            
            # Optional: evaluate client model
            if self.args.eval_clients:
                client_metrics = test_client(self.args, trained_model, client_id)
                client_metrics['round'] = round_num + 1
                client_metrics['client_id'] = client_id
                self.client_metrics.append(client_metrics)
    
    def _aggregate_models(self, selected_clients):
        """Aggregate client models using FedAvg."""
        # Calculate total samples for weighted averaging
        total_samples = sum(self.client_models[i].len for i in selected_clients)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        for name, param in self.global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param.data)
        
        # Weighted aggregation
        for client_id in selected_clients:
            client_weight = self.client_models[client_id].len / total_samples
            
            for name, param in self.client_models[client_id].named_parameters():
                aggregated_params[name] += param.data * client_weight
        
        # Update global model
        for name, param in self.global_model.named_parameters():
            param.data = aggregated_params[name].clone()
    
    def _evaluate_global_model(self, round_num):
        """Evaluate the global model on test dataset."""
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.args.B, 
            shuffle=False,
            num_workers=2
        )
        
        metrics = test_client(self.args, self.global_model, test_loader=test_loader)
        metrics['round'] = round_num
        
        return metrics
    
    def global_test(self):
        """Final evaluation of the global model."""
        print("\n" + "="*50)
        print("FINAL GLOBAL MODEL EVALUATION")
        print("="*50)
        
        final_metrics = self._evaluate_global_model(self.args.r)
        
        print(f"\nFinal Results on {self.args.dataset}:")
        print(f"  Test Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Test Loss: {final_metrics['loss']:.6f}")
        print(f"  Correct Predictions: {final_metrics['correct']}/{final_metrics['total']}")
        
        return final_metrics
    
    def save_results(self, save_dir="./fedprox_results"):
        """Save training results and model."""
        os.makedirs(save_dir, exist_ok=True)
        
        def convert_to_serializable(obj):
            """Convert numpy types and other non-serializable objects to JSON-serializable types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool)):
                return str(obj)
            else:
                return obj
        
        # Convert args to JSON-serializable format
        args_dict = convert_to_serializable(vars(self.args))
        
        # Save metrics
        results = {
            'args': args_dict,
            'round_metrics': convert_to_serializable(self.round_metrics),
            'client_metrics': convert_to_serializable(self.client_metrics),
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(save_dir, f"fedprox_{self.args.dataset.lower()}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model
        model_file = os.path.join(save_dir, f"fedprox_{self.args.dataset.lower()}_model.pth")
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'args': args_dict,
            'final_metrics': convert_to_serializable(self.round_metrics[-1]) if self.round_metrics else None
        }, model_file)
        
        print(f"\nResults saved to: {save_dir}")
        print(f"  Metrics: {results_file}")
        print(f"  Model: {model_file}")
        
        return results_file, model_file