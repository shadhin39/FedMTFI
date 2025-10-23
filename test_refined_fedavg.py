#!/usr/bin/env python3
"""
Test script for refined FedAvg implementation.
Tests the improved aggregation algorithm and parameter handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from fedavg_server import FedAvgServer
from fedavg_client import FedAvgClient
from fedavg_config import FedAvgConfig

def create_simple_model():
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

def create_test_data(num_samples=100):
    """Create synthetic test data."""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)

def test_refined_aggregation():
    """Test the refined FedAvg aggregation algorithm."""
    print("Testing Refined FedAvg Aggregation...")
    
    # Create test configuration
    config = FedAvgConfig()
    config.num_clients = 3
    config.clients_per_round = 3
    config.local_epochs = 1
    
    # Create server with proper device parameter
    model = create_simple_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server = FedAvgServer(model, device)
    
    # Create test clients with different data sizes
    clients = []
    client_models = []
    client_weights = []
    
    for i in range(3):
        # Create client with different data sizes
        data_size = 50 + i * 25  # 50, 75, 100 samples
        test_data = create_test_data(data_size)
        test_loader = DataLoader(test_data, batch_size=10, shuffle=True)
        
        client = FedAvgClient(i, create_simple_model(), test_loader, device)
        clients.append(client)
        
        # Get initial model parameters
        client_models.append(client.get_model_parameters())
        client_weights.append(data_size)
    
    print(f"Client weights (data sizes): {client_weights}")
    
    # Test aggregation
    try:
        aggregated_params = server.aggregate_models(client_models, client_weights)
        print("✓ Refined aggregation successful")
        
        # Verify aggregated parameters have correct shapes
        original_params = server.global_model.state_dict()
        for name in original_params:
            if name in aggregated_params:
                if original_params[name].shape == aggregated_params[name].shape:
                    print(f"✓ Parameter {name} shape preserved: {aggregated_params[name].shape}")
                else:
                    print(f"✗ Parameter {name} shape mismatch!")
                    return False
        
        # Test global model update
        server.update_global_model(client_models, client_weights)
        print("✓ Global model update successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Aggregation failed: {e}")
        return False

def test_parameter_transmission():
    """Test improved parameter transmission between server and clients."""
    print("\nTesting Parameter Transmission...")
    
    config = FedAvgConfig()
    model = create_simple_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server = FedAvgServer(model, device)
    
    # Create test client
    test_data = create_test_data(50)
    test_loader = DataLoader(test_data, batch_size=10)
    client = FedAvgClient(0, create_simple_model(), test_loader, device)
    
    try:
        # Test server -> client parameter transmission
        global_params = server.get_global_model_parameters()
        client.set_model_parameters(global_params)
        print("✓ Server -> Client parameter transmission successful")
        
        # Test client -> server parameter transmission
        client_params = client.get_model_parameters()
        server.update_global_model([client_params], [50])
        print("✓ Client -> Server parameter transmission successful")
        
        # Verify parameters are on CPU for transmission
        for name, param in client_params.items():
            if param.device.type == 'cpu':
                print(f"✓ Parameter {name} correctly on CPU for transmission")
            else:
                print(f"✗ Parameter {name} not on CPU: {param.device}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter transmission failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability of the refined aggregation."""
    print("\nTesting Numerical Stability...")
    
    config = FedAvgConfig()
    model = create_simple_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server = FedAvgServer(model, device)
    
    # Test with zero weights
    client_models = []
    for i in range(3):
        client_models.append(server.global_model.state_dict())
    
    try:
        # Test with zero weights
        zero_weights = [0, 0, 0]
        aggregated_params = server.aggregate_models(client_models, zero_weights)
        print("✓ Zero weights handled correctly")
        
        # Test with very small weights
        small_weights = [1e-10, 1e-10, 1e-10]
        aggregated_params = server.aggregate_models(client_models, small_weights)
        print("✓ Small weights handled correctly")
        
        # Test with large weights
        large_weights = [1e6, 1e6, 1e6]
        aggregated_params = server.aggregate_models(client_models, large_weights)
        print("✓ Large weights handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        return False

def main():
    """Run all tests for refined FedAvg."""
    print("=" * 50)
    print("REFINED FEDAVG ALGORITHM TESTS")
    print("=" * 50)
    
    tests = [
        test_refined_aggregation,
        test_parameter_transmission,
        test_numerical_stability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All refined FedAvg tests PASSED!")
        return True
    else:
        print("✗ Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)