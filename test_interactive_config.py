#!/usr/bin/env python3
"""
Test script to demonstrate interactive configuration input.
Run this script to test the interactive federated learning configuration.
"""

import sys
import os

# Add current directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_interactive_config():
    """Test the interactive configuration functionality."""
    print("Testing Interactive Federated Learning Configuration")
    print("=" * 50)
    
    # Import and test the config
    from config import CFG
    
    # Test the interactive configuration
    if hasattr(CFG, 'get_federated_config'):
        print("Interactive configuration method found!")
        
        # Get configuration interactively
        num_clients, num_clusters, clients_per_round, rounds, local_epochs = CFG.get_federated_config()
        
        # Update CFG class attributes with user input
        CFG.num_clients = num_clients
        CFG.num_clusters = num_clusters
        CFG.clients_per_round = clients_per_round
        CFG.rounds = rounds
        CFG.local_epochs = local_epochs
        
        print("\n" + "=" * 50)
        print("Configuration successfully updated!")
        print(f"CFG.num_clients = {CFG.num_clients}")
        print(f"CFG.num_clusters = {CFG.num_clusters}")
        print(f"CFG.clients_per_round = {CFG.clients_per_round}")
        print(f"CFG.rounds = {CFG.rounds}")
        print(f"CFG.local_epochs = {CFG.local_epochs}")
        
    else:
        print("Interactive configuration method not found!")
        print("Using default values:")
        print(f"CFG.num_clients = {CFG.num_clients}")
        print(f"CFG.num_clusters = {CFG.num_clusters}")
        print(f"CFG.clients_per_round = {CFG.clients_per_round}")
        print(f"CFG.rounds = {CFG.rounds}")
        print(f"CFG.local_epochs = {CFG.local_epochs}")

if __name__ == "__main__":
    test_interactive_config()