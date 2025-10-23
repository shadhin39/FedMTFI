#!/usr/bin/env python3
"""
Interactive Configuration Script for FedMTFI

Run this script to interactively set federated learning configuration parameters.
The configuration will be applied to the main config.py file.

Usage:
    python interactive_config.py
"""

import sys
import os

# Add current directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run interactive configuration."""
    print("FedMTFI Interactive Configuration")
    print("=" * 40)
    print("This script allows you to configure federated learning parameters interactively.")
    print("Press Enter to use default values shown in parentheses.\n")
    
    # Import config
    from config import CFG
    
    # Get current configuration
    print("Current Configuration:")
    print(f"  - Number of clients: {CFG.num_clients}")
    print(f"  - Number of clusters: {CFG.num_clusters}")
    print(f"  - Clients per round: {CFG.clients_per_round}")
    print(f"  - Rounds: {CFG.rounds}")
    print(f"  - Local epochs: {CFG.local_epochs}")
    
    print("\nDo you want to update the configuration? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes']:
        # Get new configuration interactively
        num_clients, num_clusters, clients_per_round, rounds, local_epochs = CFG.get_federated_config()
        
        # Update the configuration
        CFG.set_federated_config(
            num_clients=num_clients,
            num_clusters=num_clusters,
            clients_per_round=clients_per_round,
            rounds=rounds,
            local_epochs=local_epochs
        )
        
        print("\n" + "=" * 40)
        print("✓ Configuration updated successfully!")
        print("You can now run your federated learning experiments with the new settings.")
        
    else:
        print("\nConfiguration unchanged. Using current settings.")
    
    print("\nTo use these settings in your main script, simply import CFG from config:")
    print("    from config import CFG")
    print("    print(f'Using {CFG.num_clients} clients with {CFG.num_clusters} clusters')")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)