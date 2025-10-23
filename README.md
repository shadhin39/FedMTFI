# FedMTFI

A comprehensive federated learning framework implementing cluster-based federated learning with multi-teacher knowledge distillation and feature importance optimization.

## Overview

FedMTFI is a federated learning system that combines:
- **Cluster-based Federated Learning**: Organizes clients into clusters with specialized model architectures
- **Multi-Teacher Knowledge Distillation**: Uses multiple teacher models to train a unified student model or generalized server model
- **Feature Importance Optimization**: Incorporates SHAP-based feature importance for enhanced learning
- **Non-IID Data Distribution**: Handles realistic heterogeneous data scenarios

## Key Features

### Core Architecture
- **Cluster-Specific Models**: SimpleCNN, ResNet-like, MobileNet-like, and ResNet18-like architectures, etc.
- **Adaptive Input Handling**: Supports both grayscale (MNIST, Fashion-MNIST) and RGB (CIFAR-10) datasets
- **Multi-Teacher Distillation**: Aggregates knowledge from cluster models into a unified student model
- **Post-hoc Knowledge Distillation**: Final refinement phase using trained cluster models as teachers

### Federated Learning Approaches
1. **FedMTFI (Main)**: Novel cluster-based approach with multi-teacher knowledge distillation
2. **FedAvg**: Traditional federated averaging implementation using Flower framework
3. **FedProx**: Proximal federated learning with regularization terms
4. **Centralized**: Baseline centralized learning for comparison

### Advanced Features
- **Non-IID Data Distribution**: Bias-based and shard-based distribution strategies
- **Feature Importance Analysis**: SHAP and Captum integration for interpretability
- **Comprehensive Metrics**: Detailed logging of training, evaluation, and timing metrics
- **Interactive Configuration**: User-friendly setup for experiment parameters
- **Visualization**: Automated plot generation for training progress and results

## Project Structure

```
FedMTFI_Experiment_Main/
├── main.py                    # Main FedMTFI implementation
├── config.py                  # Configuration management
├── models.py                  # Neural network architectures
├── client.py                  # Client-side training logic
├── server.py                  # Server-side aggregation and distillation
├── distillation.py           # Knowledge distillation utilities
├── non_iid_distributor.py    # Data distribution strategies
├── metrics_logger.py         # Comprehensive metrics tracking
├── shap_utils.py             # Feature importance analysis
├── plotting_utils.py         # Visualization utilities
├── FedAvg/                  
│   ├── CIFAR-10/
│   ├── FMNIST/
│   └── MNIST/
├── FedProx/                  # Proximal federated learning
├── Centralized/              # Baseline centralized learning
└── data/                     # Dataset storage
```

## Supported Datasets

- **MNIST**: Handwritten digits (28x28 grayscale)
- **Fashion-MNIST**: Fashion items (28x28 grayscale)
- **CIFAR-10**: Natural images (32x32 RGB)

All datasets are automatically downloaded and preprocessed with appropriate transformations.

## Model Architectures

### Cluster Models (Teachers)
1. **Cluster 0 - SimpleCNN**: Lightweight CNN with batch normalization
2. **Cluster 1 - ResNet-like**: Residual connections with sequential processing
3. **Cluster 2 - MobileNet-like**: Depthwise separable convolutions
4. **Cluster 3 - ResNet18-like**: Full ResNet-18 inspired architecture

### Student Model
- Compact CNN architecture for knowledge distillation
- Adaptive input channels based on dataset
- Optimized for multi-teacher learning

## Installation

### Prerequisites
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install tensorflow  # For centralized baselines
pip install shap captum  # For feature importance (optional)
pip install openpyxl  # For Excel metrics export
```

### Quick Start
```bash
git clone https://github.com/shadhin39/FedMTFI.git
cd FedMTFI_Experiment_Main
python main.py
```

## Usage

### Main FedMTFI Experiment
```bash
python main.py
```

The system will prompt for configuration parameters:
- Number of clients (default: 100)
- Number of clusters (default: 4)
- Clients per round (default: 20)
- Training rounds (default: 30)
- Local epochs (default: 5)
- Distillation epochs for each dataset

### FedAvg Experiments
```bash
# CIFAR-10
cd FedAvg/CIFAR-10
python server.py  # Terminal 1
python client1.py  # Terminal 2
python client2.py  # Terminal 3

# Fashion-MNIST
cd FedAvg/FMNIST
python server.py  # Terminal 1
python client1.py  # Terminal 2
python client2.py  # Terminal 3
```

### FedProx Configuration Parameters
The FedProx experiments support the following command-line arguments:
- `--dataset`: Dataset to use (FMNIST, CIFAR10)
- `--r`: Number of communication rounds (default: 5)
- `--E`: Number of local epochs (default: 3)
- `--K`: Number of clients (default: 10)
- `--C`: Fraction of clients selected per round (default: 0.3)
- `--B`: Local batch size (default: 32)
- `--lr`: Learning rate (default: 0.01)
- `--mu`: Proximal term coefficient (default: 0.01)
- `--alpha`: Dirichlet distribution parameter for non-IID data (default: 0.5)
- `--model_type`: Model architecture type (default: simple)
- `--device`: Computing device (cpu, cuda)

### FedProx Experiments
```bash
cd FedProx

# Fashion-MNIST experiment
python3 image_main.py --dataset FMNIST --r 5 --E 3 --K 10 --C 0.3 --B 32 --lr 0.01 --mu 0.01 --alpha 0.5 --model_type simple --device cpu

# CIFAR-10 experiment
python3 image_main.py --dataset CIFAR10 --r 5 --E 3 --K 10 --C 0.3 --B 32 --lr 0.01 --mu 0.01 --alpha 0.5 --model_type simple --device cpu


### Centralized Baselines
```bash
cd Centralized/CIFAR-10
python centralized_cifar10.py

cd Centralized/FMNIST
python centralized_fminst.py
```

## Configuration

### Interactive Configuration
The main system provides interactive configuration for:
- Federated learning parameters
- Multi-teacher distillation settings
- Dataset-specific hyperparameters

### Programmatic Configuration
Modify `config.py` for batch experiments:
```python
CFG.num_clients = 50
CFG.num_clusters = 3
CFG.rounds = 20
CFG.local_epochs = 3
```

### Dataset-Specific Parameters
```python
CFG.dataset_configs = {
    "FashionMNIST": {
        "batch_size": 64,
        "lr_server": 0.001,
        "distill_epochs": 5,
        "temperature": 3.0
    },
    "CIFAR10": {
        "batch_size": 128,
        "lr_server": 0.0005,
        "distill_epochs": 10,
        "temperature": 5.0
    }
}
```

## Key Algorithms

### Cluster-Based Federated Learning
1. **Client Assignment**: Round-robin assignment to clusters
2. **Local Training**: Clients train cluster-specific models on private data
3. **Signal Generation**: Clients produce knowledge signals on public data
4. **Cluster Aggregation**: Server trains cluster models using client signals
5. **Model Update**: Updated cluster models distributed to clients

### Multi-Teacher Knowledge Distillation
1. **Teacher Training**: Cluster models trained on public datasets
2. **Confidence Weighting**: Teachers weighted by prediction confidence
3. **Feature Alignment**: Student features aligned with teacher features
4. **Combined Loss**: KD loss + feature loss + cross-entropy loss

### Non-IID Data Distribution
- **Bias-based**: Clients prefer specific labels (configurable bias strength)
- **Shard-based**: Data divided into shards, clients receive multiple shards
- **Heterogeneity Analysis**: Built-in analysis of data distribution patterns


### Non-IID Configuration
```python
# Configure data distribution
distributor = NonIIDDataDistributor(dataset, num_clients, num_classes)
client_datasets, preferences = distributor.bias_based_distribution(
    primary_bias=0.8,  # 80% preferred class
    secondary_bias=False,
    label_distribution='uniform'
)
```

## Experimental Comparisons

The framework enables comprehensive comparison between:
- **FedMTFI vs FedAvg**: Cluster-based vs traditional averaging
- **FedMTFI vs FedProx**: Knowledge distillation vs proximal terms
- **FedMTFI vs Centralized**: Federated vs centralized learning
- **Multi-teacher vs Single-teacher**: Distillation effectiveness
- **IID vs Non-IID**: Data distribution impact



## Troubleshooting

### Common Issues
1. **CUDA Memory**: Reduce batch sizes in `config.py`
2. **Model Compatibility**: Ensure consistent input channels across datasets
3. **Data Loading**: Verify dataset paths and download permissions
4. **Flower Connection**: Check network settings for FedAvg experiments

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Reduce problem size for testing
CFG.num_clients = 10
CFG.rounds = 5
CFG.local_epochs = 2
```
