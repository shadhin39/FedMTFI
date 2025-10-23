import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from config import CFG
from non_iid_distributor import NonIIDDataDistributor, create_non_iid_config


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_fmnist_mnist():
    """Load MNIST as private data for clients and FashionMNIST as public data for knowledge distillation.

    Both are converted to 3x224 and normalized with ImageNet statistics to work with MobileNetV2.
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    to_rgb_224 = transforms.Compose([
        transforms.Resize(CFG.image_size),
        transforms.Grayscale(num_output_channels=CFG.in_channels),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # MNIST as private data for client training
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=to_rgb_224
    )
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=to_rgb_224
    )
    
    # FashionMNIST as public data for knowledge distillation
    fmnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=to_rgb_224
    )
    
    return mnist_train, mnist_test, fmnist_train


def split_public_dataset(trainset):
    """Deprecated for MNIST/FashionMNIST setup; retained for backward compatibility."""
    total = len(trainset)
    public_size = int(total * CFG.public_fraction)
    indices = list(range(total))
    random.shuffle(indices)
    public_idx = indices[:public_size]
    private_idx = indices[public_size:]
    public_set = Subset(trainset, public_idx)
    private_set = Subset(trainset, private_idx)
    return public_set, private_set


def build_client_loaders(private_set, num_clients: int, heterogeneity: str = "high", strategy: str = "bias"):
    distributor = NonIIDDataDistributor(private_set, num_clients, num_classes=CFG.num_classes)
    cfg = create_non_iid_config(strategy=strategy, heterogeneity=heterogeneity)

    if cfg["distribution_strategy"] == "bias":
        client_subsets, client_prefs = distributor.bias_based_distribution(
            **cfg["bias_settings"]
        )
    else:
        client_subsets, client_prefs = distributor.shard_based_distribution(
            **cfg["shard_settings"]
        )

    loaders = [
        DataLoader(sub, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        for sub in client_subsets
    ]

    return loaders, client_prefs


def build_public_loader(public_set):
    return DataLoader(public_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)


def build_test_loader(testset):
    return DataLoader(testset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)