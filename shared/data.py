"""Dataset loaders for HTB AI Red Team exercises.

Each loader returns (train_loader, test_loader, meta_dict).
"""

from __future__ import annotations


def load_mnist(batch_size: int = 128, data_root: str = "./data"):
    """Return MNIST train/test DataLoaders + metadata.

    Downloads via torchvision on first call. Subsequent calls reuse the cache.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    meta = {
        "name": "MNIST",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }
    return train_loader, test_loader, meta
