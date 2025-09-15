import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

__all__ = ["build_datasets", "dataset_sanity_check"]


def dataset_sanity_check(root: Path, required: tuple):
    for ds in required:
        if not (root / ds).exists():
            raise RuntimeError(f"Dataset {ds} missing in {root}. Aborting.")


def build_datasets(cfg: Dict) -> Tuple[Dataset, Dataset]:
    """For the purposes of this public refactor we rely on CIFAR-100 which is
    automatically downloaded by torchvision. In the *real* cluster run the user
    would swap this call with the actual MultiCam-Overlap-V2 / AudioDoor etc.
    """
    data_root = Path(cfg["data"].get("root", "./data"))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = CIFAR100(root=str(data_root), train=True, download=True, transform=train_transform)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    # Apply test transform to validation split
    val_ds.dataset.transform = test_transform  # type: ignore

    return train_ds, val_ds
