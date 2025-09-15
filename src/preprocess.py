"""Data pre-processing / loading utilities.

For the placeholder implementation we generate a *synthetic* classification
problem so the entire pipeline can run offline in CI.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Internal helpers – dataset factory
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    class_sep: float = 5.0,
    seed: int | None = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return *(features, labels)* tensors for a toy classification problem."""

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # create a mean vector for each class located on a circle to guarantee
    # separation
    angles = torch.linspace(0, 2 * math.pi, steps=num_classes + 1)[:-1]
    means = torch.stack(
        [torch.tensor([math.cos(a), math.sin(a)] + [0.0] * (input_dim - 2)) * class_sep for a in angles]
    )

    features = torch.empty(num_samples, input_dim)
    labels = torch.empty(num_samples, dtype=torch.long)
    for i in range(num_samples):
        cls = torch.randint(0, num_classes, (1,), generator=g).item()
        features[i] = torch.randn(input_dim, generator=g) + means[cls]
        labels[i] = cls

    return features, labels


# ---------------------------------------------------------------------------
# Public API – get_dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Create train/val/test loaders according to the YAML *config*."""

    dataset_cfg = config.get("dataset", {})
    num_samples = int(dataset_cfg.get("num_samples", 1000))
    input_dim = int(dataset_cfg.get("input_dim", 20))
    num_classes = int(dataset_cfg.get("num_classes", 4))
    batch_size = int(config.get("batch_size", 32))
    seed = int(dataset_cfg.get("seed", 42))

    X, y = _make_synthetic_dataset(
        num_samples=num_samples,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=seed,
    )

    # 60 ▸ 20 ▸ 20 split
    n_train = int(0.6 * num_samples)
    n_val = int(0.2 * num_samples)
    n_test = num_samples - n_train - n_val

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train : n_train + n_val], y[n_train : n_train + n_val])
    test_ds = TensorDataset(X[-n_test:], y[-n_test:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, num_classes
