"""Evaluation helpers.

The real COSMIC-X evaluation is far more involved. For CI we expose only a
single helper that computes the *Top-1 accuracy* on a given DataLoader so the
pipeline has numerical output.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> Dict:
    """Return a dict with *top1* accuracy for the provided loader."""

    model.eval()
    device = torch.device(device)
    model.to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return {"top1_acc": correct / total if total > 0 else 0.0}
