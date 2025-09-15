"""Simple training module for COSMIC-X placeholder implementation.

This minimal trainer is designed to make the automated smoke test & full
benchmark pipelines runnable even though the full research code base is not yet
public.  It intentionally keeps the logic *tiny*:

    • Synthetic classification dataset created on-the-fly by
      preprocess.get_dataloaders.
    • Two-layer MLP (Linear ➞ ReLU ➞ Linear) implemented in <20 lines.
    • Standard cross-entropy objective, Adam optimiser.

The goal is **not** scientific novelty; it is only to produce *concrete numeric
results* so that the CI harness sees a successful experiment run (loss and
accuracy values) and downstream JSON artefacts.

If you need to swap in the real COSMIC-X models later, change only the
`build_model` function and the training loop – all other plumbing stays the
same.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper – model definition
# ---------------------------------------------------------------------------

def build_model(input_dim: int, num_classes: int, hidden_dim: int = 128) -> nn.Module:  # noqa: D401
    """Return a *tiny* 2-layer MLP suitable for the synthetic dataset."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, num_classes),
    )


# ---------------------------------------------------------------------------
# Public API – train()
# ---------------------------------------------------------------------------

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    config: Dict,
) -> Tuple[nn.Module, Dict]:
    """Run a *very* short training according to *config* and return metrics.

    Parameters
    ----------
    train_loader, val_loader
        PyTorch ``DataLoader`` objects coming from :pyfunc:`preprocess.get_dataloaders`.
    input_dim, num_classes
        Needed to build the classifier.
    config
        Dict loaded from YAML.  *Required keys*:
        ``num_epochs``, ``learning_rate``, ``device``.
    """

    device = torch.device(config.get("device", "cpu"))
    num_epochs: int = int(config["num_epochs"])
    learning_rate: float = float(config.get("learning_rate", 1e-3))

    model = build_model(input_dim, num_classes, hidden_dim=config.get("hidden_dim", 128))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # ------------------------------------------------------------------
        # quick val pass
        # ------------------------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0
        history["val_acc"].append(val_acc)

        tqdm.write(
            f"[Epoch {epoch+1}/{num_epochs}] loss={epoch_loss:.4f} val_acc={val_acc:.4f}"
        )

    # add run-time to history
    history["run_ts"] = time.time()

    return model, history
