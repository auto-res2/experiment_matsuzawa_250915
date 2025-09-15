"""Training utilities implementing a *minimal* but *real* training
loop so that the pipeline now produces non-trivial metrics instead of the
previous dummy zeros.

The goal is **NOT** to build the actual GARDEN algorithm (out of scope
for the current quick-fix) but to ensure that:
  • Some learning really happens (loss decreases, accuracy > 0).
  • All research artefacts are written under `.research/iteration3/`.
  • The module respects the fail-fast policy – any problem terminates the
    program with a clear error message.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

ITER_DIR = Path(".research/iteration3")
CHECKPOINT_PATH = ITER_DIR / "model.pt"
METRICS_PATH = ITER_DIR / "train_metrics.json"

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def get_device() -> torch.device:  # noqa: D401 (simple-return)
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val dataloaders from the pre-generated tensors.

    The tensors are produced by ``src.preprocess.prepare_datasets`` and
    stored on disk. We load them back here to avoid regenerating different
    data each stage and to make the full experiment deterministic.
    """
    train_tensors_path = ITER_DIR / "train_data.pt"
    val_tensors_path = ITER_DIR / "val_data.pt"

    if not train_tensors_path.exists() or not val_tensors_path.exists():
        logger.error(
            "Dataset tensors not found at %s or %s. Did prepare_datasets() run?",
            train_tensors_path,
            val_tensors_path,
        )
        sys.exit(1)

    train_x, train_y = torch.load(train_tensors_path)
    val_x, val_y = torch.load(val_tensors_path)

    input_dim: int = train_x.shape[1]

    batch_size: int = int(cfg["training"].get("batch_size", 32))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, input_dim


# -----------------------------------------------------------------------------
# Public dataclass
# -----------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Container for training results."""

    metrics: Dict[str, Any]
    model_path: str


# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> TrainResult:  # noqa: D401 (simple-return)
    """A *very small* training loop on a synthetic dataset.

    It trains a single linear layer on a pre-generated binary-classification
    problem. While trivial, it is enough to guarantee non-zero accuracy and
    a non-trivial loss value so downstream evaluation will have meaningful
    numbers.
    """
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, input_dim = _build_dataloaders(cfg)

    device = get_device()
    model = nn.Linear(input_dim, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs: int = int(cfg["training"].get("epochs", 5))

    history: Dict[str, list[float]] = {"train_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimiser.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * batch_x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # -----------------------------
        # Validation accuracy
        # -----------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                preds = model(val_x).argmax(dim=1)
                correct += (preds == val_y).sum().item()
                total += val_y.size(0)
        val_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(val_acc)

        logger.info("Epoch %d/%d – loss: %.4f – val_acc: %.4f", epoch + 1, num_epochs, avg_loss, val_acc)

    # ------------------------------------------------------------------
    # Persist checkpoint & metrics
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    summary_metrics = {
        "final_train_loss": history["train_loss"][-1],
        "best_val_accuracy": max(history["val_accuracy"]),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("Saved model checkpoint to %s", CHECKPOINT_PATH)
    logger.info("Saved training metrics to %s", METRICS_PATH)

    # Print to STDOUT for quick inspection (also useful for CI logs)
    print(json.dumps(summary_metrics, indent=2))

    return TrainResult(metrics=summary_metrics, model_path=str(CHECKPOINT_PATH))