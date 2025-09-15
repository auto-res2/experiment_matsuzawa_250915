"""Training utilities.
Currently no experiment code was provided, so this module only contains a
stub implementation that can be expanded once the original logic becomes
available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    """Container for training results."""

    metrics: Dict[str, Any]
    model_path: str


def train(cfg: Dict[str, Any]) -> TrainResult:
    """Main training entry-point (stub).

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed experiment configuration.
    """
    logger.warning("train() was called, but no training logic is implemented.")

    # In a real implementation we would build a model, dataloaders, optimiser,
    # loss function, iterate over epochs, etc. Here, we only emulate an output
    # structure so that downstream code will not crash.
    dummy_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
    }
    # Save an empty file to represent the model checkpoint
    checkpoint_path = ".research/iteration1/dummy_model.pt"
    torch.save({}, checkpoint_path)

    return TrainResult(metrics=dummy_metrics, model_path=checkpoint_path)