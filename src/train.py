"""Training utilities.
Currently no experiment code was provided, so this module only contains a
stub implementation that can be expanded once the original logic becomes
available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def get_device() -> torch.device:  # noqa: D401  (simple-return)
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Public dataclasses
# -----------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Container for training results."""

    metrics: Dict[str, Any]
    model_path: str


# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> TrainResult:  # noqa: D401  (simple-return)
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

    # ------------------------------------------------------------------
    # Persist dummy checkpoint so that evaluation step has a valid path.
    # All research artefacts must live under .research/iteration2/ according
    # to the task instructions.
    # ------------------------------------------------------------------
    out_dir = Path(".research/iteration2")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "dummy_model.pt"
    torch.save({}, checkpoint_path)
    logger.info("Saved dummy checkpoint to %s", checkpoint_path)

    return TrainResult(metrics=dummy_metrics, model_path=str(checkpoint_path))