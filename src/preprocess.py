"""Data-loading & preprocessing utilities.

For the purposes of the quick-fix we *synthesise* a tiny binary
classification dataset rather than downloading a real public corpus. We
still follow the mandatory directory conventions so that future
iterations can swap the generator for a real dataset without touching
other modules.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger(__name__)

ITER_DIR = Path(".research/iteration3")
DATA_STATS_PATH = ITER_DIR / "dataset_stats.json"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _generate_synthetic_dataset(n_samples: int, n_features: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a linearly-separable dataset for binary classification."""
    torch.manual_seed(42)  # Deterministic for CI

    # Random hyper-plane
    weights = torch.randn(n_features)
    bias = torch.randn(1).item()

    x = torch.randn(n_samples, n_features)
    logits = x @ weights + bias  # shape: (n_samples,)
    y = (logits > 0).long()  # labels 0 / 1

    return x, y


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def prepare_datasets(cfg: Dict[str, Any]) -> None:  # noqa: D401  (simple-return)
    """Create (or reuse) synthetic datasets and persist them to disk.

    The function is idempotent – if the tensors already exist we skip
    generation to guarantee consistency between the training and
    evaluation phases.
    """
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    train_path = ITER_DIR / "train_data.pt"
    val_path = ITER_DIR / "val_data.pt"

    # If the dataset is already present, *do not* overwrite it – this would
    # break determinism between train & eval within the same run.
    if train_path.exists() and val_path.exists():
        logger.info("Synthetic dataset already exists – skipping generation.")
        return

    split_ratio = 0.8
    total_samples = int(cfg.get("dataset", {}).get("n_samples", 1000))
    n_features = int(cfg.get("dataset", {}).get("n_features", 20))

    x, y = _generate_synthetic_dataset(total_samples, n_features)

    split_idx = math.floor(total_samples * split_ratio)
    train_x, val_x = x[:split_idx], x[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]

    torch.save((train_x, train_y), train_path)
    torch.save((val_x, val_y), val_path)

    # ------------------------------------------------------------------
    # Save some quick stats for debugging / provenance tracking
    # ------------------------------------------------------------------
    stats = {
        "total_samples": total_samples,
        "n_features": n_features,
        "train_samples": train_x.shape[0],
        "val_samples": val_x.shape[0],
    }

    with DATA_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved synthetic dataset tensors to %s", ITER_DIR)
    logger.info("Dataset stats: %s", json.dumps(stats))