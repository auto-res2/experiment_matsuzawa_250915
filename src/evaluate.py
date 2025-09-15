"""Evaluation module.
Loads the tiny linear model produced by train.py and reports Mean Absolute
Error (MAE) on a newly generated validation set.  We purposefully *do not*
re-use the training data to avoid a degenerate zero-error score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

__all__ = ["evaluate"]

MODEL_PATH = Path(".research/iteration2/model/linear_regressor.pt")


def _generate_synthetic_data(num_samples: int, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, 1)
    noise = torch.randn(num_samples, 1) * noise_std
    y = 2.0 * x + noise
    return x, y


def _load_model() -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected trained model artefact at {MODEL_PATH} – did train() run successfully?"
        )
    state = torch.load(MODEL_PATH, map_location="cpu")
    if "weight" not in state:
        raise RuntimeError("Corrupted model file – missing 'weight' key.")
    return float(state["weight"])


def evaluate(cfg: Dict) -> Dict:
    start = time.time()

    num_val: int = int(cfg.get("num_val_samples", 256))
    noise_std: float = float(cfg.get("noise_std", 0.1))

    weight = _load_model()
    x_val, y_val = _generate_synthetic_data(num_val, noise_std)

    with torch.no_grad():
        y_pred = weight * x_val
        mae = torch.mean(torch.abs(y_pred - y_val)).item()
        mse = torch.mean((y_pred - y_val) ** 2).item()

    return {
        "mae": mae,
        "mse": mse,
        "num_val_samples": num_val,
        "evaluation_time_sec": time.time() - start,
    }
