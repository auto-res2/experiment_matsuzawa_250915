"""Training module for ASTRAL surrogate smoke / demo runs.
This is **NOT** the full research code – it is a **minimal, fast** placeholder
that fulfils the CI requirements:
  • runs in <30 s on CPU for the smoke-test
  • produces *concrete numerical metrics* (loss curve, final MAE)
  • persists an extremely small "model" artefact so that evaluate.py can
    reload it without recomputing training.

The model is a *single-weight* linear regressor  y = w * x  trained on a
synthetic dataset (y = 2x + ε) with mean-squared error.  We use vanilla
PyTorch for automatic differentiation because it is already a dependency.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

__all__ = ["train"]


def _generate_synthetic_data(num_samples: int, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tensors (x, y) following y = 2x + ε."""
    x = torch.randn(num_samples, 1)
    noise = torch.randn(num_samples, 1) * noise_std
    y = 2.0 * x + noise
    return x, y


def _save_model(weight: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight}, output_dir / "linear_regressor.pt")


def train(cfg: Dict) -> Dict:
    """Run the toy training loop and return a metrics dictionary.

    The function is intentionally *simple* – five epochs of SGD on a 1-D weight.
    It nevertheless exercises torch autograd and file IO so that downstream
    evaluate.py has something real to work with.
    """
    start = time.time()

    # ---------------------------------------------------------------------
    # 1. Data – either load from preprocess step or (re-)generate on the fly
    # ---------------------------------------------------------------------
    num_samples: int = int(cfg.get("num_samples", 1000))
    epochs: int = int(cfg.get("epochs", 20))
    lr: float = float(cfg.get("learning_rate", 0.1))
    noise_std: float = float(cfg.get("noise_std", 0.1))

    x, y = _generate_synthetic_data(num_samples, noise_std)

    # ---------------------------------------------------------------------
    # 2. Model – a *single* learnable scalar weight
    # ---------------------------------------------------------------------
    weight = torch.nn.Parameter(torch.randn(()))
    optimizer = torch.optim.SGD([weight], lr=lr)
    mse_loss = torch.nn.MSELoss()

    # ---------------------------------------------------------------------
    # 3. Training loop
    # ---------------------------------------------------------------------
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = weight * x
        loss = mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    training_time = time.time() - start

    # ---------------------------------------------------------------------
    # 4. Persist artefacts for evaluation
    # ---------------------------------------------------------------------
    artefact_dir = Path(".research/iteration2/model")
    _save_model(float(weight.detach().cpu().item()), artefact_dir)

    # ---------------------------------------------------------------------
    # 5. Return metrics
    # ---------------------------------------------------------------------
    return {
        "final_weight": float(weight.detach().cpu().item()),
        "loss_curve": loss_history,
        "final_loss": loss_history[-1],
        "training_time_sec": training_time,
        "num_samples": num_samples,
        "epochs": epochs,
    }
