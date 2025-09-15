"""Evaluation utilities.

The previous stub returned constant zeros. This revision loads the model
trained by ``src.train`` and computes *real* loss & accuracy numbers on
one held-out validation split, then stores them under
`.research/iteration3/` as required by the spec.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

ITER_DIR = Path(".research/iteration3")
RESULT_PATH = ITER_DIR / "evaluation_results.json"


def _load_dataset() -> Tuple[DataLoader, int]:
    """Load the *validation* tensor dataset prepared earlier."""
    val_tensors_path = ITER_DIR / "val_data.pt"
    if not val_tensors_path.exists():
        logger.error("Validation tensor %s not found – preprocessing step missing?", val_tensors_path)
        sys.exit(1)
    val_x, val_y = torch.load(val_tensors_path)
    input_dim = val_x.shape[1]
    loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False)
    return loader, input_dim


def evaluate(cfg: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Compute loss & accuracy of the trained model on the held-out split."""
    logger.info("Starting evaluation …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader, input_dim = _load_dataset()

    model = nn.Linear(input_dim, 2).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError as err:
        logger.error("Model checkpoint %s not found: %s", model_path, err)
        sys.exit(1)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * x_batch.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0

    results: Dict[str, Any] = {
        "val_loss": round(val_loss, 4),
        "val_accuracy": round(val_acc, 4),
    }

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved evaluation results to %s", RESULT_PATH)
    print(json.dumps(results, indent=2))

    return results