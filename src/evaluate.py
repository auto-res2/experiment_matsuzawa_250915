import json
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score

from .train import Trainer


def evaluate(cfg: Dict, trainer: Trainer):
    """Run evaluation on the validation split and write results to the mandated
    research folder structure (JSON). The function prints the JSON contents to
    STDOUT so the CI runner can capture and parse the metrics immediately.
    """
    val_loader = trainer.val_loader
    device = trainer.device
    model = trainer.model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    results = {"accuracy": acc}

    # ------------------------------------------------------------------
    # Persist strictly to the required path layout
    # ------------------------------------------------------------------
    out_dir = Path(".research/iteration5")  # ← mandatory directory
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name", "experiment")
    out_path = out_dir / f"{run_name}.json"

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Echo for log-parsing
    print(json.dumps(results, indent=2))

    return results
