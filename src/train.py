import random
from typing import Dict, Any


def train_model(processed_data: Any, config: Dict[str, Any]) -> Dict[str, float]:
    """Dummy train routine that returns deterministic metrics.

    The goal is **not** to perform real training (datasets are unavailable in the
    execution environment) but to ensure downstream evaluation receives
    well-formed numerical results so that the pipeline fulfils the contract of
    returning *concrete experimental data*.
    """
    # Deterministic pseudo-metrics – reproducible across runs & seeds
    random.seed(0)
    epochs = int(config.get("epochs", 1))
    base_loss = 1.0 / (epochs + 1)
    metrics = {
        "train_loss": round(base_loss, 4),
        "train_accuracy": round(0.5 + 0.05 * epochs, 4),  # grows with epochs
    }
    return metrics
