from typing import Dict, Any


def evaluate_model(model_artifact: Any, processed_data: Any, config: Dict[str, Any]) -> Dict[str, float]:
    """Dummy evaluation – mirrors train_model for demonstration purposes."""
    # fabricate deterministic evaluation metrics
    epochs = int(config.get("epochs", 1))
    eval_metrics = {
        "eval_loss": round(1.2 / (epochs + 1), 4),
        "eval_accuracy": round(0.45 + 0.04 * epochs, 4),
    }
    return eval_metrics
