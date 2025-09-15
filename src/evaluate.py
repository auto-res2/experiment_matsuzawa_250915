"""Evaluation & plotting utilities (stub)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def evaluate(cfg: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Evaluate a trained model (stub)."""
    logger.warning("evaluate() was called, but no evaluation logic is implemented.")

    # Dummy metrics
    results = {
        "val_loss": 0.0,
        "val_accuracy": 0.0,
    }

    # Persist results so that CI / smoke-tests can at least verify I/O.
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "evaluation_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved evaluation results to %s", result_path)
    print(json.dumps(results, indent=2))
    return results