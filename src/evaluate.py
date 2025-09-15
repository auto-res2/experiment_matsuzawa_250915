"""Evaluation & plotting utilities (stub)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def evaluate(cfg: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Evaluate a trained model (stub).

    This implementation only returns dummy metrics so that the CI / smoke-test
    pipeline can verify I/O without running any heavy computation. All
    artefacts are stored under `.research/iteration2/` per the mandatory
    directory-layout requirements.
    """
    logger.warning("evaluate() was called, but no evaluation logic is implemented.")

    # ------------------------------------------------------------------
    # Dummy metrics – replace with real numbers once a model exists.
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "val_loss": 0.0,
        "val_accuracy": 0.0,
    }

    # ------------------------------------------------------------------
    # Persist results to disk so that downstream jobs (e.g. GitHub Actions)
    # can inspect them. We also echo them to stdout for quick debugging.
    # ------------------------------------------------------------------
    out_dir = Path(".research/iteration2")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "evaluation_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved evaluation results to %s", result_path)
    # Print to STDOUT for verification in log output.
    print(json.dumps(results, indent=2))
    return results