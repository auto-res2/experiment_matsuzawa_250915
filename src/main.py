"""Main orchestration script.

Supports two CLI flags:
  --smoke-test       Run a lightweight configuration defined in config/smoke_test.yaml
  --full-experiment  Run the full configuration in config/full_experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Local imports must be absolute for the -m flag
from src.preprocess import prepare_datasets
from src.train import TrainResult, train
from src.evaluate import evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Config paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT_DIR / "config"

SMOKE_CFG = CFG_DIR / "smoke_test.yaml"
FULL_CFG = CFG_DIR / "full_experiment.yaml"


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiment workflow")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full-scale experiment")

    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into memory."""
    if not path.exists():
        logger.error("Config file %s not found", path)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------

def run(cfg: Dict[str, Any]) -> None:  # noqa: D401
    """Run the full workflow: preprocessing → training → evaluation."""
    # 1. Pre-processing (idempotent)
    prepare_datasets(cfg)

    # 2. Training
    train_result: TrainResult = train(cfg)

    # 3. Evaluation
    _ = evaluate(cfg, train_result.model_path)


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = parse_args()

    cfg_path = SMOKE_CFG if args.smoke_test else FULL_CFG
    cfg = load_config(cfg_path)

    # Inject optional environment variables (e.g. HF_TOKEN) into config
    cfg["env"] = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    }

    try:
        run(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Experiment failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()