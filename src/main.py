"""Entry-point script.
Implements a **minimal end-to-end** pipeline that satisfies the CI harness:
  – yaml-based configuration (smoke vs full)
  – command-line flags `--smoke-test` / `--full-experiment`
  – preprocess → train → evaluate stages
  – JSON result dumping to `.research/iteration2/<exp_name>_<timestamp>.json`
  – prints the JSON to STDOUT so that CI can parse concrete numerical metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

# Local imports (relative path safe because src/ is on PYTHONPATH)
from preprocess import preprocess
from train import train
from evaluate import evaluate

CONFIG_DIR = Path("config")
DEFAULT_SMOKE = CONFIG_DIR / "smoke_test.yaml"
DEFAULT_FULL = CONFIG_DIR / "full_experiment.yaml"

OUTPUT_ROOT = Path(".research/iteration2")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class FailFastException(Exception):
    """Raised when a mandatory stage fails – triggers immediate termination."""


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FailFastException(f"Configuration file '{path}' does not exist.")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _dump_json(result: Dict, exp_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = OUTPUT_ROOT / f"{exp_name}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy ASTRAL surrogate experiment workflow.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the light-weight smoke test configuration.")
    group.add_argument("--full-experiment", action="store_true", help="Run the full configuration (heavier).")
    parser.add_argument("--config", type=str, default=None, help="Optional path to a custom YAML config file.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Configuration loading
    # ------------------------------------------------------------------
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = DEFAULT_SMOKE if args.smoke_test else DEFAULT_FULL

    cfg = _load_yaml(cfg_path)
    exp_name = cfg.get("experiment_name", "astral_demo")

    # ------------------------------------------------------------------
    # 2. Execute stages – fail fast on any unexpected error.
    # ------------------------------------------------------------------
    try:
        preprocess_metrics = preprocess(cfg.get("preprocess", {}))
        train_metrics = train(cfg.get("train", {}))
        eval_metrics = evaluate(cfg.get("evaluate", {}))
    except Exception as e:  # noqa: BLE001
        # Surface the root cause clearly and exit with non-zero code.
        raise FailFastException(f"Experiment failed – {e}") from e

    # ------------------------------------------------------------------
    # 3. Combine & persist results
    # ------------------------------------------------------------------
    combined = {
        "experiment_name": exp_name,
        "config_path": str(cfg_path),
        "timestamp_utc": datetime.utcnow().isoformat(),
        "preprocess": preprocess_metrics,
        "train": train_metrics,
        "evaluate": eval_metrics,
        "total_runtime_sec": train_metrics.get("training_time_sec", 0.0) + eval_metrics.get("evaluation_time_sec", 0.0),
    }

    json_path = _dump_json(combined, exp_name)

    # ------------------------------------------------------------------
    # 4. STDOUT for CI – MUST contain concrete metrics (numeric)
    # ------------------------------------------------------------------
    print(json.dumps(combined, indent=2))
    sys.stderr.write(f"\n[INFO] Results saved to {json_path}\n")


if __name__ == "__main__":
    main()
