"""Main entry-point for the COSMIC-X placeholder experiment runner.

Highlights
==========
• CLI with *--smoke-test* and *--full-experiment* flags (exactly as required).
• Loads YAML configs via PyYAML ➜ dict.
• Runs: preprocess ➞ train ➞ evaluate.
• Saves JSON artefact under .research/iteration2/ and prints it to stdout so
  CI harness can parse numerical results.
• Strict *fail-fast* – any exception bubbles up; no silent fallbacks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import yaml

# local imports – all reside in src/ directory so Python path is already set
from preprocess import get_dataloaders  # noqa: E402
from train import train  # noqa: E402
from evaluate import accuracy  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & paths (created lazily when first needed)
# ---------------------------------------------------------------------------
_JSON_DIR = Path(".research/iteration2")
_IMG_DIR = Path(".research/iteration2/images")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.as_posix()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs() -> None:
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    _IMG_DIR.mkdir(parents=True, exist_ok=True)


def _dump_json(result: Dict, stem: str) -> Path:
    _ensure_dirs()
    out_path = _JSON_DIR / f"{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401 – imperative mood
    p = argparse.ArgumentParser(description="Run COSMIC-X placeholder experiment")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--smoke-test", action="store_true", help="run quick smoke test (default)")
    g.add_argument("--full-experiment", action="store_true", help="run full benchmark")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – imperative mood
    args = parse_args()

    # Default to smoke-test when neither flag was given.
    config_path = None
    if args.full_experiment:
        config_path = Path("config/full_experiment.yaml")
    else:  # smoke-test OR unspecified
        config_path = Path("config/smoke_test.yaml")

    cfg = _load_config(config_path)
    # ------------------------------------------------------------------
    # 1) Pre-processing / data loading
    # ------------------------------------------------------------------
    t0 = time.time()
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(cfg)
    prep_time = time.time() - t0

    # ------------------------------------------------------------------
    # 2) Training
    # ------------------------------------------------------------------
    model, train_history = train(train_loader, val_loader, input_dim, num_classes, cfg)

    # ------------------------------------------------------------------
    # 3) Evaluation
    # ------------------------------------------------------------------
    test_metrics = accuracy(model, test_loader, device=cfg.get("device", "cpu"))

    # ------------------------------------------------------------------
    # 4) Collate & persist results
    # ------------------------------------------------------------------
    result = {
        "config": config_path.name,
        "prep_seconds": prep_time,
        "train_history": train_history,
        "test_metrics": test_metrics,
        "timestamp": time.time(),
    }

    out_file = _dump_json(result, stem=config_path.stem)

    # stdout for CI verification
    print(json.dumps(result, indent=2))
    print(f"Saved results ➜ {out_file.as_posix()}")


if __name__ == "__main__":
    main()
