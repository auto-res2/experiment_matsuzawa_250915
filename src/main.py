"""main.py – entry point that orchestrates preprocessing, training and (optional)
hardware evaluation according to command-line flags."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "config"
SRC_PKG = "src"  # package root for -m invocation


def _run_python(module: str, cfg_path: Path) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(cfg_path)]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRaFT-GNN experiment runner")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="Run quick validation run")
    grp.add_argument(
        "--full-experiment", action="store_true", help="Run full publication-grade experiment"
    )
    args = parser.parse_args()

    cfg_file = CFG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")

    # 1) preprocessing – idempotent
    _run_python(f"{SRC_PKG}.preprocess", cfg_file)

    # 2) training / evaluation
    _run_python(f"{SRC_PKG}.train", cfg_file)

    # 3) hardware evaluation only for full runs
    if args.full_experiment:
        from src.train import run_hardware_eval  # lazy import to avoid heavy deps in smoke test

        run_hardware_eval(str(cfg_file))


if __name__ == "__main__":
    main()
