"""Entry-point – orchestrates smoke vs. full experiments."""

import argparse, yaml
from pathlib import Path
import torch

from .train import run_experiment_1
from .evaluate import generate_figures


# ─────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────

def _load_cfg(smoke: bool) -> dict:
    cfg_file = "config/smoke_test.yaml" if smoke else "config/full_experiment.yaml"
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run R³ EdgeBench experiments")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="quick correctness run")
    g.add_argument("--full-experiment", action="store_true", help="run the full benchmark suite")
    args = parser.parse_args()

    cfg = _load_cfg(args.smoke_test)

    # Output folders
    results_dir = Path(cfg["general"]["results_dir"])
    figures_dir = Path(cfg["general"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    if cfg.get("experiment_1", {}).get("enabled", False):
        print("\n=== Experiment 1 – Variable-Rate Replay vs. Baselines ===")
        run_experiment_1(cfg["experiment_1"], device, results_dir)

    # (Exp-2 & Exp-3 omitted – refer to full code base in the paper repository.)

    # ---------------------------------------------------------
    print("\n=== Generating figures ===")
    generate_figures(results_dir, figures_dir)
    print(f"All done. Figures saved to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
