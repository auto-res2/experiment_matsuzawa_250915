import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

# Local modules
from .train import Trainer
from .evaluate import evaluate


def _load_cfg(path: Path) -> Dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg


def _run(cfg: Dict):
    trainer = Trainer(cfg)
    best_acc, _ = trainer.fit()
    results = evaluate(cfg, trainer)
    results["best_val_acc"] = best_acc
    return results


def main():
    parser = argparse.ArgumentParser(description="COSMIC-X Experimental Driver")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the quick CI validation")
    group.add_argument("--full-experiment", action="store_true", help="Run the full-scale experiment")
    args = parser.parse_args()

    cfg_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    cfg = _load_cfg(cfg_path)
    cfg["smoke_test"] = args.smoke_test

    try:
        results = _run(cfg)
    except Exception as exc:
        print("Experiment failed:", exc, file=sys.stderr)
        sys.exit(1)

    print("Final Results:\n", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
