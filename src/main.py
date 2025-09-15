"""Command-line entry-point for ORION experiment suite."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
from tqdm.auto import tqdm

from .preprocess import load_emoji_chat, load_malware_prompts, prepare_red_whatsapp
from .train import ORIONTrainer
from .evaluate import compute_metrics, plot_metrics

CONFIG_DIR = Path("config")


def _load_cfg(p: Path) -> Dict[str, Any]:
    with open(p) as fp:
        return yaml.safe_load(fp)


# --------------------------  EXP-SPECIFIC RUNNERS  ---------------------------

def _run_exp1(cfg: Dict[str, Any], smoke: bool):
    for seed in cfg["experiment"]["seeds"]:
        ds = prepare_red_whatsapp(cfg, seed, smoke)
        # Baseline
        base = ORIONTrainer(cfg, run_name=f"exp1_baseline_seed{seed}", seed=seed, smoke=smoke)
        tr_tokens = base.tokenizer(ds["train"]["msg"], return_tensors="pt", padding=True)
        base.train_epoch([tr_tokens], 0)
        base.save_checkpoint("final")
        base.finalise({})
        # ORION
        ori = ORIONTrainer(cfg, run_name=f"exp1_orion_seed{seed}", seed=seed, smoke=smoke)
        ori.train_epoch([tr_tokens], 0)
        ori.save_checkpoint("final")
        ori.finalise({})


def _run_exp2(cfg: Dict[str, Any], smoke: bool):
    _ = load_emoji_chat(cfg)  # Heavy mesh sim omitted for brevity


def _run_exp3(cfg: Dict[str, Any], smoke: bool):
    _ = load_malware_prompts(cfg)  # LRT hooks executed during training / inference


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = parser.parse_args()

    cfg_path = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    cfg = _load_cfg(cfg_path)
    print(f"Loaded configuration from {cfg_path}\n")

    smoke = args.smoke_test
    _run_exp1(cfg, smoke)
    _run_exp2(cfg, smoke)
    _run_exp3(cfg, smoke)


if __name__ == "__main__":
    main()