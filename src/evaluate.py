"""evaluate.py – aggregate *_metrics.json result files & generate a summary plot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _aggregate(files: List[Path]) -> Dict[str, Tuple[float, float]]:
    """Aggregate multiple JSON files returned by train.py.

    Returns a mapping metric → (mean, std).
    """
    all_test = [json.load(open(f))["test"] for f in files]
    keys = all_test[0].keys()
    return {
        k: (
            float(np.mean([d[k] for d in all_test])),
            float(np.std([d[k] for d in all_test])),
        )
        for k in keys
    }


def _print_and_plot(stats: Dict[str, Tuple[float, float]], n_runs: int) -> None:
    print("=== Aggregate Test Metrics ===")
    for k, (m, s) in stats.items():
        ci = 1.96 * s / np.sqrt(n_runs)
        print(f"{k}: {m:.4f} ± {ci:.4f}")

    # ---------- bar plot --------------------------------------------------
    images_dir = Path(".research/iteration4/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    sns.set()
    plt.figure(figsize=(6, 4))
    keys = list(stats.keys())
    means = [stats[k][0] for k in keys]
    sns.barplot(x=keys, y=means, palette="viridis")
    for i, m in enumerate(means):
        plt.text(i, m, f"{m:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(images_dir / "summary_metrics.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="Directory with *_metrics.json files")
    args = ap.parse_args()

    res_dir = Path(args.result_dir)
    files = sorted(res_dir.glob("seed*/metrics.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {res_dir}")

    stats = _aggregate(files)
    _print_and_plot(stats, n_runs=len(files))


if __name__ == "__main__":
    main()
