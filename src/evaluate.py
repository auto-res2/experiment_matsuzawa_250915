"""evaluate.py – aggregate JSON result files & produce a concise summary plot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _aggregate(files: List[Path]) -> Dict[str, Tuple[float, float]]:
    all_test = [json.load(open(f))["test"] for f in files]
    keys = all_test[0].keys()
    return {k: (float(np.mean([d[k] for d in all_test])), float(np.std([d[k] for d in all_test]))) for k in keys}


def _print_and_plot(stats: Dict[str, Tuple[float, float]], out_dir: Path):
    print("=== Aggregate Test Metrics ===")
    for k, (m, s) in stats.items():
        print(f"{k}: {m:.4f} ± {1.96 * s / np.sqrt(len(stats)):.4f}")
    # bar plot
    sns.set()
    plt.figure(figsize=(6, 4))
    keys = list(stats.keys())
    means = [stats[k][0] for k in keys]
    sns.barplot(x=keys, y=means)
    for i, m in enumerate(means):
        plt.text(i, m, f"{m:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_metrics.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="Directory containing *_metrics.json files")
    args = ap.parse_args()
    res_dir = Path(args.result_dir)
    files = sorted(res_dir.glob("seed*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {res_dir}")
    stats = _aggregate(files)
    _print_and_plot(stats, res_dir)


if __name__ == "__main__":
    main()
