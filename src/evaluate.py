"""Evaluation utilities: BLEU, ROUGE-L, latency & energy plots.
All images are saved under .research/iteration2/images/ as required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

FIG_DPI = 120
sns.set_theme(style="whitegrid")

# Mandatory image directory
IMG_DIR = Path(".research/iteration2/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _bleu(hyp: List[str], ref: List[str]) -> float:
    return corpus_bleu(hyp, [ref]).score


def _rouge_l(hyp: List[str], ref: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(ref, hyp)]
    return float(np.mean(scores))


def compute_metrics(
    predictions: List[str], references: List[str], energy_mj: float, latency_ms: List[float]
) -> Dict[str, Any]:
    return {
        "BLEU": _bleu(predictions, references),
        "ROUGE_L": _rouge_l(predictions, references),
        "mean_latency_ms": float(np.mean(latency_ms)),
        "energy_mj": energy_mj,
    }


# -------------------------------------------------------------------------
# Plot helpers – save into mandated directory irrespective of *out_dir*
# -------------------------------------------------------------------------

def _annotate(ax):
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)


def plot_metrics(metrics: Dict[str, Any], out_dir: Path | None = None, tag: str | None = None):
    """Bar-plot metrics and pie-chart energy split.

    Parameters
    ----------
    metrics: Dict[str, Any]
        Metric dictionary as returned by :func:`compute_metrics`.
    out_dir: Path | None
        Ignored – kept for API compatibility. All images go to the global
        .research/iteration2/images directory per task rules.
    tag: str | None
        Optional run tag used in file names / titles.
    """
    tag = tag or "untagged"
    keys = [k for k in metrics if k != "energy_mj"]

    # Bar plot for standard metrics
    fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
    sns.barplot(x=keys, y=[metrics[k] for k in keys], ax=ax, palette="Set2")
    _annotate(ax)
    ax.set_title(f"Metrics – {tag}")
    fig.tight_layout()
    fig_path = IMG_DIR / f"metrics_{tag}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Pie chart for energy budget
    fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=FIG_DPI)
    ax2.pie([metrics["energy_mj"], 1], labels=["GPU/DRAM energy (mJ)", "Baseline"], autopct="%1.1f%%")
    fig2.tight_layout()
    fig2_path = IMG_DIR / f"energy_{tag}.pdf"
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

__all__ = ["compute_metrics", "plot_metrics"]