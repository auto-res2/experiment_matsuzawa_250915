"""Evaluation & visualisation helpers for R³ experiments."""

from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ["generate_figures"]


def _annotate(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:0.1f}", (p.get_x() + p.get_width() / 2, height), ha="center", va="bottom", fontsize=8)


def plot_exp1_bar(df: pd.DataFrame, fig_dir: Path):
    plt.figure(figsize=(6, 3))
    ax = sns.barplot(data=df, x="variant", y="AA", hue="memory_kb", palette="Set2")
    _annotate(ax)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_xlabel("")
    ax.legend(title="Memory (kB)")
    plt.tight_layout()
    out = fig_dir / "accuracy_memory.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved figure {out}")


def generate_figures(results_dir: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    files = list(results_dir.glob("*.json"))
    if not files:
        print("No result JSON found – nothing to plot.")
        return

    records = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame.from_records(records)
    plot_exp1_bar(df[df.variant.isin(["R3", "R3_NoVR", "ER_JPEG"])], figures_dir)
