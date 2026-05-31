"""Plot MCI score bar charts for each dataset/seed ranking.

For each ranking.csv found under --rankings_root, produces one bar chart
showing mean_mci per feature (log-scale y-axis), ordered by rank.
Features with extremely dominant scores get a broken-axis treatment via
a secondary inset so the rest of the distribution remains readable.

Usage:
    python eval/plot_mci_bars.py \
        --rankings_root outputs/rankings_mci \
        --output_dir outputs/plots/mci_bars \
        [--top_n 30]
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Color palette per dataset
DATASET_COLORS = {
    "kitchen_complete": "#2166ac",
    "pen_human":        "#d6604d",
    "pen_expert":       "#4dac26",
    "pen_cloned":       "#8073ac",
    "taxi_noise8":      "#f4a582",
}
DEFAULT_COLOR = "#555555"


def _feature_group(name: str) -> str:
    """Return a coarse feature group label for coloring within a dataset."""
    prefixes = [
        "hand_qpos", "qpos", "qvel",
        "obj2goal_rot", "obj2goal_pos", "obj2goal",
        "obj_euler", "obj_pos", "obj",
        "goal_euler", "goal_pos", "goal",
        "contact",
    ]
    for p in prefixes:
        if name.startswith(p):
            return p
    return "other"


# Assign a stable color per group
_GROUP_COLORS = {
    "hand_qpos":    "#2166ac",
    "qpos":         "#4393c3",
    "qvel":         "#92c5de",
    "obj":          "#d6604d",
    "obj_pos":      "#f4a582",
    "obj_euler":    "#fddbc7",
    "obj2goal":     "#4dac26",
    "obj2goal_pos": "#b8e186",
    "obj2goal_rot": "#7fbc41",
    "goal":         "#762a83",
    "goal_pos":     "#af8dc3",
    "goal_euler":   "#e7d4e8",
    "contact":      "#f1a340",
    "other":        "#888888",
}


def bar_colors(names: list[str]) -> list[str]:
    return [_GROUP_COLORS.get(_feature_group(n), "#888888") for n in names]


def plot_mci_bar(df: pd.DataFrame, dataset: str, seed: int,
                 output_path: str, top_n: int = 30) -> None:
    df = df.sort_values("rank").head(top_n).reset_index(drop=True)
    names  = df["feature_name"].tolist()
    scores = df["mean_mci"].values.astype(float)
    colors = bar_colors(names)

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.38), 5))

    x = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.4)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean MCI Score (log scale)", fontsize=10)
    ax.set_title(f"MCI Feature Ranking — {dataset}  (seed {seed}, top {top_n})",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate rank numbers on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                score * 1.15, f"#{i+1}",
                ha="center", va="bottom", fontsize=6, color="#333333")

    # Legend for feature groups present in this plot
    present_groups = sorted(set(_feature_group(n) for n in names))
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1,
                       fc=_GROUP_COLORS.get(g, "#888888"), ec="white")
        for g in present_groups
    ]
    ax.legend(legend_patches, present_groups,
              title="Feature group", fontsize=7, title_fontsize=8,
              loc="upper right", framealpha=0.8, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {output_path}")


def plot_score_distribution(df: pd.DataFrame, dataset: str, seed: int,
                             output_path: str) -> None:
    """CDF of MCI scores to show how skewed the distribution is."""
    scores = np.sort(df["mean_mci"].values.astype(float))[::-1]
    cumsum = np.cumsum(scores) / scores.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(scores) + 1), cumsum,
            color=DATASET_COLORS.get(dataset, DEFAULT_COLOR), lw=2)
    ax.axhline(80, color="gray", linestyle="--", lw=1, label="80%")
    ax.axhline(95, color="gray", linestyle=":",  lw=1, label="95%")
    ax.set_xlabel("Number of top features", fontsize=10)
    ax.set_ylabel("Cumulative MCI score (%)", fontsize=10)
    ax.set_title(f"MCI Score Concentration — {dataset}  (seed {seed})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate 80% / 95% crossings
    for pct, ls in [(80, "--"), (95, ":")]:
        idx = np.searchsorted(cumsum, pct)
        if idx < len(scores):
            ax.axvline(idx + 1, color="gray", linestyle=ls, lw=1)
            ax.text(idx + 1.3, pct - 5, f"k={idx+1}", fontsize=8, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MCI score bar charts.")
    parser.add_argument("--rankings_root", default="outputs/rankings_mci",
                        help="Root directory containing dataset/seed/ranking.csv")
    parser.add_argument("--output_dir", default="outputs/plots/mci_bars",
                        help="Directory for output PNG files")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of top features to show in bar chart")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Restrict to specific dataset names")
    args = parser.parse_args()

    root = Path(args.rankings_root)
    if not root.exists():
        raise FileNotFoundError(f"Rankings root not found: {root}")

    ranking_files = sorted(root.glob("*/seed*/ranking.csv"))
    if not ranking_files:
        raise FileNotFoundError(f"No ranking.csv files found under {root}")

    for csv_path in ranking_files:
        dataset = csv_path.parts[-3]
        seed    = int(csv_path.parts[-2].replace("seed", ""))

        if args.datasets and dataset not in args.datasets:
            continue

        print(f"\n{dataset}  seed={seed}")
        df = pd.read_csv(csv_path)

        out_dir = Path(args.output_dir) / dataset / f"seed{seed}"

        # Bar chart
        plot_mci_bar(
            df, dataset, seed,
            output_path=str(out_dir / "mci_bar.png"),
            top_n=min(args.top_n, len(df)),
        )

        # CDF / concentration plot
        plot_score_distribution(
            df, dataset, seed,
            output_path=str(out_dir / "mci_concentration.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
