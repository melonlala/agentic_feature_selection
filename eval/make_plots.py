"""Generate summary plots from evaluation artifacts.

Reads CSV outputs from eval_offline.py and eval_online.py, plus ranking CSVs,
and produces the following plots (saved to output_dir):

1. return_vs_k.png         — avg online return vs number of features k
2. success_vs_k.png        — online success rate vs k
3. offline_accuracy_vs_k.png — offline test accuracy vs k
4. feature_importance_barplot.png — global SHAP feature importance
5. rank_heatmap.png        — (optional) heatmap of ranks across seeds

Uses matplotlib only.

Usage:
    python eval/make_plots.py \\
        --input_root outputs \\
        --output_dir outputs/plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir


SELECTOR_COLORS = {
    "shap":   "#2196F3",   # blue
    "oracle": "#4CAF50",   # green
    "full":   "#9C27B0",   # purple
    "random": "#FF9800",   # orange
    "mi":     "#F44336",   # red
}
SELECTOR_ORDER = ["shap", "oracle", "mi", "random", "full"]


def collect_online_metrics(input_root: Path) -> pd.DataFrame:
    """Recursively find and concatenate online_metrics.csv files."""
    frames = []
    for p in input_root.rglob("online_metrics.csv"):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_offline_metrics(input_root: Path) -> pd.DataFrame:
    """Recursively find and concatenate offline_metrics.csv files."""
    frames = []
    for p in input_root.rglob("offline_metrics.csv"):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_rankings(input_root: Path) -> list[pd.DataFrame]:
    """Find all ranking.csv files."""
    return [pd.read_csv(p) for p in input_root.rglob("ranking.csv")]


def plot_metric_vs_k(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    """Line plot of a metric vs k, one line per selector.

    Args:
        df: DataFrame with columns [selector, k, <metric_col>].
        metric_col: Column name of the metric to plot.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: File path to save PNG.
    """
    if df.empty or metric_col not in df.columns:
        print(f"[make_plots] Skipping {title}: no data.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    selectors = [s for s in SELECTOR_ORDER if s in df["selector"].unique()]
    selectors += [s for s in df["selector"].unique() if s not in SELECTOR_ORDER]

    for sel in selectors:
        sub = df[df["selector"] == sel].copy()
        # Average over seeds if multiple present
        sub_agg = sub.groupby("k")[metric_col].agg(["mean", "std"]).reset_index()
        color = SELECTOR_COLORS.get(sel, "gray")
        ax.plot(sub_agg["k"], sub_agg["mean"], marker="o", label=sel, color=color)
        if not sub_agg["std"].isna().all():
            ax.fill_between(
                sub_agg["k"],
                sub_agg["mean"] - sub_agg["std"],
                sub_agg["mean"] + sub_agg["std"],
                alpha=0.15,
                color=color,
            )

    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[make_plots] Saved {output_path}")


def plot_feature_importance(ranking_df: pd.DataFrame, output_path: str) -> None:
    """Bar plot of global SHAP feature importance.

    Args:
        ranking_df: DataFrame from ranking.csv (sorted by rank).
        output_path: File path to save PNG.
    """
    df = ranking_df.sort_values("rank").copy()
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.5), 4))

    colors = ["#2196F3" if not n.startswith("z_") else "#BDBDBD"
              for n in df["feature_name"]]
    ax.bar(df["feature_name"], df["mean_abs_shap"], color=colors)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Global SHAP Feature Importance")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[make_plots] Saved {output_path}")


def plot_rank_heatmap(ranking_dfs: list[pd.DataFrame], output_path: str) -> None:
    """Heatmap of feature ranks across multiple seeds.

    Args:
        ranking_dfs: List of ranking DataFrames from different seeds.
        output_path: File path to save PNG.
    """
    if len(ranking_dfs) < 2:
        return

    # Build aligned rank matrix
    all_names = sorted(set.union(*[set(df["feature_name"]) for df in ranking_dfs]))
    mat = np.full((len(all_names), len(ranking_dfs)), fill_value=np.nan)

    for j, df in enumerate(ranking_dfs):
        rank_map = dict(zip(df["feature_name"], df["rank"]))
        for i, name in enumerate(all_names):
            mat[i, j] = rank_map.get(name, np.nan)

    fig, ax = plt.subplots(figsize=(max(4, len(ranking_dfs) * 1.2), max(4, len(all_names) * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(ranking_dfs)))
    ax.set_xticklabels([f"seed{i}" for i in range(len(ranking_dfs))])
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names)
    ax.set_title("Feature Rank across Seeds (lower = more important)")
    plt.colorbar(im, ax=ax, label="Rank")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[make_plots] Saved {output_path}")


def run(args: argparse.Namespace) -> None:
    """Main plotting routine."""
    input_root = Path(args.input_root)
    out_dir = ensure_dir(args.output_dir)

    # --- Online metrics ---
    online_df = collect_online_metrics(input_root)
    plot_metric_vs_k(
        online_df, "avg_return", "Avg Episodic Return",
        "Online Return vs Number of Features",
        str(out_dir / "return_vs_k.png"),
    )
    plot_metric_vs_k(
        online_df, "success_rate", "Success Rate",
        "Online Success Rate vs Number of Features",
        str(out_dir / "success_vs_k.png"),
    )

    # --- Offline metrics ---
    offline_df = collect_offline_metrics(input_root)
    plot_metric_vs_k(
        offline_df, "accuracy", "Test Accuracy",
        "Offline Test Accuracy vs Number of Features",
        str(out_dir / "offline_accuracy_vs_k.png"),
    )

    # --- Feature importance ---
    ranking_dfs = collect_rankings(input_root)
    if ranking_dfs:
        # Use first ranking for bar plot (or aggregate)
        plot_feature_importance(ranking_dfs[0], str(out_dir / "feature_importance_barplot.png"))
        if len(ranking_dfs) >= 2:
            plot_rank_heatmap(ranking_dfs, str(out_dir / "rank_heatmap.png"))

    print(f"\n[make_plots] All plots saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment plots.")
    parser.add_argument("--input_root", required=True, help="Root of outputs directory.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
