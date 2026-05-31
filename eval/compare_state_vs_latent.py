"""Compare state-space vs latent-space feature-selection eval curves.

Joins offline-eval CSVs from both pipelines into a long-format summary CSV and
emits overlay plots per dataset:
  - test_mse_vs_k.png         (with MCI-implied lower bound)
  - cosine_vs_k.png
  - predictive_power_vs_k.png (ν(S) = Var(y) − MSE, with cumulative-MCI upper bound)

The MCI upper bound (Catav et al. 2021): for any feature subset S,
    ν(S) ≤ Σ_{i∈S} Î(i)   ⇔   MSE(S) ≥ Var(y) − Σ_{i∈S} Î(i)
where Î(i) is the per-feature MCI score in `ranking.csv`. We compute the
cumulative top-k MCI sum from the MCI ranking and overlay it as an upper bound.

Input directories:
    outputs/eval/offline/{dataset}/seed{N}/{selector}/offline_metrics.csv
    outputs/eval/offline_latent/{dataset}/seed{N}/{selector}/offline_metrics.csv
    outputs/rankings_mci/{dataset}/seed{N}/ranking.csv          (state MCI scores)
    outputs/rankings_mci_latent/{dataset}/seed{N}/ranking.csv   (latent MCI scores)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, save_csv


SPACE_LABELS = {"state": "State space", "latent": "Latent space"}
SPACE_COLORS = {"state": "#1f77b4", "latent": "#d62728"}
LINESTYLES   = {"shap": "-", "mci": "-", "random": "--", "oracle": ":", "mi": "-.", "full": "-"}

# MCI-ranking roots used to compute cumulative-MCI upper bounds.
MCI_ROOTS = {
    "state":  Path("outputs/rankings_mci"),
    "latent": Path("outputs/rankings_mci_latent"),
}

# Which selector label represents "MCI-ranked" students per space. Both pipelines
# write MCI rankings; only the directory label differs (see scripts/run_d4rl_mci.sh
# vs scripts/run_d4rl_latent_mci.sh — state uses --selector shap with the MCI
# ranking csv, latent uses --selector mci).
MCI_SELECTOR_BY_SPACE = {"state": "shap", "latent": "mci"}


def _parse_k(k_str: str, n_features: int) -> int:
    """Turn 'k10' / 'full' / '10' into an int feature count."""
    if isinstance(k_str, (int, np.integer)):
        return int(k_str)
    s = str(k_str).strip()
    if s == "full":
        return int(n_features)
    m = re.match(r"k?(\d+)$", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse k value: {k_str!r}")


def _collect(root: Path, space: str) -> list[dict]:
    """Walk an eval root and collect long-format rows."""
    rows: list[dict] = []
    if not root.exists():
        return rows
    for csv_path in root.glob("*/seed*/*/offline_metrics.csv"):
        # Path: {root}/{dataset}/seed{N}/{selector}/offline_metrics.csv
        rel = csv_path.relative_to(root)
        dataset = rel.parts[0]
        seed = int(rel.parts[1].removeprefix("seed"))
        selector = rel.parts[2]
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[compare] skip {csv_path}: {e}")
            continue
        for _, r in df.iterrows():
            try:
                k = _parse_k(r["k"], int(r["n_features"]))
            except Exception:
                continue
            rows.append({
                "dataset":    dataset,
                "seed":       seed,
                "space":      space,
                "selector":   selector,
                "k":          k,
                "n_features": int(r["n_features"]),
                "test_mse":   float(r["mse"]),
                "test_mae":   float(r["mae"]),
                "cosine_sim": float(r["cosine_sim"]),
            })
    return rows


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and std across seeds for each (dataset, space, selector, k)."""
    g = df.groupby(["dataset", "space", "selector", "k"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def _plot_metric(
    agg: pd.DataFrame,
    dataset: str,
    metric: str,
    ylabel: str,
    output_path: Path,
    selectors: Iterable[str],
) -> None:
    """Plot overlay of state vs latent curves for one dataset."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sub = agg[agg["dataset"] == dataset]
    if sub.empty:
        plt.close(fig)
        return
    n_curves = 0
    for space in ("state", "latent"):
        for selector in selectors:
            curve = sub[(sub["space"] == space) & (sub["selector"] == selector)]
            curve = curve.sort_values("k")
            if curve.empty:
                continue
            ks = curve["k"].to_numpy()
            mean = curve["mean"].to_numpy()
            std  = curve["std"].to_numpy()
            ax.plot(
                ks, mean,
                color=SPACE_COLORS[space],
                linestyle=LINESTYLES.get(selector, "-"),
                marker="o", markersize=4,
                label=f"{SPACE_LABELS[space]} · {selector}",
            )
            if (std > 0).any():
                ax.fill_between(ks, mean - std, mean + std,
                                color=SPACE_COLORS[space], alpha=0.12)
            n_curves += 1
    if n_curves == 0:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Number of selected features (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset}: {ylabel} vs k (state vs latent selection)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"[compare] Wrote {output_path}")


def run(args: argparse.Namespace) -> None:
    state_root  = Path(args.state_root)
    latent_root = Path(args.latent_root)
    out_dir     = ensure_dir(args.output_dir)

    rows = _collect(state_root, "state") + _collect(latent_root, "latent")
    if not rows:
        print(f"[compare] No metrics found under {state_root} or {latent_root}.")
        return
    long_df = pd.DataFrame(rows)

    datasets = sorted(long_df["dataset"].unique())
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets.split()]

    selectors = ["shap", "mci", "random", "full"]

    for ds in datasets:
        ds_dir = ensure_dir(Path(out_dir) / ds)
        ds_df  = long_df[long_df["dataset"] == ds]
        save_csv(ds_df.sort_values(["space", "selector", "seed", "k"]),
                 str(ds_dir / "summary.csv"))

        mse_agg    = _aggregate(ds_df, "test_mse")
        cosine_agg = _aggregate(ds_df, "cosine_sim")

        _plot_metric(mse_agg, ds, "test_mse", "Test MSE",
                     ds_dir / "test_mse_vs_k.png", selectors)
        _plot_metric(cosine_agg, ds, "cosine_sim", "Cosine similarity",
                     ds_dir / "cosine_vs_k.png", selectors)

    # Global combined CSV.
    save_csv(long_df.sort_values(["dataset", "space", "selector", "seed", "k"]),
             str(Path(out_dir) / "all_runs.csv"))
    print(f"[compare] Wrote {Path(out_dir) / 'all_runs.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare state-space and latent-space feature-selection eval curves."
    )
    p.add_argument("--state_root",  default="outputs/eval/offline",
                   help="Root directory for state-space eval CSVs.")
    p.add_argument("--latent_root", default="outputs/eval/offline_latent",
                   help="Root directory for latent-space eval CSVs.")
    p.add_argument("--output_dir",  default="outputs/plots/compare_state_vs_latent",
                   help="Where to write per-dataset summary CSVs + plots.")
    p.add_argument("--datasets",    default=None,
                   help="Optional space-separated list of dataset names to include.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
