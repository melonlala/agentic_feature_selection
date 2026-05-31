"""Rank stability analysis across multiple seeds.

Computes Spearman rank correlation (and optionally Kendall tau) between
feature rankings produced by different seeds. A high correlation indicates
that the SHAP-based ranking is stable and not an artefact of a particular
random sample.

Usage:
    python explain/rank_stability.py \\
        --ranking_paths outputs/rankings/taxi_noise8/seed0/ranking.csv \\
                        outputs/rankings/taxi_noise8/seed1/ranking.csv \\
                        outputs/rankings/taxi_noise8/seed2/ranking.csv \\
        --output_dir outputs/rankings/taxi_noise8/stability
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, load_csv, save_csv, save_json


def load_ranking_vector(csv_path: str) -> pd.Series:
    """Load ranking CSV and return a Series indexed by feature_name, values = rank.

    Args:
        csv_path: Path to ranking.csv from global_rank.py.

    Returns:
        Series: feature_name -> rank (1 = most important).
    """
    df = load_csv(csv_path)
    return df.set_index("feature_name")["rank"]


def run(args: argparse.Namespace) -> None:
    """Main stability analysis routine."""
    out_dir = ensure_dir(args.output_dir)

    paths = args.ranking_paths
    if len(paths) < 2:
        print("[rank_stability] Need at least 2 ranking files to compare. Exiting.")
        return

    rankings = {}
    for p in paths:
        label = Path(p).parent.name  # use parent dir name as label (e.g. seed0)
        rankings[label] = load_ranking_vector(p)

    labels = list(rankings.keys())

    # Align on common features
    common_features = sorted(
        set.intersection(*[set(r.index) for r in rankings.values()])
    )
    if len(common_features) == 0:
        raise RuntimeError("No common features found across ranking files.")

    aligned = pd.DataFrame({lbl: rankings[lbl].loc[common_features] for lbl in labels})

    # Pairwise correlations
    records = []
    for lbl_a, lbl_b in combinations(labels, 2):
        r_a = aligned[lbl_a].values
        r_b = aligned[lbl_b].values
        spear_r, spear_p = spearmanr(r_a, r_b)
        ktau, ktau_p = kendalltau(r_a, r_b)
        records.append({
            "seed_a": lbl_a,
            "seed_b": lbl_b,
            "spearman_r": float(spear_r),
            "spearman_p": float(spear_p),
            "kendall_tau": float(ktau),
            "kendall_p": float(ktau_p),
        })
        print(f"  {lbl_a} vs {lbl_b}: Spearman r={spear_r:.3f} (p={spear_p:.3g}), "
              f"Kendall tau={ktau:.3f} (p={ktau_p:.3g})")

    df_pairs = pd.DataFrame(records)
    save_csv(df_pairs, str(out_dir / "pairwise_correlations.csv"))

    # Summary statistics
    summary = {
        "n_seeds": len(labels),
        "n_common_features": len(common_features),
        "mean_spearman_r": float(df_pairs["spearman_r"].mean()),
        "std_spearman_r": float(df_pairs["spearman_r"].std()),
        "mean_kendall_tau": float(df_pairs["kendall_tau"].mean()),
        "std_kendall_tau": float(df_pairs["kendall_tau"].std()),
        "seeds": labels,
        "common_features": common_features,
    }
    save_json(summary, str(out_dir / "stability_summary.json"))

    # Also save the aligned rank matrix
    save_csv(aligned.reset_index(), str(out_dir / "aligned_ranks.csv"))

    print(f"\n[rank_stability] Mean Spearman r = {summary['mean_spearman_r']:.3f} "
          f"± {summary['std_spearman_r']:.3f}")
    print(f"[rank_stability] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank stability analysis across seeds.")
    parser.add_argument(
        "--ranking_paths", nargs="+", required=True,
        help="Paths to ranking.csv files from different seeds.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
