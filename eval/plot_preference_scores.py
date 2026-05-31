"""Plot the score (fragment-return) distribution of an offline preference dataset.

Each preference comparison produced by ``teacher/collect_ant_preferences.py`` is
labelled with ``return_a`` / ``return_b`` — the discounted ground-truth return of
the two fragments. The "score" of a fragment is that return. This script reads the
per-source ``returns_{source}.npz`` files under a preference run directory and draws
a histogram (distribution bars) of the pooled fragment scores, one panel per source.

Usage:
    python eval/plot_preference_scores.py \\
        --input_dir outputs/preferences/seals_ant/seed0 \\
        --output_dir outputs/plots

    # Pool several seeds into one figure:
    python eval/plot_preference_scores.py \\
        --input_dir outputs/preferences/seals_ant/seed0 \\
                    outputs/preferences/seals_ant/seed1 \\
                    outputs/preferences/seals_ant/seed2 \\
        --output_dir outputs/plots --tag seals_ant_all_seeds
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir

SOURCE_ORDER = ["expert", "noisy", "random"]
SOURCE_COLOR = {"expert": "#2c7fb8", "noisy": "#fec44f", "random": "#cb181d"}


def load_scores(input_dirs: "list[str]") -> "dict[str, np.ndarray]":
    """Pool fragment scores (return_a + return_b) by source across input dirs."""
    by_source: "dict[str, list[np.ndarray]]" = {}
    for d in input_dirs:
        for npz_path in sorted(glob.glob(os.path.join(d, "returns_*.npz"))):
            source = Path(npz_path).stem.replace("returns_", "")
            data = np.load(npz_path)
            scores = np.concatenate([data["return_a"], data["return_b"]])
            by_source.setdefault(source, []).append(scores)
    return {s: np.concatenate(v) for s, v in by_source.items()}


def plot(by_source: "dict[str, np.ndarray]", output_path: str, bins: int) -> None:
    sources = [s for s in SOURCE_ORDER if s in by_source]
    sources += [s for s in by_source if s not in sources]

    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    # Shared x-range so panels are visually comparable.
    all_scores = np.concatenate(list(by_source.values()))
    lo, hi = float(all_scores.min()), float(all_scores.max())
    bin_edges = np.linspace(lo, hi, bins + 1)

    for ax, source in zip(axes, sources):
        scores = by_source[source]
        color = SOURCE_COLOR.get(source, "#636363")
        ax.hist(scores, bins=bin_edges, color=color, edgecolor="black", linewidth=0.3)
        mean = float(scores.mean())
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={mean:.1f}")
        ax.set_title(f"{source}  (n={len(scores)})")
        ax.set_xlabel("fragment discounted return (score)")
        ax.set_ylabel("count")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Offline preference dataset — fragment score distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", nargs="+", required=True,
                   help="One or more preference run dirs containing returns_*.npz")
    p.add_argument("--output_dir", default="outputs/plots")
    p.add_argument("--tag", default=None,
                   help="Filename stem; defaults to the first input dir's name.")
    p.add_argument("--bins", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    by_source = load_scores(args.input_dir)
    if not by_source:
        raise SystemExit(f"No returns_*.npz found under: {args.input_dir}")

    out_dir = ensure_dir(args.output_dir)
    tag = args.tag or Path(args.input_dir[0].rstrip("/")).name
    out_path = str(out_dir / f"preference_scores_{tag}.png")
    plot(by_source, out_path, bins=args.bins)

    for source, scores in by_source.items():
        print(f"[plot_pref_scores] {source:7s} n={len(scores):5d} "
              f"min={scores.min():7.2f} max={scores.max():7.2f} "
              f"mean={scores.mean():7.2f} std={scores.std():6.2f}")
    print(f"[plot_pref_scores] wrote {out_path}")


if __name__ == "__main__":
    main()
