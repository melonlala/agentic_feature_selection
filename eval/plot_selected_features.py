"""Visualize feature importance per selector for kitchen and seals_ant.

Produces one figure per environment:
  outputs/plots/selected_features/kitchen_complete.png
  outputs/plots/selected_features/seals_ant.png

Each figure has one horizontal-bar subplot per selector. The y-axis is shared
across all subplots: every feature appears at the same row in the same order
(feature_index 0 at the top, last feature at the bottom).

Kitchen panels follow the seed=1 online-eval log (`mci_nn`, `shap` aliased to
`mci_hdc`, `random`, `oracle`, `mi`, `full`). Because not every selector has a
continuous score (`random`/`oracle`/`mi`/`full` only emit selected sets), we
plot "selection priority": the height of each bar is the number of k thresholds
in `[5, 10, 20, 30, 45, 60]` at which the feature is included. A feature first
selected at k=5 gets the tallest bar (6); a feature only included in k=60 gets
the shortest (1). For `full`, only k=60 was run, so every bar has height 1.

Seals_ant panels still show the ranker-score x-axis directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs/plots/selected_features"
RANDOM_K = 15

# seals_ant: (label, ranking.csv path) — ranker-score axis still works here.
SEALS_RANKERS = [
    ("MCI-HDC",    REPO / "outputs/rankings_mci_hdc/seals_ant/seed0/ranking.csv"),
    ("MCI-NN",     REPO / "outputs/rankings_mci_nn/seals_ant/seed0/ranking.csv"),
    ("SAGE",       REPO / "outputs/rankings_sage/seals_ant/seed0/ranking.csv"),
    ("KernelSHAP", REPO / "outputs/rankings_shap/seals_ant/seed0/ranking.csv"),
]

# kitchen: per-selector score sources (taken from the seed=1 online-eval log).
# "shap" in the log refers to the mci_hdc on-disk directory.
#   - `score_csv` selectors plot their continuous importance score directly.
#   - `compute_mi`  computes MI on the dataset (matches how the mi selector ranks).
#   - `selection`   selectors have no continuous score, so they get a binary
#                   "selected at k=10 (0/1)" bar with an explicit axis label.
KITCHEN_STUDENT_ROOT = REPO / "outputs/students/kitchen_complete/seed1"
KITCHEN_DATASET = REPO / "outputs/datasets/kitchen_complete/seed1/dataset.npz"
KITCHEN_TOPK_LIST = [5, 10, 20, 30, 45, 60]  # matches configs/kitchen_complete.yaml
KITCHEN_SELECTION_K = 10  # which k's selected set to show for selection-only selectors

KITCHEN_SELECTORS = [
    {"label": "MCI-HDC (shap)",
     "mode":  "score_csv",
     "csv":   REPO / "outputs/rankings_mci/kitchen_complete/seed1/ranking.csv"},
    {"label": "MCI-NN",
     "mode":  "score_csv",
     "csv":   REPO / "outputs/rankings_mci_nn/kitchen_complete/seed1/ranking.csv"},
    {"label": "MI",
     "mode":  "compute_mi"},
    {"label": "Oracle",
     "mode":  "selection",
     "dir":   "oracle"},
    {"label": "Random",
     "mode":  "selection",
     "dir":   "random"},
    {"label": "Full",
     "mode":  "selection",
     "dir":   "full"},
]

N_FEATURES = {"kitchen_complete": 60, "seals_ant": 29}
KITCHEN_FEATURE_NAMES = (
    [f"qpos_{i}" for i in range(9)]
    + [f"qvel_{i}" for i in range(9)]
    + [f"obj_{i}"  for i in range(42)]
)


def kitchen_group(name: str) -> str:
    if name.startswith("qpos"):
        return "qpos (robot joint)"
    if name.startswith("qvel"):
        return "qvel (joint velocity)"
    return "obj (scene state)"


KITCHEN_GROUP_COLORS = {
    "qpos (robot joint)":    "#1f77b4",
    "qvel (joint velocity)": "#2ca02c",
    "obj (scene state)":     "#d62728",
}


def _score_column(df: pd.DataFrame) -> str:
    # KernelSHAP CSVs carry mean_abs_shap; SAGE has both sage_value & mean_mci;
    # MCI rankings have mean_mci. We always use the column the ranking sorted by.
    for col in ("mean_mci", "mean_abs_shap"):
        if col in df.columns:
            return col
    raise KeyError(f"No known score column in: {list(df.columns)}")


def _load_full_ranking(path: Path, n_features: int) -> tuple[np.ndarray, list[str]]:
    """Return (scores indexed by feature_index, feature_names indexed likewise)."""
    df = pd.read_csv(path)
    score_col = _score_column(df)
    df = df.sort_values("feature_index").reset_index(drop=True)
    scores = np.zeros(n_features, dtype=np.float64)
    names = [f"feat_{i}" for i in range(n_features)]
    for _, row in df.iterrows():
        i = int(row["feature_index"])
        scores[i] = float(row[score_col])
        names[i] = str(row["feature_name"])
    return scores, names


def _random_selection_scores(
    n_features: int, k: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_features, size=k, replace=False)
    s = np.zeros(n_features, dtype=np.float64)
    s[idx] = 1.0
    return s


def _load_selected_idx(selector_dir: Path, k_label: str) -> set[int] | None:
    """Return the feature_idx set saved by train_student_continuous at k_label.

    `k_label` is "k5" / "k10" / ... / "full". Returns None if the run is missing.
    """
    f = selector_dir / k_label / "metrics.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return set(int(i) for i in json.load(fh)["feature_idx"])


def _kitchen_score_from_csv(path: Path, n_total: int) -> np.ndarray:
    """Per-feature continuous importance score from a ranking.csv file."""
    df = pd.read_csv(path)
    score_col = _score_column(df)
    df = df.sort_values("feature_index").reset_index(drop=True)
    scores = np.zeros(n_total, dtype=np.float64)
    for _, row in df.iterrows():
        scores[int(row["feature_index"])] = float(row[score_col])
    return scores


def _kitchen_mi_scores(n_total: int, seed: int = 0) -> np.ndarray:
    """Mutual-info-regression averaged over all action dims.

    This is the same scoring rule the `mi` selector uses internally
    (utils.feature_utils.get_mi_indices_multioutput). Subsamples training
    rows to keep k-NN MI fast.
    """
    from sklearn.feature_selection import mutual_info_regression

    data = np.load(KITCHEN_DATASET)
    X = data["X_train"].astype(np.float32)
    y = data["y_train"].astype(np.float32)
    if y.ndim == 1:
        y = y[:, None]
    rng = np.random.default_rng(seed)
    if X.shape[0] > 5000:
        idx = rng.choice(X.shape[0], size=5000, replace=False)
        X, y = X[idx], y[idx]
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(y.shape[1]):
        scores += mutual_info_regression(X, y[:, j], random_state=seed)
    scores /= y.shape[1]
    if X.shape[1] != n_total:
        # Pad to n_total in case the dataset has fewer features than expected.
        full = np.zeros(n_total, dtype=np.float64)
        full[: X.shape[1]] = scores
        scores = full
    return scores


def _kitchen_selection_indicator(
    selector_dir: Path, n_total: int, k: int = KITCHEN_SELECTION_K,
) -> np.ndarray:
    """1/0 mask of features selected at the given k (falls back to 'full')."""
    sel = _load_selected_idx(selector_dir, f"k{k}")
    if sel is None:
        sel = _load_selected_idx(selector_dir, "full")
    if sel is None:
        return np.zeros(n_total, dtype=np.float64)
    out = np.zeros(n_total, dtype=np.float64)
    for i in sel:
        if 0 <= i < n_total:
            out[i] = 1.0
    return out


def plot_kitchen() -> Path:
    """Kitchen figure: x-axis = per-selector feature importance score."""
    n_total = N_FEATURES["kitchen_complete"]
    feature_names = KITCHEN_FEATURE_NAMES

    panels: list[tuple[str, np.ndarray, str, bool]] = []  # (label, scores, xlabel, use_log)
    for spec in KITCHEN_SELECTORS:
        label = spec["label"]
        mode = spec["mode"]
        if mode == "score_csv":
            csv: Path = spec["csv"]
            if not csv.exists():
                print(f"  [skip] missing ranking CSV: {csv}")
                continue
            scores = _kitchen_score_from_csv(csv, n_total)
            positive = scores[scores > 0]
            use_log = (
                len(positive) > 0
                and positive.max() / max(positive.min(), 1e-12) > 200
            )
            xlabel = "log₁₀(score) − log₁₀(min⁺)" if use_log else "importance score"
            panels.append((label, scores, xlabel, use_log))
        elif mode == "compute_mi":
            scores = _kitchen_mi_scores(n_total)
            panels.append((label, scores, "MI score (avg over actions)", False))
        elif mode == "selection":
            sel_dir = KITCHEN_STUDENT_ROOT / spec["dir"]
            if not sel_dir.exists():
                print(f"  [skip] missing student dir: {sel_dir}")
                continue
            scores = _kitchen_selection_indicator(sel_dir, n_total)
            xlabel = f"selected at k={KITCHEN_SELECTION_K} (0/1)"
            if spec["dir"] == "full":
                # `full` only ran at k=60 — every feature is selected.
                scores = _kitchen_selection_indicator(sel_dir, n_total, k=60)
                xlabel = "selected at k=60 (0/1)"
            panels.append((label, scores, xlabel, False))
        else:
            raise ValueError(f"Unknown selector mode: {mode!r}")

    if not panels:
        raise RuntimeError("No kitchen selectors produced any panel.")

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(2.2 * n_panels + 1.6, max(10.0, 0.22 * n_total)),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    bar_colors = [
        KITCHEN_GROUP_COLORS[kitchen_group(n)] for n in feature_names
    ]
    y_positions = np.arange(n_total)

    for ax, (label, scores, xlabel, use_log) in zip(axes, panels):
        if use_log:
            positive = scores[scores > 0]
            baseline = float(np.log10(max(positive.min(), 1e-12)))
            plotted = np.log10(np.clip(scores, 1e-12, None)) - baseline
            plotted[scores <= 0] = 0.0
        else:
            plotted = scores

        ax.barh(y_positions, plotted, color=bar_colors,
                edgecolor="white", linewidth=0.3, height=0.78)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(feature_names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("feature", fontsize=9)

    group_handles = [
        mpatches.Patch(color=c, label=g) for g, c in KITCHEN_GROUP_COLORS.items()
    ]
    fig.legend(
        handles=group_handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9,
    )

    fig.suptitle(
        "kitchen_complete (seed=1) — per-feature importance score by selector",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "kitchen_complete.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")
    return out_path


def plot_seals_ant() -> Path:
    """seals_ant figure: per-ranker importance score in feature_index order."""
    env = "seals_ant"
    n_total = N_FEATURES[env]

    panels: list[tuple[str, np.ndarray, bool]] = []
    feature_names: list[str] | None = None
    for label, path in SEALS_RANKERS:
        if not path.exists():
            print(f"  [skip] missing {path}")
            continue
        scores, names = _load_full_ranking(path, n_total)
        if feature_names is None:
            feature_names = names
        panels.append((label, scores, False))

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_total)]
    panels.append(
        ("Random", _random_selection_scores(n_total, RANDOM_K, seed=0), True)
    )

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.0 * n_panels + 1.4, max(6.0, 0.18 * n_total)),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    cmap = plt.get_cmap("viridis")
    bar_colors = [cmap(i / max(n_total - 1, 1)) for i in range(n_total)]
    y_positions = np.arange(n_total)

    for ax, (label, scores, is_random) in zip(axes, panels):
        positive = scores[scores > 0]
        use_log = (
            not is_random and len(positive) > 0
            and (positive.max() / max(positive.min(), 1e-12)) > 200
        )
        if use_log:
            plotted = np.log10(np.clip(scores, 1e-12, None))
            baseline = float(np.log10(max(positive.min(), 1e-12)))
            plotted = plotted - baseline
            plotted[scores <= 0] = 0.0
            xlabel = f"log₁₀(score) − log₁₀({positive.min():.2g})"
        else:
            plotted = scores
            xlabel = "selection (0/1)" if is_random else "importance score"

        ax.barh(y_positions, plotted, color=bar_colors,
                edgecolor="white", linewidth=0.4, height=0.78)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(feature_names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("feature", fontsize=9)

    fig.suptitle(
        "Per-feature importance by ranker — seals_ant",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{env}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")
    return out_path


def main() -> None:
    print("\nkitchen_complete:")
    plot_kitchen()
    print("\nseals_ant:")
    plot_seals_ant()


if __name__ == "__main__":
    main()
