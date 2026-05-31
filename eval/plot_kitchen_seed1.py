"""Aggregate kitchen_complete seed 1 student metrics + emit the same plots
as outputs/compare_seals_ant/seed0/plots/online_offline_combined.png and
mci_cumulative_bound.png.

Inputs (existing artifacts):
    outputs/datasets/kitchen_complete/seed1/dataset.npz
    outputs/students/kitchen_complete/seed1/{full,mci_nn,shap,random,oracle,mi}/summary.csv
    outputs/rankings_mci/kitchen_complete/seed1/{ranking.csv, mci_scores.json}     (= mci_hdc)
    outputs/rankings_mci_nn/kitchen_complete/seed1/{ranking.csv, mci_scores.json}  (= mci_nn)

Outputs:
    outputs/compare_kitchen_complete/seed1/topk_eval.csv
    outputs/compare_kitchen_complete/seed1/plots/online_offline_combined.png
    outputs/compare_kitchen_complete/seed1/plots/mci_cumulative_bound.png

Notes:
  - The ``shap`` student directory was trained from the MCI-HDC ranking
    (run_d4rl_mci.sh passes --selector shap with that ranking.csv), so we
    relabel it ``mci_hdc`` in all outputs.
  - Kitchen seed 1 has NO online env-return data and NO latent-space artifacts.
    We replace the left panel of ``online_offline_combined`` (which shows env
    return for seals/Ant-v1) with a TEST-MSE panel; the right panel keeps
    val-MSE (log-scale) just like the seals/Ant-v1 version. Only state-space
    curves are drawn (no dotted latent overlays).
  - ``mci_cumulative_bound`` shows a single state-space panel:
      * bold markers = measured ν(S)/ν(F) at each topk evaluated by a student
                       where ν(S) = Var(y_train) - val_mse(S)
                       and    ν(F) = Var(y_train) - val_mse(full).
      * thin line   = cumulative MCI score / ν(F)   (Catav et al. 2021 bound)
    Bound lines are drawn for selectors with a ranking.csv (mci_hdc, mci_nn);
    other selectors (random, oracle, mi) get measured curves only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, load_npz, save_csv

# ─── Config ──────────────────────────────────────────────────────────────────
ROOT       = Path("outputs")
ENV_NAME   = "kitchen_complete"
SEED       = 1
OUT_ROOT   = ROOT / "compare_kitchen_complete" / f"seed{SEED}"
PLOT_DIR   = OUT_ROOT / "plots"

# Map student-dir name → canonical ranker label + ranking source.
SELECTORS: list[tuple[str, str, str | None]] = [
    # (student_dir, plot_label, ranking_csv_path or None)
    ("mci_nn", "mci_nn",  f"{ROOT}/rankings_mci_nn/{ENV_NAME}/seed{SEED}/ranking.csv"),
    ("shap",   "mci_hdc", f"{ROOT}/rankings_mci/{ENV_NAME}/seed{SEED}/ranking.csv"),
    ("random", "random",  None),
    ("oracle", "oracle",  None),
    ("mi",     "mi",      None),
]
FULL_SUMMARY = ROOT / "students" / ENV_NAME / f"seed{SEED}" / "full" / "summary.csv"
DATASET_NPZ  = ROOT / "datasets" / ENV_NAME / f"seed{SEED}" / "dataset.npz"

RANKER_COLOR = {
    "mci_nn":  "#1f77b4",
    "mci_hdc": "#2ca02c",
    "sage":    "#d62728",
    "random":  "#7f7f7f",
    "oracle":  "#9467bd",
    "mi":      "#ff7f0e",
}
RANKER_ORDER = ["mci_nn", "mci_hdc", "random", "oracle", "mi"]


# ─── Step 1: collect per-(ranker, k) eval rows ───────────────────────────────

def _parse_k_label(k_label: str) -> int | None:
    """Convert 'k5' → 5, 'full' → None (caller substitutes n_features)."""
    s = str(k_label).strip()
    if s == "full":
        return None
    if s.startswith("k"):
        s = s[1:]
    try:
        return int(s)
    except ValueError:
        return None


def collect_eval_rows() -> pd.DataFrame:
    """Concatenate per-selector offline summary.csv + (optional) online_metrics.csv
    into one long DataFrame, keyed on (ranker, k)."""
    rows: dict[tuple[str, int], dict] = {}

    # Offline (val/test MSE) — always present.
    for student_dir, label, _ in SELECTORS:
        csv = ROOT / "students" / ENV_NAME / f"seed{SEED}" / student_dir / "summary.csv"
        if not csv.exists():
            print(f"[plot] missing offline summary: {csv} — skipping {label}")
            continue
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            k = int(r["k"])
            rows[(label, k)] = {
                "space":      "state",
                "ranker":     label,
                "k":          k,
                "val_mse":    float(r["val_mse"]),
                "test_mse":   float(r["test_mse"]),
                "val_mae":    float(r["val_mae"]),
                "test_mae":   float(r["test_mae"]),
                "mean_return":  float("nan"),
                "std_return":   float("nan"),
                "success_rate": float("nan"),
            }

    # Online (env return, success) — optional; populated where available.
    n_features_default = 60  # kitchen_complete obs_dim
    for student_dir, label, _ in SELECTORS:
        online_csv = (ROOT / "eval" / "online" / ENV_NAME / f"seed{SEED}"
                      / student_dir / "online_metrics.csv")
        if not online_csv.exists():
            continue
        df = pd.read_csv(online_csv)
        for _, r in df.iterrows():
            k = _parse_k_label(r["k"])
            if k is None:
                k = n_features_default
            key = (label, k)
            if key not in rows:
                continue
            rows[key]["mean_return"]  = float(r["mean_return"])
            rows[key]["std_return"]   = float(r["std_return"])
            rows[key]["success_rate"] = float(r["success_rate"])

    return (pd.DataFrame(list(rows.values()))
            .sort_values(["ranker", "k"]).reset_index(drop=True))


# ─── Step 2: full-feature anchor (ν(F)) ──────────────────────────────────────

def full_val_mse() -> float:
    df = pd.read_csv(FULL_SUMMARY)
    return float(df["val_mse"].iloc[0])


def var_y_train() -> float:
    data = load_npz(DATASET_NPZ)
    y = np.asarray(data["y_train"], dtype=np.float64)
    return float(np.var(y, axis=0).sum())


# ─── Step 3: cumulative MCI bound per ranking ────────────────────────────────

def cumulative_mci_fraction(ranking_csv: str, ks: list[int]) -> list[float]:
    """For sorted MCI scores Î_(1) ≥ Î_(2) ≥ ..., return cumulative score
    fraction Σ_{i≤k} Î_(i) / Σ_all Î_(i) — matches the seals_ant
    'cumulative score / total' overlay (which keeps the curve in [0, 1]
    regardless of the raw MCI scale)."""
    df = pd.read_csv(ranking_csv).sort_values("rank")
    scores = df["mean_mci"].to_numpy(dtype=np.float64)
    total = float(scores.sum())
    if total <= 0:
        return [float("nan")] * len(ks)
    cum = np.cumsum(scores) / total
    return [float(cum[k - 1]) if k <= len(cum) else float(cum[-1]) for k in ks]


# ─── Plot 1: online/offline combined (kitchen variant — both panels MSE) ─────

def plot_online_offline_combined(eval_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left panel: online env return (mirrors the seals/Ant-v1 plot's left panel).
    # Source: outputs/eval/online/kitchen_complete/seed{N}/{selector}/online_metrics.csv
    # populated by eval/eval_kitchen_online_minari.py (Minari D4RL/kitchen/complete-v2).
    ax = axes[0]
    have_online = False
    for ranker in RANKER_ORDER:
        sub = eval_df[eval_df["ranker"] == ranker].sort_values("k")
        sub = sub.dropna(subset=["mean_return"])
        if sub.empty:
            continue
        have_online = True
        ax.plot(sub["k"], sub["mean_return"], marker="o", linestyle="-",
                color=RANKER_COLOR[ranker], label=f"state/{ranker}")
        if sub["std_return"].notna().any():
            ax.fill_between(
                sub["k"],
                sub["mean_return"] - sub["std_return"],
                sub["mean_return"] + sub["std_return"],
                color=RANKER_COLOR[ranker], alpha=0.12,
            )
    ax.set_xlabel("k (selected features)")
    ax.set_ylabel("Mean episodic return")
    ax.set_title(
        "Online — env return (Minari D4RL/kitchen/complete-v2)"
        if have_online else
        "Online — env return (no online_metrics.csv found)"
    )
    ax.grid(alpha=0.3)
    if have_online:
        ax.legend(fontsize=8, loc="upper right")

    # Right panel: val MSE on log y-axis (mirrors seals_ant offline panel).
    ax = axes[1]
    for ranker in RANKER_ORDER:
        sub = eval_df[eval_df["ranker"] == ranker].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["val_mse"], marker="o", linestyle="-",
                color=RANKER_COLOR[ranker], label=f"state/{ranker}")
    ax.set_yscale("log")
    ax.set_xlabel("k (selected features)")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.set_title("Offline — val MSE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{ENV_NAME} seed {SEED} — online return + offline val MSE vs k  "
        f"(solid: state; no latent runs for this seed)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ─── Plot 2: predictive power vs cumulative-MCI bound ────────────────────────

def plot_mci_cumulative_bound(
    eval_df: pd.DataFrame, nu_F: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    var_y = nu_F + full_val_mse()  # Var(y) used for ν(S) = Var(y) - val_mse(S)
    # We follow the convention from the seals_ant plot: ν(S) / ν(F).
    # ν(F) = Var(y) - val_mse(full), and ν(S) = Var(y) - val_mse(S).

    bound_lines = {}
    for student_dir, label, ranking_csv in SELECTORS:
        sub = eval_df[eval_df["ranker"] == label].sort_values("k")
        if sub.empty:
            continue

        # Measured ν(S)/ν(F) — bold markers
        nu_S       = var_y - sub["val_mse"].to_numpy(dtype=np.float64)
        norm_nu_S  = nu_S / nu_F
        ax.plot(sub["k"], norm_nu_S, marker="o", linewidth=2.0, markersize=6,
                color=RANKER_COLOR[label], label=f"{label} — ν(S)/ν(F)")

        # Cumulative MCI score / total — thin line, in [0, 1] like the seals_ant plot
        if ranking_csv is not None and Path(ranking_csv).exists():
            ks_bound = list(sub["k"].astype(int))
            cum = cumulative_mci_fraction(ranking_csv, ks_bound)
            ax.plot(ks_bound, cum, linestyle="--", linewidth=1.2,
                    color=RANKER_COLOR[label], alpha=0.7,
                    label=f"{label} — cum.MCI / total")
            bound_lines[label] = cum

    ax.set_xlabel("k (selected features)")
    ax.set_ylabel("Normalized score")
    ax.set_title(
        f"{ENV_NAME} seed {SEED} — predictive power vs cumulative MCI  "
        f"(bold-markers: measured ν(S)/ν(F); thin: cumulative score / total)   "
        f"ν(F)={nu_F:.3f}",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="k", linewidth=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(OUT_ROOT)
    ensure_dir(PLOT_DIR)

    eval_df = collect_eval_rows()
    if eval_df.empty:
        print("[plot] no eval rows found — aborting.")
        return

    csv_out = OUT_ROOT / "topk_eval.csv"
    save_csv(eval_df, str(csv_out))
    print(f"[plot] wrote {csv_out}  ({len(eval_df)} rows)")

    full_mse = full_val_mse()
    var_y    = var_y_train()
    nu_F     = var_y - full_mse
    print(f"[plot] full val MSE = {full_mse:.6f}")
    print(f"[plot] Var(y_train) = {var_y:.6f}  →  ν(F) = {nu_F:.6f}")

    plot_online_offline_combined(eval_df, PLOT_DIR / "online_offline_combined.png")
    plot_mci_cumulative_bound(eval_df, nu_F, PLOT_DIR / "mci_cumulative_bound.png")

    # Quick summary print
    print("\n[plot] ν(S)/ν(F) at each k (state):")
    pivot_nu = (eval_df.assign(nu_S=lambda d: var_y - d["val_mse"],
                                nu_norm=lambda d: (var_y - d["val_mse"]) / nu_F)
                       .pivot_table(index="k", columns="ranker",
                                    values="nu_norm", aggfunc="first"))
    print(pivot_nu.round(3).to_string())


if __name__ == "__main__":
    main()
