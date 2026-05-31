"""Extend the topk sweep for an existing compare_seals_ant seed dir.

Reuses everything already saved by ``scripts/compare_rankers_seals_ant.py``
(dataset.npz, config.yaml, rankings/, full BC checkpoint) and only trains the
subset BCs for k values not yet present under ``students/{space}/{ranker}/``.

New rows are appended to ``topk_eval.csv``; existing rows are kept untouched
unless ``--retrain`` is passed.

Usage:
    python scripts/extend_topk_seals_ant.py \\
        --seed_dir   outputs/compare_seals_ant/seed0 \\
        --state_topk 2 4 8 12 16 20 24 27 \\
        --latent_topk 2 4 8 12 16 24 32 48 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_rankers_seals_ant import (
    ENV_NAME, RANKER_COLOR, LATENT_LAYER,
    eval_policy_in_env, plot_return_vs_k,
    save_latent_subset_student, save_state_subset_student,
    train_latent_subset_bc, train_state_subset_bc,
)
from student.bc_continuous_model import BCContinuousPolicy
from utils.io import ensure_dir, load_npz, save_csv
from utils.seed import set_global_seed


def _load_full_bc(ckpt_path: Path, device: torch.device) -> BCContinuousPolicy:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = BCContinuousPolicy(
        input_dim=ckpt["input_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dims=ckpt["hidden_dims"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dir = Path(args.seed_dir)

    # --- Reuse saved artifacts ---
    data = load_npz(str(seed_dir / "dataset.npz"))
    X_train = data["X_train"].astype(np.float32)
    Y_train = data["y_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    Y_val   = data["y_val"].astype(np.float32)
    state_names = [str(n) for n in data["feature_names"]]
    obs_dim, action_dim = X_train.shape[1], Y_train.shape[1]
    print(f"[extend] obs_dim={obs_dim}, action_dim={action_dim}, "
          f"N_train={len(X_train)}, N_val={len(X_val)}, device={device}")

    full_student_path = seed_dir / "students" / "full" / "full" / "model.pt"
    full_model = _load_full_bc(full_student_path, device)
    print(f"[extend] Loaded full BC from {full_student_path}")

    rankings_dir = seed_dir / "rankings"
    rankings: dict[tuple[str, str], pd.DataFrame] = {}
    for space in ("state", "latent"):
        for csv_path in sorted((rankings_dir / space).glob("*_ranking.csv")):
            ranker = csv_path.stem.removesuffix("_ranking")
            rankings[(space, ranker)] = pd.read_csv(csv_path)
    print(f"[extend] Loaded {len(rankings)} ranking files: "
          f"{sorted(rankings.keys())}")

    # --- Existing topk_eval.csv (we'll merge into it) ---
    eval_csv = seed_dir / "topk_eval.csv"
    if eval_csv.exists():
        existing = pd.read_csv(eval_csv)
        existing_keys = set(zip(existing["space"], existing["ranker"],
                                existing["k"].astype(int)))
    else:
        existing = pd.DataFrame()
        existing_keys = set()
    print(f"[extend] {len(existing_keys)} (space, ranker, k) rows already in topk_eval.csv")

    students_dir = seed_dir / "students"
    new_rows: list[dict] = []

    space_ks = {"state": args.state_topk, "latent": args.latent_topk}

    for (space, ranker), df in rankings.items():
        max_d = df.shape[0]
        for k in space_ks[space]:
            if k > max_d:
                continue
            key = (space, ranker, int(k))
            ckpt_dir = students_dir / space / ranker / f"k{k}"
            ckpt_path = ckpt_dir / "model.pt"

            if key in existing_keys and ckpt_path.exists() and not args.retrain:
                continue

            top_idx = (df.sort_values("rank").head(k)["feature_index"]
                       .astype(int).tolist())
            print(f"[extend] {space}/{ranker}/k={k}: features={top_idx[:6]}"
                  f"{'...' if k > 6 else ''}")
            t0 = time.time()

            if space == "state":
                policy, val_mse = train_state_subset_bc(
                    X_train, Y_train, X_val, Y_val,
                    feature_idx=top_idx, hidden=tuple(args.subset_hidden),
                    epochs=args.subset_epochs, lr=1e-3, weight_decay=1e-4,
                    batch_size=256, device=device,
                )
                save_state_subset_student(
                    policy, ckpt_dir,
                    feature_idx=top_idx, action_dim=action_dim,
                    hidden_dims=args.subset_hidden, feature_names=state_names,
                )
                metrics = eval_policy_in_env(
                    policy, feature_idx=top_idx,
                    n_episodes=args.eval_episodes, device=device,
                    seed=args.seed * 1000,
                )
            else:
                policy, val_mse = train_latent_subset_bc(
                    X_train, Y_train, X_val, Y_val,
                    full_model=full_model, latent_idx=top_idx,
                    hidden=tuple(args.subset_hidden), action_dim=action_dim,
                    epochs=args.subset_epochs, lr=1e-3, weight_decay=1e-4,
                    batch_size=256, device=device,
                )
                save_latent_subset_student(
                    policy, ckpt_dir,
                    raw_D=obs_dim, action_dim=action_dim,
                    hidden_dims=args.subset_hidden, latent_idx=top_idx,
                    source_student_path=full_student_path,
                    feature_names=state_names,
                )
                metrics = eval_policy_in_env(
                    policy, feature_idx=list(range(obs_dim)),
                    n_episodes=args.eval_episodes, device=device,
                    seed=args.seed * 1000,
                )

            elapsed = time.time() - t0
            row = {
                "space":    space,
                "ranker":   ranker,
                "k":        int(k),
                "val_mse":  float(val_mse),
                **metrics,
                "elapsed_s": float(elapsed),
                "ckpt": str(ckpt_path),
            }
            new_rows.append(row)
            print(f"[extend]   val_mse={val_mse:.4f}  "
                  f"return={metrics['mean_return']:.1f}±{metrics['std_return']:.1f}  "
                  f"({elapsed:.1f}s)")

    if not new_rows:
        print("[extend] No new k values to evaluate. Done.")
        return

    new_df = pd.DataFrame(new_rows)
    if args.retrain:
        # Drop overlapping rows then append.
        keys = set(zip(new_df["space"], new_df["ranker"], new_df["k"].astype(int)))
        if not existing.empty:
            mask = ~existing.apply(
                lambda r: (r["space"], r["ranker"], int(r["k"])) in keys, axis=1,
            )
            existing = existing[mask]
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.sort_values(["space", "ranker", "k"]).reset_index(drop=True)
    save_csv(merged, str(eval_csv))
    print(f"[extend] Updated {eval_csv}  (+{len(new_rows)} new rows, "
          f"total {len(merged)})")

    # Re-plot per-space online curves with the denser k grid.
    plots_dir = ensure_dir(seed_dir / "plots")
    for space in ("state", "latent"):
        plot_return_vs_k(merged, space, plots_dir / f"return_vs_k_{space}.png")
    print(f"[extend] Re-plotted per-space curves under {plots_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_dir", required=True,
                   help="e.g. outputs/compare_seals_ant/seed0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state_topk",  type=int, nargs="+",
                   default=[2, 6, 10, 14, 20, 24])
    p.add_argument("--latent_topk", type=int, nargs="+",
                   default=[2, 12, 24, 48])
    p.add_argument("--subset_hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--subset_epochs", type=int, default=60)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--retrain", action="store_true",
                   help="Retrain checkpoints even if they exist.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
