"""SAGE feature ranking for the bc / irl / pc tasks.

SAGE (Covert et al., NeurIPS 2020) — Shapley values of the cooperative game

    v_f(S) = E[ℓ(f_∅, Y)] − E[ℓ(f_S(X_S), Y)]

where features outside S are marginalised by sampling from a background dataset
and ℓ is the task loss. Unlike MCI (which *retrains* a model per feature
subset), SAGE trains **one** full-feature model and reuses it for every subset
via marginal imputation.

To stay comparable with the MCI rankers, the one trained model is exactly the
per-task student of student/train_student.py — the *same model frame as MCI*:

    bc  : imitation BC policy (SB3 ActorCriticPolicy). Target Y = expert actions,
          loss = MSE over action dims.
    irl : RewardMLP regressing ground-truth reward. Target Y = reward, loss = MSE.
    pc  : RewardMLP trained by Bradley-Terry preference CE. The reward head's
          output is attributed against the true reward (Y = reward, loss = MSE).

Ranking time: the reported `ranking_time_sec` INCLUDES the one-time model
training (model_training_time_sec) plus the SAGE attribution
(attribution_time_sec); they are also reported separately in metadata.

Output (in --output_dir):
    ranking.csv      — feature_index, feature_name, score, rank (rank 1 = top)
    sage_values.npz  — values [d], std [d]
    metadata.json    — task, params, timing breakdown, test metrics
    resolved_config.yaml

Usage:
    python ranker/sage_rank.py \\
        --config       configs/seals_ant.yaml \\
        --task         bc \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir   outputs/rankings_sage/seals_ant/seed0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sage
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.train_student import (
    load_task_dataset,
    train_and_eval_student,
)
from utils.io import save_json
from utils.seed import set_global_seed


def build_predict_fn(task: str, model, device: torch.device):
    """Return a numpy callable f(X[N, d]) -> preds[N, out] for the trained model.

    bc → action vector via the SB3 policy; irl/pc → scalar reward via RewardMLP.
    The model is trained on the FULL feature set, so f expects full-width inputs
    (SAGE perturbs the full vector and marginalises removed columns).
    """
    if task == "bc":
        def f(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            actions, _ = model.predict(X, deterministic=True)
            return np.asarray(actions, dtype=np.float32)
        return f

    @torch.no_grad()
    def f(X: np.ndarray) -> np.ndarray:
        t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        return model(t).cpu().numpy().reshape(-1, 1)
    return f


def task_targets(task: str, data: dict, split: str) -> np.ndarray:
    """SAGE target Y for the task loss: actions (bc) or reward (irl/pc)."""
    if task == "bc":
        return data[f"y_{split}"].astype(np.float32)
    r = data[f"rewards_{split}"]
    if r is None:
        raise ValueError(f"task={task!r} needs reward arrays in the dataset.")
    return r.astype(np.float32).reshape(-1, 1)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    from student.train_bc_continuous import setup_run
    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)

    data = load_task_dataset(args.dataset_path)
    n_features = data["n_features"]
    feature_names = data["feature_names"]

    # ── One-time model training (same frame as MCI) — timed ──
    t0 = time.time()
    full_idx = list(range(n_features))
    model, test_metrics = train_and_eval_student(
        args.task, data, full_idx, cfg, args.seed, device, args,
    )
    train_time = time.time() - t0
    print(f"[sage_rank] trained {args.task} model on {n_features} features "
          f"in {train_time:.1f}s; test={test_metrics}")

    # ── SAGE attribution on the trained model — timed ──
    predict = build_predict_fn(args.task, model, device)
    X = data["X_val"].astype(np.float32)
    Y = task_targets(args.task, data, "val")

    rng = np.random.default_rng(args.seed)
    bg_n = min(args.background_size, len(data["X_train"]))
    bg_idx = rng.choice(len(data["X_train"]), size=bg_n, replace=False)
    background = data["X_train"][bg_idx].astype(np.float32)

    if args.explain_size and len(X) > args.explain_size:
        ex_idx = rng.choice(len(X), size=args.explain_size, replace=False)
        X, Y = X[ex_idx], Y[ex_idx]

    t1 = time.time()
    imputer = sage.MarginalImputer(predict, background)
    estimator = sage.PermutationEstimator(imputer, loss="mse", random_state=args.seed)
    explanation = estimator(
        X, Y,
        batch_size=args.batch_size,
        n_permutations=args.n_permutations,
        detect_convergence=(args.n_permutations is None),
        thresh=args.thresh,
        verbose=True,
        bar=True,
    )
    attribution_time = time.time() - t1
    ranking_time = time.time() - t0  # INCLUDES one-time model training

    values = np.asarray(explanation.values, dtype=np.float64)  # [d], signed
    std = np.asarray(explanation.std, dtype=np.float64)

    # Rank by SAGE value (higher = more important).
    order = sorted(range(n_features), key=lambda i: values[i], reverse=True)
    with open(out_dir / "ranking.csv", "w") as fh:
        fh.write("feature_index,feature_name,score,rank\n")
        for rank, idx in enumerate(order, start=1):
            fh.write(f"{idx},{feature_names[idx]},{values[idx]:.6f},{rank}\n")

    np.savez(out_dir / "sage_values.npz", values=values, std=std)

    save_json({
        "task": args.task,
        "ranking_method": "sage",
        "dataset_path": args.dataset_path,
        "num_features": n_features,
        "num_explain_samples": int(len(X)),
        "background_size": int(bg_n),
        "loss": "mse",
        "n_permutations": args.n_permutations,
        "thresh": args.thresh,
        "model_training_time_sec": train_time,
        "attribution_time_sec": attribution_time,
        "ranking_time_sec": ranking_time,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
    }, str(out_dir / "metadata.json"))

    print(f"[sage_rank] task={args.task} done. "
          f"ranking_time={ranking_time:.1f}s "
          f"(train={train_time:.1f}s + attribution={attribution_time:.1f}s). "
          f"Saved to {out_dir / 'ranking.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE feature ranking (bc/irl/pc).")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--task", required=True, choices=["bc", "irl", "pc"],
                   help="Which task model to train + attribute.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--background_size", type=int, default=128,
                   help="Marginal-imputer background samples (<=1024 recommended).")
    p.add_argument("--explain_size", type=int, default=512,
                   help="Number of rows to attribute (0 = all val rows).")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_permutations", type=int, default=None,
                   help="Fixed permutation count; omit to auto-detect convergence.")
    p.add_argument("--thresh", type=float, default=0.05,
                   help="Convergence threshold (used when --n_permutations omitted).")
    # pc-only knobs consumed by train_and_eval_student.
    p.add_argument("--fragment_length", type=int, default=50)
    p.add_argument("--num_pairs", type=int, default=200)
    p.add_argument("--num_eval_pairs", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
