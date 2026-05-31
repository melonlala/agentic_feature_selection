"""Legacy combined dispatcher for continuous-action BC students.

Decomposed into per-selector scripts — prefer the dedicated entry points for
new code:

    student/train_student_mci_hdc.py     (replaces --selector mci  + MCI-HDC ranking)
    student/train_student_mci_nn.py      (replaces --selector mci  + MCI-NN ranking)
    student/train_student_sage.py        (replaces --selector shap + SAGE ranking)
    student/train_student_kernelshap.py  (replaces --selector shap + KernelSHAP ranking)
    student/train_student_random.py      (replaces --selector random)
    student/train_student_full.py        (replaces --selector full)

This script is kept for backwards compatibility with shell scripts that still
pass `--selector {shap, mci, random, oracle, mi, full}` (run_d4rl_mci.sh,
run_d4rl_latent_mci.sh, run_ant_irl.sh, run_ant_latent_irl.sh). It delegates
to student/train_bc_continuous.train_bc_students, so behaviour and checkpoint
format are identical to the per-selector scripts.

Usage (legacy):
    python student/train_student_continuous.py \\
        --config configs/kitchen_complete.yaml --seed 0 \\
        --dataset_path .../dataset.npz \\
        --selector shap --ranking_path .../ranking.csv \\
        --output_dir .../students/.../shap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.train_bc_continuous import (
    load_dataset,
    resolve_topk_list,
    setup_run,
    train_bc_students,
)
from utils.feature_utils import (
    get_mi_indices,
    get_mi_indices_multioutput,
    get_random_indices,
    get_topk_indices,
)
from utils.io import load_csv


def _build_feature_idx_fn(args, dataset, oracle_indices):
    """Return a (k → list[int]) closure for the requested legacy selector."""
    sel = args.selector
    n_features = dataset["n_features"]

    if sel == "full":
        return lambda _k: list(range(n_features))  # noqa: ARG005

    if sel == "random":
        def _fn(k: int) -> list[int]:
            rng = np.random.default_rng(args.seed)
            return list(get_random_indices(n_features, k, rng=rng))
        return _fn

    if sel == "oracle":
        oracle = oracle_indices or list(range(min(n_features, dataset["action_dim"])))
        return lambda k: list(oracle[:k])

    if sel in ("shap", "mci"):
        if not args.ranking_path:
            raise ValueError(f"--ranking_path required for selector={sel}")
        ranking_df = load_csv(args.ranking_path)
        return lambda k: list(get_topk_indices(ranking_df, k))

    if sel == "mi":
        # Pre-compute once (k-NN MI is expensive).
        y_train = dataset["y_train"]
        if y_train.ndim == 2:
            ranked = get_mi_indices_multioutput(
                dataset["X_train"], y_train, n_features, seed=args.seed,
            )
        else:
            ranked = get_mi_indices(
                dataset["X_train"], dataset["action_norm_train"],
                n_features, seed=args.seed, continuous_target=True,
            )
        print(f"[train_student_continuous] MI top-5: {ranked[:5]}")
        return lambda k: list(ranked[:k])

    raise ValueError(f"Unknown selector: {sel!r}")


def run(args: argparse.Namespace) -> None:
    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    dataset = load_dataset(args.dataset_path)

    oracle_indices = cfg["env"].get("oracle_feature_indices", []) or []
    feature_idx_fn = _build_feature_idx_fn(args, dataset, oracle_indices)
    topk_list = resolve_topk_list(cfg, args.selector, dataset["n_features"])

    train_bc_students(
        cfg=cfg,
        seed=args.seed,
        device=device,
        dataset=dataset,
        feature_idx_fn=feature_idx_fn,
        topk_list=topk_list,
        selector_name=args.selector,
        output_dir=out_dir,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Legacy combined BC-student dispatcher. Prefer the "
                    "per-selector scripts for new code.",
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--dataset_path", required=True,
                   help="Path to dataset.npz (state or latent).")
    p.add_argument("--ranking_path", default=None,
                   help="Path to ranking.csv (required for --selector shap/mci).")
    p.add_argument(
        "--selector",
        default="full",
        choices=["shap", "mci", "random", "oracle", "mi", "full"],
    )
    p.add_argument("--output_dir",   required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
