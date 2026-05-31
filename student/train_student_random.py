"""Train BC students with uniformly random k-feature subsets.

Baseline selector — for each k, draw a fresh random subset of k features
from a per-seed RNG (deterministic per seed). No ranking.csv needed.

Usage:
    python student/train_student_random.py \\
        --config       configs/kitchen_complete.yaml \\
        --seed         0 \\
        --dataset_path outputs/.../dataset.npz \\
        --output_dir   outputs/students/.../random
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
from utils.feature_utils import get_random_indices


SELECTOR_NAME = "random"


def main() -> None:
    p = argparse.ArgumentParser(
        description="BC students with uniformly random k-feature subsets."
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir",   required=True)
    args = p.parse_args()

    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    dataset = load_dataset(args.dataset_path)

    # Reset RNG per-k from `args.seed` so each k's random subset is
    # reproducible and independent of the iteration order.
    def feature_idx_fn(k: int) -> list[int]:
        rng = np.random.default_rng(args.seed)
        return list(get_random_indices(dataset["n_features"], k, rng=rng))

    topk_list = resolve_topk_list(cfg, SELECTOR_NAME, dataset["n_features"])

    train_bc_students(
        cfg=cfg, seed=args.seed, device=device,
        dataset=dataset, feature_idx_fn=feature_idx_fn,
        topk_list=topk_list, selector_name=SELECTOR_NAME,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
