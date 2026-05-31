"""Train BC students with the top-k features from a KernelSHAP ranking.

Ranking source: explain/shap_rank_continuous.py (KernelSHAP on a full-feature
BC student, action-output target).

Usage:
    python student/train_student_kernelshap.py \\
        --config       configs/kitchen_complete.yaml \\
        --seed         0 \\
        --dataset_path outputs/.../dataset.npz \\
        --ranking_path outputs/rankings_shap/.../ranking.csv \\
        --output_dir   outputs/students/.../kernelshap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.train_bc_continuous import (
    load_dataset,
    resolve_topk_list,
    setup_run,
    train_bc_students,
)
from utils.feature_utils import get_topk_indices
from utils.io import load_csv


SELECTOR_NAME = "kernelshap"


def main() -> None:
    p = argparse.ArgumentParser(
        description="BC students with KernelSHAP top-k feature selection.",
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--dataset_path", required=True,
                   help="Path to dataset.npz (state or latent).")
    p.add_argument("--ranking_path", required=True,
                   help="Path to ranking.csv produced by explain/shap_rank_continuous.py.")
    p.add_argument("--output_dir",   required=True)
    args = p.parse_args()

    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    dataset = load_dataset(args.dataset_path)
    ranking_df = load_csv(args.ranking_path)

    def feature_idx_fn(k: int) -> list[int]:
        return list(get_topk_indices(ranking_df, k))

    topk_list = resolve_topk_list(cfg, SELECTOR_NAME, dataset["n_features"])

    train_bc_students(
        cfg=cfg,
        seed=args.seed,
        device=device,
        dataset=dataset,
        feature_idx_fn=feature_idx_fn,
        topk_list=topk_list,
        selector_name=SELECTOR_NAME,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
