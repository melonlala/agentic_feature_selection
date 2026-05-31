"""Train a BC student on the full feature set (upper-bound baseline).

Trains exactly once at k = n_features. Also serves as the explanation target
for selectors that need a full-feature model (sage, kernelshap).

Usage:
    python student/train_student_full.py \\
        --config       configs/kitchen_complete.yaml \\
        --seed         0 \\
        --dataset_path outputs/.../dataset.npz \\
        --output_dir   outputs/students/.../full
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


SELECTOR_NAME = "full"


def main() -> None:
    p = argparse.ArgumentParser(
        description="BC student trained on the full feature set."
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir",   required=True)
    args = p.parse_args()

    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    dataset = load_dataset(args.dataset_path)

    def feature_idx_fn(_k: int) -> list[int]:  # noqa: ARG001
        return list(range(dataset["n_features"]))

    topk_list = resolve_topk_list(cfg, SELECTOR_NAME, dataset["n_features"])

    train_bc_students(
        cfg=cfg, seed=args.seed, device=device,
        dataset=dataset, feature_idx_fn=feature_idx_fn,
        topk_list=topk_list, selector_name=SELECTOR_NAME,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
