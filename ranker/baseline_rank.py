"""Baseline feature rankings (random / mutual-information) for bc/irl/pc.

Produces a ranking.csv in the same schema as the MCI / SAGE / KernelSHAP rankers
(feature_index, feature_name, score, rank) so student/train_student.py consumes
all methods identically.

    random : task-independent uniform random ordering (seeded).
    mi     : mutual information between each feature and the task target —
               bc      → MI(feature, actions), averaged over action dims
               irl/pc  → MI(feature, ground-truth reward)
             (sklearn k-NN MI estimator; rows subsampled for tractability.)

Usage:
    python ranker/baseline_rank.py \\
        --method mi --task irl \\
        --config configs/seals_ant.yaml \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir outputs/rankings_mi/seals_ant/irl/seed0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.train_student import load_task_dataset
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed

_MI_MAX_SAMPLES = 5000  # k-NN MI is ~O(n^2); cap rows for speed.


def random_scores(d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random importance per feature (task-independent)."""
    return rng.random(d)


def mi_scores(task: str, data: dict, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Mutual information of each feature with the task target."""
    X = data["X_train"]
    n = len(X)
    if n > _MI_MAX_SAMPLES:
        idx = rng.choice(n, size=_MI_MAX_SAMPLES, replace=False)
        X = X[idx]
    else:
        idx = slice(None)

    if task == "bc":
        y = data["y_train"][idx]                       # [n, action_dim]
        scores = np.zeros(X.shape[1], dtype=np.float64)
        for j in range(y.shape[1]):
            scores += mutual_info_regression(X, y[:, j], random_state=seed)
        return scores / y.shape[1]

    # irl / pc → reward target
    r = data["rewards_train"]
    if r is None:
        raise ValueError(f"task={task!r} needs reward arrays for MI.")
    r = r[idx]
    return mutual_info_regression(X, r, random_state=seed)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    rng = np.random.default_rng(args.seed)

    data = load_task_dataset(args.dataset_path)
    d = data["n_features"]
    feature_names = data["feature_names"]

    t0 = time.time()
    if args.method == "random":
        scores = random_scores(d, rng)
    elif args.method == "mi":
        scores = mi_scores(args.task, data, rng, args.seed)
    else:
        raise ValueError(f"Unknown method {args.method!r}; expected random/mi.")
    ranking_time = time.time() - t0

    order = sorted(range(d), key=lambda i: scores[i], reverse=True)
    with open(out_dir / "ranking.csv", "w") as fh:
        fh.write("feature_index,feature_name,score,rank\n")
        for rank, idx in enumerate(order, start=1):
            fh.write(f"{idx},{feature_names[idx]},{scores[idx]:.6f},{rank}\n")

    save_json({
        "method": args.method,
        "task": args.task,
        "dataset_path": args.dataset_path,
        "num_features": d,
        "ranking_time_sec": ranking_time,
        "feature_names": feature_names,
    }, str(out_dir / "metadata.json"))
    print(f"[baseline_rank] method={args.method} task={args.task} done "
          f"({ranking_time:.2f}s). Saved to {out_dir / 'ranking.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random / MI baseline feature ranking.")
    p.add_argument("--method", required=True, choices=["random", "mi"])
    p.add_argument("--task", required=True, choices=["bc", "irl", "pc"],
                   help="Target for MI (random ignores it). bc→actions, irl/pc→reward.")
    p.add_argument("--config", default=None, help="Unused; accepted for pipeline uniformity.")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
