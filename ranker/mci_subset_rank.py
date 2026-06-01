"""MCI feature ranking by per-subset *student* retraining (continuous bc/irl/pc).

The NN counterpart of ranker/mci_rank_kernel.py: instead of a closed-form ridge,
the predictive power nu(S) of a feature subset S is obtained by actually training
the per-task student model on S and reading its held-out score — the *same model
frame* as student/train_student.py (imitation BC for bc, RewardMLP for irl/pc):

    nu(S) = bc  → test R^2 of the BC policy's action prediction
            irl → test R^2 of the reward regressor
            pc  → test preference-prediction accuracy

This generalises the classic MCI-NN (Catav et al., per-subset retraining) to the
continuous bc/irl/pc tasks, where the original ranker/mci_rank_nn.py cannot run
(its irl branch is tabular MCE-IRL; its pc branch needs array params via CLI).

MCI is estimated by permutation sampling (mirrors mci_rank_kernel.compute_mci):
for each of `--n_perms` random feature orderings, feature p[i]'s contribution is
nu(p[:i+1]) - nu(p[:i]); a set-keyed cache dedupes nu(S) across permutations.
Because each nu(S) is a full model training, keep n_perms small and use
`--subset_epochs` / `--subset_max_samples` to bound cost.

`ranking_time_sec` in metadata INCLUDES all per-subset student trainings.

Output: ranking.csv (feature_index, feature_name, mean_mci, rank), mci_scores.json,
metadata.json.

Usage:
    python ranker/mci_subset_rank.py \\
        --config configs/seals_ant.yaml --task bc \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir outputs/rankings_mci_nn/seals_ant/bc/seed0 \\
        --seed 0 --n_perms 4 --subset_epochs 20
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.train_bc_continuous import setup_run
from student.train_student import load_task_dataset, train_and_eval_student
from utils.io import save_csv, save_json
from utils.seed import set_global_seed

# nu(S) metric per task (all higher = better) + the empty-set baseline.
_NU_METRIC = {"bc": "r2", "irl": "r2", "pc": "pref_accuracy"}
_NU_EMPTY = {"bc": 0.0, "irl": 0.0, "pc": 0.5}


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    data = load_task_dataset(args.dataset_path)
    d = data["n_features"]
    feature_names = data["feature_names"]
    metric = _NU_METRIC[args.task]

    # Fast cfg for the many per-subset trainings (override epochs + sample cap).
    fast_cfg = copy.deepcopy(cfg)
    fast_cfg["student"]["epochs"] = args.subset_epochs
    cur_cap = fast_cfg["student"].get("max_train_samples") or args.subset_max_samples
    fast_cfg["student"]["max_train_samples"] = min(cur_cap, args.subset_max_samples)

    print(f"[mci_subset] task={args.task}, d={d}, n_perms={args.n_perms}, "
          f"metric={metric}, subset_epochs={args.subset_epochs}, device={device}")

    nu_cache: dict[tuple[int, ...], float] = {}

    def nu(prefix: list[int]) -> float:
        if len(prefix) == 0:
            return _NU_EMPTY[args.task]
        key = tuple(sorted(prefix))
        val = nu_cache.get(key)
        if val is None:
            _model, test = train_and_eval_student(
                args.task, data, list(prefix), fast_cfg, args.seed, device, args)
            val = float(test[metric])
            nu_cache[key] = val
        return val

    max_contrib = np.zeros(d, dtype=np.float64)
    sum_contrib = np.zeros(d, dtype=np.float64)
    n_contrib = np.zeros(d, dtype=np.float64)
    delta_lists: list[list[float]] = [[] for _ in range(d)]

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for _ in tqdm(range(args.n_perms), desc="permutations"):
        perm = [int(j) for j in rng.permutation(d)]
        prev = nu(perm[:0])
        for i in range(d):
            cur = nu(perm[: i + 1])
            contribution = cur - prev
            f = perm[i]
            delta_lists[f].append(contribution)
            if contribution > max_contrib[f]:
                max_contrib[f] = contribution
            sum_contrib[f] += contribution
            n_contrib[f] += 1
            prev = cur
    ranking_time = time.time() - t0
    shapley = sum_contrib / np.maximum(n_contrib, 1)

    order = np.argsort(max_contrib)[::-1]
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(1, d + 1)
    df = pd.DataFrame({
        "feature_index": list(range(d)),
        "feature_name": feature_names,
        "mean_mci": max_contrib,
        "rank": ranks,
    }).sort_values("rank").reset_index(drop=True)
    save_csv(df, str(out_dir / "ranking.csv"))

    save_json({
        "task": args.task,
        "feature_names": feature_names,
        "mci_scores": max_contrib.tolist(),
        "shapley_values": shapley.tolist(),
    }, str(out_dir / "mci_scores.json"))

    save_json({
        "ranking_method": "mci_nn",
        "task": args.task,
        "dataset_path": args.dataset_path,
        "num_features": d,
        "n_perms": args.n_perms,
        "subset_epochs": args.subset_epochs,
        "nu_metric": metric,
        "n_subsets_evaluated": len(nu_cache),
        "ranking_time_sec": ranking_time,
        "feature_names": feature_names,
    }, str(out_dir / "metadata.json"))

    print(f"[mci_subset] task={args.task} done. ranking_time={ranking_time:.1f}s, "
          f"{len(nu_cache)} subsets. Saved to {out_dir / 'ranking.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCI by per-subset student retraining (continuous bc/irl/pc)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, choices=["bc", "irl", "pc"])
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_perms", type=int, default=4,
                   help="Random feature permutations (each nu(S) is a full training).")
    p.add_argument("--subset_epochs", type=int, default=20,
                   help="Epochs for each per-subset student (kept small for speed).")
    p.add_argument("--subset_max_samples", type=int, default=4000,
                   help="Row cap for each per-subset training.")
    # pc-only knobs consumed by train_and_eval_student.
    p.add_argument("--fragment_length", type=int, default=50)
    p.add_argument("--num_pairs", type=int, default=200)
    p.add_argument("--num_eval_pairs", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
