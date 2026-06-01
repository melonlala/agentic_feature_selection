"""KernelSHAP feature ranking for the bc / irl / pc tasks.

KernelSHAP (Lundberg & Lee, 2017) attributes a single trained model's output to
its input features by sampling feature coalitions and solving a weighted least
squares. Unlike MCI (which *retrains* a model per feature subset), KernelSHAP
trains **one** full-feature model and explains its predictions.

To stay comparable with the MCI rankers, the one trained model is exactly the
per-task student of student/train_student.py — the *same model frame as MCI*:

    bc  : imitation BC policy (SB3 ActorCriticPolicy). Output = action vector;
          attribution aggregated over action dims.
    irl : RewardMLP regressing ground-truth reward. Output = scalar reward.
    pc  : RewardMLP trained by Bradley-Terry preference CE. Output = scalar
          reward head.

Global score: score[j] = mean over explained samples and output dims of
|SHAP_{sample, out, j}|.

Ranking time: the reported `ranking_time_sec` INCLUDES the one-time model
training (model_training_time_sec) plus the KernelSHAP attribution
(attribution_time_sec); they are also reported separately in metadata.

Output (in --output_dir):
    ranking.csv          — feature_index, feature_name, score, rank (rank 1 = top)
    kernelshap_values.npz — shap_values [out, K, d]
    metadata.json        — task, params, timing breakdown, test metrics
    resolved_config.yaml

Usage:
    python ranker/kernelshap.py \\
        --config       configs/seals_ant.yaml \\
        --task         irl \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir   outputs/rankings_kernelshap/seals_ant/seed0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.sage_rank import build_predict_fn  # shared task predict callable
from student.train_student import load_task_dataset, train_and_eval_student
from utils.io import save_json
from utils.seed import set_global_seed


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
    print(f"[kernelshap] trained {args.task} model on {n_features} features "
          f"in {train_time:.1f}s; test={test_metrics}")

    # ── KernelSHAP attribution on the trained model — timed ──
    predict = build_predict_fn(args.task, model, device)
    rng = np.random.default_rng(args.seed)

    bg_n = min(args.background_size, len(data["X_train"]))
    bg_idx = rng.choice(len(data["X_train"]), size=bg_n, replace=False)
    # Summarise the background into K centroids to keep KernelSHAP tractable.
    background = shap.kmeans(data["X_train"][bg_idx].astype(np.float32), args.n_background_clusters)

    X = data["X_val"].astype(np.float32)
    if args.explain_size and len(X) > args.explain_size:
        ex_idx = rng.choice(len(X), size=args.explain_size, replace=False)
        X = X[ex_idx]

    t1 = time.time()
    explainer = shap.KernelExplainer(predict, background, link="identity")
    shap_values = explainer.shap_values(X, nsamples=args.nsamples, silent=False)
    attribution_time = time.time() - t1
    ranking_time = time.time() - t0  # INCLUDES one-time model training

    # shap returns [N, d, out] (out=1 for irl/pc, action_dim for bc); some
    # versions drop the trailing axis for single output → [N, d].
    arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]                     # [N, d] → [N, d, 1]
    scores = np.mean(np.abs(arr), axis=(0, 2))   # → [d]

    order = sorted(range(n_features), key=lambda i: scores[i], reverse=True)
    with open(out_dir / "ranking.csv", "w") as fh:
        fh.write("feature_index,feature_name,score,rank\n")
        for rank, idx in enumerate(order, start=1):
            fh.write(f"{idx},{feature_names[idx]},{scores[idx]:.6f},{rank}\n")

    np.savez(out_dir / "kernelshap_values.npz", shap_values=arr)  # [N, d, out]

    save_json({
        "task": args.task,
        "ranking_method": "kernelshap",
        "dataset_path": args.dataset_path,
        "num_features": n_features,
        "num_explain_samples": int(len(X)),
        "background_clusters": int(args.n_background_clusters),
        "nsamples": args.nsamples,
        "model_training_time_sec": train_time,
        "attribution_time_sec": attribution_time,
        "ranking_time_sec": ranking_time,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
    }, str(out_dir / "metadata.json"))

    print(f"[kernelshap] task={args.task} done. "
          f"ranking_time={ranking_time:.1f}s "
          f"(train={train_time:.1f}s + attribution={attribution_time:.1f}s). "
          f"Saved to {out_dir / 'ranking.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KernelSHAP feature ranking (bc/irl/pc).")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--task", required=True, choices=["bc", "irl", "pc"],
                   help="Which task model to train + attribute.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--background_size", type=int, default=512,
                   help="Training rows sampled before kmeans summarisation.")
    p.add_argument("--n_background_clusters", type=int, default=32,
                   help="KernelSHAP background centroids (shap.kmeans K).")
    p.add_argument("--explain_size", type=int, default=200,
                   help="Number of rows to attribute (0 = all val rows).")
    p.add_argument("--nsamples", default="auto",
                   help="KernelSHAP coalition samples per row ('auto' or int).")
    # pc-only knobs consumed by train_and_eval_student.
    p.add_argument("--fragment_length", type=int, default=50)
    p.add_argument("--num_pairs", type=int, default=200)
    p.add_argument("--num_eval_pairs", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if isinstance(args.nsamples, str) and args.nsamples.isdigit():
        args.nsamples = int(args.nsamples)
    run(args)
