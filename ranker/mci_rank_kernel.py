"""HDC-encoded Marginal Contribution Importance (MCI) feature ranking.

This module provides a drop-in replacement for the SHAP-based explanation pipeline
(shap_behavior.py + global_rank.py). It produces a ranking.csv with the same schema,
so all downstream scripts (train_student.py, eval_*.py, make_plots.py) are unchanged.

Method overview (see docs/hdc_encoded_mci.tex for derivation):

1. Encode each feature independently with Random Fourier Features (RFF):
       z_j(x_j) = sqrt(2/D_j) * cos(W_j * x_j + b_j)

2. For any subset S, fit ridge regression (closed-form) to predict y:
       w_S* = (Z_S^T Z_S + λI)^{-1} Z_S^T y

3. Define predictive power of subset S:
       ν(S) = Var(y) - MSE(S)
   where MSE(S) = mean((y - Z_S @ w_S*)^2).

4. Score feature i via Marginal Contribution Importance (MCI):
       Î(i) = max_{S ⊆ F\{i}} [ν(S ∪ {i}) - ν(S)]

   In practice we approximate the max by sampling n_perms random subsets S.

Key advantage over SHAP:
- No MLP student pre-training required (works directly on the dataset).
- Ridge regression is closed-form → each ν(S) evaluation is O((|S|*D)^2 N).
- MCI has a theoretical upper-bound property: ν(S)/ν(F) ≤ Σ_{i∈S} Ĩ(i).

Why chosen_action_prob as the regression target:
    The teacher's chosen_action_prob = softmax(Q)[argmax(Q)] is a smooth scalar
    that reflects policy confidence. It is a better regression target than hard
    action labels (which would require multi-output or classification). The same
    rationale applies as in the SHAP pipeline.

Output files (in --output_dir):
    ranking.csv        — feature_index, feature_name, mean_mci, rank
    mci_scores.json    — per-feature MCI + marginal delta statistics
    metadata.json      — run parameters
    resolved_config.yaml

Usage:
    python explain/mci_rank.py \\
        --config configs/taxi_noise8.yaml \\
        --seed 0 \\
        --dataset_path outputs/datasets/taxi_noise8/seed0/dataset.npz \\
        --output_dir outputs/rankings_mci/taxi_noise8/seed0

Then use the ranking in student training exactly like SHAP ranking:
    python student/train_student.py \\
        --selector shap \\
        --ranking_path outputs/rankings_mci/taxi_noise8/seed0/ranking.csv ...
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

from ranker.hdc_encoder import FeatureWiseHDCEncoder
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_npz, save_csv, save_json
from utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Ridge-based predictive power
# ---------------------------------------------------------------------------

def _ridge_predictive_power(
    Z_S: torch.Tensor,
    y: torch.Tensor,
    var_y: float,
    lambda_: float,
) -> float:
    """Compute ν(S) = Var(y) - MSE(S) via closed-form ridge regression.

    Operates on GPU tensors when available; identical math to the numpy version.

    Args:
        Z_S:     Encoded subset matrix, shape [N, |S|*rff_dim] (torch.Tensor).
        y:       Regression targets, shape [N] (torch.Tensor).
        var_y:   Pre-computed Var(y) — constant across all calls.
        lambda_: Ridge regularisation strength.

    Returns:
        Scalar predictive power ν(S) ∈ (-∞, Var(y)].
    """
    p = Z_S.shape[1]

    if p == 0:
        return 0.0

    # w* = (Z^T Z + λI)^{-1} Z^T y
    A = Z_S.T @ Z_S + lambda_ * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    b = Z_S.T @ y
    w = torch.linalg.solve(A, b)

    y_hat = Z_S @ w
    mse = float(torch.mean((y - y_hat) ** 2).item())
    return float(var_y - mse)


def _ridge_predictive_power_multi(
    Z_S: torch.Tensor,
    y: torch.Tensor,
    var_y_sum: float,
    lambda_: float,
) -> float:
    """Compute ν(S) = Σ_j Var(y_j) - Σ_j MSE_j(S) for multi-output targets.

    Used for continuous-action tasks where y is [N, action_dim].
    Solves one ridge system and computes MSE across all output dimensions.

    Args:
        Z_S:       Encoded subset matrix, shape [N, |S|*rff_dim].
        y:         Multi-output targets, shape [N, action_dim].
        var_y_sum: Pre-computed Σ_j Var(y_j) — constant across calls.
        lambda_:   Ridge regularisation.

    Returns:
        Scalar predictive power (sum across output dims).
    """
    p = Z_S.shape[1]
    if p == 0:
        return 0.0

    # W* = (Z^T Z + λI)^{-1} Z^T y  — each column of y is a separate regression
    A = Z_S.T @ Z_S + lambda_ * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    W = torch.linalg.solve(A, Z_S.T @ y)  # [p, action_dim]

    y_hat = Z_S @ W                        # [N, action_dim]
    mse_sum = float(torch.mean((y - y_hat) ** 2, dim=0).sum().item())
    return float(var_y_sum - mse_sum)


# ---------------------------------------------------------------------------
# MCI computation
# ---------------------------------------------------------------------------

def compute_mci(
    encoder: FeatureWiseHDCEncoder,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    n_perms: int,
    rng: np.random.Generator,
    device: torch.device | None = None,
    multi_output: bool = False,
) -> tuple[np.ndarray, list[list[float]]]:
    """Compute MCI scores for all features via permutation sampling.

    For each feature i, we approximate:
        Î(i) = max_{S ⊆ F\\{i}} [ν(S ∪ {i}) - ν(S)]

    by sampling n_perms random subsets S (of random size) from F\\{i} and
    taking the maximum observed marginal gain.

    Args:
        encoder:      Fitted FeatureWiseHDCEncoder (call precompute first).
        X:            Training observations, shape [N, d].
        y:            Regression targets — shape [N] (scalar) or [N, action_dim] (multi).
        lambda_:      Ridge regularisation.
        n_perms:      Number of random subsets to sample per feature.
        rng:          Seeded random generator.
        device:       Torch device for ridge computation. Defaults to CUDA if available.
        multi_output: If True, use multi-output ridge (for continuous action targets).

    Returns:
        Tuple of:
          mci_scores: np.ndarray shape [d] — MCI score per feature.
          delta_lists: list of lists — raw marginal deltas for each feature.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = encoder.n_features
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    if multi_output:
        # ν(S) = Σ_j Var(y_j) - Σ_j MSE_j(S)
        y_np = np.array(y, dtype=np.float64)
        var_y_sum = float(np.var(y_np, axis=0).sum())
        power_fn = lambda Z_S, Z_Si: (  # noqa: E731
            _ridge_predictive_power_multi(Z_S,  y_t, var_y_sum, lambda_),
            _ridge_predictive_power_multi(Z_Si, y_t, var_y_sum, lambda_),
        )
    else:
        var_y = float(np.var(y))
        power_fn = lambda Z_S, Z_Si: (  # noqa: E731
            _ridge_predictive_power(Z_S,  y_t, var_y, lambda_),
            _ridge_predictive_power(Z_Si, y_t, var_y, lambda_),
        )

    mci_scores = np.zeros(d, dtype=np.float64)
    delta_lists: list[list[float]] = [[] for _ in range(d)]

    for i in range(d):
        other = [j for j in range(d) if j != i]
        deltas = []

        for _ in range(n_perms):
            size = int(rng.integers(0, len(other) + 1))
            S = list(rng.choice(other, size=size, replace=False))

            # transform_subset returns numpy; move to device for GPU ridge solve
            Z_S  = torch.tensor(encoder.transform_subset(X, S),        dtype=torch.float32, device=device)
            Z_Si = torch.tensor(encoder.transform_subset(X, S + [i]),  dtype=torch.float32, device=device)

            nu_S, nu_Si = power_fn(Z_S, Z_Si)
            deltas.append(nu_Si - nu_S)

        mci_scores[i] = max(deltas)
        delta_lists[i] = deltas

    return mci_scores, delta_lists


# ---------------------------------------------------------------------------
# Target extraction helpers
# ---------------------------------------------------------------------------

def extract_target(
    data: dict[str, np.ndarray], target: str
) -> tuple[np.ndarray, bool]:
    """Extract regression target from dataset dict.

    Args:
        data:   Dataset dict loaded from dataset.npz.
        target: One of:
                  "chosen_action_prob" — scalar policy confidence (Taxi/DQN)
                  "action_label"       — discrete action index as float (Taxi)
                  "action_norm"        — ||a||_2 per step (continuous tasks)
                  "action_multi"       — full action matrix [N, action_dim] (continuous)

    Returns:
        Tuple of (y, multi_output):
          y:            [N] float64 for scalar targets, [N, action_dim] for multi.
          multi_output: True when y is 2-D (triggers multi-output ridge in compute_mci).
    """
    if target == "chosen_action_prob":
        key = "chosen_prob_train"
        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in dataset. "
                "Ensure collect_dataset.py was run with full output schema."
            )
        return data[key].astype(np.float64).ravel(), False

    elif target == "action_label":
        key = "y_train"
        if key not in data:
            raise KeyError(f"Key '{key}' not found in dataset.")
        y = data[key].astype(np.float64)
        if y.ndim > 1:
            # Continuous task: fall back to action_norm
            return np.linalg.norm(y, axis=1), False
        return y.ravel(), False

    elif target == "action_norm":
        # Scalar L2 norm of continuous actions
        if "action_norm_train" in data:
            return data["action_norm_train"].astype(np.float64).ravel(), False
        y_train = data.get("y_train")
        if y_train is None:
            raise KeyError("Neither 'action_norm_train' nor 'y_train' in dataset.")
        y_arr = np.array(y_train, dtype=np.float64)
        if y_arr.ndim == 1:
            return y_arr, False
        return np.linalg.norm(y_arr, axis=1), False

    elif target == "action_multi":
        # Multi-output: full action matrix [N, action_dim]
        y_train = data.get("y_train")
        if y_train is None:
            raise KeyError("'y_train' not found in dataset (needed for action_multi).")
        y_arr = np.array(y_train, dtype=np.float64)
        if y_arr.ndim == 1:
            raise ValueError(
                "target='action_multi' requires continuous actions (y_train.ndim==2), "
                "but got 1-D array. Use target='action_label' for discrete tasks."
            )
        return y_arr, True

    else:
        raise ValueError(
            f"Unsupported target: {target!r}. "
            "Choose from: chosen_action_prob, action_label, action_norm, action_multi."
        )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Main MCI ranking routine."""
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    # --- Load dataset ---
    data = load_npz(args.dataset_path)
    X_train = data["X_train"].astype(np.float32)  # [N, d]
    y, multi_output = extract_target(data, args.target)

    # Feature names
    if "feature_names" in data:
        feature_names = [str(f) for f in data["feature_names"]]
    else:
        feature_names = [f"feature_{j}" for j in range(X_train.shape[1])]

    d = X_train.shape[1]
    N = X_train.shape[0]

    print(
        f"[mci_rank] N={N}, d={d}, target={args.target}, "
        f"multi_output={multi_output}, "
        f"rff_dim={args.rff_dim}, bandwidth={args.bandwidth}, "
        f"lambda={args.lambda_}, n_perms={args.n_perms}"
    )

    # Subsample to max_samples for MCI — ridge ranks features relatively,
    # so a smaller N is fine and avoids O(N·p²) blowup in Z_S.T @ Z_S.
    if args.max_samples and N > args.max_samples:
        idx = rng.choice(N, size=args.max_samples, replace=False)
        X_mci = X_train[idx]
        y_mci = y[idx]
        print(f"[mci_rank] Subsampled {N} → {args.max_samples} rows for MCI computation.")
    else:
        X_mci = X_train
        y_mci = y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mci_rank] device={device}")

    # --- HDC encoding ---
    encoder = FeatureWiseHDCEncoder(
        rff_dim=args.rff_dim,
        bandwidth=args.bandwidth,
        seed=args.seed,
    )
    encoder.fit(X_mci)
    encoder.precompute(X_mci)  # cache all Z_j blocks — avoids recomputing cos(Wx+b) per call
    print(f"[mci_rank] HDC encoder fitted and cached "
          f"({d} features × {args.rff_dim} RFF dims = {d * args.rff_dim} total dims).")

    # --- MCI computation ---
    start = time.time()
    mci_scores, delta_lists = compute_mci(
        encoder=encoder,
        X=X_mci,
        y=y_mci,
        lambda_=args.lambda_,
        n_perms=args.n_perms,
        rng=rng,
        device=device,
        multi_output=multi_output,
    )
    elapsed = time.time() - start
    print(f"[mci_rank] MCI done in {elapsed:.1f}s.")

    # --- Build ranking DataFrame ---
    # Rank 1 = most important feature (highest MCI)
    order = np.argsort(mci_scores)[::-1]        # indices sorted by descending MCI
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(1, d + 1)

    df = pd.DataFrame({
        "feature_index": list(range(d)),
        "feature_name":  feature_names,
        "mean_mci":      mci_scores,
        "rank":          ranks,
    })
    df = df.sort_values("rank").reset_index(drop=True)

    save_csv(df, str(out_dir / "ranking.csv"))

    # --- Save MCI scores detail ---
    mci_detail = {
        "feature_names": feature_names,
        "mci_scores": mci_scores.tolist(),
        "delta_stats": [
            {
                "feature": feature_names[i],
                "max_delta": float(max(delta_lists[i])),
                "mean_delta": float(np.mean(delta_lists[i])),
                "min_delta": float(min(delta_lists[i])),
            }
            for i in range(d)
        ],
        "elapsed_s": elapsed,
        "var_y": float(np.var(y_mci, axis=0).sum() if multi_output else np.var(y_mci)),
        "multi_output": multi_output,
    }
    save_json(mci_detail, str(out_dir / "mci_scores.json"))

    metadata = {
        "seed": args.seed,
        "config": args.config,
        "dataset_path": args.dataset_path,
        "output_dir": str(out_dir),
        "target": args.target,
        "rff_dim": args.rff_dim,
        "bandwidth": args.bandwidth,
        "lambda_": args.lambda_,
        "n_perms": args.n_perms,
        "n_train": N,
        "n_features": d,
        "feature_names": feature_names,
        "elapsed_s": elapsed,
    }
    save_json(metadata, str(out_dir / "metadata.json"))

    print(f"\n[mci_rank] Top {min(10, d)} features by MCI:")
    print(df.head(min(10, d)).to_string(index=False))
    print(f"\n[mci_rank] Saved ranking to {out_dir / 'ranking.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDC-encoded MCI feature ranking (drop-in for SHAP pipeline)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", required=True, help="Path to dataset.npz.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--target",
        default="chosen_action_prob",
        choices=["chosen_action_prob", "action_label", "action_norm", "action_multi"],
        help=(
            "Regression target for ridge model. "
            "'chosen_action_prob' — smooth policy confidence scalar (discrete tasks). "
            "'action_label'       — discrete action index (discrete tasks). "
            "'action_norm'        — ||a||_2 scalar (continuous tasks). "
            "'action_multi'       — multi-output ridge over all action dims (continuous tasks, default for D4RL)."
        ),
    )
    parser.add_argument(
        "--rff_dim",
        type=int,
        default=64,
        help="Number of Random Fourier Feature dimensions per input feature (D_j).",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=1.0,
        help="RFF bandwidth (std of frequency distribution). Controls kernel length-scale.",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1e-3,
        help="Ridge regularisation strength λ.",
    )
    parser.add_argument(
        "--n_perms",
        type=int,
        default=200,
        help="Number of random subset samples per feature for MCI approximation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help=(
            "Subsample training rows to this size before MCI. "
            "Avoids O(N·p²) blowup in ridge regression with large N. "
            "Set 0 or omit to use full dataset (slow for N>5000)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
