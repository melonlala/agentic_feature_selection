"""HDC-encoded Marginal Contribution Importance (MCI) feature ranking.

Kernel/regression counterpart of ranker/mci_rank_nn.py. Instead of *training* an
imitation model per feature subset, predictive power nu(S) is obtained in closed
form by ridge regression on Random-Fourier-Feature (RFF) encoded features. The
three evaluators mirror the three branches of mci_rank_nn.py — the only thing
that changes between them is the **regression target** (and how it is scored):

    bc  : target = expert actions Y.
          nu(S) = explained variance (R^2-style) of the action regression,
          summed over action dims. The regression analog of behavior cloning.

    irl : target = ground-truth per-transition reward.
          nu(S) = explained variance of the reward regression. The regression
          analog of recovering a reward from a feature subset.

    pc  : target = ground-truth per-transition reward, but scored as preference
          accuracy. A ridge reward model is fit on subset S, predicted rewards
          are summed over fragments, and nu(S) is the fraction of fragment pairs
          ranked the same way as the ground-truth (summed true reward) labels.
          The regression analog of preference-comparison reward learning.

Method (see docs/hdc_encoded_mci.tex for derivation):

1. Encode each feature independently with Random Fourier Features (RFF):
       z_j(x_j) = sqrt(2/D_j) * cos(W_j * x_j + b_j)

2. For any subset S, fit ridge regression (closed-form) to predict the target:
       w_S* = (Z_S^T Z_S + lambda I)^{-1} Z_S^T t

3. Define predictive power nu(S) per the evaluator above (explained variance for
   bc/irl, preference accuracy for pc).

4. Score feature i via Marginal Contribution Importance (MCI):
       I(i) = max_{S subset F\\{i}} [nu(S u {i}) - nu(S)]
   approximated by permutation sampling (n_perms random feature orderings; the
   context S for feature p[i] is its prefix p[:i]), mirroring mci_rank_nn.

Output files (in --output_dir):
    ranking.csv        — feature_index, feature_name, mean_mci, rank
    mci_scores.json    — per-feature MCI + marginal delta statistics
    metadata.json      — run parameters, ranking time, output file paths.

HDC params (rff_dim, bandwidth, lambda_, n_perms, max_samples) live in
configs/mci_kernel.yaml under the `mci_kernel` block; CLI flags override them.

Usage:
    python ranker/mci_rank_kernel.py \\
        --evaluator_name bc \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir outputs/rankings_mci_kernel/seals_ant/seed0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.hdc_encoder import FeatureWiseHDCEncoder
from utils.config import load_yaml_config
from utils.io import ensure_dir, load_npz, save_csv, save_json
from utils.seed import set_global_seed


# Fixed HDC / RFF-ridge + pc-fragment parameters. These are not CLI flags; they
# live in configs/mci_kernel.yaml under the `mci_kernel` block. The dict below is
# the hardcoded fallback used for any key the config omits.
HDC_PARAM_DEFAULTS = {
    "rff_dim": 64,
    "bandwidth": 1.0,
    "lambda_": 1e-3,
    "n_perms": 200,
    "max_samples": 3000,
    "fragment_length": 50,
    "num_pairs": 50,
}


def apply_hdc_config(args: argparse.Namespace) -> None:
    """Set the fixed HDC params on `args` from the `mci_kernel` config block.

    These parameters are config-only (no CLI flags). Each is read from the
    `mci_kernel` block of args.config, falling back to HDC_PARAM_DEFAULTS for any
    key the config omits.
    """
    cfg = load_yaml_config(args.config).get("mci_kernel", {}) if args.config else {}
    for key, default in HDC_PARAM_DEFAULTS.items():
        setattr(args, key, cfg.get(key, default))


# ---------------------------------------------------------------------------
# Ridge-based predictive power (explained variance) — used by bc / irl
# ---------------------------------------------------------------------------

def _ridge_predictive_power(
    Z_S: torch.Tensor,
    y: torch.Tensor,
    var_y: float,
    lambda_: float,
) -> float:
    """Compute nu(S) = Var(y) - MSE(S) via closed-form ridge regression.

    Args:
        Z_S:     Encoded subset matrix, shape [N, |S|*rff_dim].
        y:       Regression targets, shape [N].
        var_y:   Pre-computed Var(y) — constant across all calls.
        lambda_: Ridge regularisation strength.

    Returns:
        Scalar predictive power nu(S) in (-inf, Var(y)].
    """
    p = Z_S.shape[1]
    if p == 0:
        return 0.0

    A = Z_S.T @ Z_S + lambda_ * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    w = torch.linalg.solve(A, Z_S.T @ y)

    y_hat = Z_S @ w
    mse = float(torch.mean((y - y_hat) ** 2).item())
    return float(var_y - mse)


def _ridge_predictive_power_multi(
    Z_S: torch.Tensor,
    y: torch.Tensor,
    var_y_sum: float,
    lambda_: float,
) -> float:
    """Compute nu(S) = Sum_j Var(y_j) - Sum_j MSE_j(S) for multi-output targets.

    Args:
        Z_S:       Encoded subset matrix, shape [N, |S|*rff_dim].
        y:         Multi-output targets, shape [N, target_dim].
        var_y_sum: Pre-computed Sum_j Var(y_j) — constant across calls.
        lambda_:   Ridge regularisation.

    Returns:
        Scalar predictive power (sum across output dims).
    """
    p = Z_S.shape[1]
    if p == 0:
        return 0.0

    A = Z_S.T @ Z_S + lambda_ * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    W = torch.linalg.solve(A, Z_S.T @ y)  # [p, target_dim]

    y_hat = Z_S @ W                        # [N, target_dim]
    mse_sum = float(torch.mean((y - y_hat) ** 2, dim=0).sum().item())
    return float(var_y_sum - mse_sum)


# ---------------------------------------------------------------------------
# Ridge-based preference accuracy — used by pc
# ---------------------------------------------------------------------------

def _ridge_preference_accuracy(
    Z_S: torch.Tensor,
    r: torch.Tensor,
    frag_a_idx: torch.Tensor,
    frag_b_idx: torch.Tensor,
    labels: torch.Tensor,
    lambda_: float,
) -> float:
    """Compute nu(S) = preference-prediction accuracy of a ridge reward model.

    Fit closed-form ridge to predict per-transition reward r from Z_S, then for
    each fragment pair (A, B) predict A > B iff its summed predicted reward is
    larger, and score against the ground-truth labels (summed *true* reward).

    Args:
        Z_S:        Encoded subset matrix, shape [N, |S|*rff_dim].
        r:          Per-transition reward targets, shape [N].
        frag_a_idx: Transition indices of fragment A, shape [num_pairs, L] (long).
        frag_b_idx: Transition indices of fragment B, shape [num_pairs, L] (long).
        labels:     Ground-truth preferences (A > B), shape [num_pairs] (bool).
        lambda_:    Ridge regularisation strength.

    Returns:
        Preference-prediction accuracy in [0, 1]. With no features (p == 0) the
        predicted reward is constant, all fragments tie -> chance accuracy 0.5.
    """
    p = Z_S.shape[1]
    if p == 0:
        return 0.5

    A = Z_S.T @ Z_S + lambda_ * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    w = torch.linalg.solve(A, Z_S.T @ r)
    r_hat = (Z_S @ w).reshape(-1)            # [N] predicted per-transition reward

    ret_a = r_hat[frag_a_idx].sum(dim=1)     # [num_pairs] predicted fragment returns
    ret_b = r_hat[frag_b_idx].sum(dim=1)
    pred = ret_a > ret_b
    return float((pred == labels).float().mean().item())


# ---------------------------------------------------------------------------
# Evaluator factory — mirrors mci_rank_nn.create_evaluator, regression backend
# ---------------------------------------------------------------------------

def create_evaluator(evaluator_name: str, targets: dict, lambda_: float):
    """Build evaluator(Z) -> float nu(S) for the chosen branch.

    `targets` holds the pre-built (subset-independent) regression target tensors
    produced by build_targets(); only the encoded subset matrix Z varies per call.
    """
    if evaluator_name == "bc":
        y = targets["y"]                 # [N, target_dim]
        var_y_sum = targets["var_y_sum"]

        def evaluator(Z: torch.Tensor) -> float:
            return _ridge_predictive_power_multi(Z, y, var_y_sum, lambda_)

        return evaluator

    if evaluator_name == "irl":
        r = targets["r"]                 # [N]
        var_r = targets["var_r"]

        def evaluator(Z: torch.Tensor) -> float:
            return _ridge_predictive_power(Z, r, var_r, lambda_)

        return evaluator

    if evaluator_name == "pc":
        r = targets["r"]                 # [N]
        frag_a = targets["frag_a"]
        frag_b = targets["frag_b"]
        labels = targets["labels"]

        def evaluator(Z: torch.Tensor) -> float:
            return _ridge_preference_accuracy(Z, r, frag_a, frag_b, labels, lambda_)

        return evaluator

    raise ValueError(
        f"Unknown evaluator_name {evaluator_name!r}; expected 'bc', 'irl' or 'pc'."
    )


# ---------------------------------------------------------------------------
# MCI computation
# ---------------------------------------------------------------------------

def compute_mci(
    encoder: FeatureWiseHDCEncoder,
    X: np.ndarray,
    evaluator,
    n_perms: int,
    rng: np.random.Generator,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[list[float]], np.ndarray]:
    """Compute MCI scores for all features via permutation sampling.

    Mirrors mci_rank_nn.PermutationSampling. We draw `n_perms` random
    permutations of all features; in a permutation p, feature p[i] is scored by
    its marginal contribution over its prefix context:

        contribution(p[i]) = nu(p[:i+1]) - nu(p[:i])

    MCI(i) is the running max of those contributions (floored at 0, as in
    ContributionTracker), and the running mean gives the Shapley value.

    Why permutations instead of independent per-feature subsets: each
    permutation evaluates the d+1 prefixes once and reads off *all* d features'
    contributions from them, with each nu(p[:i+1]) reused as both the
    "+feature" term for position i and the baseline for position i+1. A cache
    keyed by the (order-invariant) feature set further dedupes nu(S) across
    permutations. Same per-feature subset distribution as the old independent
    sampler, ~2x fewer ridge solves, plus the Shapley average for free.

    Args:
        encoder:   Fitted FeatureWiseHDCEncoder (call precompute first).
        X:         Training observations used for encoding, shape [N, d].
        evaluator: nu(S) callable from create_evaluator (takes encoded Z tensor).
        n_perms:   Number of random permutations to sample (each gives every
                   feature one prefix-context contribution sample).
        rng:       Seeded random generator.
        device:    Torch device for the ridge solve. Defaults to CUDA if available.

    Returns:
        Tuple of:
          mci_scores:  np.ndarray [d] — max marginal contribution per feature.
          delta_lists: list of lists  — raw marginal contributions per feature.
          shapley:     np.ndarray [d] — mean marginal contribution per feature.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = encoder.n_features

    # nu(S) cache keyed by the sorted feature-index tuple (order-invariant), so
    # prefixes shared within and across permutations are evaluated only once.
    nu_cache: dict[tuple[int, ...], float] = {}

    def nu(prefix: list[int]) -> float:
        key = tuple(sorted(prefix))
        val = nu_cache.get(key)
        if val is None:
            Z = torch.tensor(
                encoder.transform_subset(X, list(prefix)),
                dtype=torch.float32,
                device=device,
            )
            val = evaluator(Z)
            nu_cache[key] = val
        return val

    max_contrib = np.zeros(d, dtype=np.float64)   # floored at 0, mirrors tracker
    sum_contrib = np.zeros(d, dtype=np.float64)
    n_contrib = np.zeros(d, dtype=np.float64)
    delta_lists: list[list[float]] = [[] for _ in range(d)]

    for _ in tqdm(range(n_perms), desc="permutations"):
        perm = [int(j) for j in rng.permutation(d)]
        prev = nu(perm[:0])                       # nu(emptyset)
        for i in range(d):
            cur = nu(perm[: i + 1])               # chains: reused as next baseline
            contribution = cur - prev
            f = perm[i]
            delta_lists[f].append(contribution)
            if contribution > max_contrib[f]:
                max_contrib[f] = contribution
            sum_contrib[f] += contribution
            n_contrib[f] += 1
            prev = cur

    shapley = sum_contrib / np.maximum(n_contrib, 1)
    return max_contrib, delta_lists, shapley


# ---------------------------------------------------------------------------
# Target extraction — the core component that differentiates the evaluators
# ---------------------------------------------------------------------------

def extract_action_target(
    data: dict[str, np.ndarray], target: str
) -> np.ndarray:
    """Extract the bc regression target (expert actions) as a 2-D array [N, k].

    Args:
        data:   Dataset dict loaded from dataset.npz.
        target: One of:
                  "action_multi"       — full action matrix [N, action_dim] (default, continuous)
                  "action_norm"        — ||a||_2 per step (continuous) -> [N, 1]
                  "action_label"       — discrete action index as float -> [N, 1]
                  "chosen_action_prob" — scalar policy confidence (Taxi/DQN) -> [N, 1]
    """
    if target == "chosen_action_prob":
        if "chosen_prob_train" not in data:
            raise KeyError(
                "Key 'chosen_prob_train' not found in dataset. "
                "Ensure collect_dataset.py was run with full output schema."
            )
        return data["chosen_prob_train"].astype(np.float64).reshape(-1, 1)

    if target == "action_label":
        if "y_train" not in data:
            raise KeyError("Key 'y_train' not found in dataset.")
        y = np.asarray(data["y_train"], dtype=np.float64)
        if y.ndim > 1:
            # Continuous task: fall back to action norm.
            return np.linalg.norm(y, axis=1, keepdims=True)
        return y.reshape(-1, 1)

    if target == "action_norm":
        if "action_norm_train" in data:
            return data["action_norm_train"].astype(np.float64).reshape(-1, 1)
        y = data.get("y_train")
        if y is None:
            raise KeyError("Neither 'action_norm_train' nor 'y_train' in dataset.")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return np.linalg.norm(y, axis=1, keepdims=True)

    if target == "action_multi":
        y = data.get("y_train")
        if y is None:
            raise KeyError("'y_train' not found in dataset (needed for action_multi).")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    raise ValueError(
        f"Unsupported bc target: {target!r}. Choose from: "
        "action_multi, action_norm, action_label, chosen_action_prob."
    )


def extract_reward(data: dict[str, np.ndarray]) -> np.ndarray:
    """Extract per-transition ground-truth reward [N] (irl / pc target)."""
    for key in ("rewards_train", "rews_train", "reward_train", "rewards", "rews"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError(
        "No reward array found (looked for rewards_train/rews_train/...). "
        "Needed for the 'irl' and 'pc' evaluators."
    )


def extract_dones(data: dict[str, np.ndarray]) -> np.ndarray:
    """Extract per-transition episode-termination flags [N] (pc target)."""
    for key in ("dones_train", "terminals_train", "dones", "terminals"):
        if key in data:
            return np.asarray(data[key]).reshape(-1).astype(bool)
    raise KeyError(
        "No done/terminal array found (looked for dones_train/terminals_train/...). "
        "Needed for the 'pc' evaluator."
    )


def _contiguous_episode_prefix(dones: np.ndarray, max_samples: int) -> slice:
    """Largest whole-episode prefix with <= max_samples transitions.

    Subsampling random rows would break the trajectory contiguity that fragment
    returns need, so for pc we instead keep a contiguous prefix that ends on an
    episode boundary.
    """
    n = len(dones)
    if not max_samples or n <= max_samples:
        return slice(0, n)
    ends = np.where(dones)[0]
    valid = ends[ends < max_samples]
    cut = int(valid[-1]) + 1 if len(valid) else max_samples
    return slice(0, cut)


def build_fragment_pairs(
    rews: np.ndarray,
    dones: np.ndarray,
    fragment_length: int,
    num_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample contiguous fragment pairs and label them by ground-truth return.

    Mirrors imitation's RandomFragmenter + SyntheticGatherer: each fragment is a
    contiguous slice of length `fragment_length` lying within a single episode;
    the label for a pair (A, B) is 1 iff the true summed reward of A exceeds B's.

    Returns:
        frag_a_idx: [num_pairs, fragment_length] transition indices of A.
        frag_b_idx: [num_pairs, fragment_length] transition indices of B.
        labels:     [num_pairs] bool, True iff return(A) > return(B).
    """
    n = len(rews)
    ends = [i for i in range(n) if dones[i]]
    if not ends or ends[-1] != n - 1:
        ends.append(n - 1)

    # Episode (start, end) inclusive ranges.
    valid_starts: list[int] = []
    start = 0
    for end in ends:
        ep_len = end - start + 1
        if ep_len >= fragment_length:
            valid_starts.extend(range(start, end - fragment_length + 2))
        start = end + 1

    if not valid_starts:
        raise ValueError(
            f"No episode >= fragment_length={fragment_length}; "
            "reduce --fragment_length."
        )

    valid_starts = np.asarray(valid_starts)
    offsets = np.arange(fragment_length)
    sa = rng.choice(valid_starts, size=num_pairs)
    sb = rng.choice(valid_starts, size=num_pairs)
    frag_a = sa[:, None] + offsets[None, :]
    frag_b = sb[:, None] + offsets[None, :]

    labels = rews[frag_a].sum(axis=1) > rews[frag_b].sum(axis=1)
    return frag_a, frag_b, labels


def build_targets(
    data: dict[str, np.ndarray],
    X_train: np.ndarray,
    evaluator_name: str,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, dict]:
    """Build the regression target tensors + select the rows to encode (X_mci).

    Returns (targets, X_mci, info):
        targets : dict of torch tensors consumed by create_evaluator.
        X_mci   : observation rows the encoder/MCI run on (subsampled).
        info    : JSON-serialisable summary for metadata.
    """
    n = X_train.shape[0]
    max_samples = args.max_samples

    if evaluator_name == "bc":
        y = extract_action_target(data, args.target)  # [N, k]
        if max_samples and n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X_mci, y = X_train[idx], y[idx]
        else:
            X_mci = X_train
        var_y_sum = float(np.var(y, axis=0).sum())
        targets = {
            "y": torch.tensor(y, dtype=torch.float32, device=device),
            "var_y_sum": var_y_sum,
        }
        info = {"target": args.target, "target_dim": int(y.shape[1]), "var_y_sum": var_y_sum}
        return targets, X_mci, info

    if evaluator_name == "irl":
        r = extract_reward(data)
        if max_samples and n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X_mci, r = X_train[idx], r[idx]
        else:
            X_mci = X_train
        var_r = float(np.var(r))
        targets = {
            "r": torch.tensor(r, dtype=torch.float32, device=device),
            "var_r": var_r,
        }
        info = {"target": "reward", "var_r": var_r}
        return targets, X_mci, info

    if evaluator_name == "pc":
        r = extract_reward(data)
        dones = extract_dones(data)
        sl = _contiguous_episode_prefix(dones, max_samples)
        X_mci, r, dones = X_train[sl], r[sl], dones[sl]
        frag_a, frag_b, labels = build_fragment_pairs(
            r, dones, args.fragment_length, args.num_pairs, rng
        )
        targets = {
            "r": torch.tensor(r, dtype=torch.float32, device=device),
            "frag_a": torch.tensor(frag_a, dtype=torch.long, device=device),
            "frag_b": torch.tensor(frag_b, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.bool, device=device),
        }
        info = {
            "target": "reward",
            "n_pairs": int(len(labels)),
            "fragment_length": int(args.fragment_length),
            "label_pos_rate": float(np.mean(labels)),
        }
        return targets, X_mci, info

    raise ValueError(
        f"Unknown evaluator_name {evaluator_name!r}; expected 'bc', 'irl' or 'pc'."
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Main MCI ranking routine."""
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)

    # --- Load dataset ---
    data = load_npz(args.dataset_path)
    X_train = data["X_train"].astype(np.float32)  # [N, d]
    n_full, d = X_train.shape

    if "feature_names" in data:
        feature_names = [str(f) for f in data["feature_names"]]
    else:
        feature_names = [f"feature_{j}" for j in range(d)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    evaluator_name = args.evaluator_name

    print(
        f"[mci_rank] evaluator={evaluator_name}, N={n_full}, d={d}, "
        f"rff_dim={args.rff_dim}, bandwidth={args.bandwidth}, "
        f"lambda={args.lambda_}, n_perms={args.n_perms}, device={device}"
    )

    # --- Build per-evaluator regression target(s) + select rows ---
    targets, X_mci, target_info = build_targets(
        data, X_train, evaluator_name, args, device, rng
    )
    n = X_mci.shape[0]
    print(f"[mci_rank] using N={n} rows for MCI; target_info={target_info}")

    # --- HDC encoding ---
    encoder = FeatureWiseHDCEncoder(
        rff_dim=args.rff_dim, bandwidth=args.bandwidth, seed=args.seed
    )
    encoder.fit(X_mci)
    encoder.precompute(X_mci)
    print(
        f"[mci_rank] HDC encoder fitted and cached "
        f"({d} features x {args.rff_dim} RFF dims = {d * args.rff_dim} total dims)."
    )

    # --- MCI computation ---
    evaluator = create_evaluator(evaluator_name, targets, args.lambda_)
    start = time.time()
    mci_scores, delta_lists, shapley = compute_mci(
        encoder=encoder,
        X=X_mci,
        evaluator=evaluator,
        n_perms=args.n_perms,
        rng=np.random.default_rng(args.seed),
        device=device,
    )
    elapsed = time.time() - start
    print(f"[mci_rank] MCI done in {elapsed:.1f}s.")

    # --- Build ranking DataFrame (rank 1 = highest MCI) ---
    order = np.argsort(mci_scores)[::-1]
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(1, d + 1)

    df = pd.DataFrame({
        "feature_index": list(range(d)),
        "feature_name":  feature_names,
        "mean_mci":      mci_scores,
        "rank":          ranks,
    }).sort_values("rank").reset_index(drop=True)
    save_csv(df, str(out_dir / "ranking.csv"))

    # --- Save MCI scores detail ---
    mci_detail = {
        "evaluator_name": evaluator_name,
        "target_info": target_info,
        "feature_names": feature_names,
        "mci_scores": mci_scores.tolist(),
        "shapley_values": shapley.tolist(),
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
    }
    save_json(mci_detail, str(out_dir / "mci_scores.json"))

    metadata = {
        "seed": args.seed,
        "dataset_path": args.dataset_path,
        "output_dir": str(out_dir),
        "config": args.config,
        "evaluator_name": evaluator_name,
        "target_info": target_info,
        "rff_dim": args.rff_dim,
        "bandwidth": args.bandwidth,
        "lambda_": args.lambda_,
        "n_perms": args.n_perms,
        "n_train": n_full,
        "n_mci": n,
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
        description="HDC-encoded MCI feature ranking (regression backend; "
        "bc / irl / pc evaluators mirror ranker/mci_rank_nn.py)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", required=True, help="Path to dataset.npz.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--config",
        default="configs/mci_kernel.yaml",
        help="YAML with the `mci_kernel` HDC param block. CLI flags override it.",
    )
    parser.add_argument(
        "--evaluator_name",
        default="bc",
        choices=["bc", "irl", "pc"],
        help=(
            "Which regression target/scoring defines nu(S): "
            "'bc' — regress expert actions, explained variance; "
            "'irl' — regress reward, explained variance; "
            "'pc' — regress reward, preference-prediction accuracy."
        ),
    )
    parser.add_argument(
        "--target",
        default="action_multi",
        choices=["action_multi", "action_norm", "action_label", "chosen_action_prob"],
        help="bc-only: which action target to regress (ignored for irl/pc).",
    )
    # rff_dim, bandwidth, lambda_, n_perms, max_samples, fragment_length and
    # num_pairs are fixed config parameters (no CLI flags); apply_hdc_config
    # loads them from the `mci_kernel` block of --config.
    args = parser.parse_args()
    apply_hdc_config(args)
    return args


if __name__ == "__main__":
    run(parse_args())
