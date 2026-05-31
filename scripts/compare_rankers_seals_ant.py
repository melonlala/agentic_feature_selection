"""Compare feature rankers on seals/Ant-v1 — state vs latent feature space.

Rewritten so that all artifacts conform to the project's standard schema and
can be passed directly to ``eval/eval_offline_continuous.py`` and
``eval/eval_online_continuous.py``.

Rankers compared
----------------
    mci_nn  — per-subset multi-output MLP retraining (Catav et al., ICML 2021)
    mci_hdc — closed-form ridge on Random Fourier Features
    sage    — Shapley aggregation of fixed-model predictive-power drops
              (uses the full-feature BC student + marginal imputation)
    random  — baseline; uniformly random feature ordering

Spaces
------
    state  — raw seals/Ant-v1 observations.
    latent — first-layer activations of a full-feature BC student
             (Linear+LayerNorm+ReLU, the "post_relu" latent).

Pipeline
--------
  1. Collect expert rollouts (HuggingFace PPO expert).
  2. Train one full-feature BC student → also acts as
        (a) the SAGE model (predictive-power evaluator),
        (b) the latent-feature extractor (its frozen layer-1).
  3. For each (ranker × space), compute a global ranking.
  4. For each (ranker × space × k), train a small top-k BC student and
     evaluate its return in the env. Validation MSE is captured too.
  5. Persist everything in eval-compatible layout (see below).

Output layout (under --output_dir)
----------------------------------
    config.yaml                          consumable by eval/*continuous.py
    dataset.npz                          X_{train,val,test}, y_{train,val,test},
                                         feature_names, action_names
    rankings/{space}/{ranker}_ranking.csv
    students/full/full/model.pt          full-feature state BC
    students/state/{ranker}/k{K}/model.pt
    students/latent/{ranker}/k{K}/model.pt
    topk_eval.csv                        per-(space, ranker, k): val_mse,
                                         mean_return, std_return, mean_length
    metadata.json
    plots/return_vs_k_{state,latent}.png

Down-stream evaluation
----------------------
Offline (test MSE / MAE / cosine on dataset.npz):
    python eval/eval_offline_continuous.py \\
        --config       outputs/compare_seals_ant/seed0/config.yaml \\
        --dataset_path outputs/compare_seals_ant/seed0/dataset.npz \\
        --student_dir  outputs/compare_seals_ant/seed0/students/state/mci_nn \\
        --output_dir   outputs/eval/offline_seals_ant/seed0/state_mci_nn

Online (env rollouts on seals/Ant-v1):
    python eval/eval_online_continuous.py \\
        --config      outputs/compare_seals_ant/seed0/config.yaml \\
        --student_dir outputs/compare_seals_ant/seed0/students/state/mci_nn \\
        --output_dir  outputs/eval/online_seals_ant/seed0/state_mci_nn \\
        --n_episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seals  # noqa: F401  — registers `seals/*` namespace
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.hdc_encoder import FeatureWiseHDCEncoder
from ranker.mci_rank_nn import compute_mci_nn
from student.bc_continuous_model import (
    BCContinuousPolicy,
    BCContinuousPolicyFromLatent,
)
from utils.io import ensure_dir, save_csv, save_json, save_npz
from utils.seed import set_global_seed

ENV_NAME = "seals/Ant-v1"
LATENT_LAYER = "post_relu"  # the compare script reads Linear+LN+ReLU as latent


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_expert_transitions(
    rng: np.random.Generator,
    n_envs: int,
    min_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build env + load PPO expert + roll out → (obs, acts) numpy."""
    env = make_vec_env(
        ENV_NAME,
        rng=rng,
        n_envs=n_envs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=ENV_NAME,
        venv=env,
    )
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    env.close()
    return (
        np.asarray(transitions.obs,  dtype=np.float32),
        np.asarray(transitions.acts, dtype=np.float32),
    )


def split_three_way(
    X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
    train_ratio: float = 0.8, val_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    """Random 3-way split. test_ratio = 1 - train - val."""
    N = len(X)
    perm = rng.permutation(N)
    X, Y = X[perm], Y[perm]
    n_tr = int(N * train_ratio)
    n_va = int(N * val_ratio)
    return {
        "X_train": X[:n_tr],          "y_train": Y[:n_tr],
        "X_val":   X[n_tr:n_tr + n_va], "y_val":   Y[n_tr:n_tr + n_va],
        "X_test":  X[n_tr + n_va:],   "y_test":  Y[n_tr + n_va:],
    }


# ---------------------------------------------------------------------------
# Training helpers (using project's BCContinuousPolicy / FromLatent)
# ---------------------------------------------------------------------------

def _train_loop(
    model: nn.Module,
    trainable_params,
    Xtr: torch.Tensor, Ytr: torch.Tensor,
    Xvl: torch.Tensor, Yvl: torch.Tensor,
    epochs: int, lr: float, weight_decay: float, batch_size: int,
) -> tuple[dict, float]:
    """Generic train loop returning (best_state_dict, best_val_mse_sum)."""
    opt = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(Xtr)
    best_val = float("inf")
    best_state: dict | None = None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, device=Xtr.device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            loss = torch.mean((model(Xtr[idx]) - Ytr[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(((model(Xvl) - Yvl) ** 2).mean(dim=0).sum().item())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    return best_state, best_val


def train_full_bc(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
    hidden: tuple[int, ...], epochs: int,
    lr: float, weight_decay: float, batch_size: int, device: torch.device,
) -> tuple[BCContinuousPolicy, float]:
    model = BCContinuousPolicy(
        input_dim=X_tr.shape[1], action_dim=Y_tr.shape[1], hidden_dims=hidden,
    ).to(device)
    Xtr = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr,  dtype=torch.float32, device=device)
    Xvl = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yvl = torch.tensor(Y_val, dtype=torch.float32, device=device)
    best_state, best_val = _train_loop(
        model, list(model.parameters()),
        Xtr, Ytr, Xvl, Yvl, epochs, lr, weight_decay, batch_size,
    )
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def train_state_subset_bc(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
    feature_idx: list[int], hidden: tuple[int, ...], epochs: int,
    lr: float, weight_decay: float, batch_size: int, device: torch.device,
) -> tuple[BCContinuousPolicy, float]:
    model = BCContinuousPolicy(
        input_dim=len(feature_idx), action_dim=Y_tr.shape[1], hidden_dims=hidden,
    ).to(device)
    Xtr = torch.tensor(X_tr[:, feature_idx],  dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr,                  dtype=torch.float32, device=device)
    Xvl = torch.tensor(X_val[:, feature_idx], dtype=torch.float32, device=device)
    Yvl = torch.tensor(Y_val,                 dtype=torch.float32, device=device)
    best_state, best_val = _train_loop(
        model, list(model.parameters()),
        Xtr, Ytr, Xvl, Yvl, epochs, lr, weight_decay, batch_size,
    )
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def _frozen_layer1_from_full(
    full_model: BCContinuousPolicy, device: torch.device,
) -> nn.Sequential:
    """Linear + LayerNorm + ReLU (post_relu latent), frozen, in eval mode."""
    net = full_model.net
    block = nn.Sequential(net[0], net[1], net[2])
    for p in block.parameters():
        p.requires_grad_(False)
    block.to(device).eval()
    return block


def train_latent_subset_bc(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
    full_model: BCContinuousPolicy, latent_idx: list[int],
    hidden: tuple[int, ...], action_dim: int, epochs: int,
    lr: float, weight_decay: float, batch_size: int, device: torch.device,
) -> tuple[BCContinuousPolicyFromLatent, float]:
    frozen = _frozen_layer1_from_full(full_model, device)
    model = BCContinuousPolicyFromLatent(
        frozen_layer1=frozen, latent_indices=latent_idx,
        action_dim=action_dim, hidden_dims=hidden,
    ).to(device)
    Xtr = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr,  dtype=torch.float32, device=device)
    Xvl = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yvl = torch.tensor(Y_val, dtype=torch.float32, device=device)
    # Only `head` has trainable params (frozen_layer1 is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    best_state, best_val = _train_loop(
        model, trainable, Xtr, Ytr, Xvl, Yvl,
        epochs, lr, weight_decay, batch_size,
    )
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


# ---------------------------------------------------------------------------
# Latent extraction (full BC layer-1 = post_relu latent)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latents(
    full_model: BCContinuousPolicy, X: np.ndarray, device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    block = _frozen_layer1_from_full(full_model, device)
    out: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        out.append(block(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

def rank_mci_nn(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
    rng: np.random.Generator, device: torch.device,
    n_perms: int, hidden: tuple[int, ...], epochs: int,
) -> np.ndarray:
    scores, _ = compute_mci_nn(
        X_tr=X_tr, Y_tr=Y_tr, X_val=X_val, Y_val=Y_val,
        n_perms=n_perms, rng=rng, device=device, hidden=hidden,
        epochs=epochs, lr=1e-3, weight_decay=1e-4, batch_size=256,
        aggregation="mean",
    )
    return scores


def _ridge_predictive_power_multi(
    Z_S: torch.Tensor, Y: torch.Tensor, var_y_sum: float, lam: float,
) -> float:
    p = Z_S.shape[1]
    if p == 0:
        return 0.0
    A = Z_S.T @ Z_S + lam * torch.eye(p, dtype=Z_S.dtype, device=Z_S.device)
    W = torch.linalg.solve(A, Z_S.T @ Y)
    Y_hat = Z_S @ W
    mse_sum = float(torch.mean((Y - Y_hat) ** 2, dim=0).sum().item())
    return float(var_y_sum - mse_sum)


def rank_mci_hdc(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    rng: np.random.Generator, device: torch.device,
    n_perms: int, rff_dim: int, bandwidth: float, lam: float,
) -> np.ndarray:
    d = X_tr.shape[1]
    encoder = FeatureWiseHDCEncoder(
        rff_dim=rff_dim, bandwidth=bandwidth,
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )
    encoder.fit(X_tr)
    encoder.precompute(X_tr)
    Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    var_y_sum = float(np.var(Y_tr, axis=0).sum())
    scores = np.zeros(d, dtype=np.float64)
    for i in range(d):
        other = [j for j in range(d) if j != i]
        deltas: list[float] = []
        for _ in range(n_perms):
            size = int(rng.integers(0, len(other) + 1))
            S = list(rng.choice(other, size=size, replace=False))
            Z_S  = torch.tensor(encoder.transform_subset(X_tr, S),       dtype=torch.float32, device=device)
            Z_Si = torch.tensor(encoder.transform_subset(X_tr, S + [i]), dtype=torch.float32, device=device)
            nu_S  = _ridge_predictive_power_multi(Z_S,  Y_t, var_y_sum, lam)
            nu_Si = _ridge_predictive_power_multi(Z_Si, Y_t, var_y_sum, lam)
            deltas.append(nu_Si - nu_S)
        scores[i] = float(np.mean(deltas))
        print(f"[mci_hdc] feature {i:>3d}: mean_delta={scores[i]:+.4f}")
    return scores


def rank_sage(
    full_model: BCContinuousPolicy,
    X_explain: np.ndarray, Y_explain: np.ndarray, background: np.ndarray,
    device: torch.device, n_permutations: int,
) -> np.ndarray:
    import sage

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=device)
            return full_model(xb).cpu().numpy()

    imputer   = sage.MarginalImputer(predict_fn, background)
    estimator = sage.PermutationEstimator(imputer, loss="mse")
    print(f"[sage] explain={len(X_explain)}, bg={len(background)}, perms={n_permutations}")
    expl = estimator(
        X_explain, Y_explain,
        batch_size=256, n_permutations=n_permutations,
        detect_convergence=False, verbose=False, bar=False,
    )
    return np.abs(expl.values)


def rank_random(d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(d)


def to_ranking_df(scores: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    d = len(scores)
    order = np.argsort(scores)[::-1]
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(1, d + 1)
    return pd.DataFrame({
        "feature_index": np.arange(d),
        "feature_name":  feature_names,
        "score":         scores,
        "rank":          ranks,
    }).sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helpers (project-standard checkpoint schema)
# ---------------------------------------------------------------------------

def save_full_student(
    model: BCContinuousPolicy, out_dir: Path,
    obs_dim: int, action_dim: int, hidden_dims: list[int],
    feature_names: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_idx":      list(range(obs_dim)),
        "feature_names":    list(feature_names),
        "input_dim":        int(obs_dim),
        "action_dim":       int(action_dim),
        "hidden_dims":      list(hidden_dims),
        "use_gaussian":     False,
        "model_class":      "BCContinuousPolicy",
    }, str(ckpt_path))
    return ckpt_path


def save_state_subset_student(
    model: BCContinuousPolicy, out_dir: Path,
    feature_idx: list[int], action_dim: int, hidden_dims: list[int],
    feature_names: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_idx":      list(feature_idx),
        "feature_names":    [feature_names[i] for i in feature_idx],
        "input_dim":        len(feature_idx),
        "action_dim":       int(action_dim),
        "hidden_dims":      list(hidden_dims),
        "use_gaussian":     False,
        "model_class":      "BCContinuousPolicy",
    }, str(ckpt_path))
    return ckpt_path


def save_latent_subset_student(
    model: BCContinuousPolicyFromLatent, out_dir: Path,
    raw_D: int, action_dim: int, hidden_dims: list[int],
    latent_idx: list[int], source_student_path: Path,
    feature_names: list[str],
) -> Path:
    """For latent students, feature_idx is the identity over raw obs (passthrough).
    `latent_idx` records the subset of layer-1 latent dims actually used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict":     model.state_dict(),
        "feature_idx":          list(range(raw_D)),
        "feature_names":        list(feature_names),
        "input_dim":            int(raw_D),
        "action_dim":           int(action_dim),
        "hidden_dims":          list(hidden_dims),
        "use_gaussian":         False,
        "model_class":          "BCContinuousPolicyFromLatent",
        "latent_idx":           list(latent_idx),
        "latent_layer":         LATENT_LAYER,
        "source_student_path":  str(source_student_path),
    }, str(ckpt_path))
    return ckpt_path


def write_eval_config(
    out_path: Path, obs_dim: int, action_dim: int,
    full_hidden: list[int], subset_hidden: list[int],
    eval_episodes: int,
) -> None:
    cfg = {
        "env": {
            "env_id":     ENV_NAME,
            "task_type":  "continuous",
            "obs_dim":    int(obs_dim),
            "action_dim": int(action_dim),
            "oracle_feature_indices": [],
        },
        "dataset": {
            "source":      "expert_rollout",
            "train_ratio": 0.8,
            "val_ratio":   0.1,
            "test_ratio":  0.1,
        },
        "student": {
            "hidden_dims":   list(subset_hidden),
            "lr":            1e-3,
            "epochs":        60,
            "batch_size":    256,
            "task_type":     "continuous",
            "topk_list":     [],
            "max_train_samples": None,
        },
        "eval": {
            "online_episodes":    int(eval_episodes),
            "offline_batch_size": 512,
        },
        "logging": {"save_csv": True, "save_json": True},
        # For reference only — not consumed by eval scripts.
        "full_student_hidden_dims": list(full_hidden),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

RANKER_COLOR = {"mci_nn": "#1f77b4", "mci_hdc": "#2ca02c",
                "sage":   "#d62728", "random":  "#7f7f7f"}


def plot_return_vs_k(df: pd.DataFrame, space: str, out_path: Path) -> None:
    sub = df[df["space"] == space]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ranker, g in sub.groupby("ranker"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["mean_return"], marker="o",
                color=RANKER_COLOR.get(ranker), label=ranker)
        ax.fill_between(g["k"],
                        g["mean_return"] - g["std_return"],
                        g["mean_return"] + g["std_return"],
                        color=RANKER_COLOR.get(ranker), alpha=0.12)
    ax.set_xlabel("k (selected features)")
    ax.set_ylabel("Mean episodic return")
    ax.set_title(f"seals/Ant-v1 — return vs k ({space} space)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Env eval (kept inline for the inline summary; eval/eval_online_continuous.py
# can also be used post-hoc on the saved checkpoints).
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_policy_in_env(
    model: nn.Module, feature_idx: list[int],
    n_episodes: int, device: torch.device, seed: int,
) -> dict:
    import gymnasium as gym
    env = gym.make(ENV_NAME)
    idx_t = torch.as_tensor(feature_idx, dtype=torch.long, device=device)
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        for _ in range(env.spec.max_episode_steps or 1000):
            x = torch.tensor(np.asarray(obs, dtype=np.float32),
                             device=device).unsqueeze(0)
            x = x.index_select(1, idx_t)
            a = model(x).cpu().numpy()[0]
            a = np.clip(a, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_ret)
        lengths.append(ep_len)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return":  float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes":  n_episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir       = ensure_dir(args.output_dir)
    rankings_dir  = ensure_dir(out_dir / "rankings")
    students_dir  = ensure_dir(out_dir / "students")
    plots_dir     = ensure_dir(out_dir / "plots")
    print(f"[compare] device={device}, out_dir={out_dir}")

    # --- 1. Collect data ---
    print(f"[compare] Collecting expert rollouts (min_episodes={args.min_episodes})...")
    t0 = time.time()
    X_obs, Y_act = collect_expert_transitions(
        rng=rng, n_envs=args.n_envs, min_episodes=args.min_episodes,
    )
    obs_dim, action_dim = X_obs.shape[1], Y_act.shape[1]
    print(f"[compare] obs={X_obs.shape}, acts={Y_act.shape}  ({time.time() - t0:.1f}s)")

    # 3-way split — matches project convention.
    splits = split_three_way(
        X_obs, Y_act, rng,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    X_train, Y_train = splits["X_train"], splits["y_train"]
    X_val,   Y_val   = splits["X_val"],   splits["y_val"]
    X_test,  Y_test  = splits["X_test"],  splits["y_test"]

    # Optional train-size cap (faster subset re-training).
    if args.max_train and len(X_train) > args.max_train:
        sel = rng.choice(len(X_train), size=args.max_train, replace=False)
        X_train, Y_train = X_train[sel], Y_train[sel]
    print(f"[compare] N_train={len(X_train)}, N_val={len(X_val)}, N_test={len(X_test)}")

    # Persist dataset + config + action-norm column (required by eval offline).
    state_names  = [f"obs_{j:02d}"    for j in range(obs_dim)]
    action_names = [f"act_{j}"        for j in range(action_dim)]
    dataset_path = out_dir / "dataset.npz"
    save_npz(
        str(dataset_path),
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=Y_train.astype(np.float32),
        y_val=Y_val.astype(np.float32),
        y_test=Y_test.astype(np.float32),
        action_norm_train=np.linalg.norm(Y_train, axis=1).astype(np.float32),
        action_norm_val  =np.linalg.norm(Y_val,   axis=1).astype(np.float32),
        action_norm_test =np.linalg.norm(Y_test,  axis=1).astype(np.float32),
        feature_names=np.array(state_names),
        action_names =np.array(action_names),
    )
    config_path = out_dir / "config.yaml"
    write_eval_config(
        config_path, obs_dim=obs_dim, action_dim=action_dim,
        full_hidden=args.full_hidden, subset_hidden=args.subset_hidden,
        eval_episodes=args.eval_episodes,
    )
    print(f"[compare] Wrote {dataset_path} and {config_path}")

    # --- 2. Train full-feature BC + save it ---
    print(f"[compare] Training full-feature BC student (hidden={args.full_hidden})...")
    t0 = time.time()
    full_model, full_val_mse = train_full_bc(
        X_train, Y_train, X_val, Y_val,
        hidden=tuple(args.full_hidden), epochs=args.full_epochs,
        lr=1e-3, weight_decay=1e-4, batch_size=256, device=device,
    )
    print(f"[compare] full BC val_mse_sum={full_val_mse:.4f}  ({time.time() - t0:.1f}s)")
    full_student_path = save_full_student(
        full_model, students_dir / "full" / "full",
        obs_dim=obs_dim, action_dim=action_dim, hidden_dims=args.full_hidden,
        feature_names=state_names,
    )

    # --- 3. Extract latents ---
    Z_train = extract_latents(full_model, X_train, device)
    Z_val   = extract_latents(full_model, X_val,   device)
    latent_dim = Z_train.shape[1]
    latent_names = [f"latent_{j:03d}" for j in range(latent_dim)]
    print(f"[compare] latents: train={Z_train.shape}, val={Z_val.shape}")

    # --- 4. Compute rankings ---
    rankings: dict[tuple[str, str], pd.DataFrame] = {}
    space_data = {
        "state":  (X_train, X_val, state_names),
        "latent": (Z_train, Z_val, latent_names),
    }
    for space, (Xt, Xv, names) in space_data.items():
        print(f"\n[compare] === Rankings on {space} (d={Xt.shape[1]}) ===")

        print(f"[compare] {space}: mci_nn (n_perms={args.mci_nn_perms}, "
              f"epochs={args.mci_nn_epochs})...")
        t0 = time.time()
        rankings[(space, "mci_nn")] = to_ranking_df(
            rank_mci_nn(Xt, Y_train, Xv, Y_val,
                        rng=rng, device=device,
                        n_perms=args.mci_nn_perms,
                        hidden=tuple(args.mci_nn_hidden),
                        epochs=args.mci_nn_epochs),
            names,
        )
        print(f"[compare]   done in {time.time() - t0:.1f}s")

        print(f"[compare] {space}: mci_hdc (n_perms={args.mci_hdc_perms})...")
        t0 = time.time()
        rankings[(space, "mci_hdc")] = to_ranking_df(
            rank_mci_hdc(Xt, Y_train, rng=rng, device=device,
                         n_perms=args.mci_hdc_perms, rff_dim=args.rff_dim,
                         bandwidth=args.bandwidth, lam=args.ridge_lambda),
            names,
        )
        print(f"[compare]   done in {time.time() - t0:.1f}s")

        if space == "state":
            print(f"[compare] {space}: sage (perms={args.sage_perms})...")
            t0 = time.time()
            bg_idx  = rng.choice(len(Xt), size=min(args.sage_bg,  len(Xt)),  replace=False)
            exp_idx = rng.choice(len(Xv), size=min(args.sage_exp, len(Xv)), replace=False)
            rankings[(space, "sage")] = to_ranking_df(
                rank_sage(full_model, Xv[exp_idx], Y_val[exp_idx], Xt[bg_idx],
                          device=device, n_permutations=args.sage_perms),
                names,
            )
            print(f"[compare]   done in {time.time() - t0:.1f}s")
        else:
            print(f"[compare] {space}: sage SKIPPED (needs latent-input model).")

        rankings[(space, "random")] = to_ranking_df(
            rank_random(Xt.shape[1], rng), names,
        )

    for (space, ranker), df in rankings.items():
        save_csv(df, str(ensure_dir(rankings_dir / space) / f"{ranker}_ranking.csv"))

    # --- 5. Train top-k subset BCs, save them, capture val_mse + env return ---
    print(f"\n[compare] === Top-k subset BC + env eval ===")
    eval_rows: list[dict] = []
    for (space, ranker), df in rankings.items():
        ks = args.state_topk if space == "state" else args.latent_topk
        for k in ks:
            if k > df.shape[0]:
                continue
            top_idx = df.sort_values("rank").head(k)["feature_index"].astype(int).tolist()
            print(f"[compare] {space}/{ranker}/k={k}: features={top_idx[:6]}"
                  f"{'...' if k > 6 else ''}")
            t0 = time.time()
            if space == "state":
                policy, val_mse = train_state_subset_bc(
                    X_train, Y_train, X_val, Y_val,
                    feature_idx=top_idx, hidden=tuple(args.subset_hidden),
                    epochs=args.subset_epochs, lr=1e-3, weight_decay=1e-4,
                    batch_size=256, device=device,
                )
                ckpt_path = save_state_subset_student(
                    policy, students_dir / "state" / ranker / f"k{k}",
                    feature_idx=top_idx, action_dim=action_dim,
                    hidden_dims=args.subset_hidden, feature_names=state_names,
                )
                metrics = eval_policy_in_env(
                    policy, feature_idx=top_idx,
                    n_episodes=args.eval_episodes, device=device,
                    seed=args.seed * 1000,
                )
            else:
                policy, val_mse = train_latent_subset_bc(
                    X_train, Y_train, X_val, Y_val,
                    full_model=full_model, latent_idx=top_idx,
                    hidden=tuple(args.subset_hidden), action_dim=action_dim,
                    epochs=args.subset_epochs, lr=1e-3, weight_decay=1e-4,
                    batch_size=256, device=device,
                )
                ckpt_path = save_latent_subset_student(
                    policy, students_dir / "latent" / ranker / f"k{k}",
                    raw_D=obs_dim, action_dim=action_dim,
                    hidden_dims=args.subset_hidden, latent_idx=top_idx,
                    source_student_path=full_student_path,
                    feature_names=state_names,
                )
                # Latent students eat the raw obs and select latents internally.
                metrics = eval_policy_in_env(
                    policy, feature_idx=list(range(obs_dim)),
                    n_episodes=args.eval_episodes, device=device,
                    seed=args.seed * 1000,
                )
            elapsed = time.time() - t0
            row = {
                "space":    space,
                "ranker":   ranker,
                "k":        int(k),
                "val_mse":  float(val_mse),
                **metrics,
                "elapsed_s": float(elapsed),
                "ckpt": str(ckpt_path),
            }
            eval_rows.append(row)
            print(f"[compare]   val_mse={val_mse:.4f}  "
                  f"return={metrics['mean_return']:.1f}±{metrics['std_return']:.1f}  "
                  f"len={metrics['mean_length']:.0f}  ({elapsed:.1f}s)")

    eval_df = pd.DataFrame(eval_rows)
    save_csv(eval_df, str(out_dir / "topk_eval.csv"))

    # --- 6. Plots ---
    for space in ("state", "latent"):
        plot_return_vs_k(eval_df, space, plots_dir / f"return_vs_k_{space}.png")

    # --- 7. Metadata ---
    save_json({
        "env_name":     ENV_NAME,
        "seed":         args.seed,
        "obs_dim":      int(obs_dim),
        "action_dim":   int(action_dim),
        "latent_dim":   int(latent_dim),
        "latent_layer": LATENT_LAYER,
        "n_train":      int(len(X_train)),
        "n_val":        int(len(X_val)),
        "n_test":       int(len(X_test)),
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, list, type(None)))},
        "full_bc_val_mse_sum": float(full_val_mse),
        "full_student_path":   str(full_student_path),
        "rankings_dir":        str(rankings_dir),
        "students_dir":        str(students_dir),
        "dataset_path":        str(dataset_path),
        "config_path":         str(config_path),
    }, str(out_dir / "metadata.json"))

    pivot = (eval_df.assign(label=lambda d: d["space"] + "/" + d["ranker"])
             .pivot_table(index="k", columns="label",
                          values="mean_return", aggfunc="first"))
    print("\n=== Return summary (mean) ===")
    print(pivot.round(1).to_string())
    save_csv(pivot.reset_index(), str(out_dir / "return_pivot.csv"))

    print(f"\n[compare] All outputs under: {out_dir}")
    print("[compare] Run downstream eval with:")
    print(f"  python eval/eval_offline_continuous.py "
          f"--config {config_path} --dataset_path {dataset_path} "
          f"--student_dir <students_dir>/<space>/<ranker> "
          f"--output_dir <eval_out>")
    print(f"  python eval/eval_online_continuous.py "
          f"--config {config_path} "
          f"--student_dir <students_dir>/<space>/<ranker> "
          f"--output_dir <eval_out> --n_episodes {args.eval_episodes}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare feature rankers (state vs latent) on seals/Ant-v1."
    )
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--output_dir",   default="outputs/compare_seals_ant/seed0")

    # Data
    p.add_argument("--n_envs",       type=int, default=8)
    p.add_argument("--min_episodes", type=int, default=50)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--max_train",    type=int, default=8000,
                   help="Cap training rows for tractable subset retraining. 0 = use all.")

    # Full BC student
    p.add_argument("--full_hidden",  type=int, nargs="+", default=[64, 64])
    p.add_argument("--full_epochs",  type=int, default=80)

    # MCI-NN
    p.add_argument("--mci_nn_perms",   type=int, default=20)
    p.add_argument("--mci_nn_hidden",  type=int, nargs="+", default=[64, 64])
    p.add_argument("--mci_nn_epochs",  type=int, default=20)

    # MCI-HDC
    p.add_argument("--mci_hdc_perms",  type=int, default=100)
    p.add_argument("--rff_dim",        type=int, default=64)
    p.add_argument("--bandwidth",      type=float, default=1.0)
    p.add_argument("--ridge_lambda",   type=float, default=1e-3)

    # SAGE
    p.add_argument("--sage_perms",     type=int, default=300)
    p.add_argument("--sage_bg",        type=int, default=200)
    p.add_argument("--sage_exp",       type=int, default=200)

    # Subset BC student + env eval
    p.add_argument("--state_topk",   type=int, nargs="+", default=[4, 8, 16, 27])
    p.add_argument("--latent_topk",  type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--subset_hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--subset_epochs", type=int, default=60)
    p.add_argument("--eval_episodes", type=int, default=20)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
