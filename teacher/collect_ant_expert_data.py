"""Collect seals MuJoCo expert demonstrations into a project-standard dataset.npz.

Works for any seals/* continuous-control env with a HumanCompatibleAI PPO expert
on the HuggingFace Hub — verified for seals/Ant-v1, seals/Walker2d-v1 and
seals/Hopper-v1. The env is read from ``env.env_id`` in the config (overridable
with ``--env_id``), defaulting to seals/Ant-v1 for backwards compatibility.

Mirrors teacher/load_d4rl_dataset.py's output schema so downstream ranking and
training scripts (ranker/mci_rank_nn.py, ranker/mci_rank_kernel.py,
ranker/sage_rank.py, ranker/kernelshap.py, student/train_student.py) operate
without modification — the (X, y, rewards, dones, next_X) keys support all three
ranking tasks: bc (X→y), irl (X→rewards), pc (rewards + dones for fragments).

Pipeline:
  1. Build the vectorized seals env with RolloutInfoWrapper.
  2. Load the HuggingFace PPO expert (HumanCompatibleAI / <env_id>).
  3. Roll out the expert until ``min_expert_episodes`` are collected.
  4. Flatten trajectories into (obs, acts, next_obs, dones, rewards) tensors.
  5. Shuffle + 3-way split, write dataset.npz + metadata.

Output schema (dataset.npz, dims shown for seals/Ant-v1):
    X_train / X_val / X_test          float32  [N, obs_dim]
    y_train / y_val / y_test          float32  [N, action_dim]
    next_X_train / val / test         float32  [N, obs_dim]
    dones_train / val / test          bool     [N]
    rewards_train / val / test        float32  [N]
    action_norm_train / val / test    float32  [N]
    feature_names                     str      [obs_dim]      obs_00 …
    action_names                      str      [action_dim]   act_0 …

Usage:
    python teacher/collect_ant_expert_data.py \\
        --config configs/seals_walker.yaml \\
        --seed 0 \\
        --output_dir outputs/datasets/seals_walker/seed0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import seals  # noqa: F401  — registers seals/* namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_json, save_npz
from utils.seed import set_global_seed


DEFAULT_ENV = "seals/Ant-v1"


def collect_expert_trajectories(
    env_name: str,
    rng: np.random.Generator,
    n_envs: int,
    min_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the HuggingFace PPO expert for `env_name` and return transitions.

    Returns:
        obs       [N, obs_dim]    float32
        acts      [N, action_dim] float32
        next_obs  [N, obs_dim]    float32
        dones     [N]             bool
        rewards   [N]             float32
    """
    env = make_vec_env(
        env_name,
        rng=rng,
        n_envs=n_envs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=env_name,
        venv=env,
    )
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    flat = rollout.flatten_trajectories_with_rew(rollouts)
    env.close()

    return (
        np.asarray(flat.obs,       dtype=np.float32),
        np.asarray(flat.acts,      dtype=np.float32),
        np.asarray(flat.next_obs,  dtype=np.float32),
        np.asarray(flat.dones,     dtype=bool),
        np.asarray(flat.rews,      dtype=np.float32),
    )


def three_way_split(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, np.ndarray]:
    """Shuffle once, then split every array in `arrays` 3-ways using the same indices."""
    N = len(next(iter(arrays.values())))
    perm = rng.permutation(N)
    n_tr = int(N * train_ratio)
    n_va = int(N * val_ratio)

    out: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        a = arr[perm]
        out[f"{key}_train"] = a[:n_tr]
        out[f"{key}_val"]   = a[n_tr:n_tr + n_va]
        out[f"{key}_test"]  = a[n_tr + n_va:]
    return out


def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    env_id = args.env_id or cfg.get("env", {}).get("env_id") or DEFAULT_ENV

    ds_cfg = cfg.get("dataset", {})
    n_envs       = int(ds_cfg.get("n_envs", 8))
    min_episodes = int(ds_cfg.get("min_expert_episodes", 60))
    train_ratio  = float(ds_cfg.get("train_ratio", 0.8))
    val_ratio    = float(ds_cfg.get("val_ratio",   0.1))

    print(f"[collect_seals] env={env_id}, n_envs={n_envs}, "
          f"min_episodes={min_episodes}, seed={args.seed}")

    obs, acts, next_obs, dones, rews = collect_expert_trajectories(
        env_name=env_id, rng=rng, n_envs=n_envs, min_episodes=min_episodes,
    )
    obs_dim    = int(obs.shape[1])
    action_dim = int(acts.shape[1])
    N = len(obs)
    print(f"[collect_seals] Collected N={N}, obs_dim={obs_dim}, action_dim={action_dim}")

    # 3-way split — all arrays use the same random permutation.
    split = three_way_split(
        {"X": obs, "y": acts, "next_X": next_obs,
         "dones": dones, "rewards": rews},
        rng=rng, train_ratio=train_ratio, val_ratio=val_ratio,
    )

    # Per-step action norm (used by mi-scalar selector in feature_utils).
    for sp in ("train", "val", "test"):
        split[f"action_norm_{sp}"] = np.linalg.norm(
            split[f"y_{sp}"], axis=1,
        ).astype(np.float32)

    feature_names = np.array([f"obs_{j:02d}" for j in range(obs_dim)])
    action_names  = np.array([f"act_{j}"    for j in range(action_dim)])

    npz_path = str(out_dir / "dataset.npz")
    save_npz(
        npz_path,
        **split,
        feature_names=feature_names,
        action_names=action_names,
    )
    print(f"[collect_seals] Saved dataset to {npz_path}")
    print(f"  train={len(split['X_train'])}, val={len(split['X_val'])}, "
          f"test={len(split['X_test'])}")

    save_json({
        "env_id":         env_id,
        "seed":           args.seed,
        "config":         args.config,
        "obs_dim":        obs_dim,
        "action_dim":     action_dim,
        "n_total":        int(N),
        "n_train":        int(len(split["X_train"])),
        "n_val":          int(len(split["X_val"])),
        "n_test":         int(len(split["X_test"])),
        "n_envs":         n_envs,
        "min_episodes":   min_episodes,
        "feature_names":  feature_names.tolist(),
        "action_names":   action_names.tolist(),
        "task_type":      "continuous",
        "expert_source":  "ppo-huggingface/HumanCompatibleAI",
    }, str(out_dir / "metadata.json"))
    print("[collect_seals] Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect a seals/* PPO-expert dataset.npz (ant/walker/hopper/...)."
    )
    p.add_argument("--config",     required=True)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--env_id", default=None,
                   help="seals/* env id; overrides env.env_id from config "
                        "(default: seals/Ant-v1).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
