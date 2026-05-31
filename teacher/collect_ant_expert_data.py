"""Collect seals/Ant-v1 expert demonstrations into a project-standard dataset.npz.

Mirrors teacher/load_d4rl_dataset.py's output schema so downstream ranking and
training scripts (mci_rank.py, mci_rank_nn.py, sage_rank_continuous.py,
shap_rank_continuous.py, train_student_continuous.py, train_student_irl.py)
operate without modification.

Pipeline:
  1. Build vectorized seals/Ant-v1 with RolloutInfoWrapper.
  2. Load the HuggingFace PPO expert (HumanCompatibleAI / seals/Ant-v1).
  3. Roll out the expert until ``min_expert_episodes`` are collected.
  4. Flatten trajectories into (obs, acts, next_obs, dones, rewards) tensors.
  5. Shuffle + 3-way split, write dataset.npz + metadata.

Output schema (dataset.npz):
    X_train / X_val / X_test          float32  [N, 29]
    y_train / y_val / y_test          float32  [N, 8]
    next_X_train / val / test         float32  [N, 29]
    dones_train / val / test          bool     [N]
    rewards_train / val / test        float32  [N]
    action_norm_train / val / test    float32  [N]
    feature_names                     str      [29]   obs_00 … obs_28
    action_names                      str      [8]    act_0 … act_7

Usage:
    python teacher/collect_ant_expert_data.py \\
        --config configs/seals_ant.yaml \\
        --seed 0 \\
        --output_dir outputs/datasets/seals_ant/seed0
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


ENV_NAME = "seals/Ant-v1"


def collect_expert_trajectories(
    rng: np.random.Generator,
    n_envs: int,
    min_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the HuggingFace PPO expert and return flattened transitions.

    Returns:
        obs       [N, 29] float32
        acts      [N,  8] float32
        next_obs  [N, 29] float32
        dones     [N]     bool
        rewards   [N]     float32
    """
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

    ds_cfg = cfg.get("dataset", {})
    n_envs       = int(ds_cfg.get("n_envs", 8))
    min_episodes = int(ds_cfg.get("min_expert_episodes", 60))
    train_ratio  = float(ds_cfg.get("train_ratio", 0.8))
    val_ratio    = float(ds_cfg.get("val_ratio",   0.1))

    print(f"[collect_ant] env={ENV_NAME}, n_envs={n_envs}, "
          f"min_episodes={min_episodes}, seed={args.seed}")

    obs, acts, next_obs, dones, rews = collect_expert_trajectories(
        rng=rng, n_envs=n_envs, min_episodes=min_episodes,
    )
    obs_dim    = int(obs.shape[1])
    action_dim = int(acts.shape[1])
    N = len(obs)
    print(f"[collect_ant] Collected N={N}, obs_dim={obs_dim}, action_dim={action_dim}")

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
    print(f"[collect_ant] Saved dataset to {npz_path}")
    print(f"  train={len(split['X_train'])}, val={len(split['X_val'])}, "
          f"test={len(split['X_test'])}")

    save_json({
        "env_id":         ENV_NAME,
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
    print("[collect_ant] Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect seals/Ant-v1 PPO-expert demos → dataset.npz."
    )
    p.add_argument("--config",     required=True)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
