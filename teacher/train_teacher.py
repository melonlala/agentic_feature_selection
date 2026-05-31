"""Train a behavioral cloning (BC) policy on the seals/Ant-v1 environment.

Uses the `imitation` package:
  1. Build a vectorized seals/Ant-v1 env wrapped in RolloutInfoWrapper.
  2. Load a pre-trained PPO expert from HuggingFace (HumanCompatibleAI).
  3. Roll out the expert to collect demonstration trajectories.
  4. Train a BC policy on the flattened transitions.
  5. Evaluate BC vs. expert and save the policy + metadata.

Usage:
    python teacher/train_teacher.py \\
        --seed 0 \\
        --output_dir outputs/teachers/seals_ant/seed0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import seals  # noqa: F401  — registers the `seals/*` gymnasium namespace
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


ENV_NAME = "seals/Ant-v1"


def train(args: argparse.Namespace) -> None:
    """Train BC policy on seals/Ant-v1 using the imitation package."""
    cfg = resolve_config(args.config) if args.config else {}
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.output_dir)
    if cfg:
        save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    t_cfg = cfg.get("teacher", {})
    n_envs       = t_cfg.get("n_envs", 8)
    min_episodes = t_cfg.get("rollout_min_episodes", 50)
    n_epochs     = t_cfg.get("bc_epochs", 10)
    n_eval       = t_cfg.get("eval_episodes", 20)

    # --- Build vectorized env with RolloutInfoWrapper (needed for rollout.rollout) ---
    env = make_vec_env(
        ENV_NAME,
        rng=rng,
        n_envs=n_envs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    print(f"[train_teacher] env={ENV_NAME}, n_envs={n_envs}, seed={args.seed}")
    print(f"[train_teacher] obs_space={env.observation_space}, "
          f"action_space={env.action_space}")

    # --- Load PPO expert from HuggingFace ---
    print("[train_teacher] Loading PPO expert from HuggingFace (HumanCompatibleAI)...")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=ENV_NAME,
        venv=env,
    )

    # --- Collect expert rollouts ---
    print(f"[train_teacher] Collecting expert rollouts (min_episodes={min_episodes})...")
    start = time.time()
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    rollout_time = time.time() - start
    print(f"[train_teacher] Collected {len(transitions)} transitions "
          f"from {len(rollouts)} episodes in {rollout_time:.1f}s.")

    # --- Train BC ---
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    print(f"[train_teacher] Training BC for {n_epochs} epochs...")
    start = time.time()
    bc_trainer.train(n_epochs=n_epochs)
    train_time = time.time() - start
    print(f"[train_teacher] BC training done in {train_time:.1f}s.")

    # --- Save policy ---
    ckpt_path = out_dir / "bc_policy.pt"
    bc_trainer.policy.save(str(ckpt_path))
    print(f"[train_teacher] BC policy saved to {ckpt_path}")

    # --- Evaluate ---
    print(f"[train_teacher] Evaluating BC vs. expert over {n_eval} episodes each...")
    bc_mean, bc_std = evaluate_policy(
        bc_trainer.policy, env, n_eval_episodes=n_eval, deterministic=True
    )
    expert_mean, expert_std = evaluate_policy(
        expert, env, n_eval_episodes=n_eval, deterministic=True
    )
    print(f"[train_teacher] BC return:     {bc_mean:.2f} ± {bc_std:.2f}")
    print(f"[train_teacher] Expert return: {expert_mean:.2f} ± {expert_std:.2f}")

    # --- Save metadata + eval ---
    metadata = {
        "seed": args.seed,
        "config": args.config,
        "output_dir": str(out_dir),
        "env_name": ENV_NAME,
        "algo": "bc",
        "expert_source": "ppo-huggingface/HumanCompatibleAI",
        "n_envs": n_envs,
        "min_rollout_episodes": min_episodes,
        "n_transitions": int(len(transitions)),
        "n_episodes": int(len(rollouts)),
        "bc_epochs": n_epochs,
        "rollout_time_s": rollout_time,
        "training_time_s": train_time,
        "checkpoint": str(ckpt_path),
        "obs_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
    }
    save_json(metadata, str(out_dir / "metadata.json"))
    save_json(
        {
            "bc_mean_return":     float(bc_mean),
            "bc_std_return":      float(bc_std),
            "expert_mean_return": float(expert_mean),
            "expert_std_return":  float(expert_std),
            "n_eval_episodes":    n_eval,
        },
        str(out_dir / "eval_metrics.json"),
    )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BC policy on seals/Ant-v1 using the imitation package."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config overriding teacher.{n_envs, bc_epochs, "
             "rollout_min_episodes, eval_episodes}.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="outputs/teachers/seals_ant/seed0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
