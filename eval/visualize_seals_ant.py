"""Render expert and BC policies on seals/Ant-v1 to MP4 videos.

Loads the PPO expert from HuggingFace and the trained BC policy from
``outputs/teachers/seals_ant/seedN/bc_policy.pt`` and rolls each out for a
configurable number of episodes, saving one MP4 per policy.

Requires offscreen MuJoCo rendering. EGL is used by default (set
``MUJOCO_GL=egl`` automatically); override via the ``MUJOCO_GL`` env var.

Usage:
    python eval/visualize_seals_ant.py \\
        --seed 0 \\
        --bc_policy outputs/teachers/seals_ant/seed0/bc_policy.pt \\
        --output_dir outputs/videos/seals_ant/seed0 \\
        --n_episodes 1
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE importing mujoco/gymnasium MuJoCo envs.
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import seals  # noqa: F401  — registers seals/* envs
import gymnasium as gym
import torch
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


ENV_NAME = "seals/Ant-v1"


def _load_bc_policy(path: str, device: str = "cpu") -> ActorCriticPolicy:
    """Reconstruct the SB3 ActorCriticPolicy saved by imitation's BC."""
    saved = torch.load(path, map_location=device, weights_only=False)
    data = saved["data"]
    # lr_schedule is callable -> not always picklable across versions; supply a no-op.
    data["lr_schedule"] = lambda _: 1.0
    policy = ActorCriticPolicy(**data)
    policy.load_state_dict(saved["state_dict"])
    policy.to(device).eval()
    return policy


def _act(policy, obs: np.ndarray) -> np.ndarray:
    """Deterministic action from an SB3 policy on a single observation."""
    with torch.no_grad():
        action, _ = policy.predict(obs, deterministic=True)
    return action


def _rollout_video(
    policy,
    output_path: Path,
    seed: int,
    n_episodes: int,
    fps: int,
    label: str,
) -> dict:
    """Record n_episodes of `policy` and concatenate frames into one MP4."""
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    returns: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0
        while True:
            frames.append(env.render())
            action = _act(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if terminated or truncated:
                frames.append(env.render())
                break
        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"[{label}] episode {ep + 1}/{n_episodes}: return={ep_return:.1f} length={ep_len}")

    env.close()

    print(f"[{label}] writing {len(frames)} frames -> {output_path}")
    imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=1)

    return {
        "label": label,
        "video": str(output_path),
        "n_episodes": n_episodes,
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)

    # Expert is loaded against a 1-env vec env (required by load_policy).
    rng = np.random.default_rng(args.seed)
    venv = make_vec_env(ENV_NAME, rng=rng, n_envs=1)
    print(f"[visualize] Loading PPO expert from HuggingFace ({ENV_NAME})...")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=ENV_NAME,
        venv=venv,
    )
    venv.close()

    print(f"[visualize] Loading BC policy from {args.bc_policy}")
    bc_policy = _load_bc_policy(args.bc_policy)

    expert_video = out_dir / "expert.mp4"
    bc_video     = out_dir / "bc.mp4"

    expert_stats = _rollout_video(expert, expert_video, args.seed,
                                  args.n_episodes, args.fps, "expert")
    bc_stats     = _rollout_video(bc_policy, bc_video, args.seed,
                                  args.n_episodes, args.fps, "bc")

    summary = {
        "env": ENV_NAME,
        "seed": args.seed,
        "bc_policy_path": str(args.bc_policy),
        "n_episodes": args.n_episodes,
        "fps": args.fps,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "expert": expert_stats,
        "bc": bc_stats,
    }
    save_json(summary, str(out_dir / "video_summary.json"))
    print(f"\n[visualize] Done.")
    print(f"  expert: return={expert_stats['mean_return']:.1f}  video={expert_video}")
    print(f"  bc:     return={bc_stats['mean_return']:.1f}  video={bc_video}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bc_policy",
        default="outputs/teachers/seals_ant/seed0/bc_policy.pt",
        help="Path to BC policy checkpoint saved by teacher/train_teacher.py.",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/videos/seals_ant/seed0",
    )
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=20,
                   help="Output video FPS. seals/Ant-v1 has dt=0.05s → 20fps is real-time.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
