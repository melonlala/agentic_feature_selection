"""Evaluate the D4RL kitchen-complete-v2 training dataset's expert reward.

Two views of "the dataset's reward":

  (1) recorded — sum the rewards stored in the HDF5, broken into episodes via
      `terminals | timeouts`. This is the expert's return as the dataset stores
      it (the v2 reward relabelling = number of subtasks completed at each step).

  (2) replay  — open-loop replay of each episode's action sequence in the gym
      env (FrankaKitchen-v1) and accumulate the env's reward. Initial states
      from D4RL are not exactly reproduced by env.reset(), so this is the
      "best-case BC online return" upper bound, not the expert's true return.

Usage:
    python eval/eval_dataset_online_kitchen.py \\
        --hdf5_path outputs/d4rl_cache/kitchen-complete-v2.hdf5 \\
        --output_dir outputs/eval/dataset_online/kitchen_complete \\
        --replay_episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, save_csv, save_json
import pandas as pd


_KITCHEN_TASKS = ["microwave", "kettle", "bottom burner", "light switch"]


def split_episodes(
    terminals: np.ndarray, timeouts: np.ndarray
) -> list[tuple[int, int]]:
    """Return list of (start, end) slice indices for each episode."""
    boundary = terminals | timeouts
    boundary_idx = np.where(boundary)[0]
    if boundary_idx.size == 0 or boundary_idx[-1] != len(terminals) - 1:
        boundary_idx = np.concatenate([boundary_idx, [len(terminals) - 1]])
    episodes = []
    start = 0
    for end in boundary_idx:
        episodes.append((start, int(end) + 1))
        start = int(end) + 1
    return episodes


def summarize_recorded(
    rewards: np.ndarray, episodes: list[tuple[int, int]]
) -> dict:
    """Per-episode return statistics from the dataset's recorded rewards."""
    returns = np.array([rewards[s:e].sum() for s, e in episodes], dtype=np.float64)
    lengths = np.array([e - s for s, e in episodes], dtype=np.int64)
    return {
        "n_episodes":   int(len(episodes)),
        "n_transitions": int(rewards.size),
        "mean_return":  float(returns.mean()),
        "std_return":   float(returns.std()),
        "min_return":   float(returns.min()),
        "max_return":   float(returns.max()),
        "median_return": float(np.median(returns)),
        "mean_length":  float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "max_reward_per_step": float(rewards.max()),
        "mean_reward_per_step": float(rewards.mean()),
        "total_reward": float(rewards.sum()),
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
    }


def make_env():
    import gymnasium as gym
    try:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    except ImportError:
        pass
    return gym.make(
        "FrankaKitchen-v1",
        render_mode=None,
        tasks_to_complete=list(_KITCHEN_TASKS),
    )


def replay_episodes(
    actions: np.ndarray,
    episodes: list[tuple[int, int]],
    n_episodes: int,
    seed: int,
) -> dict:
    """Replay each (sliced) action sequence in env and accumulate reward."""
    env = make_env()
    rng = np.random.default_rng(seed)
    # Pick a random subset (or all, capped at n_episodes).
    idx_pool = np.arange(len(episodes))
    rng.shuffle(idx_pool)
    chosen = idx_pool[:n_episodes]

    rows = []
    for ep_i in chosen:
        s, e = episodes[ep_i]
        ep_actions = actions[s:e]
        obs, _ = env.reset(seed=int(seed + ep_i))
        total_r = 0.0
        steps = 0
        completed = []
        terminated = False
        truncated = False
        # Clip to env action range to mirror eval_online_continuous.
        low, high = env.action_space.low, env.action_space.high
        for a in ep_actions:
            a_clipped = np.clip(a, low, high)
            obs, r, terminated, truncated, info = env.step(a_clipped)
            total_r += float(r)
            steps += 1
            completed = info.get("episode_task_completions", completed)
            if terminated or truncated:
                break
        rows.append({
            "episode_idx": int(ep_i),
            "dataset_length": int(e - s),
            "replay_length": int(steps),
            "replay_return": float(total_r),
            "tasks_completed": int(len(completed)),
            "success": bool(len(completed) >= len(_KITCHEN_TASKS)),
        })
    env.close()

    returns = np.array([r["replay_return"] for r in rows])
    successes = np.array([r["success"] for r in rows])
    completions = np.array([r["tasks_completed"] for r in rows])
    summary = {
        "n_episodes_replayed": int(len(rows)),
        "mean_replay_return": float(returns.mean()),
        "std_replay_return":  float(returns.std()),
        "max_replay_return":  float(returns.max()),
        "mean_tasks_completed": float(completions.mean()),
        "success_rate":       float(successes.mean()),
    }
    return {"per_episode": rows, "summary": summary}


def run(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)

    with h5py.File(args.hdf5_path, "r") as f:
        rewards   = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts  = f["timeouts"][:] if "timeouts" in f else np.zeros_like(terminals)
        actions   = f["actions"][:]

    episodes = split_episodes(terminals, timeouts)

    recorded = summarize_recorded(rewards, episodes)
    print(f"[dataset_online_kitchen] RECORDED dataset rewards")
    print(f"  n_episodes  = {recorded['n_episodes']}")
    print(f"  n_steps     = {recorded['n_transitions']}")
    print(f"  return      = {recorded['mean_return']:.2f} ± "
          f"{recorded['std_return']:.2f}  "
          f"(min={recorded['min_return']:.1f}, max={recorded['max_return']:.1f}, "
          f"median={recorded['median_return']:.1f})")
    print(f"  length      = mean={recorded['mean_length']:.1f}, "
          f"median={recorded['median_length']:.1f}")
    print(f"  reward/step = mean={recorded['mean_reward_per_step']:.4f}, "
          f"max={recorded['max_reward_per_step']:.1f}")

    save_json(recorded, str(out_dir / "recorded_metrics.json"))
    recorded_csv = pd.DataFrame({
        "episode_idx": list(range(recorded["n_episodes"])),
        "length":      recorded["lengths"],
        "return":      recorded["returns"],
    })
    save_csv(recorded_csv, str(out_dir / "recorded_per_episode.csv"))

    if args.replay_episodes > 0:
        print(f"\n[dataset_online_kitchen] REPLAY in FrankaKitchen-v1 "
              f"(n={args.replay_episodes})")
        replay = replay_episodes(
            actions, episodes, args.replay_episodes, seed=args.seed,
        )
        print(f"  online return = "
              f"{replay['summary']['mean_replay_return']:.2f} ± "
              f"{replay['summary']['std_replay_return']:.2f}  "
              f"(max={replay['summary']['max_replay_return']:.1f})")
        print(f"  mean tasks completed = "
              f"{replay['summary']['mean_tasks_completed']:.2f} / "
              f"{len(_KITCHEN_TASKS)}")
        print(f"  success rate         = {replay['summary']['success_rate']:.3f}")
        save_json(replay["summary"], str(out_dir / "replay_summary.json"))
        save_csv(pd.DataFrame(replay["per_episode"]),
                 str(out_dir / "replay_per_episode.csv"))
    print(f"\n[dataset_online_kitchen] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate kitchen-complete-v2 dataset's expert reward."
    )
    p.add_argument(
        "--hdf5_path",
        default="outputs/d4rl_cache/kitchen-complete-v2.hdf5",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/eval/dataset_online/kitchen_complete",
    )
    p.add_argument(
        "--replay_episodes", type=int, default=50,
        help="How many dataset episodes to replay in the env (0 = skip).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
