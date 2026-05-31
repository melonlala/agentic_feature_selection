"""Online evaluation of student policies in the noisy Taxi-v3 environment.

Loads all student checkpoints in student_dir, runs greedy rollouts, and
records average return, success rate, and episode length.

Usage:
    python eval/eval_online.py \\
        --config configs/taxi_noise8.yaml \\
        --student_dir outputs/students/taxi_noise8/seed0/shap \\
        --output_dir outputs/eval/online/taxi_noise8/seed0/shap
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from envs.noisy_taxi_wrapper import NoisyTaxiWrapper
from eval.eval_offline import load_student
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_json, save_csv, save_json
from utils.metrics import success_rate
from utils.seed import set_global_seed


def make_env(cfg: dict, seed: int) -> NoisyTaxiWrapper:
    env_cfg = cfg["env"]
    base_env = gym.make(env_cfg["env_id"])
    return NoisyTaxiWrapper(
        base_env,
        noise_dim=env_cfg["noise_dim"],
        noise_type=env_cfg["noise_type"],
        categorical_noise_cardinality=env_cfg.get("categorical_noise_cardinality", 5),
        noise_params=env_cfg.get("noise_params"),
        seed=seed,
    )


def run_online(
    model: "BCPolicy",
    feature_idx: list[int],
    env: NoisyTaxiWrapper,
    n_episodes: int,
    seed: int,
) -> dict:
    """Run greedy rollouts and compute online metrics.

    Args:
        model: Trained BCPolicy (eval mode).
        feature_idx: Feature indices to use.
        env: NoisyTaxiWrapper environment.
        n_episodes: Number of evaluation episodes.
        seed: Starting seed.

    Returns:
        Dict with avg_return, std_return, success_rate, avg_episode_length.
    """
    returns = []
    lengths = []

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_length = 0

    while len(returns) < n_episodes:
        obs_sub = torch.from_numpy(obs[feature_idx].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_sub)
        action = int(logits.argmax(dim=-1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward
        ep_length += 1

        if terminated or truncated:
            returns.append(ep_return)
            lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0
            obs, _ = env.reset()

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": success_rate(returns),
        "avg_episode_length": float(np.mean(lengths)),
        "n_episodes": len(returns),
    }


def run(args: argparse.Namespace) -> None:
    """Main online evaluation routine."""
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    n_episodes = cfg["eval"]["online_episodes"]

    student_dir = Path(args.student_dir)
    ckpt_dirs = sorted([d for d in student_dir.iterdir() if d.is_dir() and (d / "model.pt").exists()])
    if not ckpt_dirs:
        raise FileNotFoundError(f"No model.pt found under {student_dir}")

    rows = []
    for k_dir in ckpt_dirs:
        ckpt_path = str(k_dir / "model.pt")
        model, feature_idx, feature_names = load_student(ckpt_path)

        env = make_env(cfg, seed=args.seed + 100)
        metrics = run_online(model, feature_idx, env, n_episodes, seed=args.seed + 200)
        env.close()

        # Load k/selector info if available
        metrics_json_path = k_dir / "metrics.json"
        if metrics_json_path.exists():
            stored = load_json(str(metrics_json_path))
            k_val = stored.get("k", k_dir.name)
            selector = stored.get("selector", "unknown")
        else:
            k_val = k_dir.name
            selector = "unknown"

        row = {
            "k_label": k_dir.name,
            "k": k_val,
            "selector": selector,
            "feature_names": "|".join(feature_names),
            **metrics,
        }
        rows.append(row)
        print(f"  {k_dir.name}: avg_return={metrics['avg_return']:.2f}, "
              f"success={metrics['success_rate']:.3f}, "
              f"avg_len={metrics['avg_episode_length']:.1f}")

    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "online_metrics.csv"))
    save_json(rows, str(out_dir / "online_metrics.json"))
    print(f"\n[eval_online] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online evaluation of student policies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
