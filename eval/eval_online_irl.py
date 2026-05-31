"""Online evaluation of AIRL/GAIL students on seals/Ant-v1.

Loads each k-checkpoint saved by student/train_student_irl.py, rebuilds the
SB3 ActorCriticPolicy from ckpt['data'] + ckpt['state_dict'], applies the same
state-projection / latent-projection wrapper used at train time, and rolls out
``n_episodes`` deterministic episodes.

Output:
    {output_dir}/online_metrics.csv     k_label, mean_return, std_return, success_rate, mean_length
    {output_dir}/online_metrics.json    raw list of dicts including per-episode returns
    {output_dir}/{k}/online_eval_metrics.json   per-ckpt detail

Usage:
    python eval/eval_online_irl.py \\
        --config configs/seals_ant.yaml \\
        --student_dir outputs/students_irl/seals_ant/seed0/shap \\
        --output_dir  outputs/eval/online_irl/seals_ant/seed0/shap \\
        --n_episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import seals  # noqa: F401  — registers seals/* namespace
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.latent_extract import build_frozen_layer1
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_csv, save_json


def _load_irl_policy(ckpt_path: Path, device: torch.device):
    """Return (policy, feature_idx, latent_mode_kwargs)."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    data = dict(ckpt["data"])
    data["lr_schedule"] = lambda _: 1.0
    policy = ActorCriticPolicy(**data)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device).eval()
    return policy, list(ckpt["feature_idx"]), ckpt


def _project_state(obs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)[idx]


def _project_latent(
    obs: np.ndarray, layer1: torch.nn.Sequential,
    idx: np.ndarray, device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        z = layer1(x).squeeze(0).cpu().numpy()
    return z[idx].astype(np.float32)


def _rollout_one(
    env: gym.Env,
    policy: ActorCriticPolicy,
    feature_idx: list[int],
    layer1: torch.nn.Sequential | None,
    device: torch.device,
    max_steps: int,
    seed: int,
) -> dict:
    idx = np.asarray(feature_idx, dtype=np.int64)
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    steps = 0
    success = False
    for _ in range(max_steps):
        if layer1 is not None:
            ob_proj = _project_latent(obs, layer1, idx, device)
        else:
            ob_proj = _project_state(obs, idx)
        action, _ = policy.predict(ob_proj, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        steps += 1
        if info.get("success", False):
            success = True
        if terminated or truncated:
            break
    return {"return": total_return, "length": steps, "success": success}


def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    env_id = cfg["env"]["env_id"]

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_dir = Path(args.student_dir)
    ckpt_dirs = sorted(student_dir.glob("k*")) + sorted(student_dir.glob("full"))
    if not ckpt_dirs:
        ckpt_dirs = [student_dir]

    env = gym.make(env_id)
    max_steps = args.max_steps or (env.spec.max_episode_steps or 1000)

    rows: list[dict] = []
    for k_dir in ckpt_dirs:
        ckpt_path = k_dir / "model.pt"
        if not ckpt_path.exists():
            continue

        policy, feature_idx, ckpt = _load_irl_policy(ckpt_path, device)

        layer1 = None
        if ckpt.get("latent_mode", False):
            src = ckpt["source_student_path"]
            full_ckpt = torch.load(src, map_location=device, weights_only=False)
            layer1 = build_frozen_layer1(full_ckpt, ckpt.get("latent_layer", "pre_relu"),
                                         device)

        print(f"\n[eval_online_irl] {k_dir.name} (env={env_id}, n_episodes={args.n_episodes})")
        returns: list[float] = []
        lengths: list[int]   = []
        successes: list[bool] = []
        for ep in range(args.n_episodes):
            res = _rollout_one(
                env, policy, feature_idx, layer1, device,
                max_steps=max_steps, seed=args.seed + ep,
            )
            returns.append(res["return"])
            lengths.append(res["length"])
            successes.append(res["success"])

        metrics = {
            "mean_return":  float(np.mean(returns)),
            "std_return":   float(np.std(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_length":  float(np.mean(lengths)),
            "n_episodes":   int(args.n_episodes),
            "returns":      returns,
        }
        save_json(metrics, str(k_dir / "online_eval_metrics.json"))
        rows.append({
            "k":           k_dir.name,
            "mean_return": round(metrics["mean_return"], 3),
            "std_return":  round(metrics["std_return"], 3),
            "success_rate": round(metrics["success_rate"], 4),
            "mean_length": round(metrics["mean_length"], 1),
        })
        print(f"  return={metrics['mean_return']:.2f}±{metrics['std_return']:.2f}  "
              f"success={metrics['success_rate']:.3f}  len={metrics['mean_length']:.0f}")

    env.close()
    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "online_metrics.csv"))
    save_json(rows, str(out_dir / "online_metrics.json"))
    print(f"\n[eval_online_irl] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online evaluation of AIRL/GAIL students on seals/Ant-v1."
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--student_dir",  required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--n_episodes",   type=int, default=20)
    p.add_argument("--max_steps",    type=int, default=0,
                   help="0 → use env's default max_episode_steps.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
