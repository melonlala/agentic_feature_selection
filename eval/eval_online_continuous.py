"""Online evaluation for continuous-action BC students on D4RL pen/kitchen tasks.

Loads a student checkpoint and runs it in the gymnasium environment for N episodes.
Reports average return, success rate, and average episode length.

Supported environments (obs_dim matches D4RL datasets):
  pen-human-v1, pen-cloned-v1, pen-expert-v1  ->  AdroitHandPen-v1  (obs=45, action=24)
  kitchen-complete-v2  ->  FrankaKitchen-v1  (obs=59, padded to 60 with constant -0.06)

Kitchen obs padding: D4RL kitchen obs_dim=60, but FrankaKitchen-v1 provides 59. The 60th
feature (index 59, "obj_41") is a constant -0.06 across the entire D4RL dataset (std≈0),
so padding with -0.06 is exact. Success = all 4 tasks completed within the episode.

Usage:
    python eval/eval_online_continuous.py \\
        --config configs/pen_human.yaml \\
        --student_dir outputs/students/pen_human/seed0/shap \\
        --output_dir  outputs/eval/online/pen_human/seed0/shap \\
        --n_episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.bc_continuous_model import (
    BCContinuousPolicy,
    BCContinuousPolicyFromLatent,
    BCGaussianPolicy,
    BCGaussianPolicyFromLatent,
)
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_csv, save_json
import pandas as pd

# Map env_id -> (gymnasium env_id, kwargs).
# seals/Ant-v1 is registered by `import seals` at make_env() time.
_ENV_MAP = {
    "pen-human-v1":      ("AdroitHandPen-v1", {}),
    "pen-cloned-v1":     ("AdroitHandPen-v1", {}),
    "pen-expert-v1":     ("AdroitHandPen-v1", {}),
    "kitchen-complete-v2": (
        "FrankaKitchen-v1",
        {"tasks_to_complete": ["microwave", "kettle", "bottom burner", "light switch"]},
    ),
    "seals/Ant-v1":      ("seals/Ant-v1", {}),
}

# kitchen-complete-v2: D4RL obs_dim=60, FrankaKitchen-v1 provides 59.
# Feature index 59 ("obj_41") is constant -0.06 across the entire D4RL dataset;
# padding with this value exactly reconstructs the 60-dim D4RL observation.
_KITCHEN_OBS_PAD = -0.06


def make_env(env_id: str, seed: int = 0):
    import gymnasium as gym
    try:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    except ImportError:
        pass
    if env_id.startswith("seals/"):
        try:
            import seals  # noqa: F401  — registers seals/* envs
        except ImportError:
            pass
    entry = _ENV_MAP.get(env_id)
    if entry is None:
        raise ValueError(
            f"Online eval not supported for env_id={env_id!r}. "
            f"Supported: {list(_ENV_MAP.keys())}"
        )
    gym_id, kwargs = entry
    env = gym.make(gym_id, render_mode=None, **kwargs)
    return env


def load_student(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim    = ckpt["input_dim"]
    action_dim   = ckpt["action_dim"]
    hidden_dims  = ckpt["hidden_dims"]
    use_gaussian = ckpt.get("use_gaussian", False)
    feature_idx  = ckpt["feature_idx"]
    model_class  = ckpt.get(
        "model_class",
        "BCGaussianPolicy" if use_gaussian else "BCContinuousPolicy",
    )

    if model_class in ("BCContinuousPolicyFromLatent", "BCGaussianPolicyFromLatent"):
        # Late import to avoid circular dep when this module is used standalone.
        from ranker.latent_extract import build_frozen_layer1
        source_path  = ckpt["source_student_path"]
        latent_layer = ckpt.get("latent_layer", "pre_relu")
        latent_idx   = list(ckpt["latent_idx"])
        full_ckpt    = torch.load(source_path, map_location=device, weights_only=False)
        frozen_layer1 = build_frozen_layer1(full_ckpt, latent_layer, device)
        if model_class == "BCGaussianPolicyFromLatent":
            model: torch.nn.Module = BCGaussianPolicyFromLatent(
                frozen_layer1=frozen_layer1, latent_indices=latent_idx,
                action_dim=action_dim, hidden_dims=hidden_dims,
            )
        else:
            model = BCContinuousPolicyFromLatent(
                frozen_layer1=frozen_layer1, latent_indices=latent_idx,
                action_dim=action_dim, hidden_dims=hidden_dims,
            )
    elif use_gaussian:
        model = BCGaussianPolicy(
            input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims,
        )
    else:
        model = BCContinuousPolicy(
            input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, feature_idx, use_gaussian


def _flatten_obs(obs, env_id: str, expected_dim: int) -> np.ndarray:
    """Flatten env observation to a 1-D float32 array of expected_dim."""
    if isinstance(obs, dict):
        flat = obs["observation"].astype(np.float32)
    else:
        flat = np.asarray(obs, dtype=np.float32)
    # Kitchen: FrankaKitchen-v1 gives 59 dims; pad to 60 with the constant value.
    if len(flat) < expected_dim:
        pad = np.full(expected_dim - len(flat), _KITCHEN_OBS_PAD, dtype=np.float32)
        flat = np.concatenate([flat, pad])
    return flat


def run_episode(
    env,
    model: torch.nn.Module,
    feature_idx: list[int],
    use_gaussian: bool,
    device: torch.device,
    env_id: str = "",
    expected_obs_dim: int = 0,
    max_steps: int = 200,
    action_clip: float = 1.0,
) -> dict:
    obs, _ = env.reset()
    total_return = 0.0
    success = False
    steps = 0
    n_tasks = 4 if "kitchen" in env_id else 0

    for _ in range(max_steps):
        flat = _flatten_obs(obs, env_id, expected_obs_dim)
        x = torch.from_numpy(flat[feature_idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            if use_gaussian:
                action, _ = model(x)
            else:
                action = model(x)
        action = action.squeeze(0).cpu().numpy()
        # Clip to the env's action range. action_clip is the legacy default of
        # ±1 used by D4RL envs; for envs with wider ranges (e.g. seals/Ant-v1)
        # we honour the env's low/high.
        try:
            low, high = env.action_space.low, env.action_space.high
            action = np.clip(action, low, high)
        except AttributeError:
            action = np.clip(action, -action_clip, action_clip)

        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        steps += 1

        if n_tasks:
            # Kitchen: success = all required tasks completed this episode
            completed = info.get("episode_task_completions", [])
            if len(completed) >= n_tasks:
                success = True
        elif info.get("success", False):
            success = True

        if terminated or truncated:
            break

    return {"return": total_return, "success": success, "length": steps}


def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    env_id = cfg["env"]["env_id"]

    if env_id not in _ENV_MAP:
        print(f"[eval_online_continuous] Skipping {env_id}: not in ENV_MAP "
              f"(supported: {list(_ENV_MAP.keys())})")
        return

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_dir = Path(args.student_dir)
    ckpt_dirs = sorted(student_dir.glob("k*")) + sorted(student_dir.glob("full"))
    if not ckpt_dirs:
        ckpt_dirs = [student_dir]

    # Create env once and reuse across all k checkpoints to avoid repeated
    # MuJoCo model compilation (which dominates per-ckpt runtime).
    env = make_env(env_id, seed=args.seed)

    # Determine full D4RL obs dim as max(feature_idx)+1 across ALL k-checkpoints.
    # This matters for kitchen where the shap selector's k=15 checkpoint includes
    # feature index 59 but k=5 does not — we must pad the gymnasium obs (59-dim)
    # to 60 even when processing the k=5 checkpoint.
    dataset_obs_dim = 0
    for d in ckpt_dirs:
        cp = d / "model.pt"
        if not cp.exists():
            continue
        _c = torch.load(str(cp), map_location="cpu", weights_only=False)
        if _c["feature_idx"]:
            dataset_obs_dim = max(dataset_obs_dim, max(_c["feature_idx"]) + 1)
    if dataset_obs_dim == 0:
        print("[eval_online_continuous] No model.pt found, skipping.")
        env.close()
        return

    rows = []
    for k_dir in ckpt_dirs:
        ckpt_path = k_dir / "model.pt"
        if not ckpt_path.exists():
            continue

        model, feature_idx, use_gaussian = load_student(str(ckpt_path), device)

        print(f"\n[eval_online_continuous] {k_dir.name} ({env_id}) "
              f"n_episodes={args.n_episodes}")

        returns, successes, lengths = [], [], []
        for _ in range(args.n_episodes):
            result = run_episode(
                env, model, feature_idx, use_gaussian, device,
                env_id=env_id,
                expected_obs_dim=dataset_obs_dim,
                max_steps=args.max_steps,
            )
            returns.append(result["return"])
            successes.append(result["success"])
            lengths.append(result["length"])

        metrics = {
            "mean_return":  float(np.mean(returns)),
            "std_return":   float(np.std(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_length":  float(np.mean(lengths)),
            "n_episodes":   args.n_episodes,
            "returns":      returns,
        }
        row = {
            "k":           k_dir.name,
            "mean_return": round(metrics["mean_return"], 3),
            "std_return":  round(metrics["std_return"], 3),
            "success_rate": round(metrics["success_rate"], 4),
            "mean_length": round(metrics["mean_length"], 1),
        }
        rows.append(row)
        save_json(metrics, str(k_dir / "online_eval_metrics.json"))
        print(f"  return={metrics['mean_return']:.2f}±{metrics['std_return']:.2f}  "
              f"success={metrics['success_rate']:.3f}  "
              f"len={metrics['mean_length']:.0f}")

    env.close()
    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "online_metrics.csv"))
    save_json(rows, str(out_dir / "online_metrics.json"))
    print(f"\n[eval_online_continuous] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--student_dir",  required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--n_episodes",   type=int, default=50)
    p.add_argument("--max_steps",    type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
