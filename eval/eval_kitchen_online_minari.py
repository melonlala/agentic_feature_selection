"""Online eval for kitchen BC students against the Minari D4RL kitchen env.

Loads ``D4RL/kitchen/complete-v2`` via Minari, recovers the env with
``dataset.recover_environment()``, then rolls each BC student checkpoint
saved under ``outputs/students/kitchen_complete/seed{N}/{selector}/k*/`` for
a configurable number of episodes.

Per-student outputs (one online_metrics.csv per selector dir):
    outputs/eval/online/kitchen_complete/seed{N}/{selector}/online_metrics.csv
        columns: k, mean_return, std_return, mean_length, success_rate, n_episodes
    outputs/eval/online/kitchen_complete/seed{N}/{selector}/k*/online_eval_metrics.json
        per-episode returns + metrics

Notes
-----
  * Dataset obs is 60-D (D4RL convention).  FrankaKitchen-v1 from
    gymnasium_robotics gives 59-D — we pad with the constant 60th feature
    (``obj_41`` is a constant -0.06 across the entire D4RL dataset).
    The student's stored ``feature_idx`` slices the padded 60-D vector.
  * "Success" = all four kitchen tasks completed within the episode.
  * BC student checkpoints are SB3 ActorCriticPolicy (from imitation.bc.BC)
    and are reconstructed via ``ActorCriticPolicy(**ckpt['data'])`` +
    ``load_state_dict(ckpt['state_dict'])``.

Usage:
    python eval/eval_kitchen_online_minari.py \\
        --seed 1 \\
        --selectors mci_nn shap random oracle mi full \\
        --n_episodes 20
"""

from __future__ import annotations

import argparse
import os
# Headless MuJoCo rendering — avoid the script crashing on machines without an X server.
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import ensure_dir, save_csv, save_json
from utils.seed import set_global_seed


DATASET_ID  = "D4RL/kitchen/complete-v2"
OBS_PAD     = -0.06   # dataset's obj_41 is constant -0.06 → exact 59→60 pad
N_TASKS     = 4       # microwave + kettle + light switch + slide cabinet
MAX_STEPS_DEFAULT = 280  # FrankaKitchen-v1 spec.max_episode_steps


# ─── checkpoint loading ──────────────────────────────────────────────────────

def load_sb3_policy(ckpt_path: Path, device: torch.device):
    """Reconstruct the SB3 ActorCriticPolicy saved by train_bc_continuous.py."""
    from stable_baselines3.common.policies import ActorCriticPolicy
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    data = dict(ckpt["data"])
    # lr_schedule may not survive across SB3 versions — replace with a no-op.
    data["lr_schedule"] = lambda _progress: 1.0
    policy = ActorCriticPolicy(**data)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device).eval()
    return policy, list(ckpt["feature_idx"])


# ─── obs handling ────────────────────────────────────────────────────────────

def flatten_obs(obs) -> np.ndarray:
    """Extract the 59-D `observation` key from FrankaKitchen-v1's Dict obs and
    pad to 60 with the constant -0.06 (matches the D4RL convention)."""
    flat = obs["observation"] if isinstance(obs, dict) else np.asarray(obs)
    flat = np.asarray(flat, dtype=np.float32)
    if flat.size == 59:
        flat = np.concatenate([flat, np.array([OBS_PAD], dtype=np.float32)])
    return flat


# ─── rollout ─────────────────────────────────────────────────────────────────

def run_episode(env, policy, feature_idx: list[int], max_steps: int,
                device: torch.device) -> dict:
    obs, _ = env.reset()
    total_return = 0.0
    n_completed_max = 0
    steps = 0
    idx_t = np.asarray(feature_idx, dtype=np.int64)

    for _ in range(max_steps):
        flat = flatten_obs(obs)
        x = flat[idx_t][None].astype(np.float32)
        with torch.no_grad():
            a, _ = policy.predict(x, deterministic=True)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        # The kitchen env expects 9-D in [-1, 1]; deterministic predict already
        # respects the action_space bounds but clip defensively.
        a = np.clip(a, env.action_space.low, env.action_space.high)

        obs, r, term, trunc, info = env.step(a)
        total_return += float(r)
        steps += 1
        n_completed = len(info.get("episode_task_completions") or [])
        n_completed_max = max(n_completed_max, n_completed)
        if term or trunc:
            break

    return {
        "return":       total_return,
        "length":       steps,
        "tasks_done":   n_completed_max,
        "success":      n_completed_max >= N_TASKS,
    }


def eval_one_ckpt(env, ckpt_path: Path, n_episodes: int, max_steps: int,
                  device: torch.device, seed_base: int) -> dict:
    policy, feature_idx = load_sb3_policy(ckpt_path, device)
    returns: list[float] = []
    lengths: list[int]   = []
    successes: list[bool] = []
    tasks_done: list[int] = []
    for ep in range(n_episodes):
        env.reset(seed=seed_base + ep)            # deterministic per-ep seed
        r = run_episode(env, policy, feature_idx, max_steps, device)
        returns.append(r["return"])
        lengths.append(r["length"])
        successes.append(r["success"])
        tasks_done.append(r["tasks_done"])
    return {
        "n_episodes":      int(n_episodes),
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_length":     float(np.mean(lengths)),
        "success_rate":    float(np.mean(successes)),
        "mean_tasks_done": float(np.mean(tasks_done)),
        "returns":         returns,
        "lengths":         lengths,
        "tasks_done":      tasks_done,
        "feature_idx":     feature_idx,
    }


# ─── driver ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    import minari
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[online_kitchen] minari {minari.__version__}; loading {DATASET_ID}...")
    dataset = minari.load_dataset(DATASET_ID, download=True)
    env = dataset.recover_environment()
    max_steps = (env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps
                 else MAX_STEPS_DEFAULT)
    print(f"[online_kitchen] env={env.spec.id if env.spec else type(env).__name__}, "
          f"max_steps={max_steps}, action_space={env.action_space}")

    student_root = Path("outputs/students/kitchen_complete") / f"seed{args.seed}"
    if not student_root.exists():
        raise FileNotFoundError(f"No students dir: {student_root}")

    selectors = args.selectors
    if not selectors:
        selectors = sorted(d.name for d in student_root.iterdir() if d.is_dir())
    print(f"[online_kitchen] selectors: {selectors}")

    for selector in selectors:
        sel_dir = student_root / selector
        if not sel_dir.is_dir():
            print(f"[online_kitchen] skip {selector} — no such dir.")
            continue

        out_dir = ensure_dir(
            Path("outputs/eval/online/kitchen_complete") / f"seed{args.seed}" / selector
        )

        # k subdirs (k5, k10, ..., or 'full')
        k_dirs = sorted(
            (d for d in sel_dir.iterdir()
             if d.is_dir() and (d / "model.pt").exists()),
            key=lambda d: (d.name != "full", d.name),  # 'full' last
        )
        if not k_dirs:
            print(f"[online_kitchen] {selector}: no model.pt found, skipping.")
            continue

        rows: list[dict] = []
        for k_dir in k_dirs:
            ckpt = k_dir / "model.pt"
            print(f"\n[online_kitchen] {selector}/{k_dir.name}  "
                  f"(rollouts: {args.n_episodes})...")
            t0 = time.time()
            m = eval_one_ckpt(
                env, ckpt,
                n_episodes=args.n_episodes,
                max_steps=max_steps,
                device=device,
                seed_base=args.seed * 1000,
            )
            dt = time.time() - t0
            save_json(m, str(ensure_dir(out_dir / k_dir.name) / "online_eval_metrics.json"))
            print(f"  return={m['mean_return']:.2f}±{m['std_return']:.2f}  "
                  f"tasks_done={m['mean_tasks_done']:.2f}/4  "
                  f"success={m['success_rate']:.2f}  "
                  f"len={m['mean_length']:.0f}  ({dt:.1f}s)")
            rows.append({
                "k":               k_dir.name,
                "mean_return":     round(m["mean_return"], 3),
                "std_return":      round(m["std_return"], 3),
                "success_rate":    round(m["success_rate"], 4),
                "mean_length":     round(m["mean_length"], 1),
                "mean_tasks_done": round(m["mean_tasks_done"], 3),
                "n_episodes":      m["n_episodes"],
            })

        df = pd.DataFrame(rows)
        save_csv(df, str(out_dir / "online_metrics.csv"))
        print(f"[online_kitchen] wrote {out_dir / 'online_metrics.csv'}")

    env.close()
    print("\n[online_kitchen] all selectors done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online eval for kitchen BC students via Minari."
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--selectors", nargs="*", default=None,
                   help="Selector dir names under outputs/students/kitchen_complete/seed{N}/. "
                        "Defaults to all subdirs.")
    p.add_argument("--n_episodes", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
