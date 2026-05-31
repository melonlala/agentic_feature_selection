"""Load D4RL offline datasets and convert to the project's .npz schema.

Supports kitchen-complete-v2 and pen-{human,cloned,expert}-v1 datasets.

Two loading modes (tried in order):
  1. d4rl package: ``import d4rl; gym.make(env_id).get_dataset()``
  2. Direct HDF5:  download the file from rail.eecs.berkeley.edu with requests/h5py

The output .npz schema mirrors collect_dataset.py for continuous-action tasks:
  X_train / X_val / X_test      float32  [N, obs_dim]
  y_train / y_val / y_test      float32  [N, action_dim]  (continuous actions)
  action_norm_train/val/test    float32  [N]              ||a||_2 per step
  rewards_train/val/test        float32  [N]
  terminals_train/val/test      bool     [N]
  feature_names                 str      [obs_dim]
  action_names                  str      [action_dim]

Usage:
    python teacher/load_d4rl_dataset.py \\
        --config configs/kitchen_complete.yaml \\
        --seed 0 \\
        --output_dir outputs/datasets/kitchen_complete/seed0

    # Or supply an already-downloaded HDF5 file:
    python teacher/load_d4rl_dataset.py \\
        --config configs/kitchen_complete.yaml \\
        --seed 0 \\
        --hdf5_path /path/to/kitchen-complete-v2.hdf5 \\
        --output_dir outputs/datasets/kitchen_complete/seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed

# ─── D4RL dataset URLs ────────────────────────────────────────────────────────
# Verified working URLs from rail.eecs.berkeley.edu (checked 2025-04).
#
# kitchen-complete-v2:
#   The "complete" variant from d4rl v2 uses the same underlying HDF5 as
#   kitchen-complete-v0 (microwave + kettle + bottomburner + light demos).
#   No standalone kitchen-complete-v2.hdf5 is available for direct download;
#   the v2 API only relabels rewards.  We use the v0 HDF5 directly (same obs/act).
#
# pen-*-v1:
#   Available at hand_dapg_v1/ (v1 naming, confirmed 200 responses).
_D4RL_URLS: dict[str, str] = {
    "kitchen-complete-v2": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/"
        "kitchen_microwave_kettle_bottomburner_light-v0.hdf5"
    ),
    # Legacy v0 name resolves to same file as v2 (same observations/actions)
    "kitchen-complete-v0": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/"
        "kitchen_microwave_kettle_bottomburner_light-v0.hdf5"
    ),
    "pen-human-v1": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/"
        "pen-human-v1.hdf5"
    ),
    "pen-cloned-v1": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/"
        "pen-cloned-v1.hdf5"
    ),
    "pen-expert-v1": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/"
        "pen-expert-v1.hdf5"
    ),
}

# ─── Feature name templates ───────────────────────────────────────────────────
def _kitchen_feature_names() -> list[str]:
    names: list[str] = []
    for i in range(9):
        names.append(f"qpos_{i}")   # robot joint positions
    for i in range(9):
        names.append(f"qvel_{i}")   # robot joint velocities
    for i in range(42):
        names.append(f"obj_{i}")    # object/goal state (kitchen-specific)
    return names  # 60 total


def _pen_feature_names() -> list[str]:
    names: list[str] = []
    for i in range(24):
        names.append(f"hand_qpos_{i}")    # hand joint positions
    for i in range(3):
        names.append(f"obj_pos_{i}")      # object position
    for i in range(3):
        names.append(f"obj_euler_{i}")    # object euler angles
    for i in range(3):
        names.append(f"goal_pos_{i}")     # goal position
    for i in range(3):
        names.append(f"goal_euler_{i}")   # goal euler angles
    for i in range(3):
        names.append(f"obj2goal_pos_{i}") # object-to-goal position delta
    for i in range(3):
        names.append(f"obj2goal_rot_{i}") # object-to-goal rotation delta
    for i in range(3):
        names.append(f"contact_{i}")      # contact features
    return names  # 45 total


def _generic_feature_names(obs_dim: int, prefix: str = "obs") -> list[str]:
    return [f"{prefix}_{i}" for i in range(obs_dim)]


def _kitchen_action_names() -> list[str]:
    return [f"act_{i}" for i in range(9)]


def _pen_action_names() -> list[str]:
    return [f"act_{i}" for i in range(24)]


def get_feature_names(env_id: str, obs_dim: int) -> list[str]:
    if "kitchen" in env_id:
        names = _kitchen_feature_names()
    elif "pen" in env_id:
        names = _pen_feature_names()
    else:
        names = _generic_feature_names(obs_dim)
    # Trim or pad to actual obs_dim
    if len(names) > obs_dim:
        names = names[:obs_dim]
    while len(names) < obs_dim:
        names.append(f"obs_{len(names)}")
    return names


def get_action_names(env_id: str, action_dim: int) -> list[str]:
    if "kitchen" in env_id:
        return _kitchen_action_names()[:action_dim]
    elif "pen" in env_id:
        return _pen_action_names()[:action_dim]
    return [f"act_{i}" for i in range(action_dim)]


# ─── Loading backends ─────────────────────────────────────────────────────────

def _load_via_d4rl(env_id: str) -> dict[str, np.ndarray]:
    """Attempt to load dataset via the d4rl package."""
    import gym  # type: ignore[import]
    import d4rl  # type: ignore[import]  # noqa: F401
    env = gym.make(env_id)
    raw = env.get_dataset()
    env.close()
    return raw


def _download_hdf5(env_id: str, cache_dir: Path) -> Path:
    """Download the HDF5 file for env_id if not cached."""
    import requests  # type: ignore[import]

    url = _D4RL_URLS.get(env_id)
    if url is None:
        raise ValueError(
            f"No download URL for env_id={env_id!r}. "
            f"Known IDs: {list(_D4RL_URLS)}"
        )
    fname = cache_dir / f"{env_id}.hdf5"
    if fname.exists():
        print(f"[load_d4rl] Using cached HDF5: {fname}")
        return fname

    ensure_dir(cache_dir)
    print(f"[load_d4rl] Downloading {url} → {fname} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(fname, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    return fname


def _load_via_hdf5(hdf5_path: Path) -> dict[str, np.ndarray]:
    """Load dataset from an HDF5 file (d4rl format)."""
    import h5py

    raw: dict[str, np.ndarray] = {}
    with h5py.File(hdf5_path, "r") as f:
        for key in ("observations", "actions", "rewards", "terminals", "timeouts"):
            if key in f:
                raw[key] = f[key][:]
    return raw


def load_raw_dataset(
    env_id: str,
    hdf5_path: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load raw D4RL dataset.  Returns a dict with keys:
       observations, actions, rewards, terminals, [timeouts].
    """
    if hdf5_path is not None:
        return _load_via_hdf5(Path(hdf5_path))

    # Try d4rl package first
    try:
        raw = _load_via_d4rl(env_id)
        print("[load_d4rl] Loaded via d4rl package.")
        return {
            "observations": np.array(raw["observations"], dtype=np.float32),
            "actions": np.array(raw["actions"], dtype=np.float32),
            "rewards": np.array(raw["rewards"], dtype=np.float32),
            "terminals": np.array(raw["terminals"], dtype=bool),
        }
    except ImportError:
        print("[load_d4rl] d4rl not available; falling back to HDF5 download.")

    # Fall back to direct download
    if cache_dir is None:
        cache_dir = Path("outputs/d4rl_cache")
    local_path = _download_hdf5(env_id, cache_dir)
    return _load_via_hdf5(local_path)


# ─── Dataset conversion ───────────────────────────────────────────────────────

def split_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    rng: np.random.Generator | None = None,
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Shuffle and split arrays into train/val/test.

    Args:
        obs:         [N, obs_dim] observations.
        actions:     [N, action_dim] continuous actions.
        rewards:     [N] rewards.
        terminals:   [N] terminal flags.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation (test = 1 - train - val).
        rng:         Seeded RNG for shuffling.
        max_samples: Cap total N before splitting (None = use all).

    Returns:
        Dict with X_train/val/test, y_train/val/test, action_norm_*,
        rewards_*, terminals_* keys.
    """
    N = len(obs)

    if max_samples and N > max_samples:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_samples, replace=False)
        obs = obs[idx]
        actions = actions[idx]
        rewards = rewards[idx]
        terminals = terminals[idx]
        N = max_samples

    if rng is None:
        rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    obs = obs[perm]
    actions = actions[perm]
    rewards = rewards[perm]
    terminals = terminals[perm]

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    def _split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    X_tr, X_va, X_te = _split(obs)
    y_tr, y_va, y_te = _split(actions)
    r_tr, r_va, r_te = _split(rewards)
    t_tr, t_va, t_te = _split(terminals)

    norm_tr = np.linalg.norm(y_tr, axis=1).astype(np.float32)
    norm_va = np.linalg.norm(y_va, axis=1).astype(np.float32)
    norm_te = np.linalg.norm(y_te, axis=1).astype(np.float32)

    return {
        "X_train": X_tr.astype(np.float32),
        "X_val":   X_va.astype(np.float32),
        "X_test":  X_te.astype(np.float32),
        "y_train": y_tr.astype(np.float32),
        "y_val":   y_va.astype(np.float32),
        "y_test":  y_te.astype(np.float32),
        "action_norm_train": norm_tr,
        "action_norm_val":   norm_va,
        "action_norm_test":  norm_te,
        "rewards_train":   r_tr.astype(np.float32),
        "rewards_val":     r_va.astype(np.float32),
        "rewards_test":    r_te.astype(np.float32),
        "terminals_train": t_tr.astype(bool),
        "terminals_val":   t_va.astype(bool),
        "terminals_test":  t_te.astype(bool),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    env_cfg = cfg["env"]
    env_id = env_cfg["env_id"]
    ds_cfg = cfg.get("dataset", {})

    # ── Load raw data ────────────────────────────────────────────────────────
    raw = load_raw_dataset(
        env_id=env_id,
        hdf5_path=args.hdf5_path,
        cache_dir=Path("outputs/d4rl_cache"),
    )

    obs     = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    rewards = raw["rewards"].astype(np.float32)
    terminals = raw.get("terminals", np.zeros(len(obs), dtype=bool))
    if "timeouts" in raw:
        terminals = terminals | raw["timeouts"].astype(bool)

    N, obs_dim    = obs.shape
    action_dim    = actions.shape[1]

    print(f"[load_d4rl] {env_id}: N={N}, obs_dim={obs_dim}, action_dim={action_dim}")

    # ── Feature names ────────────────────────────────────────────────────────
    feature_names = get_feature_names(env_id, obs_dim)
    action_names  = get_action_names(env_id, action_dim)

    # ── Split ────────────────────────────────────────────────────────────────
    max_samples = ds_cfg.get("max_samples", None)
    arrays = split_dataset(
        obs, actions, rewards, terminals,
        train_ratio=ds_cfg.get("train_ratio", 0.8),
        val_ratio=ds_cfg.get("val_ratio", 0.1),
        rng=rng,
        max_samples=max_samples,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    npz_path = str(out_dir / "dataset.npz")
    np.savez(
        npz_path,
        **arrays,
        feature_names=np.array(feature_names),
        action_names=np.array(action_names),
    )
    print(f"[load_d4rl] Saved dataset to {npz_path}")
    print(f"  train={len(arrays['X_train'])}, val={len(arrays['X_val'])}, "
          f"test={len(arrays['X_test'])}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "env_id": env_id,
        "seed": args.seed,
        "config": args.config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_total": N,
        "n_train": int(len(arrays["X_train"])),
        "n_val":   int(len(arrays["X_val"])),
        "n_test":  int(len(arrays["X_test"])),
        "feature_names": feature_names,
        "action_names":  action_names,
        "task_type": "continuous",
    }
    save_json(meta, str(out_dir / "metadata.json"))
    print("[load_d4rl] Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert D4RL offline dataset to project .npz format."
    )
    p.add_argument("--config",     required=True, help="YAML config path.")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--hdf5_path",  default=None,
                   help="Path to a pre-downloaded HDF5 file. If omitted, auto-downloads.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
