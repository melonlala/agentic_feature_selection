"""Continuous-action BC training via the `imitation` package.

Every per-selector entry point —
    student/train_student_mci_hdc.py
    student/train_student_mci_nn.py
    student/train_student_sage.py
    student/train_student_kernelshap.py
    student/train_student_random.py
    student/train_student_full.py
— calls `train_bc_continuous_student(...)` from this module to produce the
*best* student model under its feature subset. The training method is fixed:

    imitation.algorithms.bc.BC          (the imitation package's BC trainer)
        + stable_baselines3 ActorCriticPolicy
        + AdamW (lr / weight_decay from cfg)
        + best-validation-MSE state restoration

The *only* per-selector difference is `feature_idx`. By construction this
means every selector is benchmarked against the same training pipeline —
comparison across selectors measures the value of the chosen feature subset,
not differences in optimizer, architecture, or learning rate.

Headline call (per-k):
    policy, metrics = train_bc_continuous_student(
        X_train, y_train, X_val, y_val,
        feature_idx=...,
        hidden_dims=..., epochs=..., lr=..., batch_size=...,
        seed=..., device=...,
    )

Sweep call (over the full topk list — used by per-selector scripts):
    df = train_bc_students(
        cfg, seed, device, dataset,
        feature_idx_fn=lambda k: ...,   # selector-specific
        topk_list=..., selector_name=..., output_dir=...,
    )

`train_bc_students` is a thin loop over k that calls
`train_bc_continuous_student` once per k, then writes
`{output_dir}/k{K}/model.pt`, `metrics.json`, and `summary.csv`.

Checkpoint format (one model.pt per k_label) — same as before:
    state_dict, data                    SB3 ActorCriticPolicy reconstruction
    model_state_dict                    alias of state_dict (project convention)
    feature_idx, feature_names          int + str lists, parallel
    input_dim, action_dim, hidden_dims  ints / list
    use_gaussian                        True
    model_class                         "ImitationBCPolicy"
    policy_class                        "ActorCriticPolicy"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_npz, save_csv, save_json
from utils.seed import set_global_seed


# ─── Headline: train ONE best BC student given selected features ─────────────

def train_bc_continuous_student(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_idx: list[int],
    *,
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_train_samples: int | None = None,
) -> tuple[ActorCriticPolicy, dict]:
    """Train one BC student via `imitation.bc.BC` on the selected features.

    The training method is identical regardless of selector — only
    `feature_idx` varies. After training, the policy's state_dict is rolled
    back to the epoch with the lowest validation MSE (= "best student model").

    Args:
        X_train, y_train: Training arrays in the FULL feature space.
        X_val,   y_val:   Validation arrays in the FULL feature space.
        feature_idx:      Indices to slice from X_train / X_val. This is the
                          only argument that differs across selectors.
        hidden_dims:      Policy MLP hidden layer sizes.
        epochs:           Number of `imitation.bc.BC` training epochs.
        lr, batch_size:   AdamW + bc.BC hyperparameters.
        seed:             Random seed (controls AdamW init + bc.BC's RNG).
        device:           Torch device.
        max_train_samples: Optional cap on training rows for speed.

    Returns:
        (best_policy, {"best_val_mse": float, "val_mses": list[float]}).
    """
    set_global_seed(seed)

    if max_train_samples and len(X_train) > max_train_samples:
        sub_rng = np.random.default_rng(seed)
        idx = sub_rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    X_tr_sub = X_train[:, feature_idx].astype(np.float32)
    y_tr     = y_train.astype(np.float32)

    input_dim  = len(feature_idx)
    action_dim = y_tr.shape[1]

    obs_space, act_space = _make_spaces(input_dim, action_dim)
    transitions = _make_transitions(X_tr_sub, y_tr)

    # Build the policy explicitly so we control net_arch — bc.BC's default
    # FeedForward32Policy is [32, 32], which is too narrow for our tasks.
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _progress: lr,
        net_arch=list(hidden_dims),
    ).to(device)

    rng = np.random.default_rng(seed)
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions,
        rng=rng,
        policy=policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        device=device,
    )

    # bc.BC has no built-in validation hook — we track best-MSE state via
    # on_epoch_end and restore at the end. This is what produces the
    # "best student model" the rest of the pipeline consumes.
    best_val_mse = float("inf")
    best_state: dict | None = None
    val_mses: list[float] = []

    def on_epoch_end() -> None:
        nonlocal best_val_mse, best_state
        m = evaluate_continuous(bc_trainer.policy, X_val, y_val, feature_idx)
        val_mses.append(m["mse"])
        if m["mse"] < best_val_mse:
            best_val_mse = m["mse"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in bc_trainer.policy.state_dict().items()
            }

    print(f"  Training BC via imitation.bc.BC for {epochs} epochs "
          f"(N_train={len(X_tr_sub)}, input_dim={input_dim}, "
          f"action_dim={action_dim}, hidden_dims={hidden_dims})...")
    bc_trainer.train(
        n_epochs=epochs,
        on_epoch_end=on_epoch_end,
        progress_bar=False,
    )

    if best_state is not None:
        bc_trainer.policy.load_state_dict(best_state)
    bc_trainer.policy.set_training_mode(False)

    return bc_trainer.policy, {
        "best_val_mse": best_val_mse,
        "val_mses":     val_mses,
    }


# ─── Setup / dataset / topk-list helpers used by per-selector scripts ────────

def setup_run(
    config_path: str, seed: int, output_dir: str,
) -> tuple[dict, torch.device, Path]:
    """Resolve config, seed RNGs, pick device, persist config snapshot."""
    cfg = resolve_config(config_path)
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))
    return cfg, device, out_dir


def load_dataset(dataset_path: str) -> dict[str, Any]:
    """Load a continuous-action dataset.npz into the shape this module uses."""
    data = load_npz(dataset_path)
    y_train = data["y_train"].astype(np.float32)
    return {
        "X_train":       data["X_train"].astype(np.float32),
        "X_val":         data["X_val"].astype(np.float32),
        "X_test":        data["X_test"].astype(np.float32),
        "y_train":       y_train,
        "y_val":         data["y_val"].astype(np.float32),
        "y_test":        data["y_test"].astype(np.float32),
        "feature_names": [str(f) for f in data["feature_names"]],
        "n_features":    int(data["X_train"].shape[1]),
        "action_dim":    int(y_train.shape[1]),
        "action_norm_train": data.get(
            "action_norm_train",
            np.linalg.norm(y_train, axis=1).astype(np.float32),
        ),
    }


def resolve_topk_list(
    cfg: dict, selector_name: str, n_features: int,
) -> list[int]:
    """For `full`, sweep once at k=n_features; otherwise use cfg.student.topk_list."""
    if selector_name == "full":
        return [n_features]
    return list(cfg["student"]["topk_list"])


# ─── Space + transition helpers ──────────────────────────────────────────────

def _make_spaces(
    input_dim: int, action_dim: int,
) -> tuple[gym.spaces.Box, gym.spaces.Box]:
    """Build gymnasium Box spaces matching the sliced obs and action dims."""
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(input_dim,), dtype=np.float32,
    )
    # D4RL Kitchen / pen actions live in [-1, 1]; matches the post-hoc clip
    # used by eval/eval_online_continuous.py.
    act_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32,
    )
    return obs_space, act_space


def _make_transitions(X: np.ndarray, y: np.ndarray) -> Transitions:
    """Wrap (obs, acts) arrays as imitation.data.types.Transitions."""
    # bc.BC's DataLoader collate fn indexes next_obs/dones unconditionally,
    # so we populate them with self-refs / zeros (rather than using
    # TransitionsMinimal which would KeyError at batch time).
    N = len(X)
    X32 = X.astype(np.float32)
    return Transitions(
        obs=X32,
        acts=y.astype(np.float32),
        infos=np.array([{} for _ in range(N)], dtype=object),
        next_obs=X32,
        dones=np.zeros(N, dtype=bool),
    )


# ─── Eval + checkpoint persistence ───────────────────────────────────────────

@torch.no_grad()
def evaluate_continuous(
    policy: ActorCriticPolicy,
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: list[int],
    batch_size: int = 4096,
) -> dict[str, float]:
    """Compute MSE / MAE on a data split via deterministic policy.predict()."""
    policy.set_training_mode(False)
    X_sub = X[:, feature_idx].astype(np.float32)
    preds_list: list[np.ndarray] = []
    for start in range(0, len(X_sub), batch_size):
        xb = X_sub[start:start + batch_size]
        actions, _ = policy.predict(xb, deterministic=True)
        preds_list.append(np.asarray(actions, dtype=np.float32))
    preds = np.concatenate(preds_list, axis=0)
    err = preds - y.astype(np.float32)
    return {
        "mse":         float((err ** 2).mean()),
        "mae":         float(np.abs(err).mean()),
        "per_dim_mse": (err ** 2).mean(axis=0).tolist(),
    }


def save_imitation_ckpt(
    policy: ActorCriticPolicy,
    out_path: Path,
    feature_idx: list[int],
    feature_names: list[str],
    input_dim: int,
    action_dim: int,
    hidden_dims: list[int],
) -> None:
    """Save SB3 ActorCriticPolicy + project metadata into one torch file."""
    sb3_ctor = policy._get_constructor_parameters()
    payload = {
        # SB3 reconstruction keys (BasePolicy.save convention):
        "state_dict": policy.state_dict(),
        "data":       sb3_ctor,
        # Project-standard metadata:
        "model_state_dict": policy.state_dict(),
        "feature_idx":      list(feature_idx),
        "feature_names":    list(feature_names),
        "input_dim":        int(input_dim),
        "action_dim":       int(action_dim),
        "hidden_dims":      list(hidden_dims),
        "use_gaussian":     True,
        "model_class":      "ImitationBCPolicy",
        "policy_class":     "ActorCriticPolicy",
    }
    torch.save(payload, str(out_path))


# ─── Top-k sweep — thin wrapper called by per-selector scripts ──────────────

def train_bc_students(
    cfg: dict,
    seed: int,
    device: torch.device,
    dataset: dict[str, Any],
    feature_idx_fn: Callable[[int], list[int]],
    topk_list: list[int],
    selector_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """For each k in `topk_list`, call `train_bc_continuous_student(...)`
    with `feature_idx = feature_idx_fn(k)` and persist the best model.

    The training method is fixed (imitation.bc.BC + SB3 ActorCriticPolicy,
    cfg.student.{hidden_dims, lr, batch_size, epochs, max_train_samples}).
    Across selectors, only `feature_idx_fn` varies — so this loop is the
    apples-to-apples comparison surface.

    Returns:
        DataFrame of one summary row per k (also written to
        `{output_dir}/summary.csv`). Per-k artifacts at
        `{output_dir}/{k_label}/{model.pt, metrics.json}`.
    """
    n_features    = dataset["n_features"]
    action_dim    = dataset["action_dim"]
    feature_names = dataset["feature_names"]
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val,   y_val   = dataset["X_val"],   dataset["y_val"]
    X_test,  y_test  = dataset["X_test"],  dataset["y_test"]

    s_cfg       = cfg["student"]
    hidden_dims = list(s_cfg["hidden_dims"])
    epochs      = int(s_cfg["epochs"])
    lr          = float(s_cfg["lr"])
    batch_size  = int(s_cfg["batch_size"])
    max_train   = s_cfg.get("max_train_samples", None)

    print(f"[train_bc_continuous] selector={selector_name}, device={device}, "
          f"n_features={n_features}, action_dim={action_dim}, "
          f"trainer=imitation.bc.BC, hidden_dims={hidden_dims}")

    summary_rows: list[dict] = []
    for k in topk_list:
        k_eff = n_features if selector_name == "full" else k

        feature_idx = list(feature_idx_fn(k_eff))
        selected_names = [feature_names[i] for i in feature_idx]

        k_label = "full" if selector_name == "full" else f"k{k}"
        k_dir   = ensure_dir(Path(output_dir) / k_label)

        print(f"\n[train_bc_continuous] selector={selector_name}, k={k_eff}, "
              f"features={selected_names[:6]}"
              f"{'...' if len(selected_names) > 6 else ''}")

        # ▼ The single point where training happens — same for every selector.
        policy, tr_metrics = train_bc_continuous_student(
            X_train, y_train,
            X_val,   y_val,
            feature_idx=feature_idx,
            hidden_dims=hidden_dims,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_train_samples=max_train,
        )

        save_imitation_ckpt(
            policy,
            out_path=k_dir / "model.pt",
            feature_idx=feature_idx,
            feature_names=selected_names,
            input_dim=len(feature_idx),
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

        val_metrics  = evaluate_continuous(policy, X_val,  y_val,  feature_idx)
        test_metrics = evaluate_continuous(policy, X_test, y_test, feature_idx)

        save_json({
            "selector":      selector_name,
            "k":             k_eff,
            "feature_idx":   feature_idx,
            "feature_names": selected_names,
            "val_mse":       val_metrics["mse"],
            "val_mae":       val_metrics["mae"],
            "test_mse":      test_metrics["mse"],
            "test_mae":      test_metrics["mae"],
            "best_val_mse":  tr_metrics["best_val_mse"],
        }, str(k_dir / "metrics.json"))

        summary_rows.append({
            "selector":      selector_name,
            "k":             k_eff,
            "feature_names": "|".join(selected_names),
            **{f"val_{kk}":  vv for kk, vv in val_metrics.items()  if kk != "per_dim_mse"},
            **{f"test_{kk}": vv for kk, vv in test_metrics.items() if kk != "per_dim_mse"},
        })

        print(f"  val_mse={val_metrics['mse']:.6f}, "
              f"test_mse={test_metrics['mse']:.6f}, "
              f"test_mae={test_metrics['mae']:.6f}")

        if selector_name == "full":
            break

    df = pd.DataFrame(summary_rows)
    save_csv(df, str(Path(output_dir) / "summary.csv"))
    print(f"\n[train_bc_continuous] Summary saved to {Path(output_dir) / 'summary.csv'}")
    return df
