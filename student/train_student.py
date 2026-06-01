"""Train a feature-subset student for the bc / irl / pc tasks.

Unified student trainer that mirrors the three evaluator branches of the MCI
rankers (ranker/mci_rank_nn.py, ranker/mci_rank_kernel.py). Given a ranking.csv
produced by *any* ranker (mci_nn / mci_kernel / sage / kernelshap / ...), it
selects the top-k features and trains the downstream student for the chosen
task, then evaluates that student on the held-out **test** split:

    bc  : behavioral-cloning policy (imitation.bc.BC + SB3 ActorCriticPolicy)
          predicting expert actions from the feature subset.
          test metrics — action MSE / MAE / cosine-sim / R^2.

    irl : reward regressor (MLP) recovering the ground-truth per-transition
          reward from the feature subset (the supervised / regression analog of
          IRL reward recovery, matching mci_rank_kernel's irl branch).
          test metrics — reward MSE / MAE / R^2.

    pc  : preference-comparison reward model (MLP) trained by Bradley-Terry
          cross-entropy over contiguous trajectory-fragment pairs, labelled by
          ground-truth summed reward.
          test metric — preference-prediction accuracy on held-out fragment pairs.

Feature selection: rank-1 = most important; the top-k feature indices are read
from ranking.csv (`feature_index` column, with a `feature_name` fallback). With
no --ranking_path the student uses the full feature set (upper-bound baseline).

The headline library call returns the trained student model *and* its test
results:

    model, test_results = train_and_eval_student(
        task, data, feature_idx, cfg, seed, device, args=...)

Output files (per k, under --output_dir/{k_label}/):
    model.pt        — student checkpoint (+ feature_idx / feature_names / arch)
    metrics.json    — selected features + val/test metrics
and one summary.csv across all k at the run root.

Usage:
    python student/train_student.py \\
        --config       configs/seals_ant.yaml \\
        --task         pc \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --ranking_path outputs/rankings_mci_kernel/seals_ant/seed0/ranking.csv \\
        --output_dir   outputs/students_pc/seals_ant/seed0/mci_kernel \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.mci_rank_kernel import (
    _contiguous_episode_prefix,
    build_fragment_pairs,
)
from student.train_bc_continuous import (
    evaluate_continuous,
    save_imitation_ckpt,
    setup_run,
    train_bc_continuous_student,
)
from utils.io import ensure_dir, load_csv, load_npz, save_csv, save_json
from utils.seed import set_global_seed


# ─── Dataset loading ─────────────────────────────────────────────────────────

def _split(data: dict, base: str, split: str) -> np.ndarray | None:
    """Return data[f'{base}_{split}'] (e.g. rewards_test) as float, or None."""
    key = f"{base}_{split}"
    if key not in data:
        return None
    return np.asarray(data[key])


def load_task_dataset(dataset_path: str) -> dict[str, Any]:
    """Load the train/val/test arrays + (optional) rewards/dones for all tasks.

    bc needs X/y; irl needs X + rewards; pc needs X + rewards + dones. Reward and
    done arrays are returned as None when absent so the bc path stays usable on
    action-only datasets.
    """
    data = load_npz(dataset_path)
    y_train = data["y_train"].astype(np.float32)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    def _y(split: str) -> np.ndarray:
        y = np.asarray(data[f"y_{split}"], dtype=np.float32)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    out: dict[str, Any] = {
        "X_train": data["X_train"].astype(np.float32),
        "X_val":   data["X_val"].astype(np.float32),
        "X_test":  data["X_test"].astype(np.float32),
        "y_train": _y("train"),
        "y_val":   _y("val"),
        "y_test":  _y("test"),
        "feature_names": [str(f) for f in data.get("feature_names", [])],
        "n_features": int(data["X_train"].shape[1]),
        "action_dim": int(_y("train").shape[1]),
    }
    if not out["feature_names"]:
        out["feature_names"] = [f"feature_{j}" for j in range(out["n_features"])]

    for split in ("train", "val", "test"):
        r = _split(data, "rewards", split)
        if r is None:
            r = _split(data, "rews", split)
        out[f"rewards_{split}"] = None if r is None else r.astype(np.float64).reshape(-1)
        d = _split(data, "dones", split)
        if d is None:
            d = _split(data, "terminals", split)
        out[f"dones_{split}"] = None if d is None else d.reshape(-1).astype(bool)
    return out


# ─── Feature selection (works across all ranker CSV schemas) ─────────────────

def resolve_feature_idx(
    ranking_df: pd.DataFrame | None,
    k: int,
    feature_names: list[str],
    n_features: int,
) -> list[int]:
    """Top-k feature indices from a ranking.csv, or all features if None.

    Prefers the `feature_index` column (mci_nn / mci_kernel / sage / kernelshap);
    falls back to mapping `feature_name` -> dataset index for rankers that emit
    only names (e.g. SHAP global_rank).
    """
    if ranking_df is None:
        return list(range(n_features))

    top = ranking_df.sort_values("rank").head(k)
    if "feature_index" in top.columns:
        return [int(i) for i in top["feature_index"]]
    if "feature_name" in top.columns:
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        return [name_to_idx[str(n)] for n in top["feature_name"]]
    raise ValueError(
        "ranking.csv must contain a 'feature_index' or 'feature_name' column."
    )


# ─── Metric helpers ──────────────────────────────────────────────────────────

def _explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2-style explained variance, summed/averaged over output dims."""
    var = np.var(y_true, axis=0)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    denom = var.sum()
    if denom <= 0:
        return 0.0
    return float(1.0 - mse.sum() / denom)


# ─── Reward MLP (irl / pc backbone) ──────────────────────────────────────────

class RewardMLP(nn.Module):
    """Scalar reward head r_theta(s) over a feature subset (irl / pc student)."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [N]


# ─── bc task ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _bc_test_metrics(policy, X_test, y_test, feature_idx) -> dict[str, float]:
    """Action MSE / MAE / cosine-sim / R^2 from policy.predict on the test split."""
    base = evaluate_continuous(policy, X_test, y_test, feature_idx)  # mse, mae
    X_sub = X_test[:, feature_idx].astype(np.float32)
    preds = np.concatenate(
        [policy.predict(X_sub[i:i + 4096], deterministic=True)[0]
         for i in range(0, len(X_sub), 4096)],
        axis=0,
    ).astype(np.float32)
    y = y_test.astype(np.float32)
    num = (preds * y).sum(axis=1)
    den = np.linalg.norm(preds, axis=1) * np.linalg.norm(y, axis=1) + 1e-8
    return {
        "mse":        base["mse"],
        "mae":        base["mae"],
        "cosine_sim": float(np.mean(num / den)),
        "r2":         _explained_variance(y, preds),
    }


def _train_bc(data, feature_idx, cfg, seed, device):
    s = cfg["student"]
    policy, _tr = train_bc_continuous_student(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        feature_idx=feature_idx,
        hidden_dims=list(s["hidden_dims"]),
        epochs=int(s["epochs"]),
        lr=float(s["lr"]),
        batch_size=int(s["batch_size"]),
        seed=seed,
        device=device,
        max_train_samples=s.get("max_train_samples"),
    )
    test = _bc_test_metrics(policy, data["X_test"], data["y_test"], feature_idx)
    val  = evaluate_continuous(policy, data["X_val"], data["y_val"], feature_idx)
    test["val_mse"] = val["mse"]
    return policy, test


# ─── irl task ────────────────────────────────────────────────────────────────

def _train_reward_regressor(Xtr, rtr, Xva, rva, cfg, seed, device):
    """Fit RewardMLP to regress ground-truth reward; restore best-val-MSE state."""
    set_global_seed(seed)
    s = cfg["student"]
    model = RewardMLP(Xtr.shape[1], list(s["hidden_dims"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(s["lr"]))

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    rtr_t = torch.tensor(rtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    rva_t = torch.tensor(rva, dtype=torch.float32, device=device)

    bs = int(s["batch_size"])
    n = len(Xtr_t)
    best_val, best_state = float("inf"), None
    rng = np.random.default_rng(seed)
    for _ in range(int(s["epochs"])):
        model.train()
        perm = torch.tensor(rng.permutation(n), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), rtr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(Xva_t), rva_t).item())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def _train_irl(data, feature_idx, cfg, seed, device):
    if data["rewards_train"] is None:
        raise ValueError("irl task needs reward arrays (rewards_train/...) in the dataset.")
    s = cfg["student"]
    cap = s.get("max_train_samples")
    Xtr = data["X_train"][:, feature_idx]
    rtr = data["rewards_train"]
    if cap and len(Xtr) > cap:
        idx = np.random.default_rng(seed).choice(len(Xtr), size=cap, replace=False)
        Xtr, rtr = Xtr[idx], rtr[idx]

    model, best_val = _train_reward_regressor(
        Xtr, rtr,
        data["X_val"][:, feature_idx], data["rewards_val"],
        cfg, seed, device,
    )
    with torch.no_grad():
        Xte = torch.tensor(data["X_test"][:, feature_idx], dtype=torch.float32, device=device)
        pred = model(Xte).cpu().numpy()
    rte = data["rewards_test"]
    err = pred - rte
    test = {
        "reward_mse": float(np.mean(err ** 2)),
        "reward_mae": float(np.mean(np.abs(err))),
        "r2":         _explained_variance(rte.reshape(-1, 1), pred.reshape(-1, 1)),
        "val_reward_mse": best_val,
    }
    return model, test


# ─── pc task ─────────────────────────────────────────────────────────────────

def _fragment_pairs(rews, dones, fragment_length, num_pairs, seed, device):
    """Build fragment-pair index/label tensors on `device` (reuses kernel ranker)."""
    fa, fb, lab = build_fragment_pairs(
        rews, dones, fragment_length, num_pairs, np.random.default_rng(seed)
    )
    return (
        torch.tensor(fa, dtype=torch.long, device=device),
        torch.tensor(fb, dtype=torch.long, device=device),
        torch.tensor(lab, dtype=torch.float32, device=device),
    )


def _pref_accuracy(model, X_t, fa, fb, lab) -> float:
    with torch.no_grad():
        r = model(X_t)
        pred = r[fa].sum(1) > r[fb].sum(1)
        return float((pred.float() == lab).float().mean().item())


def _train_pc(data, feature_idx, cfg, seed, device, args):
    if data["rewards_train"] is None or data["dones_train"] is None:
        raise ValueError("pc task needs reward + done arrays in the dataset.")
    s = cfg["student"]

    # Keep a contiguous, whole-episode prefix so fragment slices stay valid.
    cap = s.get("max_train_samples")
    sl = _contiguous_episode_prefix(data["dones_train"], cap or 0)
    Xtr = data["X_train"][sl][:, feature_idx]
    rtr, dtr = data["rewards_train"][sl], data["dones_train"][sl]

    fa_tr, fb_tr, lab_tr = _fragment_pairs(
        rtr, dtr, args.fragment_length, args.num_pairs, seed, device)
    fa_va, fb_va, lab_va = _fragment_pairs(
        data["rewards_val"], data["dones_val"],
        args.fragment_length, args.num_eval_pairs, seed + 1, device)
    fa_te, fb_te, lab_te = _fragment_pairs(
        data["rewards_test"], data["dones_test"],
        args.fragment_length, args.num_eval_pairs, seed + 2, device)

    set_global_seed(seed)
    model = RewardMLP(len(feature_idx), list(s["hidden_dims"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(s["lr"]))
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(data["X_val"][:, feature_idx], dtype=torch.float32, device=device)
    Xte_t = torch.tensor(data["X_test"][:, feature_idx], dtype=torch.float32, device=device)

    best_acc, best_state = -1.0, None
    for _ in range(int(s["epochs"])):
        model.train()
        opt.zero_grad()
        r = model(Xtr_t)                       # [N]
        logits = r[fa_tr].sum(1) - r[fb_tr].sum(1)   # Bradley-Terry logit
        loss = nn.functional.binary_cross_entropy_with_logits(logits, lab_tr)
        loss.backward()
        opt.step()
        model.eval()
        acc = _pref_accuracy(model, Xva_t, fa_va, fb_va, lab_va)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test = {
        "pref_accuracy": _pref_accuracy(model, Xte_t, fa_te, fb_te, lab_te),
        "val_pref_accuracy": best_acc,
        "n_eval_pairs": int(len(lab_te)),
    }
    return model, test


# ─── Headline: train one student for a task + evaluate on test ───────────────

def train_and_eval_student(
    task: str,
    data: dict[str, Any],
    feature_idx: list[int],
    cfg: dict,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    """Train the student for `task` on `feature_idx` and return (model, test_results)."""
    if task == "bc":
        return _train_bc(data, feature_idx, cfg, seed, device)
    if task == "irl":
        return _train_irl(data, feature_idx, cfg, seed, device)
    if task == "pc":
        return _train_pc(data, feature_idx, cfg, seed, device, args)
    raise ValueError(f"Unknown task {task!r}; expected 'bc', 'irl' or 'pc'.")


# ─── Checkpoint persistence ──────────────────────────────────────────────────

def _save_ckpt(task, model, k_dir, feature_idx, feature_names, cfg, action_dim):
    """Persist the student. bc reuses the SB3 imitation checkpoint format."""
    out_path = k_dir / "model.pt"
    if task == "bc":
        save_imitation_ckpt(
            model, out_path=out_path,
            feature_idx=feature_idx, feature_names=feature_names,
            input_dim=len(feature_idx), action_dim=action_dim,
            hidden_dims=list(cfg["student"]["hidden_dims"]),
        )
    else:
        torch.save({
            "model_state_dict": model.state_dict(),
            "feature_idx":      list(feature_idx),
            "feature_names":    list(feature_names),
            "input_dim":        len(feature_idx),
            "hidden_dims":      list(cfg["student"]["hidden_dims"]),
            "task":             task,
            "model_class":      "RewardMLP",
        }, str(out_path))


# ─── CLI driver ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> pd.DataFrame:
    """Resolve features, train + evaluate a student per k, persist artifacts."""
    cfg, device, out_dir = setup_run(args.config, args.seed, args.output_dir)
    data = load_task_dataset(args.dataset_path)
    n_features    = data["n_features"]
    feature_names = data["feature_names"]
    action_dim    = data["action_dim"]

    ranking_df = load_csv(args.ranking_path) if args.ranking_path else None

    # k schedule: explicit --topk > full (no ranking) > cfg topk_list (ranking).
    if args.topk is not None:
        ks = [min(args.topk, n_features)]
    elif ranking_df is None:
        ks = [n_features]
    else:
        ks = list(cfg["student"]["topk_list"])

    print(f"[train_student] task={args.task}, ranker={'full' if ranking_df is None else args.ranking_path}, "
          f"n_features={n_features}, ks={ks}, device={device}")

    rows = []
    k_records = []
    t_start = time.time()
    for k in ks:
        feature_idx = resolve_feature_idx(ranking_df, k, feature_names, n_features)
        sel_names = [feature_names[i] for i in feature_idx]
        k_label = "full" if (ranking_df is None and k == n_features) else f"k{len(feature_idx)}"
        k_dir = ensure_dir(out_dir / k_label)

        print(f"\n[train_student] task={args.task}, k={len(feature_idx)}, "
              f"features={sel_names[:6]}{'...' if len(sel_names) > 6 else ''}")

        t_k = time.time()
        model, test = train_and_eval_student(
            args.task, data, feature_idx, cfg, args.seed, device, args)
        train_time_sec = time.time() - t_k

        _save_ckpt(args.task, model, k_dir, feature_idx, sel_names, cfg, action_dim)
        save_json({
            "task":           args.task,
            "k":              len(feature_idx),
            "feature_idx":    feature_idx,
            "feature_names":  sel_names,
            "test":           test,
            "train_time_sec": train_time_sec,
        }, str(k_dir / "metrics.json"))

        rows.append({
            "task": args.task, "k": len(feature_idx),
            "feature_names": "|".join(sel_names),
            **{f"test_{kk}": vv for kk, vv in test.items()},
            "train_time_sec": train_time_sec,
        })
        k_records.append({
            "k": len(feature_idx), "feature_idx": feature_idx,
            "test": test, "train_time_sec": train_time_sec,
        })
        print(f"  test: " + ", ".join(
            f"{kk}={vv:.4f}" for kk, vv in test.items() if isinstance(vv, (int, float)))
            + f" | train_time={train_time_sec:.1f}s")

    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "summary.csv"))

    total_train_time = sum(r["train_time_sec"] for r in k_records)
    save_json({
        "task":                  args.task,
        "dataset_path":          args.dataset_path,
        "ranking_path":          args.ranking_path,
        "selector":              "full" if ranking_df is None else "ranking",
        "seed":                  args.seed,
        "n_features":            n_features,
        "ks":                    [r["k"] for r in k_records],
        "k_records":             k_records,
        "total_train_time_sec":  total_train_time,
        "elapsed_sec":           time.time() - t_start,
    }, str(out_dir / "metadata.json"))

    print(f"\n[train_student] Summary saved to {out_dir / 'summary.csv'}; "
          f"metadata to {out_dir / 'metadata.json'} "
          f"(total train_time={total_train_time:.1f}s).")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a feature-subset student for the bc/irl/pc tasks "
        "from a ranker's ranking.csv, and evaluate it on the test split."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", required=True)
    p.add_argument(
        "--task", required=True, choices=["bc", "irl", "pc"],
        help="Which student to train: bc (action cloning), irl (reward "
        "regression), or pc (preference reward model).",
    )
    p.add_argument(
        "--ranking_path", default=None,
        help="ranking.csv from any ranker (top-k by rank). Omit to use all features.",
    )
    p.add_argument(
        "--topk", type=int, default=None,
        help="Train a single student at this k. Omit to sweep cfg.student.topk_list "
        "(or train on the full feature set when --ranking_path is absent).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--fragment_length", type=int, default=50,
        help="pc-only: length of each contiguous trajectory fragment.",
    )
    p.add_argument(
        "--num_pairs", type=int, default=200,
        help="pc-only: number of training fragment pairs.",
    )
    p.add_argument(
        "--num_eval_pairs", type=int, default=1000,
        help="pc-only: number of val/test fragment pairs scored for accuracy.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
