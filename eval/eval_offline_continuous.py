"""Offline evaluation for continuous-action BC students.

Evaluates one or more student checkpoints on the test split using MSE, MAE, and
the cosine similarity between predicted and expert actions.  Results are saved
as CSV + JSON in the output directory.

Usage:
    python eval/eval_offline_continuous.py \\
        --config configs/kitchen_complete.yaml \\
        --dataset_path outputs/datasets/kitchen_complete/seed0/dataset.npz \\
        --student_dir outputs/students/kitchen_complete/seed0/shap \\
        --output_dir  outputs/eval/offline/kitchen_complete/seed0/shap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.bc_continuous_model import (
    BCContinuousPolicy,
    BCContinuousPolicyFromLatent,
    BCGaussianPolicy,
    BCGaussianPolicyFromLatent,
)
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_npz, save_csv, save_json


def load_student(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, list[int], bool]:
    """Load a continuous student checkpoint.

    Handles both state-space students (BC{Continuous,Gaussian}Policy) and
    latent-space students (BC{Continuous,Gaussian}PolicyFromLatent). For the
    latter, the frozen layer-1 is reconstructed from the source full student
    referenced by `ckpt['source_student_path']`.

    Returns:
        Tuple of (model, feature_idx, use_gaussian).
    """
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
        # Late import to avoid circular dep when eval is invoked without the
        # explain package installed (it's a sibling dir, always present in repo).
        from ranker.latent_extract import build_frozen_layer1
        source_path = ckpt["source_student_path"]
        latent_layer = ckpt.get("latent_layer", "pre_relu")
        latent_idx   = list(ckpt["latent_idx"])
        full_ckpt    = torch.load(source_path, map_location=device, weights_only=False)
        frozen_layer1 = build_frozen_layer1(full_ckpt, latent_layer, device)
        if model_class == "BCGaussianPolicyFromLatent":
            model: torch.nn.Module = BCGaussianPolicyFromLatent(
                frozen_layer1=frozen_layer1,
                latent_indices=latent_idx,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
            )
        else:
            model = BCContinuousPolicyFromLatent(
                frozen_layer1=frozen_layer1,
                latent_indices=latent_idx,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
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


def evaluate_one(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_idx: list[int],
    use_gaussian: bool,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """Compute MSE, MAE, and cosine similarity on the test split.

    Args:
        model:       Trained continuous policy.
        X_test:      Full observations [N, D].
        y_test:      Expert actions [N, action_dim].
        feature_idx: Feature indices to slice.
        use_gaussian: Whether model is BCGaussianPolicy.
        device:      Torch device.
        batch_size:  Inference batch size.

    Returns:
        Dict with mse, mae, cosine_sim, per_dim_mse.
    """
    X_sub = torch.from_numpy(X_test[:, feature_idx]).float()
    y_t   = torch.from_numpy(y_test).float()

    preds = []
    loader = DataLoader(TensorDataset(X_sub), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            if use_gaussian:
                mean, _ = model(xb)
                preds.append(mean.cpu())
            else:
                preds.append(model(xb).cpu())
    preds = torch.cat(preds, dim=0)   # [N, action_dim]

    err          = preds - y_t
    mse          = float((err ** 2).mean())
    mae          = float(err.abs().mean())
    per_dim_mse  = (err ** 2).mean(dim=0).tolist()

    # Cosine similarity between predicted and expert action vectors
    cos = torch.nn.functional.cosine_similarity(preds, y_t, dim=1)
    cosine_sim = float(cos.mean())

    return {
        "mse":         mse,
        "mae":         mae,
        "cosine_sim":  cosine_sim,
        "per_dim_mse": per_dim_mse,
    }


def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data   = load_npz(args.dataset_path)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    feature_names = [str(f) for f in data.get("feature_names", [])]

    student_dir = Path(args.student_dir)
    ckpt_dirs   = sorted(student_dir.glob("k*")) + sorted(student_dir.glob("full"))
    if not ckpt_dirs:
        ckpt_dirs = [student_dir]

    rows = []
    for k_dir in ckpt_dirs:
        ckpt_path = k_dir / "model.pt"
        if not ckpt_path.exists():
            continue

        model, feature_idx, use_gaussian = load_student(str(ckpt_path), device)
        sel_names = [feature_names[i] for i in feature_idx] if feature_names else []

        metrics = evaluate_one(
            model, X_test, y_test, feature_idx, use_gaussian, device,
            batch_size=cfg["eval"]["offline_batch_size"],
        )
        k_label = k_dir.name
        row = {
            "k":           k_label,
            "n_features":  len(feature_idx),
            "feature_names": "|".join(sel_names),
            **{kk: vv for kk, vv in metrics.items() if kk != "per_dim_mse"},
        }
        rows.append(row)
        save_json(metrics, str(k_dir / "offline_eval_metrics.json"))
        print(f"[eval_offline_continuous] {k_label}: mse={metrics['mse']:.6f}, "
              f"mae={metrics['mae']:.6f}, cosine_sim={metrics['cosine_sim']:.4f}")

    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "offline_metrics.csv"))
    save_json(rows, str(out_dir / "offline_metrics.json"))
    print(f"[eval_offline_continuous] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline eval for continuous-action BC students."
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--student_dir",  required=True,
                   help="Directory containing k* subdirectories with model.pt checkpoints.")
    p.add_argument("--output_dir",   required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
