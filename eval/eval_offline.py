"""Offline evaluation of student checkpoints on the test split.

Iterates over all k-subdirectories in student_dir, loads each checkpoint,
and evaluates on the test set from the provided dataset.

Metrics computed:
  - accuracy
  - macro F1
  - optional KL divergence to teacher action probabilities

Usage:
    python eval/eval_offline.py \\
        --config configs/taxi_noise8.yaml \\
        --dataset_path outputs/datasets/taxi_noise8/seed0/dataset.npz \\
        --student_dir outputs/students/taxi_noise8/seed0/shap \\
        --output_dir outputs/eval/offline/taxi_noise8/seed0/shap
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.bc_model import BCPolicy
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_json, load_npz, save_csv, save_json
from utils.metrics import accuracy, macro_f1
from utils.seed import set_global_seed


def load_student(ckpt_path: str) -> tuple[BCPolicy, list[int], list[str]]:
    """Load a student checkpoint.

    Args:
        ckpt_path: Path to model.pt saved by train_student.py.

    Returns:
        Tuple of (model, feature_idx, feature_names).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BCPolicy(
        input_dim=ckpt["input_dim"],
        n_actions=ckpt["n_actions"],
        hidden_dims=ckpt["hidden_dims"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["feature_idx"], ckpt["feature_names"]


def eval_checkpoint(
    model: BCPolicy,
    X_test: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray | None,
    feature_idx: list[int],
) -> dict:
    """Evaluate a single checkpoint on the test split.

    Args:
        model: Trained BCPolicy.
        X_test: Test observations [N, D].
        y_test: Test labels [N].
        p_test: Teacher softmax probs [N, 6] (optional, for KL).
        feature_idx: Feature indices to slice.

    Returns:
        Dict with accuracy, macro_f1, and optionally kl_to_teacher.
    """
    X_sub = torch.from_numpy(X_test[:, feature_idx].astype(np.float32))
    with torch.no_grad():
        logits = model(X_sub)

    preds = logits.argmax(dim=-1).numpy()
    acc = accuracy(y_test, preds)
    f1 = macro_f1(y_test, preds)

    result = {"accuracy": acc, "macro_f1": f1}

    if p_test is not None:
        # KL(teacher || student): E[P * log(P/Q)]
        student_log_probs = F.log_softmax(logits, dim=-1)
        teacher_probs = torch.from_numpy(p_test.astype(np.float32)).clamp(min=1e-8)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)
        result["kl_to_teacher"] = float(kl.item())

    return result


def run(args: argparse.Namespace) -> None:
    """Main offline evaluation routine."""
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    # Load dataset
    data = load_npz(args.dataset_path)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)
    p_test  = data["p_test"].astype(np.float32) if "p_test" in data else None

    student_dir = Path(args.student_dir)
    rows = []

    # Find all k-subdirectories containing model.pt
    ckpt_dirs = sorted([d for d in student_dir.iterdir() if d.is_dir() and (d / "model.pt").exists()])
    if not ckpt_dirs:
        raise FileNotFoundError(f"No model.pt found under {student_dir}")

    for k_dir in ckpt_dirs:
        ckpt_path = str(k_dir / "model.pt")
        model, feature_idx, feature_names = load_student(ckpt_path)

        metrics = eval_checkpoint(model, X_test, y_test, p_test, feature_idx)

        # Load k from stored metrics if available
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
        print(f"  {k_dir.name}: acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}"
              + (f", kl={metrics.get('kl_to_teacher', float('nan')):.4f}" if "kl_to_teacher" in metrics else ""))

    df = pd.DataFrame(rows)
    save_csv(df, str(out_dir / "offline_metrics.csv"))
    save_json(rows, str(out_dir / "offline_metrics.json"))
    print(f"\n[eval_offline] Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation of student checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
