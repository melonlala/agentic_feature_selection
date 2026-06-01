"""Train discrete-action (Taxi) behavioral cloning students with feature selectors.

Discrete BC trainer for the Taxi-v3 pipeline (logits + cross-entropy, accuracy /
macro-F1). For the continuous bc/irl/pc imitation framework see
student/train_student.py.

For each k in topk_list, trains one student on the selected top-k features
and saves a checkpoint, metrics, and the feature indices used.

Selectors:
  - shap:   top-k features ranked by SHAP (requires --ranking_path)
  - random: uniformly random k features
  - oracle: first k oracle features [row, col, passenger_loc, destination]
  - mi:     top-k by mutual information with teacher action labels
  - full:   all features (k is ignored)

Usage:
    python student/train_student_discrete.py \\
        --config configs/taxi_noise8.yaml \\
        --seed 0 \\
        --dataset_path outputs/datasets/taxi_noise8/seed0/dataset.npz \\
        --ranking_path outputs/rankings/taxi_noise8/seed0/ranking.csv \\
        --selector shap \\
        --output_dir outputs/students/taxi_noise8/seed0/shap
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.bc_model import BCPolicy
from student.distill_losses import combined_loss, cross_entropy_loss
from utils.config import resolve_config, save_resolved_config
from utils.feature_utils import dispatch_selector
from utils.io import ensure_dir, load_csv, load_npz, save_json
from utils.metrics import accuracy, macro_f1
from utils.seed import set_global_seed


def train_one_student(
    X_train: np.ndarray,
    y_train: np.ndarray,
    p_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_idx: list[int],
    cfg: dict,
    seed: int,
    device: str = "cpu",
) -> tuple[BCPolicy, dict]:
    """Train a single BCPolicy on a feature subset.

    Args:
        X_train: Full training observations [N_tr, D].
        y_train: Training action labels [N_tr].
        p_train: Teacher softmax probs [N_tr, 6].
        X_val: Full validation observations [N_val, D].
        y_val: Validation action labels [N_val].
        feature_idx: Indices of selected features.
        cfg: Resolved config dict.
        seed: Random seed.
        device: Torch device.

    Returns:
        Tuple of (best_model, metrics_dict).
    """
    set_global_seed(seed)
    s_cfg = cfg["student"]

    # Slice to selected features
    X_tr = X_train[:, feature_idx]
    X_va = X_val[:, feature_idx]

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_train),
        torch.from_numpy(p_train),
    )
    va_ds = TensorDataset(
        torch.from_numpy(X_va),
        torch.from_numpy(y_val),
    )

    tr_loader = DataLoader(tr_ds, batch_size=s_cfg["batch_size"], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=s_cfg["batch_size"] * 4)

    model = BCPolicy(
        input_dim=len(feature_idx),
        n_actions=6,
        hidden_dims=s_cfg["hidden_dims"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=s_cfg["lr"])

    use_soft = s_cfg.get("use_soft_distill", False)
    alpha = s_cfg.get("distill_alpha", 0.5)

    best_val_acc = -1.0
    best_state = None
    train_losses, val_accs = [], []

    for epoch in range(s_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, p_b in tr_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            p_b = p_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            if use_soft:
                loss, _ = combined_loss(logits, y_b, p_b, alpha=alpha)
            else:
                loss = cross_entropy_loss(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)

        epoch_loss /= len(X_tr)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in va_loader:
                logits = model(X_b.to(device))
                preds.append(logits.argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(preds)
        val_acc = accuracy(y_val, preds)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    metrics = {
        "best_val_acc": best_val_acc,
        "final_train_loss": train_losses[-1],
        "train_losses": train_losses,
        "val_accs": val_accs,
    }
    return model, metrics


def evaluate_split(model: BCPolicy, X: np.ndarray, y: np.ndarray, feature_idx: list[int]) -> dict:
    """Evaluate model on a data split.

    Args:
        model: Trained BCPolicy.
        X: Observations [N, D].
        y: Labels [N].
        feature_idx: Feature indices to slice.

    Returns:
        Dict with accuracy and macro_f1.
    """
    X_sub = torch.from_numpy(X[:, feature_idx])
    with torch.no_grad():
        logits = model(X_sub)
    preds = logits.argmax(dim=-1).numpy()
    return {
        "accuracy": accuracy(y, preds),
        "macro_f1": macro_f1(y, preds),
    }


def run(args: argparse.Namespace) -> None:
    """Main training routine."""
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    # Load dataset
    data = load_npz(args.dataset_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    p_train = data["p_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.int64)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)
    feature_names = [str(f) for f in data["feature_names"]]

    n_features = X_train.shape[1]

    # Load ranking if needed
    ranking_df = None
    if args.selector == "shap":
        if not args.ranking_path:
            raise ValueError("--ranking_path is required for selector=shap")
        ranking_df = load_csv(args.ranking_path)

    s_cfg = cfg["student"]
    topk_list = s_cfg["topk_list"] if args.selector != "full" else [n_features]

    summary_rows = []

    for k in topk_list:
        k_eff = n_features if args.selector == "full" else k

        feature_idx = dispatch_selector(
            selector=args.selector,
            k=k_eff,
            n_features=n_features,
            X_train=X_train,
            y_train=y_train,
            ranking_df=ranking_df,
            seed=args.seed,
        )
        selected_names = [feature_names[i] for i in feature_idx]

        k_label = "full" if args.selector == "full" else f"k{k}"
        k_dir = ensure_dir(out_dir / k_label)

        print(f"\n[train_student] selector={args.selector}, k={k_eff}, "
              f"features={selected_names}")

        model, tr_metrics = train_one_student(
            X_train, y_train, p_train,
            X_val, y_val,
            feature_idx, cfg, seed=args.seed,
        )

        # Save checkpoint
        ckpt_path = str(k_dir / "model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "feature_idx": feature_idx,
            "feature_names": selected_names,
            "input_dim": len(feature_idx),
            "n_actions": 6,
            "hidden_dims": s_cfg["hidden_dims"],
        }, ckpt_path)

        # Evaluate on all splits
        val_metrics = evaluate_split(model, X_val, y_val, feature_idx)
        test_metrics = evaluate_split(model, X_test, y_test, feature_idx)

        metrics = {
            "selector": args.selector,
            "k": k_eff,
            "feature_idx": feature_idx,
            "feature_names": selected_names,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "best_val_acc": tr_metrics["best_val_acc"],
        }
        save_json(metrics, str(k_dir / "metrics.json"))

        row = {
            "selector": args.selector,
            "k": k_eff,
            "feature_names": "|".join(selected_names),
            **{f"val_{kk}": vv for kk, vv in val_metrics.items()},
            **{f"test_{kk}": vv for kk, vv in test_metrics.items()},
        }
        summary_rows.append(row)

        print(f"  val_acc={val_metrics['accuracy']:.4f}, "
              f"test_acc={test_metrics['accuracy']:.4f}, "
              f"test_f1={test_metrics['macro_f1']:.4f}")

        if args.selector == "full":
            break  # only one run for full

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    from utils.io import save_csv
    save_csv(df, str(out_dir / "summary.csv"))
    print(f"\n[train_student] Summary saved to {out_dir / 'summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC students with feature selection.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--ranking_path", default=None, help="Path to ranking.csv (for shap selector).")
    parser.add_argument(
        "--selector",
        default="full",
        choices=["shap", "random", "oracle", "mi", "full"],
        help="Feature selector. Defaults to 'full' (all features).",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
