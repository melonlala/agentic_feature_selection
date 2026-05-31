"""Extract first-layer latent embeddings from a trained full-feature BC student.

Produces a latent_dataset.npz whose `X_*` keys hold the layer-1 activations of
the full student (instead of raw observations), so that downstream scripts
(`mci_rank.py`, `train_student_continuous.py --latent_mode`,
`eval_offline_continuous.py`) work on the latent representation with minimal
changes.

Latent layer choice:
    --latent_layer pre_relu  → output of net[:2]  (Linear → LayerNorm)   [default]
    --latent_layer post_relu → output of net[:3]  (Linear → LayerNorm → ReLU)

`pre_relu` is preferred for MCI: LayerNorm outputs are zero-mean / unit-variance
per sample (good RFF conditioning) and have no dead-unit degeneracy.

Output schema (.npz):
    X_train, X_val, X_test         latents,   [N, hidden_dims[0]]
    y_train, y_val, y_test         actions,   [N, action_dim] (copied through)
    action_norm_train              scalar action norm (copied if present)
    feature_names                  ["latent_000", ..., "latent_{H-1}"]
    raw_X_train, raw_X_val, raw_X_test   original raw observations (for student training)
    latent_layer, source_student_path    metadata (1-D arrays of dtype=str)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student.bc_continuous_model import BCContinuousPolicy, BCGaussianPolicy
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, load_npz, save_json, save_npz
from utils.seed import set_global_seed


# --- Module-level utility (also used by the student trainer) ---------------

def build_frozen_layer1(
    ckpt: dict,
    latent_layer: str,
    device: torch.device,
) -> torch.nn.Sequential:
    """Reconstruct the full student and return its first-layer block (frozen).

    Args:
        ckpt:         Loaded torch checkpoint dict for a full-feature student.
        latent_layer: "pre_relu" → net[:2]; "post_relu" → net[:3].
        device:       Torch device to put the layer on.

    Returns:
        A `nn.Sequential` containing the first Linear + LayerNorm (+ ReLU),
        with all parameters set to `requires_grad=False` and in eval mode.
    """
    model_class = ckpt.get("model_class", "BCContinuousPolicy")
    input_dim   = int(ckpt["input_dim"])
    action_dim  = int(ckpt["action_dim"])
    hidden_dims = list(ckpt["hidden_dims"])

    if model_class == "BCGaussianPolicy":
        full = BCGaussianPolicy(
            input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims,
        )
        full.load_state_dict(ckpt["model_state_dict"])
        net = full.mean_net
    elif model_class == "BCContinuousPolicy":
        full = BCContinuousPolicy(
            input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims,
        )
        full.load_state_dict(ckpt["model_state_dict"])
        net = full.net
    else:
        raise ValueError(
            f"Unsupported model_class {model_class!r} for latent extraction. "
            "Expected BCContinuousPolicy or BCGaussianPolicy (a full student)."
        )

    if latent_layer == "pre_relu":
        block = torch.nn.Sequential(net[0], net[1])
    elif latent_layer == "post_relu":
        block = torch.nn.Sequential(net[0], net[1], net[2])
    else:
        raise ValueError(
            f"Unknown latent_layer {latent_layer!r}; choose pre_relu or post_relu."
        )

    for p in block.parameters():
        p.requires_grad_(False)
    block.to(device).eval()
    return block


def _forward_latent(
    layer1: torch.nn.Sequential,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Forward `X` through `layer1` in batches; return latents as np.float32."""
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            zb = layer1(xb)
            out.append(zb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# --- CLI entry point -------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)

    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)
    save_resolved_config(cfg, str(out_path.parent / "resolved_config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full-student checkpoint.
    ckpt = torch.load(args.full_student_path, map_location=device, weights_only=False)
    feat_idx = list(ckpt.get("feature_idx", []))
    raw_D    = int(ckpt["input_dim"])

    # Sanity: the full student must have been trained on the full raw feature set.
    if feat_idx and feat_idx != list(range(raw_D)):
        raise ValueError(
            f"full_student_path checkpoint has feature_idx={feat_idx[:6]}... "
            f"of length {len(feat_idx)}, but expected list(range({raw_D})). "
            "Use a 'full'-selector student for latent extraction."
        )

    layer1 = build_frozen_layer1(ckpt, args.latent_layer, device)
    hidden_dim = ckpt["hidden_dims"][0]

    # Load source dataset.
    data = load_npz(args.dataset_path)
    X_train = data["X_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    X_test  = data["X_test"].astype(np.float32)
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    if X_train.shape[1] != raw_D:
        raise ValueError(
            f"Dataset feature dim {X_train.shape[1]} != student input_dim {raw_D}."
        )

    print(f"[latent_extract] device={device}, raw_D={raw_D}, hidden_dim={hidden_dim}, "
          f"latent_layer={args.latent_layer}")

    Z_train = _forward_latent(layer1, X_train, device)
    Z_val   = _forward_latent(layer1, X_val,   device)
    Z_test  = _forward_latent(layer1, X_test,  device)

    # Sanity stats on training latents.
    per_dim_std = Z_train.std(axis=0)
    alive       = int((per_dim_std > 1e-6).sum())
    print(f"[latent_extract] Z_train.shape={Z_train.shape}, "
          f"alive_dims(std>1e-6)={alive}/{hidden_dim}, "
          f"mean_std={float(per_dim_std.mean()):.4f}, has_nan={bool(np.isnan(Z_train).any())}")

    width = max(3, len(str(hidden_dim - 1)))
    latent_names = np.array(
        [f"latent_{i:0{width}d}" for i in range(hidden_dim)], dtype=object,
    )

    npz_payload = {
        # Treated as the "features" by mci_rank.py and train_student_continuous.py.
        "X_train":       Z_train,
        "X_val":         Z_val,
        "X_test":        Z_test,
        "y_train":       np.asarray(y_train),
        "y_val":         np.asarray(y_val),
        "y_test":        np.asarray(y_test),
        "feature_names": latent_names,
        # Raw observations preserved so the latent student can read them at train time.
        "raw_X_train":   X_train,
        "raw_X_val":     X_val,
        "raw_X_test":    X_test,
        # Metadata as 0-d / 1-d arrays for round-trip via np.savez.
        "latent_layer":          np.array(args.latent_layer),
        "source_student_path":   np.array(str(args.full_student_path)),
        "raw_feature_dim":       np.array(raw_D),
    }
    if "action_norm_train" in data:
        npz_payload["action_norm_train"] = np.asarray(data["action_norm_train"])

    save_npz(out_path, **npz_payload)

    meta = {
        "seed": args.seed,
        "config": args.config,
        "dataset_path": args.dataset_path,
        "full_student_path": str(args.full_student_path),
        "output_path": str(out_path),
        "latent_layer": args.latent_layer,
        "raw_feature_dim": raw_D,
        "latent_dim": int(hidden_dim),
        "n_train": int(Z_train.shape[0]),
        "n_val":   int(Z_val.shape[0]),
        "n_test":  int(Z_test.shape[0]),
        "alive_dims": alive,
    }
    save_json(meta, str(out_path.parent / "metadata.json"))

    print(f"[latent_extract] Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract first-layer latents from a full-feature BC student."
    )
    p.add_argument("--config",            required=True)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--dataset_path",      required=True,
                   help="Path to the source dataset.npz (raw features).")
    p.add_argument("--full_student_path", required=True,
                   help="Path to the trained full-feature student model.pt.")
    p.add_argument("--output_path",       required=True,
                   help="Destination .npz path for the latent dataset.")
    p.add_argument("--latent_layer",      default="pre_relu",
                   choices=["pre_relu", "post_relu"],
                   help="Which layer-1 output to use as the latent.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())