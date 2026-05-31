"""KernelSHAP feature ranking for a continuous predictor.

Input:
    - dataset.npz with X_train, X_val, feature_names
    - model_ckpt: checkpoint (model.pt) of a full-feature fully_trained model. 

Target:
    For each sample x, the predicted vector f(x) ∈ R^{d_y}.
    KernelSHAP attributes each feature's contribution to f. Local
    attributions form a [K, d_y, d_x] array. Global ranking aggregates
    over both samples and action dims:
        score[j] = mean_{k, y} |SHAP_{k,y,j}|


Output (in --output_dir):
    ranking.csv      — feature_index, feature_name, score, rank
    kernelshap_values.npz  — kernelshap_values [d_y, K, d_x]  for downstream plotting
    metadata.json — metadata about model_ckpt, dataset_path, and computation details

Usage:
    python explain/kernelshap.py \\
        --model_ckpt outputs/seals_ant/model.pt \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir outputs/rankings_shap/seals_ant/seed0
"""

import shap
import numpy as np
import argparse
from pathlib import Path
import json
import torch
import time

def run(args: argparse.Namespace) -> None:
    # prepare output directory and save config
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    

    # prepare dataset and model
    data = np.load(args.dataset_path)
    X = data["X_train"]
    y = data["y_train"]
    feature_names = [str(f) for f in data.get("feature_names", [])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_ckpt, map_location=device)

    # use Kernel SHAP to explain dataset predictions
    start_time = time.time()
    explainer = shap.KernelExplainer(model.predict_proba, X, link="logit")
    shap_values = explainer.shap_values(X)

    # aggregate SHAP values to get global feature importance scores
    # (average over samples and action dimensions)
    shap_values = np.array(shap_values)  # [d_y, K, d_x]
    scores = np.mean(np.abs(shap_values), axis=(0, 1))  # save ranking
    ranking = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    with open(out_dir / "ranking.csv", "w") as f:
        f.write("feature_index,feature_name,score,rank\n")
        for rank, (idx, score) in enumerate(ranking):
            name = feature_names[idx] if idx < len(feature_names) else ""
            f.write(f"{idx},{name},{score:.6f},{rank+1}\n")
    end_time = time.time()

    # save SHAP values for downstream plotting
    np.savez(out_dir / "kernelshap_values.npz", shap_values=shap_values)    

    # save metadata about computation time
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "model_ckpt": str(args.model_ckpt),
            "dataset_path": str(args.dataset_path),
            "ranking_method": "kernelshap",
            "ranking_results_file": str(out_dir / "ranking.csv"),
            "raw shap_values_file": str(out_dir / "kernelshap_values.npz"),
            "computation_time_sec": end_time - start_time,
        }, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KernelSHAP feature ranking")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to model checkpoint (model.pt)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset.npz with X_train, y_train, feature_names")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs (ranking.csv, kernelshap_values.npz)")
    args = parser.parse_args()
    run(args)
