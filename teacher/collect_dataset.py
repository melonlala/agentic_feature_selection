"""Collect a teacher-labeled dataset from the noisy Taxi environment.

For each environment step, we record:
  - obs_full:          float32 observation vector (structured + noise features)
  - action_teacher:    int64 greedy action chosen by the teacher
  - q_values:          float32 Q-values for all actions [N, 6]
  - action_probs:      float32 softmax probabilities [N, 6]
  - chosen_action_prob: float32 probability of the chosen action [N]
  - done:              bool — True at episode boundary

The dataset is split into train / val / test and saved as compressed .npz.

Data schema (keys in dataset.npz):
  X_train, X_val, X_test          — float32 [N, obs_dim]
  y_train, y_val, y_test          — int64   [N]
  q_train, q_val, q_test          — float32 [N, 6]
  p_train, p_val, p_test          — float32 [N, 6]
  chosen_prob_train, ..._val, ..._test — float32 [N]
  feature_names                   — str array [obs_dim]

Usage:
    python teacher/collect_dataset.py \\
        --config configs/taxi_noise8.yaml \\
        --seed 0 \\
        --teacher_ckpt outputs/teachers/taxi_noise8/seed0/model.zip \\
        --output_dir outputs/datasets/taxi_noise8/seed0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from envs.noisy_taxi_wrapper import NoisyTaxiWrapper
from teacher.teacher_policy import TeacherPolicy
from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_json, save_npz
from utils.seed import set_global_seed


def make_env(cfg: dict, seed: int) -> NoisyTaxiWrapper:
    env_cfg = cfg["env"]
    base_env = gym.make(env_cfg["env_id"])
    return NoisyTaxiWrapper(
        base_env,
        noise_dim=env_cfg["noise_dim"],
        noise_type=env_cfg["noise_type"],
        categorical_noise_cardinality=env_cfg.get("categorical_noise_cardinality", 5),
        noise_params=env_cfg.get("noise_params"),
        seed=seed,
    )


def collect(args: argparse.Namespace) -> None:
    """Main collection routine."""
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    ds_cfg = cfg["dataset"]
    n_steps = ds_cfg["n_collect_steps"]
    train_size = ds_cfg["train_size"]
    val_size = ds_cfg["val_size"]
    test_size = ds_cfg["test_size"]
    deduplicate = ds_cfg.get("deduplicate", False)

    env = make_env(cfg, seed=args.seed)
    teacher = TeacherPolicy(args.teacher_ckpt, device="cpu")

    feature_names = env.all_feature_names
    obs_dim = env.observation_dim

    # Detect how to feed observations to the teacher.
    #
    # Case A — Discrete teacher (trained on native Taxi-v3):
    #   teacher._is_discrete == True. TeacherPolicy.predict_q accepts [N, 4]
    #   structured obs and internally converts to state integers. Always pass
    #   the first 4 (clean) features from the noisy env obs.
    #
    # Case B — Box teacher trained on clean 4-dim obs (noise_dim == 0):
    #   teacher obs shape is (4,) < env obs_dim. Pass first 4 features.
    #
    # Case C — Box teacher trained on noisy obs matching env obs_dim:
    #   Pass the full observation.
    import gymnasium.spaces as gym_spaces
    teacher_obs_space = teacher.model.observation_space
    if teacher._is_discrete:
        # Native Taxi-v3 teacher: pass clean 4-dim structured obs
        teacher_obs_dim = 4
        use_clean_obs_for_teacher = True
        print(f"[collect_dataset] Discrete-obs teacher (native Taxi-v3). "
              f"Passing first {teacher_obs_dim} clean features to teacher; "
              f"recording full {obs_dim}-dim noisy obs for student.")
    else:
        teacher_obs_dim = int(teacher_obs_space.shape[0])
        use_clean_obs_for_teacher = teacher_obs_dim < obs_dim
        if use_clean_obs_for_teacher:
            print(f"[collect_dataset] Box teacher trained on {teacher_obs_dim}-dim clean obs; "
                  f"env provides {obs_dim}-dim noisy obs. "
                  f"Passing first {teacher_obs_dim} features to teacher.")

    # --- Collect transitions ---
    obs_list, action_list, q_list, prob_list, chosen_prob_list, done_list = [], [], [], [], [], []

    obs, _ = env.reset(seed=args.seed)
    print(f"[collect_dataset] Collecting {n_steps} steps...")

    for step in tqdm(range(n_steps)):
        # Extract clean obs for teacher if it was trained without noise
        obs_for_teacher = obs[:teacher_obs_dim] if use_clean_obs_for_teacher else obs

        # Teacher inference
        q_vals = teacher.predict_q(obs_for_teacher[np.newaxis])[0]        # [6]
        probs = teacher.predict_probs(obs_for_teacher[np.newaxis])[0]     # [6]
        action = int(q_vals.argmax())
        chosen_p = float(probs[action])

        obs_list.append(obs.copy())
        action_list.append(action)
        q_list.append(q_vals)
        prob_list.append(probs)
        chosen_prob_list.append(chosen_p)

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_list.append(done)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()

    X = np.array(obs_list, dtype=np.float32)         # [N, obs_dim]
    y = np.array(action_list, dtype=np.int64)         # [N]
    Q = np.array(q_list, dtype=np.float32)            # [N, 6]
    P = np.array(prob_list, dtype=np.float32)         # [N, 6]
    C = np.array(chosen_prob_list, dtype=np.float32)  # [N]
    D = np.array(done_list, dtype=bool)               # [N]

    N = len(X)
    print(f"[collect_dataset] Collected {N} transitions.")

    # --- Optional deduplication ---
    if deduplicate:
        _, unique_idx = np.unique(X, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        X, y, Q, P, C, D = X[unique_idx], y[unique_idx], Q[unique_idx], P[unique_idx], C[unique_idx], D[unique_idx]
        print(f"[collect_dataset] After dedup: {len(X)} transitions.")
        N = len(X)

    # --- Split ---
    total_needed = train_size + val_size + test_size
    if N < total_needed:
        raise RuntimeError(
            f"Collected {N} samples but need {total_needed}. "
            "Increase n_collect_steps or reduce split sizes."
        )

    # Shuffle with seed before splitting
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    X, y, Q, P, C, D = X[perm], y[perm], Q[perm], P[perm], C[perm], D[perm]

    i_tr = train_size
    i_va = train_size + val_size
    i_te = i_va + test_size

    splits = {
        "X_train": X[:i_tr],       "X_val": X[i_tr:i_va],       "X_test": X[i_va:i_te],
        "y_train": y[:i_tr],       "y_val": y[i_tr:i_va],       "y_test": y[i_va:i_te],
        "q_train": Q[:i_tr],       "q_val": Q[i_tr:i_va],       "q_test": Q[i_va:i_te],
        "p_train": P[:i_tr],       "p_val": P[i_tr:i_va],       "p_test": P[i_va:i_te],
        "chosen_prob_train": C[:i_tr],
        "chosen_prob_val":   C[i_tr:i_va],
        "chosen_prob_test":  C[i_va:i_te],
        "feature_names": np.array(feature_names),
    }

    dataset_path = str(out_dir / "dataset.npz")
    save_npz(dataset_path, **splits)
    print(f"[collect_dataset] Dataset saved to {dataset_path}")

    # --- Metadata ---
    metadata = {
        "seed": args.seed,
        "config": args.config,
        "teacher_ckpt": str(args.teacher_ckpt),
        "n_collect_steps": n_steps,
        "n_collected": int(N),
        "deduplicated": deduplicate,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "obs_dim": obs_dim,
        "n_actions": 6,
        "feature_names": feature_names,
        "dataset_path": dataset_path,
        "schema": {
            "X_{split}": "float32 [N, obs_dim] — full observation vector",
            "y_{split}": "int64  [N] — teacher greedy action",
            "q_{split}": "float32 [N, 6] — Q-values for all actions",
            "p_{split}": "float32 [N, 6] — softmax action probabilities",
            "chosen_prob_{split}": "float32 [N] — prob of chosen action",
            "feature_names": "str array — feature names in obs order",
        },
    }
    save_json(metadata, str(out_dir / "metadata.json"))
    print("[collect_dataset] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teacher-labeled dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher_ckpt", required=True, help="Path to model.zip")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
