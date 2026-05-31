"""SAGE feature ranking for a continuous-action BC student.

SAGE (Covert et al., NeurIPS 2020) — Shapley values of the cooperative game:
    v_f(S) = E[ℓ(f_∅, Y)] − E[ℓ(f_S(X_S), Y)]
where features outside S are marginalised by sampling from a background dataset
and ℓ is per-step squared error summed across action dims.

This is the continuous-action analogue of explain/sage_rank.py. Key changes:
  - Wrapped model returns a *vector* of action predictions, not a scalar
    probability. sage.MarginalImputer + sage.PermutationEstimator handle
    multi-output models correctly.
  - Loss = MSE summed across action dims (matches MCI's predictive-power
    definition).
  - Supports both project-native BC*Policy and SB3 ActorCriticPolicy
    (the latter is what train_student_continuous.py saves).

Output (in --output_dir):
    ranking.csv     — feature_index, feature_name, mean_mci (=|sage|), sage_value, rank
    sage_values.json
    metadata.json
    resolved_config.yaml

Usage:
    python explain/sage_rank_continuous.py \\
        --config configs/seals_ant.yaml \\
        --seed 0 \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --student_dir outputs/students/seals_ant/seed0/full \\
        --output_dir outputs/rankings_sage/seals_ant/seed0
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import torch
import core
from utils import rank_utils


def calculate_A(num_features):  # noqa:N802
    """Calculate A parameter's exact form."""
    p_coaccur = (
        np.sum(
            (np.arange(2, num_features) - 1)
            / (num_features - np.arange(2, num_features))
        )
    ) / (
        num_features
        * (num_features - 1)
        * np.sum(
            1
            / (np.arange(1, num_features) * (num_features - np.arange(1, num_features)))
        )
    )
    A = np.eye(num_features) * 0.5 + (1 - np.eye(num_features)) * p_coaccur
    return A


def estimate_constraints(imputer, X, Y, batch_size, loss_fn):
    """
    Estimate loss when no features are included and when all features are
    included. This is used to enforce constraints.
    """
    N = 0
    mean_loss = 0
    marginal_loss = 0
    num_features = imputer.num_groups
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size : (i + 1) * batch_size]
        y = Y[i * batch_size : (i + 1) * batch_size]
        N += len(x)

        # All features.
        pred = imputer(x, np.ones((len(x), num_features), dtype=bool))
        loss = loss_fn(pred, y)
        mean_loss += np.sum(loss - mean_loss) / N

        # No features.
        pred = imputer(x, np.zeros((len(x), num_features), dtype=bool))
        loss = loss_fn(pred, y)
        marginal_loss += np.sum(loss - marginal_loss) / N

    return -marginal_loss, -mean_loss


def calculate_result(A, b, total, b_sum_squares, n):
    """Calculate regression coefficients and uncertainty estimates."""
    num_features = A.shape[1]
    A_inv_one = np.linalg.solve(A, np.ones(num_features))
    A_inv_vec = np.linalg.solve(A, b)
    values = A_inv_vec - A_inv_one * (np.sum(A_inv_vec) - total) / np.sum(A_inv_one)

    # Calculate variance.
    try:
        b_sum_squares = 0.5 * (b_sum_squares + b_sum_squares.T)
        b_cov = b_sum_squares / (n**2)
        # TODO this fails in situations where model is invariant to features.
        cholesky = np.linalg.cholesky(b_cov)
        L = np.linalg.solve(A, cholesky) + np.matmul(
            np.outer(A_inv_one, A_inv_one), cholesky
        ) / np.sum(A_inv_one)
        beta_cov = np.matmul(L, L.T)
        var = np.diag(beta_cov)
        std = var**0.5
    except np.linalg.LinAlgError:
        # b_cov likely is not PSD due to insufficient samples.
        std = np.ones(num_features) * np.nan

    return values, std

import joblib
import numpy as np
from tqdm.auto import tqdm

from sage import core, utils


class PermutationEstimator:
    """
    Estimate SAGE values by unrolling permutations of feature indices.

    Args:
      imputer: model that accommodates held out features.
      loss: loss function ('mse', 'cross entropy', 'zero one').
      n_jobs: number of jobs for parallel processing.
      random_state: random seed, enables reproducibility.
    """

    def __init__(self, imputer, loss="cross entropy", n_jobs=1, random_state=None):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction="none")
        self.random_state = random_state
        self.n_jobs = joblib.effective_n_jobs(n_jobs)
        if n_jobs != 1:
            print(f"PermutationEstimator will use {self.n_jobs} jobs")

    def __call__(
        self,
        X,
        Y=None,
        batch_size=512,
        detect_convergence=True,
        thresh=0.025,
        n_permutations=None,
        min_coalition=0.0,
        max_coalition=1.0,
        verbose=False,
        bar=True,
    ):
        """
        Estimate SAGE values.

        Args:
          X: input data.
          Y: target data. If None, model output will be used.
          batch_size: number of examples to be processed in parallel, should be
            set to a large value.
          detect_convergence: whether to stop when approximately converged.
          thresh: threshold for determining convergence.
          n_permutations: number of permutations to unroll.
          min_coalition: minimum coalition size (int or float).
          max_coalition: maximum coalition size (int or float).
          verbose: print progress messages.
          bar: display progress bar.

        The default behavior is to detect convergence based on the width of the
        SAGE values' confidence intervals. Convergence is defined by the ratio
        of the maximum standard deviation to the gap between the largest and
        smallest values.

        Returns: Explanation object.
        """
        # Set random state.
        self.rng = np.random.default_rng(seed=self.random_state)

        # Determine explanation type.
        if Y is not None:
            explanation_type = "SAGE"
        else:
            explanation_type = "Shapley Effects"

        # Verify model.
        N, _ = X.shape
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(self.imputer, X, Y, self.loss_fn, batch_size)

        # Determine min/max coalition sizes.
        if isinstance(min_coalition, float):
            min_coalition = int(min_coalition * num_features)
        if isinstance(max_coalition, float):
            max_coalition = int(max_coalition * num_features)
        assert min_coalition >= 0
        assert max_coalition <= num_features
        assert min_coalition < max_coalition
        if min_coalition > 0 or max_coalition < num_features:
            explanation_type = "Relaxed " + explanation_type

        # Possibly force convergence detection.
        if n_permutations is None:
            n_permutations = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print("Turning convergence detection on")

        if detect_convergence:
            assert 0 < thresh < 1

        # Set up bar.
        n_loops = int(np.ceil(n_permutations / (batch_size * self.n_jobs)))
        if bar:
            if detect_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * self.n_jobs * batch_size)

        # Setup.
        tracker = utils.ImportanceTracker()

        for it in range(n_loops):
            # Sample data.
            batches = []
            for _ in range(self.n_jobs):
                idxs = self.rng.choice(N, batch_size)
                batches.append((X[idxs], Y[idxs]))

            # Get results from parallel processing of batches.
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._process_sample)(
                    x, y, num_features, min_coalition, max_coalition
                )
                for x, y in batches
            )

            for scores, sample_counts in results:
                tracker.update(scores, sample_counts)

            # Calculate progress.
            std = np.max(tracker.std)
            gap = max(tracker.values.max() - tracker.values.min(), 1e-12)
            ratio = std / gap

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print(f"StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})")
                else:
                    print(f"StdDev Ratio = {ratio:.4f}")

            # Check for convergence.
            if detect_convergence:
                if ratio < thresh:
                    if verbose:
                        print("Detected convergence")

                    # Skip bar ahead.
                    if bar:
                        bar.n = bar.total
                        bar.refresh()
                    break

            # Update progress bar.
            if bar and detect_convergence:
                # Update using convergence estimation.
                N_est = (it + 1) * (ratio / thresh) ** 2
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
            if bar and not detect_convergence:
                # Simply update number of permutations.
                bar.update(self.n_jobs)

        if bar:
            bar.close()

        return core.Explanation(tracker.values, tracker.std, explanation_type)

    def _process_sample(self, x, y, num_features, min_coalition, max_coalition):
        # Setup.
        batch_size = len(x)
        arange = np.arange(batch_size)
        scores = np.zeros((batch_size, num_features))
        S = np.zeros((batch_size, num_features), dtype=bool)
        permutations = np.tile(np.arange(num_features), (batch_size, 1))

        # Sample permutations.
        for i in range(batch_size):
            self.rng.shuffle(permutations[i])

        # Calculate sample counts.
        if min_coalition > 0 or max_coalition < num_features:
            sample_counts = np.zeros(num_features, dtype=int)
            for i in range(batch_size):
                sample_counts[permutations[i, min_coalition:max_coalition]] += 1
        else:
            sample_counts = None

        # Add necessary features to minimum coalition.
        for i in range(min_coalition):
            # Add next feature.
            inds = permutations[:, i]
            S[arange, inds] = 1

        # Make prediction with minimum coalition.
        y_hat = self.imputer(x, S)
        prev_loss = self.loss_fn(y_hat, y)

        # Add all remaining features.
        for i in range(min_coalition, max_coalition):
            # Add next feature.
            inds = permutations[:, i]
            S[arange, inds] = 1

            # Make prediction with missing features.
            y_hat = self.imputer(x, S)
            loss = self.loss_fn(y_hat, y)

            # Calculate delta sample.
            scores[arange, inds] = prev_loss - loss
            prev_loss = loss

        return scores, sample_counts
    


def run(args: argparse.Namespace) -> None:
    cfg = rank_utils.resolve_config(args.config)
    out_dir = rank_utils.ensure_dir(args.output_dir)
    rank_utils.save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = rank_utils.load_npz(args.dataset_path)
    X_val = data["X_val"].astype(np.float32)
    Y_val = data["y_val"].astype(np.float32)

    student_dir = Path(args.student_dir)
    ckpt_path = student_dir / "model.pt"
    model, feature_idx, use_gaussian = rank_utils.load_student(str(ckpt_path), device)

    imputer = rank_utils.MarginalImputer(model, feature_idx, use_gaussian, device)
    estimator = PermutationEstimator(imputer, loss=args.loss, n_jobs=cfg["eval"]["kernelshap_n_jobs"], random_state=args.seed)
    kernelshap_values = estimator(
        X_val,
        Y_val,
        batch_size=cfg["eval"]["kernelshap_batch_size"],
        n_samples=cfg["eval"]["kernelshap_n_samples"],
        detect_convergence=cfg["eval"]["kernelshap_detect_convergence"],
        thresh=cfg["eval"]["kernelshap_convergence_thresh"],
        verbose=True,
        bar=True,
    )

    np.savez(
        out_dir / "kernelshap_values.npz",
        values=kernelshap_values.values,
        std=kernelshap_values.std,
    )
    print(f"[kernelshap] Done. Values saved to {out_dir / 'kernelshap_values.npz'}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SAGE feature ranking")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset.npz")
    parser.add_argument("--student_dir", type=str, required=True, help="Path to student model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--loss', type=str, default='mse', help="Loss function for SAGE ('mse', 'cross entropy', 'zero one')")
    args = parser.parse_args()

    run(args)