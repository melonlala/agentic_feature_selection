"""Feature-wise Hyperdimensional Computing (HDC) encoder using Random Fourier Features (RFF).

Each feature j is encoded independently via a random cosine projection (Rahimi & Recht, 2007):

    z_j(x_j) = sqrt(2 / D_j) * cos(W_j * x_j + b_j)

where:
  - W_j ~ N(0, bandwidth^2) of shape [D_j]  (frequency samples)
  - b_j ~ Uniform[0, 2π]   of shape [D_j]   (phase shifts)
  - D_j = rff_dim           (number of RFF components per feature)

This approximates the RBF kernel k(x_j, x_j') ≈ z_j(x_j)^T z_j(x_j').

The full encoding is:
    z(x) = [z_1(x_1), z_2(x_2), ..., z_d(x_d)]  ∈ R^{d * D_j}

The feature-wise block structure means any subset S produces:
    z_S(x) = [z_j(x_j)]_{j ∈ S}  ∈ R^{|S| * D_j}

which can be formed by column-selecting from the pre-computed full encoding Z, enabling
efficient closed-form ridge regression without retraining for each subset S.

Design note — why feature-wise, not full joint RFF:
    A joint RFF encoder z(x) = cos(Wx + b) would mix feature contributions and make it
    impossible to identify which raw feature j drives the prediction. The feature-wise
    structure preserves feature modularity required by MCI computation.
"""

from __future__ import annotations

import numpy as np


class FeatureWiseHDCEncoder:
    """Feature-wise Random Fourier Feature encoder.

    Each input feature is independently mapped to an RFF block of size rff_dim,
    approximating a per-feature RBF kernel.

    Args:
        rff_dim: Number of RFF components per feature (D_j in the paper).
        bandwidth: Standard deviation of the Gaussian frequency distribution.
                   Controls the kernel length-scale. Larger bandwidth → shorter
                   length-scale → more sensitive to small differences.
        seed: Random seed for reproducible frequency sampling.
    """

    def __init__(self, rff_dim: int = 64, bandwidth: float = 1.0, seed: int = 0) -> None:
        self.rff_dim = rff_dim
        self.bandwidth = bandwidth
        self.seed = seed

        # Populated by fit()
        self._W: list[np.ndarray] = []   # W_j ∈ R^{rff_dim} for each feature j
        self._b: list[np.ndarray] = []   # b_j ∈ R^{rff_dim} for each feature j
        self._n_features: int = 0
        self._scale: float = 0.0         # sqrt(2 / rff_dim), precomputed

    def fit(self, X: np.ndarray) -> "FeatureWiseHDCEncoder":
        """Sample per-feature RFF parameters from training data.

        The sampling only requires the number of features (X.shape[1]); actual
        feature values are not used. Call fit on training data only.

        Args:
            X: Training observations, shape [N, d]. Only d is used.

        Returns:
            self (for method chaining).
        """
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)

        self._n_features = d
        self._scale = np.sqrt(2.0 / self.rff_dim)
        self._W = []
        self._b = []

        for _ in range(d):
            # Frequency vector: W_j ~ N(0, bandwidth^2) ∈ R^{rff_dim}
            W_j = rng.normal(loc=0.0, scale=self.bandwidth, size=self.rff_dim)
            # Phase shift: b_j ~ Uniform[0, 2π] ∈ R^{rff_dim}
            b_j = rng.uniform(low=0.0, high=2.0 * np.pi, size=self.rff_dim)
            self._W.append(W_j)
            self._b.append(b_j)

        return self

    def transform_feature(self, X: np.ndarray, j: int) -> np.ndarray:
        """Encode a single feature j across all samples.

        z_j(x_j) = sqrt(2/D_j) * cos(W_j * x_j + b_j)

        Args:
            X: Observations, shape [N, d].
            j: Feature index.

        Returns:
            Encoded feature block, shape [N, rff_dim].
        """
        self._check_fitted()
        x_j = X[:, j].reshape(-1, 1)            # [N, 1]
        proj = x_j * self._W[j] + self._b[j]    # [N, rff_dim]
        return (self._scale * np.cos(proj)).astype(np.float32)

    def precompute(self, X: np.ndarray) -> None:
        """Pre-encode all features and cache the result blocks.

        After calling this, transform_subset uses cached blocks instead of
        recomputing cos(Wx+b) every call — critical for MCI where the same X
        is used thousands of times.

        Args:
            X: Observations to cache encodings for, shape [N, d].
        """
        self._check_fitted()
        self._cache = [self.transform_feature(X, j) for j in range(self._n_features)]

    def transform_subset(self, X: np.ndarray, S: list[int]) -> np.ndarray:
        """Encode a subset of features S.

        Z_S = [z_j(x_j)]_{j ∈ S} ∈ R^{N, |S| * rff_dim}

        Uses pre-computed cache if available (call precompute(X) first).

        Args:
            X: Observations, shape [N, d]. Ignored if cache is populated.
            S: List of feature indices to include.

        Returns:
            Encoded subset matrix, shape [N, len(S) * rff_dim].
            Returns shape [N, 0] if S is empty.
        """
        self._check_fitted()
        if len(S) == 0:
            return np.zeros((len(X), 0), dtype=np.float32)
        if hasattr(self, "_cache"):
            return np.concatenate([self._cache[j] for j in S], axis=1)
        blocks = [self.transform_feature(X, j) for j in S]
        return np.concatenate(blocks, axis=1)  # [N, |S|*rff_dim]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode all features.

        Args:
            X: Observations, shape [N, d].

        Returns:
            Full HDC encoding, shape [N, d * rff_dim].
        """
        return self.transform_subset(X, list(range(self._n_features)))

    @property
    def n_features(self) -> int:
        """Number of input features (set after fit)."""
        return self._n_features

    def _check_fitted(self) -> None:
        if self._n_features == 0:
            raise RuntimeError("FeatureWiseHDCEncoder must be fit before transform.")
