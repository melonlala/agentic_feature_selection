"""Imputers / masking strategies for SHAP KernelExplainer.

These imputers are used to define what value a feature takes when it is
"absent" from a SHAP coalition. Three strategies are implemented:

1. marginal:       Replace absent features with samples from the marginal
                   distribution (i.e. random rows from a background dataset).
                   This is the default and matches SHAP's built-in behaviour.

2. replay_random:  Replace absent features with values drawn uniformly from
                   all observed values of that feature in the background set.
                   Equivalent to marginal but samples each feature independently.

3. conditional_knn: Replace absent features with values from the k-nearest
                    neighbours (in the *present* feature subspace) in the
                    background set. Approximates E[absent | present features].

All imputers implement:
    __call__(x_masked, mask) -> imputed_x

where:
  x_masked : np.ndarray [1, D] — the query point (absent features zeroed)
  mask     : np.ndarray [D]   — binary mask, 1 = feature present

Note: SHAP's KernelExplainer handles the masking internally; these classes
are provided as callable prediction wrappers rather than being passed directly
as masking objects. See shap_behavior.py for integration.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class MarginalImputer:
    """Marginal imputer: samples missing features from background rows.

    For each masked query, a random background row is used to fill in the
    absent features. This is mathematically equivalent to sampling from the
    marginal distribution of each feature *jointly* (since we sample whole
    rows from the background).

    Args:
        background: Reference dataset, shape [M, D].
        seed: Random seed.
    """

    def __init__(self, background: np.ndarray, seed: int = 0) -> None:
        self.background = np.asarray(background, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def impute(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill absent features (mask=0) with random background values.

        Args:
            x: Query point, shape [D] or [1, D].
            mask: Binary mask, shape [D]. 1 = present, 0 = absent.

        Returns:
            Imputed observation, shape [D].
        """
        x = np.asarray(x, dtype=np.float32).flatten()
        mask = np.asarray(mask, dtype=bool).flatten()
        idx = self._rng.integers(0, len(self.background))
        bg_row = self.background[idx].copy()
        result = bg_row.copy()
        result[mask] = x[mask]
        return result

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.impute(x, mask)


class ReplayRandomImputer:
    """Replay-random imputer: samples each absent feature independently.

    Unlike MarginalImputer, absent features are filled by sampling each
    independently from its marginal distribution in the background set.
    This decorrelates the imputed values.

    Args:
        background: Reference dataset, shape [M, D].
        seed: Random seed.
    """

    def __init__(self, background: np.ndarray, seed: int = 0) -> None:
        self.background = np.asarray(background, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def impute(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill absent features by sampling each independently.

        Args:
            x: Query point, shape [D] or [1, D].
            mask: Binary mask, shape [D]. 1 = present, 0 = absent.

        Returns:
            Imputed observation, shape [D].
        """
        x = np.asarray(x, dtype=np.float32).flatten()
        mask = np.asarray(mask, dtype=bool).flatten()
        result = x.copy()
        absent_idx = np.where(~mask)[0]
        for j in absent_idx:
            row_idx = self._rng.integers(0, len(self.background))
            result[j] = self.background[row_idx, j]
        return result

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.impute(x, mask)


class ConditionalKNNImputer:
    """Conditional k-NN imputer: fills absent features from nearest neighbours.

    Finds the k nearest neighbours in the background set using only the
    *present* features, then samples absent feature values from those
    neighbours. This provides a crude approximation to E[absent | present].

    Args:
        background: Reference dataset, shape [M, D].
        k: Number of nearest neighbours.
        seed: Random seed.
    """

    def __init__(self, background: np.ndarray, k: int = 5, seed: int = 0) -> None:
        self.background = np.asarray(background, dtype=np.float32)
        self.k = min(k, len(background))
        self._rng = np.random.default_rng(seed)

    def impute(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill absent features from k-NN in the present feature subspace.

        Args:
            x: Query point, shape [D] or [1, D].
            mask: Binary mask, shape [D]. 1 = present, 0 = absent.

        Returns:
            Imputed observation, shape [D].
        """
        x = np.asarray(x, dtype=np.float32).flatten()
        mask = np.asarray(mask, dtype=bool).flatten()
        present_idx = np.where(mask)[0]
        absent_idx = np.where(~mask)[0]

        if len(absent_idx) == 0:
            return x.copy()

        if len(present_idx) == 0:
            # No present features: fall back to random background row
            idx = self._rng.integers(0, len(self.background))
            result = self.background[idx].copy()
            return result

        # Build kNN on present features only
        bg_present = self.background[:, present_idx]
        query = x[present_idx].reshape(1, -1)

        nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto").fit(bg_present)
        _, indices = nn.kneighbors(query)
        neighbour_idx = indices[0]

        # Sample one of the k neighbours uniformly
        chosen = self._rng.choice(neighbour_idx)
        result = x.copy()
        result[absent_idx] = self.background[chosen, absent_idx]
        return result

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.impute(x, mask)


def get_imputer(name: str, background: np.ndarray, seed: int = 0, **kwargs):
    """Factory function for imputers.

    Args:
        name: One of {"marginal", "replay_random", "conditional_knn"}.
        background: Background dataset, shape [M, D].
        seed: Random seed.
        **kwargs: Additional kwargs passed to the imputer.

    Returns:
        Imputer instance.
    """
    if name == "marginal":
        return MarginalImputer(background, seed=seed)
    elif name == "replay_random":
        return ReplayRandomImputer(background, seed=seed)
    elif name == "conditional_knn":
        return ConditionalKNNImputer(background, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown imputer: {name!r}. Choose marginal/replay_random/conditional_knn.")
