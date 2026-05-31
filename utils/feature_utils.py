"""Feature selection utilities.

Feature ordering convention for Taxi-v3 (must be consistent everywhere):
  [taxi_row, taxi_col, passenger_loc, destination, z_0, z_1, ..., z_{noise_dim-1}]

The first 4 features are the oracle features (sufficient for solving Taxi-v3).

For D4RL continuous tasks, oracle_feature_indices are supplied at call-time via config.
"""

from typing import Union, Iterable, Tuple


import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# Indices of oracle features within the full feature vector (Taxi-v3 default)
ORACLE_FEATURE_INDICES = [0, 1, 2, 3]
ORACLE_FEATURE_NAMES = ["taxi_row", "taxi_col", "passenger_loc", "destination"]

MultiVariateArray = Union[pd.DataFrame, pd.Series, np.ndarray]
UniVariateArray = Union[pd.Series, np.ndarray, list]
def context_to_key(context: Iterable[str]) -> Tuple[str]:
    return tuple(sorted(context))

def get_oracle_indices(k: int) -> list[int]:
    """Return oracle feature indices, truncated to k if k < 4.

    The oracle features are the 4 structured Taxi features. If k >= 4, all 4
    are returned. If k < 4, only the first k are returned (priority: row, col,
    passenger_loc, destination).

    Args:
        k: Number of features to select.

    Returns:
        List of selected feature indices.
    """
    return ORACLE_FEATURE_INDICES[:k]


def get_topk_indices(ranking_df: pd.DataFrame, k: int) -> list[int]:
    """Return the top-k feature indices from a ranking DataFrame.

    Args:
        ranking_df: DataFrame with columns [feature_name, mean_abs_shap, rank, ...].
                    Sorted ascending by rank (rank 1 = most important).
        k: Number of top features to select.

    Returns:
        List of feature indices in ranked order (most important first).
    """
    top = ranking_df.sort_values("rank").head(k)
    return list(top["feature_index"].astype(int))


def get_random_indices(n_features: int, k: int, rng: np.random.Generator | None = None) -> list[int]:
    """Sample k feature indices uniformly at random (without replacement).

    Args:
        n_features: Total number of features.
        k: Number of features to select.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        Sorted list of selected feature indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(n_features, size=min(k, n_features), replace=False)
    return sorted(indices.tolist())


def get_mi_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int = 0,
    continuous_target: bool = False,
) -> list[int]:
    """Return top-k feature indices ranked by mutual information with labels.

    For discrete targets (Taxi) uses mutual_info_classif.
    For continuous scalar targets (action norm) uses mutual_info_regression.

    Args:
        X_train:           Training observations, shape [N, D].
        y_train:           Training labels [N] (discrete) or [N] scalar (continuous).
        k:                 Number of features to select.
        seed:              Random seed for MI estimation.
        continuous_target: If True, use mutual_info_regression instead of classif.

    Returns:
        List of top-k feature indices (most informative first).
    """
    if continuous_target:
        mi_scores = mutual_info_regression(X_train, y_train, random_state=seed)
    else:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    ranked = np.argsort(mi_scores)[::-1]  # descending
    return [int(i) for i in ranked[:k]]


_MI_MAX_SAMPLES = 5000  # k-NN MI is O(n²) in high-D; subsample to stay fast


def get_mi_indices_multioutput(
    X_train: np.ndarray,
    y_matrix: np.ndarray,
    k: int,
    seed: int = 0,
) -> list[int]:
    """Return top-k feature indices by average MI across all action dimensions.

    For each action dimension, compute MI_regression(X, a_j) and average.
    Subsamples to _MI_MAX_SAMPLES rows so k-NN estimation stays tractable in
    high-dimensional observation spaces (e.g. kitchen obs_dim=60).

    Args:
        X_train:  Training observations [N, D].
        y_matrix: Continuous actions [N, action_dim].
        k:        Number of features to select.
        seed:     Random seed.

    Returns:
        List of top-k feature indices.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    if n > _MI_MAX_SAMPLES:
        idx = rng.choice(n, size=_MI_MAX_SAMPLES, replace=False)
        X_mi = X_train[idx]
        y_mi = y_matrix[idx]
    else:
        X_mi = X_train
        y_mi = y_matrix

    action_dim = y_mi.shape[1]
    scores = np.zeros(X_mi.shape[1], dtype=np.float64)
    for j in range(action_dim):
        scores += mutual_info_regression(X_mi, y_mi[:, j], random_state=seed)
    scores /= action_dim
    ranked = np.argsort(scores)[::-1]
    return [int(i) for i in ranked[:k]]


def dispatch_selector(
    selector: str,
    k: int,
    n_features: int,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    ranking_df: pd.DataFrame | None = None,
    seed: int = 0,
) -> list[int]:
    """Dispatch feature selection by selector name.

    Supported selectors:
      - "shap":   top-k from SHAP ranking (requires ranking_df)
      - "random": uniformly random k features
      - "oracle": first k oracle features [row, col, passenger_loc, dest]
      - "mi":     top-k by mutual information (requires X_train, y_train)
      - "full":   all n_features features

    Args:
        selector: One of {"shap", "random", "oracle", "mi", "full"}.
        k: Number of features to select (ignored for "full").
        n_features: Total feature dimensionality.
        X_train: Training observations (needed for "mi").
        y_train: Training labels (needed for "mi").
        ranking_df: SHAP ranking DataFrame (needed for "shap").
        seed: Random seed.

    Returns:
        List of selected feature indices.
    """
    if selector == "full":
        return list(range(n_features))
    elif selector == "oracle":
        return get_oracle_indices(k)
    elif selector == "shap":
        if ranking_df is None:
            raise ValueError("ranking_df is required for selector='shap'")
        return get_topk_indices(ranking_df, k)
    elif selector == "random":
        rng = np.random.default_rng(seed)
        return get_random_indices(n_features, k, rng=rng)
    elif selector == "mi":
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for selector='mi'")
        return get_mi_indices(X_train, y_train, k, seed=seed)
    else:
        raise ValueError(f"Unknown selector: {selector!r}. Choose from shap/random/oracle/mi/full.")


def dispatch_selector_continuous(
    selector: str,
    k: int,
    n_features: int,
    X_train: np.ndarray | None = None,
    y_scalar: np.ndarray | None = None,
    y_matrix: np.ndarray | None = None,
    ranking_df: pd.DataFrame | None = None,
    oracle_indices: list[int] | None = None,
    seed: int = 0,
) -> list[int]:
    """Dispatch feature selection for continuous-action tasks.

    Same 5 selectors as dispatch_selector but adapted for continuous actions:
      - "oracle": uses task-specific oracle_indices from config (not hardcoded Taxi)
      - "mi":     averages mutual_info_regression over all action dimensions
      - others:   identical to dispatch_selector

    Args:
        selector:       One of {"shap", "random", "oracle", "mi", "full"}.
        k:              Number of features to select (ignored for "full").
        n_features:     Total feature dimensionality.
        X_train:        Training observations [N, D].
        y_scalar:       Scalar target per step (e.g. action norm) for MI scalar mode.
        y_matrix:       Continuous actions [N, action_dim] for MI multi-output mode.
        ranking_df:     MCI/SHAP ranking DataFrame (for "shap" selector).
        oracle_indices: Task-specific oracle feature indices from config.
        seed:           Random seed.

    Returns:
        List of selected feature indices.
    """
    if selector == "full":
        return list(range(n_features))

    elif selector == "oracle":
        if not oracle_indices:
            oracle_indices = list(range(min(k, n_features)))
        return oracle_indices[:k]

    elif selector in ("shap", "mci"):
        if ranking_df is None:
            raise ValueError(f"ranking_df is required for selector={selector!r}")
        return get_topk_indices(ranking_df, k)

    elif selector == "random":
        rng = np.random.default_rng(seed)
        return get_random_indices(n_features, k, rng=rng)

    elif selector == "mi":
        if X_train is None:
            raise ValueError("X_train is required for selector='mi'")
        if y_matrix is not None and y_matrix.ndim == 2:
            return get_mi_indices_multioutput(X_train, y_matrix, k, seed=seed)
        if y_scalar is None:
            raise ValueError("y_scalar or y_matrix required for selector='mi'")
        return get_mi_indices(X_train, y_scalar, k, seed=seed, continuous_target=True)

    else:
        raise ValueError(
            f"Unknown selector: {selector!r}. Choose from shap/mci/random/oracle/mi/full."
        )
