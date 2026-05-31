"""Evaluation metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth integer labels, shape [N].
        y_pred: Predicted integer labels, shape [N].

    Returns:
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score.

    Args:
        y_true: Ground-truth integer labels, shape [N].
        y_pred: Predicted integer labels, shape [N].

    Returns:
        Macro F1 in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def success_rate(rewards: list[float], threshold: float = -199.0) -> float:
    """Compute fraction of episodes where total reward exceeds threshold.

    In Taxi-v3 a successful delivery yields +20 but costs -1/step, so the
    maximum episodic return is ~+15 (optimal ~5-step delivery). A timeout
    (200 steps without delivery) gives exactly -200. The default threshold
    of -199 separates all successful deliveries from timeouts regardless of
    how many steps the agent took.

    Args:
        rewards: List of episodic returns.
        threshold: Minimum return to count as success. Default -199 works
            for Taxi-v3 (timeout = -200, any delivery > -200).

    Returns:
        Success rate in [0, 1].
    """
    if len(rewards) == 0:
        return 0.0
    return float(np.mean([r >= threshold for r in rewards]))
