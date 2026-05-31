"""Global seed utilities for reproducibility."""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed random, numpy, and torch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic CUDA ops (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
