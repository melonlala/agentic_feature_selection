"""Custom Q-network architecture.

This file provides a standalone QNetwork module that mirrors the architecture
used by Stable-Baselines3 for DQN. It exists to allow direct weight loading
and batch inference outside of SB3.

Note: The actual teacher training uses SB3's DQN implementation. This module
is used by TeacherPolicy for inference.
"""

from typing import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Multi-layer perceptron Q-network.

    Maps observations to Q-values for each action.

    Args:
        input_dim: Dimensionality of the observation vector.
        n_actions: Number of discrete actions (6 for Taxi-v3).
        hidden_dims: Sizes of hidden layers (default [128, 128]).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 6,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_size = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            x: Observation tensor, shape [N, input_dim].

        Returns:
            Q-value tensor, shape [N, n_actions].
        """
        return self.net(x)
