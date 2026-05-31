"""Behavioral cloning MLP policy.

Implements a configurable MLP that maps an observation (or a subset of its
features) to action logits. Training is handled by train_student.py.
"""

from typing import Sequence

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """Multi-layer perceptron behavioral cloning policy.

    Args:
        input_dim: Number of input features (may be a subset of obs_dim).
        n_actions: Number of discrete actions (6 for Taxi-v3).
        hidden_dims: Sizes of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 6,
        hidden_dims: Sequence[int] = (64, 64),
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
        self.input_dim = input_dim
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            x: Input tensor, shape [N, input_dim].

        Returns:
            Logits tensor, shape [N, n_actions].
        """
        return self.net(x)
