"""Behavioral cloning MLP for continuous action spaces.

Four variants:
  - BCContinuousPolicy:           deterministic MLP, forward → action mean [N, act_dim]
  - BCGaussianPolicy:             Gaussian MLP,       forward → (mean, log_std)
  - BCContinuousPolicyFromLatent: frozen layer-1 of a pre-trained full student +
                                  fresh head over selected latent dims.
  - BCGaussianPolicyFromLatent:   same, Gaussian variant.

Training is handled by train_student_continuous.py.
Both variants support variable-width hidden layers and input feature subsets.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_size = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.LayerNorm(h))  # layer norm stabilises continuous BC
        layers.append(nn.ReLU())
        in_size = h
    layers.append(nn.Linear(in_size, output_dim))
    return nn.Sequential(*layers)


class BCContinuousPolicy(nn.Module):
    """Deterministic behavioral cloning policy for continuous actions.

    Produces a single action vector; trained with MSE loss.

    Args:
        input_dim:   Number of selected input features.
        action_dim:  Dimensionality of the action space.
        hidden_dims: MLP hidden layer sizes.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 9,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = _build_mlp(input_dim, action_dim, hidden_dims)
        self.input_dim  = input_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted actions.

        Args:
            x: Input tensor [N, input_dim].

        Returns:
            Action tensor [N, action_dim].
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward (deterministic)."""
        return self.forward(x)


class BCGaussianPolicy(nn.Module):
    """Gaussian behavioral cloning policy for continuous actions.

    Outputs a diagonal Gaussian distribution over actions.
    During inference use the mean; log_std is learned globally (state-independent
    baseline that can optionally be made state-dependent by switching shared_std=False).

    Args:
        input_dim:   Number of selected input features.
        action_dim:  Dimensionality of the action space.
        hidden_dims: MLP hidden layer sizes.
        log_std_min: Lower clamp for log_std (numerical stability).
        log_std_max: Upper clamp for log_std.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 9,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_net   = _build_mlp(input_dim, action_dim, hidden_dims)
        # Shared (state-independent) log std — simple and effective for BC
        self.log_std    = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.input_dim  = input_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian parameters.

        Args:
            x: Input tensor [N, input_dim].

        Returns:
            Tuple of (mean, log_std), each [N, action_dim].
        """
        mean = self.mean_net(x)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def predict(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Sample or return mean action.

        Args:
            x:             Input tensor [N, input_dim].
            deterministic: If True, return the mean (no noise).

        Returns:
            Action tensor [N, action_dim].
        """
        mean, log_std = self.forward(x)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * log_std.exp()

    def log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given actions under the Gaussian.

        Args:
            x:       Input tensor [N, input_dim].
            actions: Target actions  [N, action_dim].

        Returns:
            Log-prob per sample [N].
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        # Gaussian log-prob (summed over action dims)
        log_p = -0.5 * ((actions - mean) / std) ** 2 - log_std - 0.5 * torch.log(
            2 * torch.tensor(torch.pi)
        )
        return log_p.sum(dim=-1)


class BCContinuousPolicyFromLatent(nn.Module):
    """BC policy that consumes a frozen layer-1 embedding of a pre-trained student.

    Forward: x → frozen_layer1(x) → select latent_indices → head → action.
    The frozen layer is treated as a fixed feature extractor; only `head` trains.
    """

    def __init__(
        self,
        frozen_layer1: nn.Sequential,
        latent_indices: Sequence[int],
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.frozen_layer1 = frozen_layer1
        for p in self.frozen_layer1.parameters():
            p.requires_grad_(False)
        idx = torch.as_tensor(list(latent_indices), dtype=torch.long)
        self.register_buffer("latent_indices", idx)
        self.head = _build_mlp(int(idx.numel()), action_dim, hidden_dims)
        self.input_dim  = int(idx.numel())
        self.action_dim = action_dim

    def train(self, mode: bool = True) -> "BCContinuousPolicyFromLatent":  # noqa: D401
        # Keep the frozen extractor in eval mode regardless of caller intent.
        super().train(mode)
        self.frozen_layer1.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.frozen_layer1(x)
        return self.head(z.index_select(1, self.latent_indices))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class BCGaussianPolicyFromLatent(nn.Module):
    """Gaussian-output variant of `BCContinuousPolicyFromLatent`."""

    def __init__(
        self,
        frozen_layer1: nn.Sequential,
        latent_indices: Sequence[int],
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.frozen_layer1 = frozen_layer1
        for p in self.frozen_layer1.parameters():
            p.requires_grad_(False)
        idx = torch.as_tensor(list(latent_indices), dtype=torch.long)
        self.register_buffer("latent_indices", idx)
        self.mean_net = _build_mlp(int(idx.numel()), action_dim, hidden_dims)
        self.log_std    = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.input_dim  = int(idx.numel())
        self.action_dim = action_dim

    def train(self, mode: bool = True) -> "BCGaussianPolicyFromLatent":  # noqa: D401
        super().train(mode)
        self.frozen_layer1.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.frozen_layer1(x)
        mean = self.mean_net(z.index_select(1, self.latent_indices))
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def predict(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(x)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * log_std.exp()
