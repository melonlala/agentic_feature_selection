"""Teacher policy wrapper for inference.

Wraps a trained Stable-Baselines3 DQN model and exposes a clean, backend-
independent inference API used by dataset collection and SHAP explanation.

Supports two teacher observation modes:

1. Discrete obs (native Taxi-v3, noise_dim == 0):
   The teacher was trained on Gymnasium's raw Discrete(500) observation, which
   SB3 one-hot encodes to a 500-dim vector. At inference, this class accepts
   structured 4-dim observations [row, col, passenger_loc, destination] and
   converts them back to state integers, which are then passed through the
   same one-hot → Q-net pipeline.

2. Box obs (NoisyTaxiWrapper, noise_dim > 0):
   The teacher was trained on a float32 Box observation. Observations are
   passed directly to the Q-network.
"""

from pathlib import Path

import gymnasium.spaces as gym_spaces
import numpy as np
import torch
from stable_baselines3 import DQN


class TeacherPolicy:
    """Inference wrapper around a trained SB3 DQN policy.

    All methods accept batches of float32 numpy arrays (structured 4-dim obs
    or full noisy obs) and return numpy arrays. The underlying model is placed
    in eval mode and no gradients are computed.

    Args:
        model_path: Path to the saved SB3 DQN model (.zip).
        device: Torch device string (e.g. "cpu" or "cuda").
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.model = DQN.load(str(model_path), device=device)
        self.model.policy.set_training_mode(False)
        self._n_actions = self.model.action_space.n

        # Detect observation mode
        obs_space = self.model.observation_space
        self._is_discrete = isinstance(obs_space, gym_spaces.Discrete)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _structured_to_int(self, X: np.ndarray) -> np.ndarray:
        """Convert structured 4-dim obs to Taxi-v3 state integers.

        Inverts the decode_taxi_state mapping:
            state = ((row * 5 + col) * 5 + passenger_loc) * 4 + destination

        Args:
            X: float32 array of shape [N, 4] — [row, col, passenger_loc, dest].

        Returns:
            int64 array of shape [N] — state integers in [0, 499].
        """
        rows      = X[:, 0].clip(0, 4).astype(np.int64)
        cols      = X[:, 1].clip(0, 4).astype(np.int64)
        pass_locs = X[:, 2].clip(0, 4).astype(np.int64)
        dests     = X[:, 3].clip(0, 3).astype(np.int64)
        return ((rows * 5 + cols) * 5 + pass_locs) * 4 + dests

    def _obs_to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy obs array to the correct tensor type for this teacher.

        For Discrete teachers: X is [N, 4] structured obs → converted to
        [N] int64 state integers (SB3 q_net handles one-hot encoding internally).

        For Box teachers: X is [N, D] float obs → float32 tensor.

        Args:
            X: Input observations, shape [N, 4] or [N, D].

        Returns:
            Tensor ready for self.model.policy.q_net.
        """
        if self._is_discrete:
            X = np.asarray(X, dtype=np.float32)
            state_ints = self._structured_to_int(X)  # [N]
            return torch.as_tensor(state_ints, dtype=torch.long, device=self.device)
        else:
            return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    def predict_q(self, X: np.ndarray) -> np.ndarray:
        """Compute Q-values for a batch of observations.

        Args:
            X: Structured observations [N, 4] (both discrete and Box teachers)
               or full noisy observations [N, D] (Box teachers only).

        Returns:
            Q-values, shape [N, n_actions].
        """
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(X)
            # q_net.forward calls extract_features → preprocess_obs internally,
            # which one-hot encodes Discrete obs before the MLP head.
            q_values = self.model.policy.q_net(obs_tensor)
        return q_values.cpu().numpy()

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute softmax action probabilities.

        Uses a numerically stable softmax over Q-values. This is the SVERL-style
        behaviour explanation target: we model the teacher's action distribution,
        not a hard argmax, making SHAP attributions smooth and meaningful.

        Args:
            X: Observations, shape [N, 4] or [N, D].

        Returns:
            Action probabilities, shape [N, n_actions].
        """
        q = self.predict_q(X)
        q_shifted = q - q.max(axis=1, keepdims=True)
        exp_q = np.exp(q_shifted)
        return (exp_q / exp_q.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict_actions(self, X: np.ndarray) -> np.ndarray:
        """Predict greedy actions (argmax over Q-values).

        Args:
            X: Observations, shape [N, 4] or [N, D].

        Returns:
            Action indices, shape [N].
        """
        return self.predict_q(X).argmax(axis=1).astype(np.int64)

    def predict_chosen_action_prob(self, X: np.ndarray) -> np.ndarray:
        """Compute the softmax probability of the greedy (chosen) action.

        Primary SHAP explanation target (SVERL-style).

        Args:
            X: Observations, shape [N, 4] or [N, D].

        Returns:
            Probability of the chosen action, shape [N].
        """
        probs = self.predict_probs(X)
        actions = probs.argmax(axis=1)
        return probs[np.arange(len(probs)), actions].astype(np.float32)

    def predict_chosen_action_q(self, X: np.ndarray) -> np.ndarray:
        """Compute the Q-value of the greedy (chosen) action.

        Args:
            X: Observations, shape [N, 4] or [N, D].

        Returns:
            Q-value of the chosen action, shape [N].
        """
        q = self.predict_q(X)
        actions = q.argmax(axis=1)
        return q[np.arange(len(q)), actions].astype(np.float32)
