"""Noisy Taxi-v3 wrapper.

Wraps the Gymnasium Taxi-v3 discrete observation space into a continuous
float32 observation vector:

    [taxi_row, taxi_col, passenger_loc, destination, z_0, ..., z_{noise_dim-1}]

The first four entries are the *structured* features decoded from the integer
state. The remaining entries are noise features appended according to the
configured noise type.

Single noise types:
  - "gaussian":         i.i.d. N(0, 1).
  - "categorical":      i.i.d. uniform discrete in {0, ..., cardinality-1}.
  - "uniform":          i.i.d. U[low, high] continuous.
  - "laplace":          i.i.d. Laplace(0, scale) — heavy-tailed.
  - "bernoulli":        i.i.d. Bernoulli(p) — binary {0, 1}.
  - "correlated":       MVN with a fixed random correlation matrix. Noise
                        features are correlated with each other but not with
                        oracle features. Tests SHAP under collinearity.
  - "state_correlated": z_i = alpha * oracle[i % 4] + sqrt(1-alpha²) * N(0,1).
                        Noise carries partial state information. Hardest case
                        for feature selection.

Mixed noise type:
  - "mixed":  Concatenates multiple independently-configured noise blocks.
              Requires ``noise_params["components"]``, a list of dicts each
              with a ``type``, ``dim``, and any type-specific keys.  The total
              ``noise_dim`` is inferred as the sum of component dims (or must
              match if provided explicitly).

              Feature names encode the component type:
                  z_{type}_{local_index}
              e.g. z_gaussian_0, z_bernoulli_1, z_state_correlated_0.
              This makes SHAP rankings interpretable across noise families.

              Example noise_params for four simultaneous types:
                  components:
                    - type: gaussian
                      dim: 2
                    - type: bernoulli
                      dim: 2
                      p: 0.5
                    - type: correlated
                      dim: 2
                      correlation_strength: 0.8
                    - type: state_correlated
                      dim: 2
                      state_correlation_strength: 0.5

Dynamics, rewards, and termination conditions are unchanged for all types.

Extra parameters for single types are passed via ``noise_params``:
  - "uniform":          low (default 0.0), high (default 1.0)
  - "laplace":          scale (default 1.0)
  - "bernoulli":        p (default 0.5)
  - "correlated":       correlation_strength (default 0.8, range [0, 1))
  - "state_correlated": state_correlation_strength (default 0.5, range [0, 1])
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Types valid for top-level noise_type
SUPPORTED_NOISE_TYPES = (
    "gaussian",
    "categorical",
    "uniform",
    "laplace",
    "bernoulli",
    "correlated",
    "state_correlated",
    "mixed",
)

# Types valid inside a "mixed" component (cannot nest "mixed")
_COMPONENT_NOISE_TYPES = tuple(t for t in SUPPORTED_NOISE_TYPES if t != "mixed")


def decode_taxi_state(state_int: int) -> tuple[int, int, int, int]:
    """Decode a Taxi-v3 integer state into structured features.

    Taxi-v3 encodes its state as a single integer using mixed-radix encoding:
        state = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

    Ranges:
        taxi_row:      0..4
        taxi_col:      0..4
        passenger_loc: 0..4  (0-3 = pickup locations, 4 = in taxi)
        destination:   0..3

    Args:
        state_int: Integer state from Taxi-v3.

    Returns:
        Tuple (taxi_row, taxi_col, passenger_loc, destination).
    """
    destination = state_int % 4
    state_int //= 4
    passenger_loc = state_int % 5
    state_int //= 5
    taxi_col = state_int % 5
    taxi_row = state_int // 5
    return taxi_row, taxi_col, passenger_loc, destination


def _make_correlated_cov(noise_dim: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Build a random positive-definite correlation matrix for correlated noise.

    Constructs a matrix with off-diagonal entries drawn from U[0, strength],
    symmetrised, then made PD via eigenvalue clipping.

    Args:
        noise_dim: Number of noise features.
        strength: Target magnitude of off-diagonal correlations (0 = diagonal).
        rng: Seeded RNG for reproducibility.

    Returns:
        Float64 correlation matrix of shape [noise_dim, noise_dim].
    """
    if noise_dim == 1:
        return np.ones((1, 1))
    raw = rng.uniform(0, strength, size=(noise_dim, noise_dim))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-6, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _init_component(spec: dict, rng: np.random.Generator) -> dict:
    """Initialise a single noise component from its spec dict.

    Validates the spec and pre-computes any state that is fixed for the
    lifetime of the wrapper (e.g. the correlated covariance matrix).

    Args:
        spec: Dict with keys ``type``, ``dim``, and optional type params.
        rng: Shared RNG; used to derive per-component seeds deterministically.

    Returns:
        Enriched component dict ready for use in ``_sample_component``.
    """
    t = spec.get("type")
    d = int(spec.get("dim", 1))
    assert t in _COMPONENT_NOISE_TYPES, (
        f"Component type must be one of {_COMPONENT_NOISE_TYPES}, got {t!r}"
    )
    assert d >= 1, f"Component dim must be >= 1, got {d}"

    comp: dict = {"type": t, "dim": d}

    if t == "correlated":
        strength = float(spec.get("correlation_strength", 0.8))
        assert 0.0 <= strength < 1.0, "correlation_strength must be in [0, 1)"
        comp["corr_cov"] = _make_correlated_cov(d, strength, rng)

    elif t == "state_correlated":
        alpha = float(spec.get("state_correlation_strength", 0.5))
        assert 0.0 <= alpha <= 1.0, "state_correlation_strength must be in [0, 1]"
        comp["state_alpha"] = alpha
        comp["state_residual"] = float(np.sqrt(max(1.0 - alpha ** 2, 0.0)))

    elif t == "uniform":
        lo = float(spec.get("low", 0.0))
        hi = float(spec.get("high", 1.0))
        assert lo < hi, f"uniform noise requires low < high, got {lo} >= {hi}"
        comp["low"] = lo
        comp["high"] = hi

    elif t == "laplace":
        comp["scale"] = float(spec.get("scale", 1.0))

    elif t == "bernoulli":
        comp["p"] = float(spec.get("p", 0.5))

    elif t == "categorical":
        comp["cardinality"] = int(spec.get("cardinality", 5))

    return comp


def _component_bounds(comp: dict) -> tuple[list[float], list[float]]:
    """Return (low, high) bound lists for a single component."""
    t, d = comp["type"], comp["dim"]
    if t in ("gaussian", "laplace", "correlated", "state_correlated"):
        return [-np.inf] * d, [np.inf] * d
    if t == "categorical":
        return [0.0] * d, [float(comp["cardinality"] - 1)] * d
    if t == "uniform":
        return [comp["low"]] * d, [comp["high"]] * d
    if t == "bernoulli":
        return [0.0] * d, [1.0] * d
    raise ValueError(f"Unknown component type: {t!r}")


def _sample_component(comp: dict, base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a noise sample for one component.

    Args:
        comp: Initialised component dict from ``_init_component``.
        base: Oracle features [taxi_row, taxi_col, passenger_loc, destination].
        rng: Shared RNG.

    Returns:
        Float32 array of length comp["dim"].
    """
    t, d = comp["type"], comp["dim"]

    if t == "gaussian":
        return rng.standard_normal(d).astype(np.float32)

    if t == "categorical":
        return rng.integers(0, comp["cardinality"], size=d).astype(np.float32)

    if t == "uniform":
        return rng.uniform(comp["low"], comp["high"], size=d).astype(np.float32)

    if t == "laplace":
        return rng.laplace(loc=0.0, scale=comp["scale"], size=d).astype(np.float32)

    if t == "bernoulli":
        return rng.binomial(n=1, p=comp["p"], size=d).astype(np.float32)

    if t == "correlated":
        return rng.multivariate_normal(
            mean=np.zeros(d), cov=comp["corr_cov"]
        ).astype(np.float32)

    # state_correlated
    indices = np.arange(d) % 4
    oracle_vals = base[indices]
    gaussian_part = rng.standard_normal(d).astype(np.float32)
    return (comp["state_alpha"] * oracle_vals
            + comp["state_residual"] * gaussian_part).astype(np.float32)


class NoisyTaxiWrapper(gym.Wrapper):
    """Gymnasium wrapper that augments Taxi-v3 with noise features.

    Converts the integer observation into a float32 vector:
        [taxi_row, taxi_col, passenger_loc, destination, *noise_features]

    For ``noise_type="mixed"``, noise features are drawn from multiple
    independently-configured blocks; feature names encode the block type.
    For all other types, noise features are named z_0, z_1, ...

    Args:
        env: A Taxi-v3 gymnasium environment.
        noise_dim: Number of noise features. For "mixed", inferred from the
            sum of component dims if not provided; must match if provided.
        noise_type: One of SUPPORTED_NOISE_TYPES.
        categorical_noise_cardinality: Category count for "categorical" noise.
        noise_params: Dict of extra parameters. For "mixed", must contain a
            ``components`` list. See module docstring for per-type keys.
        seed: RNG seed for reproducible noise generation.
    """

    def __init__(
        self,
        env: gym.Env,
        noise_dim: int = 8,
        noise_type: str = "gaussian",
        categorical_noise_cardinality: int = 5,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)

        assert noise_type in SUPPORTED_NOISE_TYPES, (
            f"noise_type must be one of {SUPPORTED_NOISE_TYPES}, got {noise_type!r}"
        )

        self.noise_type = noise_type
        self.categorical_noise_cardinality = categorical_noise_cardinality
        self._noise_params = noise_params or {}
        self._rng = np.random.default_rng(seed)

        n_base = 4  # taxi_row, taxi_col, passenger_loc, destination
        base_low  = [0.0, 0.0, 0.0, 0.0]
        base_high = [4.0, 4.0, 4.0, 3.0]
        self._base_feature_names = ["taxi_row", "taxi_col", "passenger_loc", "destination"]

        # ----------------------------------------------------------------
        # "mixed" initialisation — build a list of component descriptors
        # ----------------------------------------------------------------
        if noise_type == "mixed":
            raw_components = self._noise_params.get("components")
            assert raw_components, (
                'noise_type="mixed" requires noise_params["components"] list'
            )
            self._components = [_init_component(c, self._rng) for c in raw_components]
            inferred_dim = sum(c["dim"] for c in self._components)

            assert noise_dim == 0 or noise_dim == inferred_dim, (
                f"noise_dim={noise_dim} does not match sum of component dims "
                f"({inferred_dim}). Either omit noise_dim or set it to {inferred_dim}."
            )
            self.noise_dim = inferred_dim

            # Bounds: concatenate per-component bounds
            n_low, n_high = [], []
            for comp in self._components:
                lo, hi = _component_bounds(comp)
                n_low.extend(lo)
                n_high.extend(hi)

            # Feature names: z_{type}_{local_index}
            # Duplicate type names get a disambiguating counter suffix.
            type_counters: dict[str, int] = {}
            noise_names = []
            for comp in self._components:
                t = comp["type"]
                count = type_counters.get(t, 0)
                for local_i in range(comp["dim"]):
                    noise_names.append(f"z_{t}_{local_i + count}")
                type_counters[t] = count + comp["dim"]
            self._noise_feature_names = noise_names

        # ----------------------------------------------------------------
        # Single-type initialisation (unchanged from before)
        # ----------------------------------------------------------------
        else:
            assert noise_dim >= 0, f"noise_dim must be >= 0, got {noise_dim}"
            self.noise_dim = noise_dim
            self._components = None  # unused for single types

            # Per-type pre-computation
            if noise_type == "correlated" and noise_dim > 0:
                strength = float(self._noise_params.get("correlation_strength", 0.8))
                assert 0.0 <= strength < 1.0, "correlation_strength must be in [0, 1)"
                self._corr_cov = _make_correlated_cov(noise_dim, strength, self._rng)
            else:
                self._corr_cov = None

            if noise_type == "state_correlated":
                alpha = float(self._noise_params.get("state_correlation_strength", 0.5))
                assert 0.0 <= alpha <= 1.0, "state_correlation_strength must be in [0, 1]"
                self._state_alpha = alpha
                self._state_residual = float(np.sqrt(max(1.0 - alpha ** 2, 0.0)))
            else:
                self._state_alpha = 0.0
                self._state_residual = 1.0

            # Bounds
            if noise_type == "gaussian":
                n_low  = [-np.inf] * noise_dim
                n_high = [ np.inf] * noise_dim
            elif noise_type == "categorical":
                n_low  = [0.0] * noise_dim
                n_high = [float(categorical_noise_cardinality - 1)] * noise_dim
            elif noise_type == "uniform":
                lo = float(self._noise_params.get("low", 0.0))
                hi = float(self._noise_params.get("high", 1.0))
                assert lo < hi, f"uniform noise requires low < high"
                n_low  = [lo] * noise_dim
                n_high = [hi] * noise_dim
            elif noise_type == "laplace":
                n_low  = [-np.inf] * noise_dim
                n_high = [ np.inf] * noise_dim
            elif noise_type == "bernoulli":
                n_low  = [0.0] * noise_dim
                n_high = [1.0] * noise_dim
            elif noise_type in ("correlated", "state_correlated"):
                n_low  = [-np.inf] * noise_dim
                n_high = [ np.inf] * noise_dim
            else:
                n_low  = [-np.inf] * noise_dim
                n_high = [ np.inf] * noise_dim

            self._noise_feature_names = [f"z_{i}" for i in range(noise_dim)]

        # ----------------------------------------------------------------
        # Shared: observation space + feature name consistency check
        # ----------------------------------------------------------------
        self._observation_dim = n_base + self.noise_dim
        self.observation_space = spaces.Box(
            low=np.array(base_low + n_low, dtype=np.float32),
            high=np.array(base_high + n_high, dtype=np.float32),
            dtype=np.float32,
        )

        assert len(self.all_feature_names) == self._observation_dim, (
            "Feature name count mismatch — check noise_dim / component dims."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Total observation dimensionality (base + noise)."""
        return self._observation_dim

    @property
    def base_feature_names(self) -> list[str]:
        """Names of the four structured (oracle) features."""
        return list(self._base_feature_names)

    @property
    def all_feature_names(self) -> list[str]:
        """Names of all features in observation order."""
        return self._base_feature_names + self._noise_feature_names

    @property
    def noise_components(self) -> list[dict] | None:
        """Component descriptors for 'mixed' noise; None for single types."""
        return self._components

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._augment(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_noise(self, base: np.ndarray) -> np.ndarray:
        """Sample noise features for a single-type wrapper.

        Args:
            base: Oracle features [taxi_row, taxi_col, passenger_loc, dest].

        Returns:
            Float32 noise array of length noise_dim.
        """
        d = self.noise_dim
        t = self.noise_type

        if t == "gaussian":
            return self._rng.standard_normal(d).astype(np.float32)

        if t == "categorical":
            return self._rng.integers(
                0, self.categorical_noise_cardinality, size=d
            ).astype(np.float32)

        if t == "uniform":
            lo = float(self._noise_params.get("low", 0.0))
            hi = float(self._noise_params.get("high", 1.0))
            return self._rng.uniform(lo, hi, size=d).astype(np.float32)

        if t == "laplace":
            scale = float(self._noise_params.get("scale", 1.0))
            return self._rng.laplace(loc=0.0, scale=scale, size=d).astype(np.float32)

        if t == "bernoulli":
            p = float(self._noise_params.get("p", 0.5))
            return self._rng.binomial(n=1, p=p, size=d).astype(np.float32)

        if t == "correlated":
            return self._rng.multivariate_normal(
                mean=np.zeros(d), cov=self._corr_cov
            ).astype(np.float32)

        # state_correlated
        indices = np.arange(d) % 4
        oracle_vals = base[indices]
        gaussian_part = self._rng.standard_normal(d).astype(np.float32)
        return (self._state_alpha * oracle_vals
                + self._state_residual * gaussian_part).astype(np.float32)

    def _augment(self, state_int: int) -> np.ndarray:
        """Decode state and append noise features.

        Args:
            state_int: Raw Taxi-v3 integer observation.

        Returns:
            Float32 observation vector of length observation_dim.
        """
        row, col, passenger, dest = decode_taxi_state(int(state_int))
        base = np.array([row, col, passenger, dest], dtype=np.float32)

        if self.noise_dim == 0:
            return base

        if self.noise_type == "mixed":
            parts = [_sample_component(c, base, self._rng) for c in self._components]
            noise = np.concatenate(parts, axis=0)
        else:
            noise = self._sample_noise(base)

        return np.concatenate([base, noise], axis=0)
