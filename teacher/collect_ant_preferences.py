"""Collect an *offline* preference-learning dataset from seals/Ant-v1.

This is the preference-comparisons analogue of ``teacher/collect_ant_expert_data.py``.
Instead of producing a flat (obs, act) imitation dataset, it produces a
``PreferenceDataset`` — pairs of trajectory fragments labelled with synthetic
preferences derived from the ground-truth environment reward.

It is "offline" in the same sense as the AIRL pipeline's demo collection: the
trajectory pool is gathered once from fixed policies (no reward model is trained
in the loop), then fragmented and labelled. This mirrors the building blocks
used by imitation's ``scripts/train_preference_comparisons.py``
(``RandomFragmenter`` + ``SyntheticGatherer`` + ``PreferenceDataset``) but skips
the on-policy ``AgentTrainer`` so the result is a static dataset.

The trajectory pool is drawn from three policies of differing quality, and the
dataset is divided into one preference file *per source* so each quality regime
can be consumed independently downstream:
  - ``expert``: the HuggingFace PPO expert (high reward),
  - ``noisy`` : a noisy expert (expert actions + Gaussian noise, medium reward),
  - ``random``: a uniform-random policy (low reward).
The episode/pair budgets per source are configurable via the ``preferences``
config block. Within a single source there is still reward spread (e.g. expert
episode returns vary widely), so the synthetic preferences remain informative.

Labels are the two fragments' (discounted) ground-truth returns — ``return_a``
and ``return_b`` — rather than a binary preference. Downstream code can derive a
preference at any temperature via ``sigmoid((return_a - return_b) / T)`` or train
a reward model by regression onto the returns.

Pipeline (run once per source):
  1. Build vectorized seals/Ant-v1 with RolloutInfoWrapper.
  2. Roll out the source policy into a pool of TrajectoryWithRew.
  3. RandomFragmenter → fragment pairs of fixed length.
  4. Discounted-return labelling → (return_a, return_b) per pair.
  5. Save preferences_{source}.pkl + returns_{source}.npz + trajectories_{source}.pkl.

Output artifacts (in --output_dir), for each source in {expert, noisy, random}:
    preferences_{source}.pkl    pickled dict: fragments1/2 (TrajectoryWithRew) +
                                return_a / return_b (np.float32 [N]) + discount_factor
    returns_{source}.npz        just the numeric labels: return_a, return_b
    trajectories_{source}.pkl   pickled Sequence[TrajectoryWithRew] (that source's pool)
    metadata.json               provenance + per-source summary statistics
    resolved_config.yaml

Usage:
    python teacher/collect_ant_preferences.py \\
        --config configs/seals_ant.yaml \\
        --seed 0 \\
        --output_dir outputs/preferences/seals_ant/seed0
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import seals  # noqa: F401  — registers seals/* namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imitation.data.serialize as data_serialize
from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from utils.config import resolve_config, save_resolved_config
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


ENV_NAME = "seals/Ant-v1"


def _make_noisy_callable(
    expert,
    noise_std: float,
    rng: np.random.Generator,
) -> Callable:
    """Wrap an SB3 policy into an imitation PolicyCallable that adds action noise.

    imitation's ``policy_to_callable`` expects ``(obs, state, episode_starts) ->
    (acts, state)``, so we mirror that signature.
    """

    def predict(observations, states, episode_starts) -> Tuple[np.ndarray, None]:
        acts, _ = expert.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=False,
        )
        acts = np.asarray(acts, dtype=np.float32)
        acts = acts + rng.normal(0.0, noise_std, size=acts.shape).astype(np.float32)
        return acts, None

    return predict


def collect_trajectories_by_source(
    rng: np.random.Generator,
    n_envs: int,
    expert_episodes: int,
    noisy_episodes: int,
    random_episodes: int,
    noise_std: float,
) -> "dict[str, List[TrajectoryWithRew]]":
    """Roll out each policy separately, returning trajectories keyed by source.

    Sources:
      - ``expert``: the HuggingFace PPO expert (high reward).
      - ``noisy`` : expert actions + Gaussian noise (medium reward).
      - ``random``: uniform-random actions (low reward).

    Each source contributes (at least) the requested number of episodes; the
    rollout helper may return a few extra to avoid biasing toward short episodes.
    """
    venv = make_vec_env(
        ENV_NAME,
        rng=rng,
        n_envs=n_envs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=ENV_NAME,
        venv=venv,
    )

    # (label, policy, n_episodes) — policy=None means uniform-random actions.
    sources = [
        ("expert", expert, expert_episodes),
        ("noisy", _make_noisy_callable(expert, noise_std, rng), noisy_episodes),
        ("random", None, random_episodes),
    ]
    by_source: "dict[str, List[TrajectoryWithRew]]" = {}
    for label, policy, n_eps in sources:
        if n_eps <= 0:
            continue
        trajs = list(
            rollout.rollout(
                policy,
                venv,
                rollout.make_sample_until(min_timesteps=None, min_episodes=n_eps),
                rng=rng,
            )
        )
        returns = [float(t.rews.sum()) for t in trajs]
        print(
            f"[collect_prefs] source={label:7s} episodes={len(trajs):3d} "
            f"mean_return={np.mean(returns):8.2f} "
            f"min={np.min(returns):8.2f} max={np.max(returns):8.2f}"
        )
        by_source[label] = trajs

    venv.close()
    return by_source


def build_return_labeled_pairs(
    pool: Sequence[TrajectoryWithRew],
    rng: np.random.Generator,
    fragment_length: int,
    num_pairs: int,
    discount_factor: float,
    max_size: Optional[int],
) -> "dict":
    """Fragment the pool and label each pair with the two fragments' returns.

    Instead of the binary/Bradley-Terry preference produced by
    ``SyntheticGatherer``, each comparison is labelled with ``return_a`` and
    ``return_b`` — the (discounted) ground-truth return of fragment A and B
    respectively. Downstream code can derive a preference at any temperature via
    ``sigmoid((return_a - return_b) / T)`` or train a reward model by regression.

    Returns a plain dict (no custom class, so it unpickles without imports):
        fragments1       List[TrajectoryWithRew]   length N
        fragments2       List[TrajectoryWithRew]   length N
        return_a         np.ndarray [N] float32    discounted return of fragment A
        return_b         np.ndarray [N] float32    discounted return of fragment B
        discount_factor  float
    """
    fragmenter = preference_comparisons.RandomFragmenter(rng=rng, warning_threshold=0)
    fragment_pairs = fragmenter(pool, fragment_length=fragment_length, num_pairs=num_pairs)

    if max_size is not None and len(fragment_pairs) > max_size:
        fragment_pairs = list(fragment_pairs)[-max_size:]

    frag_a, frag_b = zip(*fragment_pairs)
    return_a = np.array(
        [rollout.discounted_sum(f.rews, discount_factor) for f in frag_a],
        dtype=np.float32,
    )
    return_b = np.array(
        [rollout.discounted_sum(f.rews, discount_factor) for f in frag_b],
        dtype=np.float32,
    )

    return {
        "fragments1": list(frag_a),
        "fragments2": list(frag_b),
        "return_a": return_a,
        "return_b": return_b,
        "discount_factor": float(discount_factor),
    }


def run(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.output_dir)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    ds_cfg = cfg.get("dataset", {})
    pref_cfg = cfg.get("preferences", {})

    n_envs = int(ds_cfg.get("n_envs", 8))

    # How many episodes to gather, and the policy mixture that produces them.
    total_episodes = int(pref_cfg.get("total_episodes", 120))
    expert_frac = float(pref_cfg.get("expert_frac", 0.5))
    noisy_frac = float(pref_cfg.get("noisy_frac", 0.3))
    random_frac = float(pref_cfg.get("random_frac", 0.2))
    noise_std = float(pref_cfg.get("noise_std", 0.5))

    expert_episodes = int(round(total_episodes * expert_frac))
    noisy_episodes = int(round(total_episodes * noisy_frac))
    random_episodes = int(round(total_episodes * random_frac))

    # Fragmentation + return-labelling knobs.
    fragment_length = int(pref_cfg.get("fragment_length", 50))
    num_pairs = int(pref_cfg.get("num_pairs", 2000))
    discount_factor = float(pref_cfg.get("discount_factor", 0.99))
    max_size = pref_cfg.get("max_size", None)
    max_size = int(max_size) if max_size is not None else None

    # Per-source comparison budget — scale total num_pairs by each episode fraction.
    pairs_by_source = {
        "expert": int(round(num_pairs * expert_frac)),
        "noisy": int(round(num_pairs * noisy_frac)),
        "random": int(round(num_pairs * random_frac)),
    }

    print(
        f"[collect_prefs] env={ENV_NAME}, seed={args.seed}, n_envs={n_envs}\n"
        f"  episodes: expert={expert_episodes} noisy={noisy_episodes} "
        f"random={random_episodes} (noise_std={noise_std})\n"
        f"  pairs:    expert={pairs_by_source['expert']} "
        f"noisy={pairs_by_source['noisy']} random={pairs_by_source['random']}\n"
        f"  labels: return_a/return_b  fragment_length={fragment_length} "
        f"discount={discount_factor}"
    )

    by_source = collect_trajectories_by_source(
        rng=rng,
        n_envs=n_envs,
        expert_episodes=expert_episodes,
        noisy_episodes=noisy_episodes,
        random_episodes=random_episodes,
        noise_std=noise_std,
    )

    obs_dim = action_dim = None
    source_summaries: "dict[str, dict]" = {}

    # Build and save one self-contained return-labelled dataset per source.
    # Within a source there is still reward spread (e.g. expert episode returns
    # vary widely), so return_a vs return_b carries a usable training signal.
    for source, trajs in by_source.items():
        n_pairs = pairs_by_source.get(source, 0)
        if n_pairs <= 0 or len(trajs) == 0:
            continue

        if obs_dim is None:
            obs_dim = int(np.asarray(trajs[0].obs).shape[-1])
            action_dim = int(np.asarray(trajs[0].acts).shape[-1])

        data = build_return_labeled_pairs(
            pool=trajs,
            rng=rng,
            fragment_length=fragment_length,
            num_pairs=n_pairs,
            discount_factor=discount_factor,
            max_size=max_size,
        )
        return_a = data["return_a"]
        return_b = data["return_b"]
        n_comp = len(return_a)
        ep_returns = np.array([float(t.rews.sum()) for t in trajs], dtype=np.float32)
        # Fraction of pairs where A's return strictly beats B's (vs ties).
        frac_a_wins = float(np.mean(return_a > return_b))
        print(
            f"[collect_prefs] {source:7s} → {n_comp} comparisons, "
            f"return_a mean={return_a.mean():.2f} return_b mean={return_b.mean():.2f} "
            f"(A>B in {frac_a_wins:.2f})"
        )

        # Self-describing pickle (fragments + return labels) and a lightweight
        # npz with just the numeric labels for quick analysis.
        with open(out_dir / f"preferences_{source}.pkl", "wb") as fh:
            pickle.dump(data, fh)
        np.savez(
            out_dir / f"returns_{source}.npz",
            return_a=return_a,
            return_b=return_b,
        )
        data_serialize.save(str(out_dir / f"trajectories_{source}.pkl"), trajs)

        source_summaries[source] = {
            "n_trajectories": len(trajs),
            "n_comparisons": int(n_comp),
            "episode_return_mean": float(ep_returns.mean()),
            "episode_return_std": float(ep_returns.std()),
            "return_a_mean": float(return_a.mean()),
            "return_b_mean": float(return_b.mean()),
            "frac_a_wins": frac_a_wins,
        }

    save_json(
        {
            "env_id": ENV_NAME,
            "seed": args.seed,
            "config": args.config,
            "task": "preference_comparisons",
            "label_type": "returns",
            "label_keys": ["return_a", "return_b"],
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "sources": list(source_summaries.keys()),
            "source_summaries": source_summaries,
            "pool_episodes": {
                "expert": expert_episodes,
                "noisy": noisy_episodes,
                "random": random_episodes,
            },
            "pairs_by_source": pairs_by_source,
            "noise_std": noise_std,
            "fragment_length": fragment_length,
            "num_pairs": num_pairs,
            "discount_factor": discount_factor,
            "max_size": max_size,
            "expert_source": "ppo-huggingface/HumanCompatibleAI",
            "feature_names": [f"obs_{j:02d}" for j in range(obs_dim or 0)],
            "action_names": [f"act_{j}" for j in range(action_dim or 0)],
        },
        str(out_dir / "metadata.json"),
    )
    print(
        f"[collect_prefs] Done. Wrote per-source files for: "
        f"{', '.join(source_summaries.keys())}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect a seals/Ant-v1 offline preference-comparisons dataset."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
