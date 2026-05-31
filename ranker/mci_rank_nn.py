"""Marginal Contribution Importance (MCI) feature ranking

Adapts the MCI-NN method (Catav et al., ICML 2021) so that the per-subset
predictive-power evaluation uses the model from the `imitation` package (`imitation.algorithms`).

Input:
    - dataset.npz — pre-collected dataset with keys:
        - X: (N, d_x) array of N samples of d_x-dimensional observations.
        - Y: (N, d_y) array of N samples of d_y-dimensional expert actions.
        - feature_names: (d_x,) array of strings with feature names.
    - evaluator

Method:
    For each observation feature i, sample n_perms random subsets
     S ⊆ F\\{i} and compute:

         Î(i) = max_S [ν(S∪{i}) − ν(S)]

     where ν(S) is the predictive
     power of subset S from evaluator (sum across action dims). For every (S, S∪{i}) pair
     two evaluations are trained from scratch on the
     reduced-dimensional clean input via `evaluator`. No masking and no
     imputation — each policy receives a clean subset of features.


Output:
    ranking.csv     — feature_index, feature_name, mean_mci, rank
    mci_scores.json — [d_y, d_x] array of per-feature MCI scores
    metadata.json — dataset path, score_path, n_perms, ranker_name, ranking time. 

Usage:
    python explain/mci_rank_nn.py \\
        --seed 0 \\
        --dataset_path outputs/datasets/seals_ant/seed0/dataset.npz \\
        --output_dir outputs/rankings_mci_nn/seals_ant/seed0
"""

from time import time

import numpy as np
import os
from typing import Optional, Sequence, Callable, Optional, Set, List, Dict, Tuple, Iterable
from pandas import DataFrame, Series
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.feature_utils import MultiVariateArray, UniVariateArray, context_to_key
from utils.io import save_json
from utils.rank_utils import multi_process_lst

from abc import ABC, abstractmethod


EvaluationFunction = Callable[[MultiVariateArray, UniVariateArray,
                               Optional[MultiVariateArray], Optional[UniVariateArray]], float]


def is_empty(array: MultiVariateArray):
    if isinstance(array, DataFrame) or isinstance(array, Series):
        return array.empty
    else:
        return len(array) == 0

class MciValues:

    """contain MCI values and project relevant plots from them"""

    def __init__(self,
                 mci_values: Sequence[float],
                 feature_names: Sequence[str],
                 contexts: Sequence[Tuple[str, ...]],
                 additional_values: Optional[Sequence[Sequence[float]]] = None,
                 additional_contexts: Optional[Sequence[Sequence[Tuple[str, ...]]]] = None,
                 shapley_values: Optional[Sequence[float]] = None):
        """
        :param mci_values: array of MCI values for each feature
        :param feature_names: array of features names (corresponds to the values)
        :param contexts: array of argmax contribution contexts for each feature (corresponds to the values)
        :param additional_values: placeholder for additional MCI values per feature (for non optimal values)
        :param additional_contexts: placeholder for additional MCI contexts per feature (for non optimal values)
        :param shapley_values: shapley values for comparison (optional)
        """
        self.mci_values = mci_values
        self.feature_names = feature_names
        self.contexts = contexts
        self.additional_values = additional_values
        self.additional_contexts = additional_contexts
        self.shapley_values = shapley_values

    @classmethod
    def create_from_tracker(cls, tracker: "ContributionTracker", feature_names: Sequence[str]):
        return cls(mci_values=tracker.max_contributions,
                   feature_names=feature_names,
                   contexts=tracker.argmax_contexts,
                   additional_values=tracker.all_contributions,
                   additional_contexts=tracker.all_contexts,
                   shapley_values=tracker.avg_contributions)

    def plot_values(self, plot_contexts: bool = False, score_name="MCI", file_path: Optional[str] = None):
        """Simple bar plot for MCI values per feature name"""
        score_features = sorted([(score, feature, context) for score, feature, context
                                 in zip(self.mci_values, self.feature_names, self.contexts)],
                                key=lambda x: x[0])

        if plot_contexts:
            features = [f"{f} ({', '.join(context)})" for score, f, context in score_features]
        else:
            features = [f for score, f, context in score_features]
        plt.barh(y=features, width=[score for score, f, context in score_features])
        plt.title(f"{score_name} feature importance")
        plt.xlabel(f"{score_name} value")
        plt.ylabel("Feature name")

        if file_path:
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_shapley_values(self, file_path: Optional[str] = None):
        score_features = sorted([(score, feature) for score, feature
                                 in zip(self.shapley_values, self.feature_names)],
                                key=lambda x: x[0])
        features = [f for score, f in score_features]
        plt.barh(y=features, width=[score for score, f in score_features])
        plt.title(f"Shapley feature importance")
        plt.xlabel(f"Shapley value")
        plt.ylabel("Feature name")
        if file_path:
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def results_dict(self) -> dict:
        results = {
            "feature_names": self.feature_names,
            "mci_values": self.mci_values,
            "contexts": self.contexts,
            "shapley_values": self.shapley_values
        }
        return results
    


class ContributionTracker:

    def __init__(self, n_features: int, track_all: bool = False):
        """
        :param n_features: number of features to track contributions for
        :param track_all: if true, saves all observed contributions and not only max per feature
        """
        self._n_features = n_features
        self.track_all = track_all

        self.max_contributions = [0.0]*self._n_features
        self.sum_contributions = [0.0]*self._n_features
        self.n_contributions = [0.0]*self._n_features
        self.argmax_contexts = [set() for _ in range(self._n_features)]

        self.all_contributions = [[] for _ in range(self._n_features)]
        self.all_contexts = [[] for _ in range(self._n_features)]

    def update_value(self, feature_idx: int, contribution: float, context: Set[str], noise_tolerance: float = 0.0):
        if contribution > self.max_contributions[feature_idx] + noise_tolerance:
            self.max_contributions[feature_idx] = contribution
            self.argmax_contexts[feature_idx] = context

        self.n_contributions[feature_idx] += 1
        self.sum_contributions[feature_idx] += contribution

        if self.track_all:
            self.all_contributions[feature_idx].append(contribution)
            self.all_contexts[feature_idx].append(context)

    def update_tracker(self, tracker: 'ContributionTracker'):
        if self.track_all and tracker.track_all:
            for feature_idx, (f_conts, f_contexts) in enumerate(zip(tracker.all_contributions, tracker.all_contexts)):
                for cont, context in zip(f_conts, f_contexts):
                    self.update_value(feature_idx, cont, context)
        else:
            for feature_idx, (cont, context) in enumerate(zip(tracker.max_contributions, tracker.argmax_contexts)):
                self.update_value(feature_idx, cont, context)

    def save_to_file(self, feature_names: List[str], file_path: str):
        state = {}

        state["max_contributions"] = self.max_contributions
        state["sum_contributions"] = self.sum_contributions
        state["n_contributions"] = self.n_contributions
        state["argmax_contexts"] = [list(c) for c in self.argmax_contexts]
        state["feature_names"] = feature_names

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @staticmethod
    def load_from_file(file_path: str, feature_names: List[str]) -> 'ContributionTracker':
        with open(file_path) as f:
            state = json.load(f)

        assert feature_names == state["feature_names"]
        tracker = ContributionTracker(n_features=len(feature_names), track_all=False)
        tracker.max_contributions = state["max_contributions"]
        tracker.sum_contributions = state["sum_contributions"]
        tracker.n_contributions = state["n_contributions"]
        tracker.argmax_contexts = state["argmax_contexts"]
        return tracker

    @property
    def avg_contributions(self):
        return [s/max(n, 1) for s, n in zip(self.sum_contributions, self.n_contributions)]
    

class BaseEstimator(ABC):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_processes: int = 5,
                 chunk_size: int = 20,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False):
        """
        :param evaluator: features subsets evaluation function
        :param n_processes: number of process to use
        :param chunk_size: max number of subsets to evaluate at each process at a time
        :param max_context_size: max feature subset size to evaluate as feature context
        :param noise_confidence: PAC learning error bound confidence (usually noted as delta for PAC)
        :param noise_factor: a scalar to multiple by the PAC learning error bound
        :param track_all: a bool indicates whether to save all observed contributions and not just max
        """

        self._evaluator = evaluator
        self._n_processes = n_processes
        self._chunk_size = chunk_size
        self._max_context_size = max_context_size
        self._noise_factor = noise_factor
        self._noise_confidence = noise_confidence
        self._track_all = track_all

    @abstractmethod
    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:

        raise NotImplementedError()

    def _multiprocess_eval_subsets(self,
                                   subsets: List[Iterable[str]],
                                   x: DataFrame,
                                   y: UniVariateArray,
                                   x_test: Optional[DataFrame],
                                   y_test: Optional[UniVariateArray] = None) -> Dict[Tuple[str, ...], float]:
        subsets = list(set(context_to_key(c) for c in subsets))  # remove duplications

        evaluations: Dict[Tuple[str, ...], float] = {}
        pbar = tqdm(total=len(subsets))
        for eval_results in multi_process_lst(lst=subsets, apply_on_chunk=self._evaluate_subsets_chunk,
                                              chunk_size=self._chunk_size, n_processes=self._n_processes,
                                              args=(x, y, x_test, y_test)):
            evaluations.update(eval_results)
            pbar.update(len(eval_results))
        return evaluations

    def _evaluate_subsets_chunk(self,
                                subsets: List[Iterable[str]],
                                x: DataFrame,
                                y: UniVariateArray,
                                x_test: Optional[DataFrame],
                                y_test: Optional[UniVariateArray]) -> Dict[Tuple[str], float]:
        evaluations: Dict[Tuple[str, ...], float] = {}
        for s in subsets:
            evaluations[context_to_key(s)] = self._evaluator(x[list(s)], y, x_test[list(s)] if x_test is not None
            else None, y_test)
        return evaluations

class PermutationSampling(BaseEstimator):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_permutations: int,
                 out_dir: Optional[str] = None,
                 n_processes: int = 1,
                 chunk_size: int = 2**8,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False,
                 permutations_batch_size: int = 200):

        super(PermutationSampling, self).__init__(evaluator=evaluator,
                                                  n_processes=n_processes,
                                                  chunk_size=chunk_size,
                                                  max_context_size=max_context_size,
                                                  noise_confidence=noise_confidence,
                                                  noise_factor=noise_factor,
                                                  track_all=track_all)
        self._n_permutations = n_permutations
        self._out_dir = out_dir
        self._n_permutations_done = 0
        self._permutations_batch_size = permutations_batch_size

    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:
        if not isinstance(x, DataFrame):
            assert x is not None, "feature names must be provided if x is not a dataframe"
            x = DataFrame(x, columns=feature_names)
            if x_test is not None and not isinstance(x_test, DataFrame):
                x_test = DataFrame(x_test, columns=feature_names)

        feature_names = list(x.columns)
        if self._out_dir and os.path.isdir(self._out_dir) and len(os.listdir(self._out_dir)) > 0:
            files = [int(f.replace(".json", "")) for f in os.listdir(self._out_dir) if f.endswith(".json")]
            self._n_permutations_done = sorted(files)[-1]
            most_updated_file = os.path.join(self._out_dir, f"{self._n_permutations_done}.json")
            print(f"loading results checkpoint from {most_updated_file}")
            tracker = ContributionTracker.load_from_file(most_updated_file, feature_names)
        else:
            if self._out_dir and not os.path.isdir(self._out_dir):
                os.mkdir(self._out_dir)
            tracker = ContributionTracker(n_features=len(feature_names), track_all=self._track_all)

        while self._n_permutations > self._n_permutations_done:
            np.random.seed(self._n_permutations_done)
            perm_sample_size = min(self._permutations_batch_size, self._n_permutations - self._n_permutations_done)
            permutations_sample = [list(np.random.permutation(feature_names)) for _ in range(perm_sample_size)]
            suffixes = [p[:i] for p in permutations_sample for i in range(len(p)+1)]
            evaluations = self._multiprocess_eval_subsets(suffixes, x, y, x_test, y_test)

            for p in tqdm(permutations_sample):
                for i in range(len(p)):
                    suffix = p[:i]
                    suffix_with_f = p[:i+1]
                    contribution = evaluations[context_to_key(suffix_with_f)] - evaluations[context_to_key(suffix)]
                    tracker.update_value(feature_idx=feature_names.index(p[i]),
                                         contribution=contribution,
                                         context=set(suffix))
            self._n_permutations_done += perm_sample_size
            out_path = os.path.join(self._out_dir, f"{self._n_permutations_done}.json")
            print(f"saving results for {self._n_permutations_done} permutations into {out_path}")
            tracker.save_to_file(feature_names, out_path)
        return MciValues.create_from_tracker(tracker, feature_names)
    
def load_dataset(dataset_path: str) -> Tuple[MultiVariateArray, UniVariateArray, Sequence[str]]:
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    feature_names = data["feature_names"] if "feature_names" in data else [f"feature_{i}" for i in range(X.shape[1])]
    return X, Y, feature_names

def create_evaluator(X, Y, feature_names, evaluator_name: str, evaluator_params: dict) -> EvaluationFunction:
    if evaluator_name == "bc":
        from imitation.algorithms import bc
        from imitation.data import rollout
        from imitation.util import util

        model = bc.BC(
            observation_space=util.space_from_box(X.shape[1:]),
            action_space=util.space_from_box(Y.shape[1:]),
            **evaluator_params,
        )

        return model

    # Max Causal Entropy IRL (imitation.algorithms.mce_irl) as the per-subset
    # predictive-power evaluator. Follows the imitation MCE-IRL example:
    #   mce_partition_fh -> mce_occupancy_measures -> BasicRewardNet -> MCEIRL
    #   -> rollout.generate_trajectories -> rollout.rollout_stats
    #
    # The feature subset S selects which columns of the tabular env's
    # `observation_matrix` the reward net is allowed to see. The transition
    # dynamics, reward_matrix and the expert occupancy measures are unchanged
    # by S, so the expert side is computed once and reused for every subset.
    # The predictive power nu(S) is the mean ground-truth return achieved by the
    # policy MCEIRL recovers when its reward net only observes features in S.
    #
    # Required `evaluator_params`:
    #   env_creator   : () -> seals.base_envs.TabularModelPOMDP factory.
    # Optional `evaluator_params`:
    #   feature_names      : ordered names matching observation_matrix columns
    #                        (defaults to the env's natural column order).
    #   rng                : np.random.Generator (defaults to seed 0).
    #   discount           : MCE discount (default 1.0).
    #   n_eval_timesteps   : min timesteps to sample for nu(S) (default 5000).
    #   reward_net_kwargs  : passed to BasicRewardNet (default {"hid_sizes": [256]}).
    #   mceirl_kwargs      : passed to MCEIRL (default log_interval/optimizer lr).
    elif evaluator_name == "irl":
        from imitation.algorithms.mce_irl import (
            MCEIRL,
            mce_occupancy_measures,
            mce_partition_fh,
        )
        from imitation.data import rollout
        from imitation.rewards import reward_nets
        from seals import base_envs
        from stable_baselines3.common.vec_env import DummyVecEnv

        params = dict(evaluator_params or {})
        env_creator = params["env_creator"]
        rng = params.get("rng") or np.random.default_rng(0)
        discount = params.get("discount", 1.0)
        n_eval_timesteps = params.get("n_eval_timesteps", 5000)
        reward_net_kwargs = params.get(
            "reward_net_kwargs",
            {"hid_sizes": [256]},
        )
        mceirl_kwargs = params.get(
            "mceirl_kwargs",
            {"log_interval": 250, "optimizer_kwargs": {"lr": 0.01}},
        )

        # Reference env (full feature set) — defines dynamics and the ground-truth
        # reward, hence the expert policy / occupancy that all subsets imitate.
        full_env = env_creator()
        feature_names = list(
            params.get(
                "feature_names",
                [f"feature_{i}" for i in range(full_env.observation_matrix.shape[1])],
            )
        )
        name_to_col = {name: i for i, name in enumerate(feature_names)}

        # Expert side is subset-independent — compute once and reuse.
        _, _, expert_pi = mce_partition_fh(full_env, discount=discount)
        _, expert_om = mce_occupancy_measures(
            full_env, pi=expert_pi, discount=discount
        )

        def _subset_env(columns):
            """Clone `full_env` with observations restricted to `columns`."""
            obs = full_env.observation_matrix
            if len(columns) == 0:
                # nu(emptyset): a single constant feature -> constant reward ->
                # MCEIRL recovers a (near-)uniform policy as the baseline.
                sub_obs = np.zeros((obs.shape[0], 1), dtype=obs.dtype)
            else:
                sub_obs = obs[:, columns]
            return base_envs.TabularModelPOMDP(
                transition_matrix=full_env.transition_matrix,
                observation_matrix=sub_obs,
                reward_matrix=full_env.reward_matrix,
                horizon=full_env.horizon,
                initial_state_dist=full_env.initial_state_dist,
            )

        # The feature subset S is carried by the column names of X.
        columns = [name_to_col[name] for name in X.columns]
        sub_env = _subset_env(columns)

        reward_net = reward_nets.BasicRewardNet(
            sub_env.observation_space,
            sub_env.action_space,
            use_action=False,
            use_done=False,
            use_next_state=False,
            **reward_net_kwargs,
        )

        mce_irl = MCEIRL(
            expert_om,
            sub_env,
            reward_net,
            rng=rng,
            discount=discount,
            **mceirl_kwargs,
        )
        
        return mce_irl

    else:
        raise ValueError(
            f"Unknown evaluator_name {evaluator_name!r}; expected 'bc' or 'irl'."
        )

    return evaluator

def run(args):
    # preprocess data and create ranker
    X, Y, feature_names = load_dataset(args.dataset_path)
    evaluator = create_evaluator(X, Y, feature_names, args.evaluator_name, args.evaluator_params)
    ranker = PermutationSampling(evaluator=evaluator, n_permutations=args.n_permutations, out_dir=args.output_dir, n_processes=args.n_processes,
                                  chunk_size=args.chunk_size, max_context_size=args.max_context_size, noise_confidence=args.noise_confidence, noise_factor=args.noise_factor, track_all=args.track_all, permutations_batch_size=args.permutations_batch_size)
    
    start_time = time()
    mci_values = ranker.mci_values(X, Y, feature_names=feature_names)
    ranking_time = time() - start_time


    mci_values.plot_values(file_path=os.path.join(args.output_dir, "mci_values.png"))
    if mci_values.shapley_values is not None:
        mci_values.plot_shapley_values(file_path=os.path.join(args.output_dir, "shapley_values.png"))
    
    # save mci_values as json with keys: feature_names, mci_values, contexts, shapley_values
    save_json(mci_values.results_dict(), os.path.join(args.output_dir, "mci_values.json"))
    
    # save ranking.csv with columns: feature_index, feature_name, mean_mci, rank
    ranking_df = DataFrame({
        "feature_index": list(range(len(feature_names))),
        "feature_name": feature_names,
        "mean_mci": mci_values.mci_values,
    }).sort_values("mean_mci", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1
    ranking_df.to_csv(os.path.join(args.output_dir, "ranking.csv"), index=False)

    # save metadata.json with dataset path, score path, n_permutations, ranker name, ranking time
    metadata = {
        "dataset_path": args.dataset_path,
        "score_path": os.path.join(args.output_dir, "mci_values.json"), 
        "n_permutations": args.n_permutations,
        "ranker_name": "PermutationSampling",
        "ranking_time": time.time(),
    }

    save_json(metadata, os.path.join(args.output_dir, "metadata.json"))
