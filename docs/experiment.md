# Experiments

This document specifies the experimental design for feature attribution / ranking applied
to imitation learning. It complements the paper's `paper/sections/experiment.tex` and the
D4RL write-up in `docs/d4rl_results_analysis.md`. Where the plan describes capabilities
that are not yet implemented in code, this is flagged explicitly so the doc doubles as a
to-do map.

## 1. Overview

Every ranking method is decomposed into **two orthogonal axes**:

1. **Predictive-power computation strategy** — how the value `ν(S)` of a feature coalition
   `S` is estimated (black-box model padding / masking, ridge regression, or model
   retraining).
2. **Feature-set search strategy** — how subsets `S ⊆ F` are explored to attribute credit
   to each feature (weighted coalition regression, permutation sampling, nearest-neighbor,
   greedy local refinement).

Each ranker emits a shared `ranking.csv`; the top-`k` features then feed a downstream
imitation learner (BC / preference learning / IRL) across several benchmarks. The driver
is `scripts/run_unified_feature_pipeline.sh` (sweeps `env × seed × space × selector`), and
feature ordering + selector dispatch are centralized in `utils/feature_utils.py`. All runs
use the `il` conda env, launched from the repo root.

## 2. Baselines (ranking methods)

| ranking method | predict-power computation | feature-set search | entry point | status |
| --- | --- | --- | --- | --- |
| kernelShap | model padding (marginal imputation) | weighted coalition regression (SHAP `KernelExplainer`),  `nsamples = 2 * X.shape[1] + 2048` | `explain/kernelshap.py` | implemented |
| mi | kNN entropy estimator | 3-nearest-neighbor mutual information (`mutual_info_*`, `n_neighbors=3`) | `utils/feature_utils.py` dispatch | **selector only — no `ranking.csv`** |
| sage | model padding (`sage.MarginalImputer`) | permutation sampling (Shapley aggregation) | `explain/sage_rank.py` | implemented |
| mci_nn | model retraining (per-subset `imitation.bc.BC`) | permutation sampling of random subsets | `explain/mci_rank_nn.py` | implemented |
| mci_kernel | ridge regression on RFF/HDC encoding (closed form) | permutation sampling **(+ greedy local refinement — planned)** | `explain/mci_rank.py`, `explain/hdc_encoder.py` | **greedy refinement not yet coded** |

**Method notes.**

- **kernelShap** — absent features are filled by marginal imputation from a background set
  (`explain/imputers.py`); the SHAP `KernelExplainer` fits per-feature attributions by
  weighted regression over sampled coalitions. Per-sample local values are averaged into a
  global ranking by `explain/global_rank.py`. Key knobs: `--nsamples`, `--background_size`,
  `--explain_size`. Target is the smooth `chosen_action_prob` (discrete) or the continuous
  action vector. Output column: `mean_abs_shap`.
- **mi** — estimates each feature's mutual information with the action label using
  scikit-learn's kNN-based estimator (`mutual_info_classif` / `mutual_info_regression`,
  `n_neighbors=3`). It is currently dispatched as a **selector** during student training
  (pick top-`k` by MI), not as a ranker that writes `ranking.csv`. Promoting it to a full
  ranker is future work.
- **sage** — global Shapley importance via `sage.MarginalImputer` (same padding family as
  kernelShap) plus permutation sampling; values sum to the model's total predictive power.
  Natively global (no per-sample stage), much cheaper than SHAP. Knobs: `--n_permutations`
  (0 = auto-converge), `--convergence_thresh`, `--background_size`. Output column:
  `sage_value` (ranked by `|sage_value|`, aliased to `mean_mci`).
- **mci_nn** — the faithful but expensive baseline: for each feature `i`, sample random
  subsets `S ⊆ F\{i}`, retrain a BC policy (`imitation.bc.BC` over an SB3
  `ActorCriticPolicy`) on `S` and on `S ∪ {i}`, and measure the marginal gain in
  variance-explained. Knobs: `--n_perms`, `--aggregation {mean,max}`, `--epochs`,
  `--hidden`, `--max_samples`. Output column: `mean_mci`.
- **mci_kernel (MCI-HDC)** — replaces retraining with a closed-form ridge fit on a
  Random-Fourier-Feature / HDC encoding of each feature (`explain/hdc_encoder.py`):
  `ν(S) = Var(y) − MSE(S)` where `w_S = (Z_Sᵀ Z_S + λI)⁻¹ Z_Sᵀ y`. Subsets are explored by
  permutation sampling; the planned **greedy local refinement** of the best subset per
  feature is not yet implemented. Knobs: `--rff_dim`, `--bandwidth`, `--lambda_`,
  `--n_perms`, `--max_samples`. Requires only the dataset (no student model). Output
  column: `mean_mci`.

**Shared `ranking.csv` schema** — `feature_index, feature_name, mean_mci` (or
`mean_abs_shap`), `rank`. The common schema makes every ranker a drop-in input to the
top-`k` selectors used by the downstream learners.

**Deltas vs. the plan.** (1) `mci_kernel` greedy local refinement is designed but not yet
coded — current search is permutation sampling only. (2) `mi` is currently a feature
selector, not a ranker producing `ranking.csv`.

## 3. Datasets / learning settings

### 3.1 Behavior cloning (BC)

Data source: `(state, action)` transitions in `dataset.npz`.

- **Discrete — Taxi-v3** (`envs/noisy_taxi_wrapper.py`): integer state decoded to the
  oracle vector `[taxi_row, taxi_col, passenger_loc, destination]` (indices 0–3), with
  appended noise features `z_0 … z_{noise_dim-1}`. Schema: `X_{train,val,test}` `[N, D]`,
  `y_*` `[N]` action indices, `p_train` `[N, 6]` teacher softmax. Trained by
  `student/train_student.py`.
- **Continuous — seals/Ant-v1 and D4RL kitchen/pen** (`student/train_bc_continuous.py`,
  loaded via `teacher/collect_ant_expert_data.py` / `teacher/load_d4rl_dataset.py`):
  `X` `[N, obs_dim]`, `y` `[N, action_dim]` continuous actions, plus `next_X`, `dones`,
  `rewards`. Uses `imitation.bc.BC` over an SB3 `ActorCriticPolicy`.
- **Selectors** (`--selector`): `shap` (needs `--ranking_path`), `oracle` (first `k` of
  the oracle indices), `mi`, `random`, `full` (upper bound).

### 3.2 Preference learning (PL)

Data source: `(fragment_A, fragment_B, return_a, return_b, preference_label)` with
**50-step fragments** (`fragment_length: 50` in the `preferences:` block of
`configs/seals_ant.yaml`). Collected by `teacher/collect_ant_preferences.py`, which rolls
out three policy sources — expert PPO, noisy expert, random — and per source runs a
`RandomFragmenter` + `SyntheticGatherer` to produce:

- `preferences_{source}.pkl` — `fragments1`, `fragments2`, `return_a`, `return_b`,
  `discount_factor`.
- `returns_{source}.npz` — lightweight `return_a`, `return_b` (and `preference` once
  labeled).

The binary preference label is injected post-hoc by `teacher/add_preference_labels.py`:
`1.0` if `return_a > return_b`, `0.0` if `<`, `0.5` on a tie. Environment: seals/Ant-v1.

### 3.3 Inverse RL (IRL)

Data source: trajectory transitions `(obs, action, next_obs, reward, done)`. Expert demos
collected by `teacher/collect_ant_expert_data.py` (HuggingFace PPO expert, flattened via
`imitation.data.rollout`) into `dataset.npz` with `obs/acts/next_obs/dones/rews`. The
adversarial pipeline (AIRL/GAIL, configurable via the `irl:` block of
`configs/seals_ant.yaml`) runs through `scripts/run_ant_irl.sh`; the latent-feature
variant is `scripts/run_ant_latent_irl.sh` (reuses stage-1 dataset + full BC, adds
`--latent_mode`). Environment: seals/Ant-v1.

### 3.4 Benchmarks

| benchmark | type | obs / action dim | oracle features | configs | status |
| --- | --- | --- | --- | --- | --- |
| Taxi-v3 | discrete, toy + noise | `4 + noise_dim` / 6 | indices 0–3 | `taxi_clean`, `taxi_noise{4,8,16}`, `taxi_noise8_{bernoulli,correlated,laplace,uniform,state_correlated,mixed}`, `taxi_noise4_mixed` | implemented |
| seals/Ant-v1 | continuous, MuJoCo | 29 / 8 | none declared | `configs/seals_ant.yaml` | implemented |
| Walker / other MuJoCo | continuous, MuJoCo | — | — | — | **planned (only Ant wired up)** |
| D4RL kitchen-complete-v2 | continuous, offline | 60 / 9 | `qpos_0..8` (0–8) | `configs/kitchen_complete.yaml` | implemented |
| D4RL pen-{human,cloned,expert}-v1 | continuous, offline | 45 / 24 | hand joints (0–23) | `configs/pen_{human,cloned,expert}.yaml` | implemented |

Taxi noise variants cover Gaussian, Bernoulli, uniform, Laplace, correlated, and
state-correlated noise (plus mixed). D4RL HDF5 downloads are cached under
`outputs/d4rl_cache/`.

## 4. Evaluation metrics

### 4.1 BC — action-prediction accuracy *(implemented)*

- Offline (`eval/eval_offline.py`): `accuracy`, `macro_f1` (`utils/metrics.py`), optional
  `kl_to_teacher`. Continuous variant (`eval/eval_offline_continuous.py`): `mse`, `mae`,
  `cosine_sim`.
- Online (`eval/eval_online.py`, `eval/eval_online_continuous.py`): `avg_return`,
  `std_return`, `success_rate`, episode length.

### 4.2 IRL — state-visitation KL *(NOT implemented — gap)*

The plan calls for the KL distance between the state-visitation distribution of the target
policy and the soft-optimal policy. **No such metric exists in the codebase.** Current IRL
evaluation (`eval/eval_online_irl.py`) reports only `mean_return`, `std_return`,
`success_rate`, `mean_length` from deterministic rollouts. The state-visitation KL must be
built.

### 4.3 PL — preference-prediction accuracy *(NOT implemented — gap)*

The plan calls for preference-prediction accuracy. **No preference model is trained or
evaluated.** Today the repo only collects preference data and visualizes fragment-return
distributions (`eval/plot_preference_scores.py`). A preference predictor trained on
selected features and a held-out accuracy metric must be added.

### 4.4 Implemented vs. gaps

| setting | metric (plan) | status |
| --- | --- | --- |
| BC | action-prediction accuracy | implemented (offline + online) |
| IRL | state-visitation KL (target vs. soft-optimal) | **gap** — only online return/success exist |
| PL | preference-prediction accuracy | **gap** — only data collection + histograms exist |

## 5. Pipeline & reproduction

The single driver is `scripts/run_unified_feature_pipeline.sh`:

```bash
bash scripts/run_unified_feature_pipeline.sh \
  --envs "kitchen_complete seals_ant" \
  --seeds "0 1 2" \
  --spaces "state latent" \
  --selectors "mci_hdc mci_nn sage kernelshap random full" \
  --output_root outputs/unified
```

Outputs land under
`outputs/unified/{env}/seed{N}/{space}/{rankings|students}/{selector}/`. Per-setting
scripts: `scripts/run_d4rl_mci.sh` (D4RL ranking), `scripts/run_ant_irl.sh` /
`scripts/run_ant_latent_irl.sh` (AIRL), `scripts/run_ant_preferences.sh` (preference
collection). Run everything from the repo root in the `il` conda env.

## Future work (gaps identified above)

1. IRL state-visitation distribution KL metric.
2. PL preference-prediction model + accuracy metric.
3. `mci_kernel` greedy local refinement of sampled subsets.
4. Promote `mi` from selector to full ranker emitting `ranking.csv`.
5. Wire up additional MuJoCo benchmarks (e.g. Walker) beyond seals/Ant-v1.
