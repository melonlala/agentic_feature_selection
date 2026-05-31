# SVERL-Style Feature Distillation for Imitation Reinforcement Learning

## Motivation

Standard imitation learning trains a student on the teacher's full observation
vector. However, many features may be irrelevant or noisy. This project asks:

> **Can importance-ranked features from a behavior-explanation model recover the
> teacher policy better than random features, and approach oracle performance?**

We follow the SVERL (Shapley Values for Explaining Reinforcement Learning)
design philosophy: explain *chosen-action probability*, not expected returns
or hard argmax. This produces smoother, more reliable attributions.

We implement and compare four feature-ranking methods — **SHAP**, **SAGE**,
**MCI-NN**, and **MCI-HDC** — to evaluate how different importance definitions
(Shapley weighted-average vs. marginal-contribution maximum) and subset
evaluation strategies (fixed model with masking vs. per-subset retraining)
affect downstream imitation performance.

---

## Method Overview

### Pipeline

```
Taxi-v3 (integer obs)
       │
       ▼
DQN Teacher (SB3)           trained on clean 4-feature vector
       │
       ▼
NoisyTaxiWrapper             [row, col, pass_loc, dest, z_0, ..., z_{n-1}]
       │
       ▼
Collect dataset              (obs, action, Q, probs, chosen_prob)
       │
       ├────────────────┬────────────────┬────────────────┐
       ▼                ▼                ▼                ▼
   KernelSHAP         SAGE           MCI-NN           MCI-HDC
   (surrogate MLP)   (permutation)  (retrain MLP)  (ridge on RFF)
       │                │                │                │
       ▼                ▼                ▼                ▼
   ranking.csv       ranking.csv     ranking.csv     ranking.csv
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                                │
                                ▼
                     top-k feature subset
                                │
                                ▼
              BC Student (MLP on masked obs)
                                │
                                ▼
              Offline + Online Evaluation
         (accuracy / F1 / return / success rate)
```

### Explanation Target: Behavior, Not Outcome

All four explanation methods share the same regression target: the scalar
`P(chosen_action | obs)` — the softmax probability the teacher assigns to
the action it actually selects.

- **Why not argmax?** The greedy action is a step function; attributions on a
  step function are noisy and unstable.
- **Why not expected return?** We want to understand *what drives the policy's
  decisions*, not what makes the environment reward high.

This follows the SVERL design: attributions describe the policy's reasoning,
enabling compact policy-relevant feature selection.

---

## Explanation Methods

All four methods consume the same dataset and produce the same output schema —
a `ranking.csv` with columns `(feature_index, feature_name, score, rank)` — so
all downstream training and evaluation scripts are agnostic to which ranker
was used.

### 1. SHAP (KernelSHAP)

A **local, per-sample** attribution method grounded in cooperative game theory.

A lightweight MLP surrogate is trained to predict `P(a | x)`. KernelExplainer
then estimates each feature's Shapley value by evaluating the surrogate on
coalitions of features, with held-out features replaced by background samples.
For each explained sample, a weighted least-squares problem is solved over
coalition evaluations to recover `d` local Shapley values. Global importance
is the mean absolute local attribution:

```
φ_j^SHAP = (1/N) Σ_i |SHAP_j(x_i)|
```

**Cost:** scales with `N_explain × C_coalitions × M_background`. In our
noise-8 setting (d=12), explaining 500 samples ≈ 207M surrogate evaluations,
all CPU-bound (pure NumPy).

**Properties:** efficiency (sums to total), symmetry, dummy. Credit dilution
under correlated features.

### 2. SAGE (Shapley Additive Global importancE)

A **global** importance method that applies Shapley values directly to a
set-function measuring predictive power, bypassing per-sample explanation.

Define model-based predictive power of feature subset S as:

```
v_f(S) = E[ℓ(f_∅, Y)] − E[ℓ(f_S(X_S), Y)]
```

where `f_S` marginalizes out features not in S. SAGE values are the Shapley
values of this cooperative game. In practice, SAGE's sampling algorithm
enumerates random permutations of feature indices, walking through each
permutation and accumulating per-feature marginal gains. Convergence is
monitored via confidence intervals.

**Cost:** targets global importance directly — never explains individual
samples. The SAGE paper reports 2–4 orders of magnitude fewer model
evaluations than computing the same values via per-sample SHAP and averaging.

**Properties:** all Shapley axioms (efficiency, symmetry, dummy, monotonicity,
linearity). Values sum to total predictive power `v_f(D)`. Same credit-dilution
behavior as SHAP under correlated features.

### 3. MCI-NN (Neural-Network Marginal Contribution Importance)

Replaces Shapley's **weighted-average** marginal contribution with a
**maximum** marginal contribution. Following Catav et al. (ICML 2021), MCI-NN
evaluates **universal predictive power** — for each feature subset, a fresh
model is trained from scratch using only that subset's features. This means
MCI-NN **explains the data**, not any single pre-trained model.

For each feature i, we sample `n_perms` random subsets `S ⊆ F\{i}` and
compute:

```
Î(i) = max_S [ν(S ∪ {i}) − ν(S)]
```

where `ν(S)` is the **universal predictive power** of subset S:

```
ν(S) = Var(y) − MSE_S
```

and `MSE_S` is the test loss of a small MLP trained from scratch using **only
the features in S**. For every `(S, S∪{i})` pair, two separate MLPs are
trained on the corresponding feature subsets, and their test MSE is compared.
No masking or imputation is involved — each model receives a clean,
reduced-dimensional input.

**Cost:** `d × n_perms × 2` model trainings. Each training fits a small MLP
(e.g. [64, 64]) on the subset features for a fixed number of epochs. With
d=12 and n_perms=100, this is 2,400 MLP trainings. Each individual training
is fast (small model, small data, few epochs on GPU), but the aggregate cost
is higher than SHAP or SAGE which never retrain. For Taxi-v3 scale (d ≤ 12,
N ≤ 5000, 2-layer MLP), wall-clock time remains practical (minutes).

**Properties:** dummy, symmetry, **super-efficiency** (`Σ I(i) ≥ ν(F)`),
sub-additivity, **duplication invariance**. The max operator means any
sampling estimate is a lower bound that tightens with more permutations.

**Key distinction from SHAP/SAGE:** SHAP and SAGE fix one trained model and
simulate feature absence via masking (replacing held-out features with
marginal values). This evaluates *model-based* predictive power — how the
specific model degrades under input perturbation. MCI-NN retrains per subset,
evaluating *universal* predictive power — how much information the feature
subset intrinsically contains. This makes MCI-NN robust to artifacts of a
particular model's learned representations.

### 4. MCI-HDC (Hyperdimensional-Computing-Encoded MCI)

Removes the need for a pre-trained neural network by replacing the student
model with **closed-form ridge regression on Random Fourier Features (RFF)**.

Each scalar feature is encoded via RFF:

```
z_j(x_j) = sqrt(2/D_j) · cos(W_j · x_j + b_j)
```

For any subset S, the ridge solution is computed in closed form:

```
w_S* = (Z_S^T Z_S + λI)^{-1} Z_S^T y
```

MCI scores are computed identically to MCI-NN (max over sampled subsets).

**Cost:** `d × n_perms × 2` ridge solves, each `O((|S|·D_rff)² · N)`.
No iterative training, no gradient computation. RFF blocks are precomputed
and cached.

**Properties:** all MCI axioms. Distinguishing advantage: **requires no
pre-trained model at all** — operates directly on `(X, y)` dataset. The
RFF encoding approximates a Gaussian kernel, capturing non-linear
relationships that raw linear models would miss.

### Comparison Table

| Property | SHAP | SAGE | MCI-NN | MCI-HDC |
|---|---|---|---|---|
| Scope | Local → global | Global | Global | Global |
| Explains | Model | Model | Data | Data |
| Pre-trained model required | Yes (surrogate) | Yes (any) | No (retrains per subset) | No (ridge on RFF) |
| Retrains per subset | No | No | Yes (small MLP) | Yes (closed-form ridge) |
| Subset evaluation | Coalition + WLS | Marginal masking | Fresh MLP on subset features | Ridge regression on RFF |
| Aggregation | Weighted avg (Shapley) | Weighted avg (Shapley) | Max | Max |
| Scales with N_explain | Yes | No | No | No |
| GPU-batchable | No (NumPy) | Partially | Yes (batch training) | Yes (torch.linalg.solve) |
| Correlated-feature robust | Low (dilution) | Low (dilution) | High (invariant) | High (invariant) |
| Efficiency axiom | Sums to total | Sums to total | Super-efficiency | Super-efficiency |

---

## Feature Selectors

| Selector | Description |
|----------|-------------|
| `shap`   | Top-k features ranked by mean \|SHAP\| (SVERL-style behavior explanation) |
| `sage`   | Top-k features ranked by SAGE values (global Shapley on predictive power) |
| `mci_nn` | Top-k features ranked by MCI via per-subset MLP retraining (explains data) |
| `mci_hdc`| Top-k features ranked by MCI via ridge regression on RFF encodings |
| `oracle` | First k of {taxi_row, taxi_col, passenger_loc, destination} — ground-truth |
| `mi`     | Top-k features ranked by mutual information with teacher action labels |
| `random` | Uniformly random k features (baseline) |
| `full`   | All features (upper bound baseline) |

**Main hypothesis:** importance-based selectors (shap, sage, mci_nn, mci_hdc)
should significantly outperform `random` and approach `oracle`/`full`, with MCI
variants showing particular advantage when noise features are correlated with
informative ones.

---

## Repo Layout

```
sverl_feature_distill/
├── configs/            YAML configs (base + noise variants)
├── envs/               NoisyTaxiWrapper
├── teacher/            DQN training, inference wrapper, dataset collection
├── explain/
│   ├── shap_behavior.py    KernelSHAP explanation
│   ├── global_rank.py      Aggregate SHAP → ranking.csv
│   ├── sage_rank.py        SAGE explanation → ranking.csv
│   ├── mci_rank_nn.py      MCI-NN explanation → ranking.csv
│   ├── mci_rank.py         MCI-HDC explanation → ranking.csv
│   └── hdc_encoder.py      Random Fourier Feature encoder
├── student/            BC policy, distillation losses, training script
├── eval/               Offline/online evaluation, plotting
├── utils/              Config, I/O, metrics, seed, feature selection
├── scripts/            End-to-end shell scripts
└── outputs/            All experiment artifacts (auto-created)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

---

## Run Commands (Stage by Stage)

All commands are run from the `sverl_feature_distill/` directory.

### 1. Train teacher

```bash
python teacher/train_teacher.py \
    --config configs/taxi_noise4.yaml \
    --seed 0
```

### 2. Collect dataset with augmented noises

```bash
python teacher/collect_dataset.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --teacher_ckpt outputs/teachers/taxi_noise4/seed0/model.zip \
    --output_dir outputs/datasets/taxi_noise4/seed0
```

### 3. Train full-feature IL student (needed by SHAP and SAGE)

SHAP requires a surrogate MLP; SAGE requires a trained model to evaluate.
MCI-NN and MCI-HDC do **not** need this step — they retrain fresh models
per subset internally.

```bash
python student/train_student.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --selector full \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/students/taxi_noise4/seed0/
```

### 4a. Feature ranking — SHAP

```bash
python explain/shap_behavior.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --student_data outputs/students/taxi_noise4/seed0/ \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/shap/taxi_noise4/seed0

python explain/global_rank.py \
    --input outputs/shap/taxi_noise4/seed0/shap_values.npz \
    --output_dir outputs/rankings/taxi_noise4/seed0
```

### 4b. Feature ranking — SAGE

```bash
python explain/sage_rank.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/rankings_sage/taxi_noise4/seed0
```

### 4c. Feature ranking — MCI-NN

```bash
python explain/mci_rank_nn.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/rankings_mci_nn/taxi_noise4/seed0
```

MCI-NN retrains a small MLP for every evaluated subset internally — no
external model checkpoint is needed.

### 4d. Feature ranking — MCI-HDC

```bash
python explain/mci_rank.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/rankings_mci_hdc/taxi_noise4/seed0
```

### 5. Train IL students with ranked features

```bash
python student/train_student.py \
    --config configs/taxi_noise4.yaml \
    --seed 0 \
    --selector shap \
    --ranking_path outputs/rankings/taxi_noise4/seed0/ranking.csv \
    --dataset_path outputs/datasets/taxi_noise4/seed0/dataset.npz \
    --output_dir outputs/students/taxi_noise4/seed0/
```

Replace `--selector shap` and `--ranking_path` with the appropriate ranker
and path for sage / mci_nn / mci_hdc experiments.

### 6. Generate plots

```bash
python eval/make_plots.py \
    --input_root outputs \
    --output_dir outputs/plots
```

### Full pipeline (all seeds, all selectors)

```bash
bash scripts/run_all.sh
```

---

## Saved Artifacts

| Path | Description |
|------|-------------|
| `outputs/teachers/*/model.zip` | SB3 DQN checkpoint |
| `outputs/datasets/*/dataset.npz` | Teacher-labeled dataset (train/val/test splits) |
| `outputs/shap/*/shap_values.npz` | Local SHAP values [K, D] or [6, K, D] |
| `outputs/rankings/*/ranking.csv` | SHAP global feature ranking |
| `outputs/rankings_sage/*/ranking.csv` | SAGE global feature ranking |
| `outputs/rankings_mci_nn/*/ranking.csv` | MCI-NN global feature ranking |
| `outputs/rankings_mci_hdc/*/ranking.csv` | MCI-HDC global feature ranking |
| `outputs/students/*/*/model.pt` | Student BC policy checkpoint |
| `outputs/eval/offline/*/offline_metrics.csv` | Test accuracy and F1 per k |
| `outputs/eval/online/*/online_metrics.csv` | Return, success rate, episode length per k |
| `outputs/plots/` | Summary figures |

### Dataset schema (`dataset.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `X_{split}` | [N, obs_dim] | float32 observation vectors |
| `y_{split}` | [N] | int64 teacher greedy actions |
| `q_{split}` | [N, 6] | float32 Q-values |
| `p_{split}` | [N, 6] | float32 softmax probabilities |
| `chosen_prob_{split}` | [N] | float32 probability of chosen action |
| `feature_names` | [obs_dim] | string feature names |