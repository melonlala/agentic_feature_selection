# D4RL Results Analysis

**Date:** 2026-04-22
**Pipeline:** MCI feature ranking + BC student distillation
**Seeds:** 0 only (single run)

---

## Dataset Coverage

| Dataset | Selectors | Seeds |
|---|---|---|
| kitchen_complete | shap, random, oracle | 0 |
| pen_human | shap, random, oracle | 0 |
| pen_expert | shap only | 0 |
| pen_cloned | rankings only (no students) | 0 |

Missing: `mi`, `full` selectors for all datasets; student training for pen_cloned.

---

## kitchen_complete

**Feature space:** 60 features (`qpos_0..8`, `qvel_0..8`, `obj_0..41`)
**Oracle definition:** `qpos_0..8` (9 features); oracle saturates at k ≥ 10.

### Test MSE vs k

| k | shap | random | oracle |
|---|---|---|---|
| 5 | 0.1010 | 0.1142 | **0.0276** |
| 10 | 0.0852 | 0.0495 | **0.0161** |
| 20 | 0.0401 | 0.0380 | **0.0161** |
| 30 | 0.0194 | 0.0358 | **0.0161** |
| 45 | 0.0163 | **0.0148** | 0.0161 |
| 60 | **0.0119** | **0.0119** | 0.0161 |

### Cosine Similarity vs k

| k | shap | random | oracle |
|---|---|---|---|
| 5 | 0.476 | 0.371 | **0.849** |
| 10 | 0.565 | 0.743 | **0.914** |
| 20 | 0.802 | 0.809 | **0.914** |
| 30 | 0.897 | 0.805 | **0.914** |
| 45 | 0.911 | **0.917** | 0.914 |
| 60 | **0.932** | **0.932** | 0.914 |

### MCI Top-10 Features

| Rank | Feature | MCI Score |
|---|---|---|
| 1 | obj_25 | 671,021 |
| 2 | obj_23 | 114,624 |
| 3 | qpos_6 | 73,484 |
| 4 | obj_18 | 68,472 |
| 5 | qpos_8 | 42,510 |
| 6 | qvel_3 | 9,531 |
| 7 | obj_31 | 8,279 |
| 8 | qvel_1 | 8,167 |
| 9 | obj_9 | 5,336 |
| 10 | obj_35 | 5,237 |

### Findings

- **Oracle dominates at small k.** At k=5, oracle MSE (0.028) is 3.7× lower than SHAP (0.101). The qpos features are genuinely the most action-predictive, but MCI does not rank them first.
- **MCI ranking is distorted by obj_25.** Its score (671k) is 6× the 2nd-place feature (obj_23, 115k), suggesting one object feature has extreme marginal conditional information — possibly a contact or end-effector coordinate correlated with the action signal in this dataset.
- **SHAP recovers oracle performance only at k ≈ 45.** Oracle achieves its best MSE (0.016) with 9 features; SHAP needs 45 features to reach the same level.
- **Random beats SHAP at k=45** (0.0148 vs 0.0163), indicating the MCI ranking does not provide a useful ordering for this task beyond helping identify the very top feature.
- **All selectors converge at k=60** (full feature set, MSE ≈ 0.012).

---

## pen_human

**Feature space:** 45 features (`hand_qpos_0..23`, `obj_pos`, `obj_euler`, `goal_pos`, `goal_euler`, `obj2goal_pos`, `obj2goal_rot`, `contact_0..2`)
**Oracle definition:** `hand_qpos_0..23` (24 features); oracle saturates at k ≥ 30.

### Test MSE vs k

| k | shap | random | oracle |
|---|---|---|---|
| 5 | 0.0157 | 0.0146 | **0.0137** |
| 10 | 0.0057 | **0.0052** | 0.0059 |
| 15 | 0.0043 | 0.0050 | **0.0038** |
| 20 | 0.0042 | **0.0033** | 0.0035 |
| 30 | **0.0033** | 0.0037 | **0.0033** |
| 45 | 0.0031 | **0.0029** | 0.0033 |

### Cosine Similarity vs k

| k | shap | random | oracle |
|---|---|---|---|
| 5 | 0.988 | 0.989 | **0.990** |
| 10 | 0.996 | 0.996 | 0.996 |
| 15 | 0.997 | 0.996 | **0.997** |
| 20 | 0.997 | 0.997 | 0.997 |
| 30 | 0.997 | 0.997 | **0.998** |
| 45 | 0.997 | **0.998** | 0.998 |

### MCI Top-10 Features

| Rank | Feature | MCI Score |
|---|---|---|
| 1 | hand_qpos_9 | 28,572,056 |
| 2 | hand_qpos_7 | 3,244,507 |
| 3 | hand_qpos_13 | 82,439 |
| 4 | hand_qpos_16 | ~60k |
| 5 | hand_qpos_5 | ~40k |
| 6 | hand_qpos_12 | — |
| 7 | obj_euler_2 | — |
| 8 | hand_qpos_18 | — |
| 9 | hand_qpos_15 | — |
| 10 | hand_qpos_6 | — |

### Findings

- **All selectors are competitive throughout.** The maximum gap between selectors at any k is 0.004 MSE — no selector has a decisive advantage.
- **Random beats oracle at k=10, 20, 45.** The oracle definition (hand_qpos only) misses obj, goal, and contact features that carry genuine action information for human demonstrations. The human policy is not purely joint-angle-conditioned.
- **Cosine similarity is uniformly very high (>0.988).** Even the worst-performing subset predicts the correct action direction; MSE differences are driven by action magnitude errors, not direction errors.
- **SHAP selects a different but equivalent ordering.** Top-5: `hand_qpos_9, 7, 13, 16, 5` vs oracle `hand_qpos_0..4`. Both reach the same MSE by k=30, implying the first few finger joints are interchangeable for cloning this policy.
- **SHAP slightly outperforms random at k=30** (0.0033 vs 0.0037), the only k where MCI ranking provides a meaningful edge.

---

## pen_expert

**Feature space:** 45 features (same as pen_human)
**Selectors available:** shap only

### Test MSE vs k

| k | shap |
|---|---|
| 5 | 0.1377 |
| 10 | 0.1118 |
| 15 | 0.1068 |
| 20 | 0.1027 |
| 30 | 0.0958 |
| 45 | **0.0955** |

### MCI Top-5 Features

| Rank | Feature | MCI Score |
|---|---|---|
| 1 | goal_pos_1 | 925,918 |
| 2 | obj2goal_pos_0 | 378,209 |
| 3 | obj_euler_0 | 218,174 |
| 4 | hand_qpos_13 | — |
| 5 | contact_1 | — |

### Findings

- **MSE is 20–30× higher than pen_human.** Expert demonstrations are significantly harder to clone with BC — the expert policy may be more stochastic or require temporal context that a single-step BC model cannot capture.
- **Goal/object features dominate, not hand joints.** Unlike pen_human (where hand_qpos top-ranked), MCI for pen_expert picks `goal_pos_1`, `obj2goal_pos_0`, `obj_euler_0` as top features. The expert policy is more explicitly goal-conditioned in its feature reliance.
- **Diminishing returns after k=30.** MSE drops from 0.138 at k=5 to 0.096 at k=30, then only 0.0003 improvement from k=30→45. The informative features are concentrated in the top ~30 by MCI ranking.
- **Cannot compare to oracle/random** — only shap was run for this dataset.

---

## pen_cloned

Rankings computed (seed=0), but no student training was run. No student metrics available.

---

## Cross-Dataset Summary

### SHAP vs Oracle at k=10

| Dataset | shap MSE | oracle MSE | oracle advantage |
|---|---|---|---|
| kitchen_complete | 0.0852 | 0.0161 | 5.3× |
| pen_human | 0.0057 | 0.0059 | ≈ parity |

### Where MCI/SHAP Helps

- **pen_human:** SHAP identifies an alternative compact subset of hand joints that is as good as the oracle ordering by k=30. Useful when the ground-truth oracle is unknown.

### Where MCI/SHAP Struggles

- **kitchen_complete:** MCI scores are dominated by a single outlier feature (obj_25, score 6× the 2nd place), distorting the ranking. SHAP deprioritizes qpos features that oracle knows are ground-truth critical. Needs k≈45 to recover oracle-at-k=10 performance.
- **pen_expert:** High MSE throughout suggests the fundamental bottleneck is BC model capacity or data quality, not feature selection.

### Caveats

- All results are **single seed (seed=0)** — no variance estimates, rankings may not be stable.
- `mi` and `full` selectors were not run for any dataset — the comparison is incomplete.
- `pen_cloned` has no student results.
- The oracle definition for kitchen_complete (qpos only) may itself be approximate — qvel features may also be informative.
