# Debug RL Teacher Policy

Diagnose why the DQN teacher failed to learn an effective policy in the
`sverl_feature_distill` project. Run all checks, interpret findings, and
recommend concrete fixes.

## Instructions

The user may optionally provide a teacher output directory as an argument
(e.g. `/debug-teacher outputs/teachers/clean/seed0`). If no argument is given,
default to the most recently modified teacher directory under `outputs/teachers/`.

Work from the `sverl_feature_distill/` project root. Use the `il` conda environment.

---

### Step 1 — Read training artifacts

Read these files from the teacher directory:
- `eval_metrics.json` — final mean_reward, success_rate
- `train_summary.json` — total_timesteps, training_time_s
- `metadata.json` — seed, noise_dim, obs_dim, config path
- `resolved_config.yaml` — all hyperparameters actually used

Print a summary table:
```
Metric            | Value
------------------|---------
mean_reward       | ...
std_reward        | ...
success_rate      | ...
total_timesteps   | ...
obs_dim / noise   | ...
```

---

### Step 2 — Classify the failure mode

Based on eval_metrics.json, classify:

| mean_reward range | Diagnosis |
|-------------------|-----------|
| ≤ −190            | **timeout every episode** — agent never delivers passenger; likely exploration or architecture problem |
| −190 to −50       | **partial learning** — occasionally succeeds but inconsistently; may need more timesteps or tuned LR |
| −50 to +5         | **near-optimal** — borderline; evaluate success_rate to confirm |
| ≥ +5              | **converged** — teacher is usable; check if the problem is downstream |

Print the classification and a one-sentence interpretation.

---

### Step 3 — Hyperparameter audit

Read `resolved_config.yaml`. Check each of these known failure points for
DQN on Taxi-v3 and flag any that are suspicious:

1. **`learning_starts`** — should be ≤ 10000 for Taxi-v3. SB3 default of 50000
   means the buffer fills with random transitions before any learning begins,
   which hurts on sparse-reward tasks. Flag if > 10000.

2. **`exploration_fraction`** — controls how many of `total_timesteps` are
   used for epsilon decay. For 300k steps, 0.3 means epsilon decays over 90k
   steps. Flag if < 0.15 (decays too fast, not enough exploration) or > 0.6
   (stays random too long).

3. **`exploration_final_eps`** — final epsilon. Flag if > 0.15 (too much
   residual randomness) or < 0.01 (might not explore enough early on).

4. **`total_timesteps`** — Taxi-v3 with continuous 4-dim obs needs ~200k–500k.
   With native Discrete obs (noise_dim=0), SB3 one-hot encodes to 500-dim,
   which converges faster (~100k). Flag if < 100000.

5. **`learning_rate`** — typical range 1e-4 to 1e-3. Flag if > 5e-3
   (unstable) or < 1e-5 (too slow).

6. **`net_arch`** — for Discrete obs + one-hot, [64,64] is enough. For
   continuous Box obs with noise, [128,128] or larger is better. Check that
   arch matches obs type.

7. **`noise_dim` vs obs type** — if noise_dim == 0 but `NoisyTaxiWrapper` is
   still used, the teacher gets a 4-dim float obs instead of the SB3 one-hot
   500-dim encoding, which dramatically slows convergence. Read `metadata.json`
   key `uses_native_discrete_obs` to verify.

Print: `[OK]` or `[WARN]` for each item with a brief note.

---

### Step 4 — Behavioral probe

Run this quick probe to check if the saved model can at least act greedily.
Write a temporary inline Python script (print it, then execute it via bash):

```python
import sys, gymnasium as gym, numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

CKPT = "<teacher_dir>/model.zip"   # fill in actual path
model = DQN.load(CKPT)

env = Monitor(gym.make("Taxi-v3"))
mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
print(f"Probe: mean_reward={mean_r:.2f} std={std_r:.2f}")

# Check action distribution — a stuck policy often picks the same action always
obs, _ = env.reset(seed=42)
actions = []
for _ in range(200):
    a, _ = model.predict(obs, deterministic=True)
    actions.append(int(a))
    obs, _, term, trunc, _ = env.step(a)
    if term or trunc:
        obs, _ = env.reset()
from collections import Counter
print("Action distribution (200 steps):", dict(Counter(actions)))
env.close()
```

Interpret:
- If all 200 steps use the same 1-2 actions → **degenerate policy** (collapsed Q-values)
- If std_r == 0 and mean_r == −200 → **always timing out**, policy never delivers
- If mean_r improves over eval_metrics → training eval had a bug (check eval seed)

---

### Step 5 — Check SB3 training log (if available)

Look for a `monitor.csv` or SB3 stdout log in the teacher directory or any
subdirectory. If found, read the last 20 lines and check if episode rewards
were trending upward during training. A flat line at −200 throughout indicates
the network never got a positive learning signal.

If no log is available, note it and suggest re-running with:
```bash
python teacher/train_teacher.py --config <config> --seed <seed> --output_dir <dir>
```
and checking the verbose=1 output for reward improvement.

---

### Step 6 — Root cause summary and recommended fixes

Based on findings from Steps 2–5, print a prioritised fix list:

**Fix format:**
```
[Priority N] <short title>
  Problem : <what went wrong>
  Fix     : <exact config change or code change>
  Expected: <what you expect to see after the fix>
```

Common fixes to consider (only include those relevant to what was found):

- Increase `total_timesteps` to 500000 for continuous obs
- Lower `learning_starts` to 5000–10000
- Ensure `noise_dim == 0` config uses native Discrete obs (not NoisyTaxiWrapper)
- Adjust `exploration_fraction` so epsilon decays over at least 30% of training
- Switch to a simpler architecture [64, 64] if obs is already one-hot 500-dim
- Re-run with a different seed if the issue is seed-specific collapse
- Add a `VecEnv` with `n_envs=4` to diversify experience (more trajectory variety)

---

### Step 7 — Optional: quick retrain with patched config

If the root cause is clearly a hyperparameter issue, offer to patch the config
and immediately re-run training:

```bash
python teacher/train_teacher.py \
    --config configs/taxi_clean.yaml \
    --seed 0 \
    --output_dir outputs/teachers/clean/seed0_debug
```

Ask the user before running. If they confirm, run it and report the new
eval_metrics.json when done.
