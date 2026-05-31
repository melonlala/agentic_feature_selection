#!/usr/bin/env bash
# Collect offline preference-comparisons datasets for seals/Ant-v1 across seeds.
#
# Produces, per seed, under outputs/preferences/seals_ant/seed{N}/, one
# preference file per source (expert / noisy / random):
#   preferences_{source}.pkl   pickled PreferenceDataset (fragment pairs + labels)
#   trajectories_{source}.pkl  that source's raw TrajectoryWithRew pool
#   metadata.json, resolved_config.yaml
#
# Usage: bash scripts/run_ant_preferences.sh [seed ...]   (default seeds 0-4)
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/seals_ant.yaml"
SEEDS=("${@:-0 1 2}")
# Expand a single "0 1 2 3 4" string into words if no args were passed.
read -r -a SEEDS <<< "${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  echo "=== seals/Ant-v1 preferences | seed ${seed} ==="
  python teacher/collect_ant_preferences.py \
    --config "${CONFIG}" \
    --seed "${seed}" \
    --output_dir "outputs/preferences/seals_ant/seed${seed}"
done

echo "Done. Datasets under outputs/preferences/seals_ant/seed{N}/."
