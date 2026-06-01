#!/usr/bin/env bash
# Collect continuous-control datasets that support the bc/irl/pc ranking tasks.
#
# Two backends, one dataset.npz schema:
#   seals MuJoCo (ant / walker / hopper)  → teacher/collect_ant_expert_data.py
#       rolls out the HuggingFace PPO expert (HumanCompatibleAI) per env.
#   D4RL offline (kitchen / pen)          → teacher/load_d4rl_dataset.py
#       downloads / loads the offline HDF5.
#
# Every dataset.npz carries X / y / rewards / dones(or terminals) / next_X, which
# is exactly what the rankers need:
#   bc  → X→y           (ranker/{mci_rank_*,sage_rank,kernelshap}.py --task bc)
#   irl → X→rewards     (--task irl)
#   pc  → rewards+dones (--task pc; fragment pairs)
#
# Outputs: outputs/datasets/{name}/seed{N}/dataset.npz (+ metadata.json).
# Run from the sverl_feature_distill/ directory in the `il` conda env:
#   bash scripts/run_collect_datasets.sh                 # all envs, seeds 0..2
#   SEEDS="0 1 2 3 4" bash scripts/run_collect_datasets.sh
#   ENVS="seals_ant kitchen_complete" bash scripts/run_collect_datasets.sh

set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
# name:config:backend  (backend = seals | d4rl)
ALL_TARGETS=(
    "seals_ant:configs/seals_ant.yaml:seals"
    "seals_walker:configs/seals_walker.yaml:seals"
    "seals_hopper:configs/seals_hopper.yaml:seals"
    "kitchen_complete:configs/kitchen_complete.yaml:d4rl"
    "pen_human:configs/pen_human.yaml:d4rl"
    "pen_cloned:configs/pen_cloned.yaml:d4rl"
    "pen_expert:configs/pen_expert.yaml:d4rl"
)

# Optional ENVS filter (space-separated names); default = all targets.
ENVS="${ENVS:-}"

OUT_ROOT="outputs/datasets"

for entry in "${ALL_TARGETS[@]}"; do
    IFS=":" read -r NAME CONFIG BACKEND <<< "$entry"

    if [ -n "$ENVS" ] && [[ " $ENVS " != *" $NAME "* ]]; then
        continue
    fi

    for SEED in $SEEDS; do
        OUT_DIR="${OUT_ROOT}/${NAME}/seed${SEED}"
        echo "===== Collecting ${NAME} (${BACKEND}) seed=${SEED} → ${OUT_DIR} ====="

        if [ "$BACKEND" = "seals" ]; then
            python teacher/collect_ant_expert_data.py \
                --config "$CONFIG" \
                --seed "$SEED" \
                --output_dir "$OUT_DIR"
        else
            python teacher/load_d4rl_dataset.py \
                --config "$CONFIG" \
                --seed "$SEED" \
                --output_dir "$OUT_DIR"
        fi
    done
done

echo ""
echo "All datasets collected under ${OUT_ROOT}/."
echo "Next: rank with e.g."
echo "  python ranker/mci_rank_kernel.py --evaluator_name bc \\"
echo "      --dataset_path ${OUT_ROOT}/seals_ant/seed0/dataset.npz \\"
echo "      --output_dir outputs/rankings_mci_kernel/seals_ant/seed0"
