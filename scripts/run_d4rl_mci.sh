#!/usr/bin/env bash
# D4RL MCI feature-ranking pipeline.
#
# Runs the full IL + MCI pipeline on D4RL kitchen/complete-v2 and pen/* datasets.
# Feature selection methods are identical to the Taxi pipeline:
#   shap (MCI ranking), random, oracle, mi, full.
# The imitation-learning network is BCContinuousPolicy / BCGaussianPolicy
# instead of the Taxi DQN, matching the continuous-action D4RL tasks.
#
# Prerequisites:
#   - conda environment 'il' activated  (Python 3.10, torch, sklearn, h5py, …)
#   - Run from the sverl_feature_distill/ directory
#
# Stages:
#   1. Download + convert D4RL datasets → dataset.npz
#   2. MCI feature ranking             → ranking.csv
#   3. Rank stability across seeds     → stability/
#   4. Train selector students         → students/
#   5. Offline evaluation              → eval/offline/
#
# Usage:
#   bash scripts/run_d4rl_mci.sh [--datasets "kitchen_complete pen_human"]
#   bash scripts/run_d4rl_mci.sh  # runs all 4 datasets

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS=(0 1 2 3 4)
SELECTORS=("shap" "random" "oracle" "mi" "full")

# MCI hyperparameters
RFF_DIM=64
BANDWIDTH=1.0
LAMBDA=0.001
N_PERMS=200
MAX_SAMPLES=3000

# Dataset list (config_name : yaml_path pairs)
declare -A DATASETS=(
    [kitchen_complete]="configs/kitchen_complete.yaml"
    [pen_human]="configs/pen_human.yaml"
    [pen_cloned]="configs/pen_cloned.yaml"
    [pen_expert]="configs/pen_expert.yaml"
)

# Optional: filter to specific datasets via first argument
if [[ $# -ge 1 ]]; then
    IFS=' ' read -r -a REQUESTED <<< "$1"
    declare -A DATASETS_FILTERED
    for name in "${REQUESTED[@]}"; do
        if [[ -v "DATASETS[$name]" ]]; then
            DATASETS_FILTERED[$name]="${DATASETS[$name]}"
        else
            echo "WARNING: Unknown dataset '$name'. Known: ${!DATASETS[*]}"
        fi
    done
    declare -A DATASETS
    for k in "${!DATASETS_FILTERED[@]}"; do
        DATASETS[$k]="${DATASETS_FILTERED[$k]}"
    done
fi

echo "===== D4RL MCI Pipeline ====="
echo "Datasets: ${!DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Selectors: ${SELECTORS[*]}"
echo "MCI: rff_dim=$RFF_DIM bandwidth=$BANDWIDTH lambda=$LAMBDA n_perms=$N_PERMS"
echo ""

for DATASET_NAME in "${!DATASETS[@]}"; do
    CONFIG="${DATASETS[$DATASET_NAME]}"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "DATASET: $DATASET_NAME  (config: $CONFIG)"
    echo "══════════════════════════════════════════════"

    COMPLETED_SEEDS=()

    for SEED in "${SEEDS[@]}"; do
        DATASET_DIR="outputs/datasets/${DATASET_NAME}/seed${SEED}"
        DATASET_NPZ="${DATASET_DIR}/dataset.npz"
        RANKING_DIR="outputs/rankings_mci/${DATASET_NAME}/seed${SEED}"
        RANKING_CSV="${RANKING_DIR}/ranking.csv"

        echo ""
        echo "── Seed $SEED ──────────────────────────────────"

        # ── Stage 1: Load/download D4RL dataset ─────────────────────────────
        if [[ -f "$DATASET_NPZ" ]]; then
            echo "[skip] Dataset already exists: $DATASET_NPZ"
        else
            echo "[run]  Loading D4RL dataset → $DATASET_NPZ"
            python teacher/load_d4rl_dataset.py \
                --config "$CONFIG" \
                --seed   "$SEED" \
                --output_dir "$DATASET_DIR"
        fi

        if [[ ! -f "$DATASET_NPZ" ]]; then
            echo "ERROR: Dataset not found after load attempt: $DATASET_NPZ"
            echo "       Check download errors above or supply --hdf5_path manually."
            continue
        fi

        # ── Stage 2: MCI ranking ─────────────────────────────────────────────
        if [[ -f "$RANKING_CSV" ]]; then
            echo "[skip] Ranking already exists: $RANKING_CSV"
        else
            echo "[run]  MCI ranking seed=$SEED"
            python explain/mci_rank.py \
                --config       "$CONFIG" \
                --seed         "$SEED" \
                --dataset_path "$DATASET_NPZ" \
                --output_dir   "$RANKING_DIR" \
                --target       "action_multi" \
                --rff_dim      "$RFF_DIM" \
                --bandwidth    "$BANDWIDTH" \
                --lambda_      "$LAMBDA" \
                --n_perms      "$N_PERMS" \
                --max_samples  "$MAX_SAMPLES"
        fi

        COMPLETED_SEEDS+=("$SEED")
    done

    # ── Stage 3: Rank stability ───────────────────────────────────────────────
    if [[ ${#COMPLETED_SEEDS[@]} -ge 2 ]]; then
        echo ""
        echo "[run]  Rank stability for $DATASET_NAME (seeds: ${COMPLETED_SEEDS[*]})"
        RANKING_PATHS=()
        for SEED in "${COMPLETED_SEEDS[@]}"; do
            RANKING_PATHS+=("outputs/rankings_mci/${DATASET_NAME}/seed${SEED}/ranking.csv")
        done
        python explain/rank_stability.py \
            --ranking_paths "${RANKING_PATHS[@]}" \
            --output_dir "outputs/rankings_mci/${DATASET_NAME}/stability"
    else
        echo "[skip] Rank stability — need ≥2 completed seeds."
    fi

    # ── Stages 4 + 5: Train students + offline eval ───────────────────────────
    for SEED in "${COMPLETED_SEEDS[@]}"; do
        DATASET_NPZ="outputs/datasets/${DATASET_NAME}/seed${SEED}/dataset.npz"
        RANKING_CSV="outputs/rankings_mci/${DATASET_NAME}/seed${SEED}/ranking.csv"

        for SELECTOR in "${SELECTORS[@]}"; do
            STUDENT_DIR="outputs/students/${DATASET_NAME}/seed${SEED}/${SELECTOR}"
            SUMMARY_CSV="${STUDENT_DIR}/summary.csv"

            echo ""
            echo "[run]  Train student: dataset=$DATASET_NAME seed=$SEED selector=$SELECTOR"

            RANKING_ARG=""
            if [[ "$SELECTOR" == "shap" ]]; then
                RANKING_ARG="--ranking_path $RANKING_CSV"
            fi

            if [[ -f "$SUMMARY_CSV" ]]; then
                echo "[skip] Summary already exists: $SUMMARY_CSV"
            else
                python student/train_student_continuous.py \
                    --config       "$CONFIG" \
                    --seed         "$SEED" \
                    --dataset_path "$DATASET_NPZ" \
                    --selector     "$SELECTOR" \
                    $RANKING_ARG \
                    --output_dir   "$STUDENT_DIR"
            fi

            # Offline eval
            EVAL_DIR="outputs/eval/offline/${DATASET_NAME}/seed${SEED}/${SELECTOR}"
            EVAL_CSV="${EVAL_DIR}/offline_metrics.csv"

            if [[ -f "$EVAL_CSV" ]]; then
                echo "[skip] Offline eval already exists: $EVAL_CSV"
            else
                echo "[run]  Offline eval: dataset=$DATASET_NAME seed=$SEED selector=$SELECTOR"
                python eval/eval_offline_continuous.py \
                    --config       "$CONFIG" \
                    --dataset_path "$DATASET_NPZ" \
                    --student_dir  "$STUDENT_DIR" \
                    --output_dir   "$EVAL_DIR"
            fi
        done
    done

    echo ""
    echo "Done: $DATASET_NAME"
done

echo ""
echo "═══════════════════════════════════════════════"
echo "D4RL MCI pipeline complete."
echo ""
echo "Results layout:"
echo "  outputs/datasets/{dataset}/seed{N}/dataset.npz"
echo "  outputs/rankings_mci/{dataset}/seed{N}/ranking.csv"
echo "  outputs/students/{dataset}/seed{N}/{selector}/summary.csv"
echo "  outputs/eval/offline/{dataset}/seed{N}/{selector}/offline_metrics.csv"
echo ""
echo "To run plotting (if make_plots.py supports continuous tasks):"
echo "  python eval/make_plots.py --input_root outputs --output_dir outputs/plots"
