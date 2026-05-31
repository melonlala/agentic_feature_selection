#!/usr/bin/env bash
# D4RL latent-space MCI pipeline.
#
# Mirrors scripts/run_d4rl_mci.sh but ranks first-layer latent embeddings of a
# pre-trained full-feature student (instead of raw observation features). All
# outputs go to *_latent paths so the state-space pipeline is untouched.
#
# Stages per (dataset, seed):
#   1. (assume) state-space dataset.npz already exists.
#   2. (assume) full-feature student already trained:
#        outputs/students/{ds}/seed{N}/full/full/model.pt
#   3. Extract latents → outputs/datasets_latent/{ds}/seed{N}/latent_dataset.npz
#   4. MCI ranking on latents → outputs/rankings_mci_latent/{ds}/seed{N}/ranking.csv
#   5. Rank stability across seeds → outputs/rankings_mci_latent/{ds}/stability/
#   6. Train latent-mode students (selectors: mci, random, full)
#      → outputs/students_latent/{ds}/seed{N}/{selector}/k*/
#   7. Offline eval → outputs/eval/offline_latent/{ds}/seed{N}/{selector}/
#
# Prerequisites:
#   - conda env 'il' activated.
#   - Run from sverl_feature_distill/.
#   - State-space datasets and full students must already exist; this script
#     does not retrain them.
#
# Usage:
#   bash scripts/run_d4rl_latent_mci.sh
#   bash scripts/run_d4rl_latent_mci.sh "kitchen_complete pen_human"

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS=(0 1 2 3 4)
# Oracle/MI have no meaningful analog in latent space (no ground-truth informative
# latent dim; MI on latents already trained to predict actions is degenerate).
SELECTORS=("mci" "random" "full")
LATENT_LAYER="pre_relu"

# MCI hyperparameters (smaller RFF / fewer perms — 256-D latent is wider than raw obs)
RFF_DIM=32
BANDWIDTH=1.0
LAMBDA=0.001
N_PERMS=100
MAX_SAMPLES=3000

declare -A DATASETS=(
    [kitchen_complete]="configs/kitchen_complete.yaml"
    [pen_human]="configs/pen_human.yaml"
    [pen_cloned]="configs/pen_cloned.yaml"
    [pen_expert]="configs/pen_expert.yaml"
)

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

echo "===== D4RL Latent-MCI Pipeline ====="
echo "Datasets:     ${!DATASETS[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Selectors:    ${SELECTORS[*]}"
echo "Latent layer: $LATENT_LAYER"
echo "MCI:          rff_dim=$RFF_DIM bandwidth=$BANDWIDTH lambda=$LAMBDA n_perms=$N_PERMS"
echo ""

for DATASET_NAME in "${!DATASETS[@]}"; do
    CONFIG="${DATASETS[$DATASET_NAME]}"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "DATASET: $DATASET_NAME  (config: $CONFIG)"
    echo "══════════════════════════════════════════════"

    COMPLETED_SEEDS=()

    for SEED in "${SEEDS[@]}"; do
        STATE_DATASET="outputs/datasets/${DATASET_NAME}/seed${SEED}/dataset.npz"
        FULL_STUDENT="outputs/students/${DATASET_NAME}/seed${SEED}/full/full/model.pt"
        LATENT_DIR="outputs/datasets_latent/${DATASET_NAME}/seed${SEED}"
        LATENT_NPZ="${LATENT_DIR}/latent_dataset.npz"
        RANKING_DIR="outputs/rankings_mci_latent/${DATASET_NAME}/seed${SEED}"
        RANKING_CSV="${RANKING_DIR}/ranking.csv"

        echo ""
        echo "── Seed $SEED ──────────────────────────────────"

        if [[ ! -f "$STATE_DATASET" ]]; then
            echo "[skip] State-space dataset missing: $STATE_DATASET (run run_d4rl_mci.sh first)"
            continue
        fi
        if [[ ! -f "$FULL_STUDENT" ]]; then
            echo "[skip] Full-feature student missing: $FULL_STUDENT (run run_d4rl_mci.sh first)"
            continue
        fi

        # ── Stage 3: Extract latents ─────────────────────────────────────────
        if [[ -f "$LATENT_NPZ" ]]; then
            echo "[skip] Latent dataset exists: $LATENT_NPZ"
        else
            echo "[run]  Extract latents → $LATENT_NPZ"
            python explain/latent_extract.py \
                --config            "$CONFIG" \
                --seed              "$SEED" \
                --dataset_path      "$STATE_DATASET" \
                --full_student_path "$FULL_STUDENT" \
                --output_path       "$LATENT_NPZ" \
                --latent_layer      "$LATENT_LAYER"
        fi

        if [[ ! -f "$LATENT_NPZ" ]]; then
            echo "ERROR: Latent dataset not found after extraction: $LATENT_NPZ"
            continue
        fi

        # ── Stage 4: MCI ranking on latents ───────────────────────────────────
        if [[ -f "$RANKING_CSV" ]]; then
            echo "[skip] Ranking exists: $RANKING_CSV"
        else
            echo "[run]  MCI ranking on latents seed=$SEED"
            python explain/mci_rank.py \
                --config       "$CONFIG" \
                --seed         "$SEED" \
                --dataset_path "$LATENT_NPZ" \
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

    # ── Stage 5: Rank stability ───────────────────────────────────────────────
    if [[ ${#COMPLETED_SEEDS[@]} -ge 2 ]]; then
        echo ""
        echo "[run]  Rank stability (latent) for $DATASET_NAME (seeds: ${COMPLETED_SEEDS[*]})"
        RANKING_PATHS=()
        for SEED in "${COMPLETED_SEEDS[@]}"; do
            RANKING_PATHS+=("outputs/rankings_mci_latent/${DATASET_NAME}/seed${SEED}/ranking.csv")
        done
        python explain/rank_stability.py \
            --ranking_paths "${RANKING_PATHS[@]}" \
            --output_dir "outputs/rankings_mci_latent/${DATASET_NAME}/stability"
    else
        echo "[skip] Rank stability — need ≥2 completed seeds."
    fi

    # ── Stages 6 + 7: Train latent students + offline eval ────────────────────
    for SEED in "${COMPLETED_SEEDS[@]}"; do
        LATENT_NPZ="outputs/datasets_latent/${DATASET_NAME}/seed${SEED}/latent_dataset.npz"
        RANKING_CSV="outputs/rankings_mci_latent/${DATASET_NAME}/seed${SEED}/ranking.csv"
        FULL_STUDENT="outputs/students/${DATASET_NAME}/seed${SEED}/full/full/model.pt"
        STATE_DATASET="outputs/datasets/${DATASET_NAME}/seed${SEED}/dataset.npz"

        for SELECTOR in "${SELECTORS[@]}"; do
            STUDENT_DIR="outputs/students_latent/${DATASET_NAME}/seed${SEED}/${SELECTOR}"
            SUMMARY_CSV="${STUDENT_DIR}/summary.csv"

            echo ""
            echo "[run]  Train latent student: dataset=$DATASET_NAME seed=$SEED selector=$SELECTOR"

            RANKING_ARG=""
            if [[ "$SELECTOR" == "mci" ]]; then
                RANKING_ARG="--ranking_path $RANKING_CSV"
            fi

            if [[ -f "$SUMMARY_CSV" ]]; then
                echo "[skip] Summary already exists: $SUMMARY_CSV"
            else
                python student/train_student_continuous.py \
                    --config            "$CONFIG" \
                    --seed              "$SEED" \
                    --dataset_path      "$LATENT_NPZ" \
                    --selector          "$SELECTOR" \
                    $RANKING_ARG \
                    --output_dir        "$STUDENT_DIR" \
                    --latent_mode \
                    --full_student_path "$FULL_STUDENT" \
                    --latent_layer      "$LATENT_LAYER"
            fi

            # Offline eval: evaluate on the *raw* state-space test split.
            # Eval slices X_test[:, feature_idx] — for latent students feature_idx
            # = range(raw_D), so the slice is a no-op pass-through. The internal
            # latent extraction + selection runs inside the network.
            EVAL_DIR="outputs/eval/offline_latent/${DATASET_NAME}/seed${SEED}/${SELECTOR}"
            EVAL_CSV="${EVAL_DIR}/offline_metrics.csv"

            if [[ -f "$EVAL_CSV" ]]; then
                echo "[skip] Offline eval already exists: $EVAL_CSV"
            else
                echo "[run]  Offline eval (latent): dataset=$DATASET_NAME seed=$SEED selector=$SELECTOR"
                python eval/eval_offline_continuous.py \
                    --config       "$CONFIG" \
                    --dataset_path "$STATE_DATASET" \
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
echo "D4RL latent-MCI pipeline complete."
echo ""
echo "Results layout:"
echo "  outputs/datasets_latent/{ds}/seed{N}/latent_dataset.npz"
echo "  outputs/rankings_mci_latent/{ds}/seed{N}/ranking.csv"
echo "  outputs/students_latent/{ds}/seed{N}/{selector}/summary.csv"
echo "  outputs/eval/offline_latent/{ds}/seed{N}/{selector}/offline_metrics.csv"
echo ""
echo "To produce the state-vs-latent comparison:"
echo "  python eval/compare_state_vs_latent.py --output_dir outputs/plots/compare_state_vs_latent"
