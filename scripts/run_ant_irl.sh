#!/usr/bin/env bash
# State-space AIRL pipeline on seals/Ant-v1.
#
# Compares feature rankers (mci_nn, mci_hdc, sage, kernel_shap, random, full)
# under the same imitation-package AIRL training as the rest of the project.
#
# Stages per seed:
#   1. Collect expert demos                       → dataset.npz
#   2. Train a full-feature BC student            (used to power SAGE/SHAP)
#   3. Compute 5 rankings (mci_nn, mci_hdc, sage, kernel_shap, random)
#   4. Train AIRL students per (ranker, k) + the `full` baseline
#   5. Online eval per (ranker, k)
#
# Prerequisites:
#   - conda env `il` activated
#   - HuggingFace expert reachable (HumanCompatibleAI/ppo-seals-Ant-v1)
#   - Run from sverl_feature_distill/
#
# Usage:
#   bash scripts/run_ant_irl.sh                 # all seeds
#   bash scripts/run_ant_irl.sh "0 1"           # subset of seeds

set -euo pipefail

CONFIG="configs/seals_ant.yaml"
SEEDS=(0 1 2)
SELECTORS=(mci_nn mci_hdc sage kernel_shap random full)

# Ranking hyperparameters
MCI_NN_PERMS=20
MCI_NN_EPOCHS=20
MCI_HDC_PERMS=100
MCI_HDC_RFF=64
MCI_HDC_BW=1.0
MCI_HDC_LAM=0.001
SAGE_PERMS=300
SHAP_BG=200
SHAP_EXP=200

# IRL hyperparameters (read from config[irl] by train_student_irl.py)
QUICK_EVAL_EPISODES=10
ONLINE_EVAL_EPISODES=20

if [[ $# -ge 1 ]]; then
    IFS=' ' read -r -a SEEDS <<< "$1"
fi

echo "===== seals/Ant-v1 AIRL pipeline (state space) ====="
echo "Config:    $CONFIG"
echo "Seeds:     ${SEEDS[*]}"
echo "Selectors: ${SELECTORS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  SEED $SEED"
    echo "═══════════════════════════════════════════════"

    DATASET_DIR="outputs/datasets/seals_ant/seed${SEED}"
    DATASET_NPZ="${DATASET_DIR}/dataset.npz"

    # ── Stage 1: Expert dataset ────────────────────────────────────────────
    if [[ -f "$DATASET_NPZ" ]]; then
        echo "[skip] Dataset already exists: $DATASET_NPZ"
    else
        echo "[run]  Collect expert demos → $DATASET_NPZ"
        python teacher/collect_ant_expert_data.py \
            --config "$CONFIG" --seed "$SEED" \
            --output_dir "$DATASET_DIR"
    fi

    # ── Stage 2: Full BC student (powers SAGE/SHAP) ────────────────────────
    FULL_BC_DIR="outputs/students/seals_ant/seed${SEED}/full"
    FULL_BC_PT="${FULL_BC_DIR}/full/model.pt"
    if [[ -f "$FULL_BC_PT" ]]; then
        echo "[skip] Full BC student exists: $FULL_BC_PT"
    else
        echo "[run]  Train full-feature BC student"
        python student/train_student_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$DATASET_NPZ" \
            --selector full \
            --output_dir "$FULL_BC_DIR"
    fi

    # ── Stage 3: Rankings (5 methods) ──────────────────────────────────────
    RANK_MCI_NN="outputs/rankings_mci_nn/seals_ant/seed${SEED}"
    RANK_MCI_HDC="outputs/rankings_mci_hdc/seals_ant/seed${SEED}"
    RANK_SAGE="outputs/rankings_sage/seals_ant/seed${SEED}"
    RANK_SHAP="outputs/rankings_shap/seals_ant/seed${SEED}"

    if [[ -f "${RANK_MCI_NN}/ranking.csv" ]]; then
        echo "[skip] MCI-NN ranking exists: ${RANK_MCI_NN}/ranking.csv"
    else
        echo "[run]  MCI-NN ranking"
        python explain/mci_rank_nn.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$DATASET_NPZ" \
            --n_perms "$MCI_NN_PERMS" --epochs "$MCI_NN_EPOCHS" \
            --aggregation mean \
            --output_dir "$RANK_MCI_NN"
    fi

    if [[ -f "${RANK_MCI_HDC}/ranking.csv" ]]; then
        echo "[skip] MCI-HDC ranking exists: ${RANK_MCI_HDC}/ranking.csv"
    else
        echo "[run]  MCI-HDC ranking"
        python explain/mci_rank.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$DATASET_NPZ" \
            --output_dir "$RANK_MCI_HDC" \
            --target action_multi \
            --rff_dim "$MCI_HDC_RFF" --bandwidth "$MCI_HDC_BW" \
            --lambda_ "$MCI_HDC_LAM" --n_perms "$MCI_HDC_PERMS"
    fi

    if [[ -f "${RANK_SAGE}/ranking.csv" ]]; then
        echo "[skip] SAGE ranking exists: ${RANK_SAGE}/ranking.csv"
    else
        echo "[run]  SAGE ranking"
        python explain/sage_rank_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$DATASET_NPZ" \
            --student_dir "$FULL_BC_DIR" \
            --n_permutations "$SAGE_PERMS" \
            --output_dir "$RANK_SAGE"
    fi

    if [[ -f "${RANK_SHAP}/ranking.csv" ]]; then
        echo "[skip] KernelSHAP ranking exists: ${RANK_SHAP}/ranking.csv"
    else
        echo "[run]  KernelSHAP ranking"
        python explain/shap_rank_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$DATASET_NPZ" \
            --student_dir "$FULL_BC_DIR" \
            --background_size "$SHAP_BG" --explain_size "$SHAP_EXP" \
            --output_dir "$RANK_SHAP"
    fi

    # ── Stage 4: AIRL students per (selector, k) ───────────────────────────
    declare -A RANK_PATHS=(
        [mci_nn]="${RANK_MCI_NN}/ranking.csv"
        [mci_hdc]="${RANK_MCI_HDC}/ranking.csv"
        [sage]="${RANK_SAGE}/ranking.csv"
        [kernel_shap]="${RANK_SHAP}/ranking.csv"
    )

    for SELECTOR in "${SELECTORS[@]}"; do
        STUDENT_DIR="outputs/students_irl/seals_ant/seed${SEED}/${SELECTOR}"
        SUMMARY_CSV="${STUDENT_DIR}/summary.csv"

        # Map selector → (--selector flag, --ranking_path) understood by
        # train_student_irl.py (which knows shap / mci / random / oracle / mi / full).
        case "$SELECTOR" in
            mci_nn|mci_hdc)
                SEL_FLAG="mci"; RANK_PATH="${RANK_PATHS[$SELECTOR]}" ;;
            sage|kernel_shap)
                SEL_FLAG="shap"; RANK_PATH="${RANK_PATHS[$SELECTOR]}" ;;
            random|full)
                SEL_FLAG="$SELECTOR"; RANK_PATH="" ;;
            *)
                echo "Unknown selector: $SELECTOR" ; exit 1 ;;
        esac

        if [[ -f "$SUMMARY_CSV" ]]; then
            echo "[skip] IRL student exists: $SUMMARY_CSV"
        else
            echo "[run]  IRL student: selector=$SELECTOR (flag=$SEL_FLAG)"
            RANK_ARG=""
            if [[ -n "$RANK_PATH" ]]; then RANK_ARG="--ranking_path $RANK_PATH"; fi
            python student/train_student_irl.py \
                --config "$CONFIG" --seed "$SEED" \
                --dataset_path "$DATASET_NPZ" \
                --selector "$SEL_FLAG" $RANK_ARG \
                --quick_eval_episodes "$QUICK_EVAL_EPISODES" \
                --output_dir "$STUDENT_DIR"
        fi

        # ── Stage 5: Online eval ───────────────────────────────────────────
        EVAL_DIR="outputs/eval/online_irl/seals_ant/seed${SEED}/${SELECTOR}"
        if [[ -f "${EVAL_DIR}/online_metrics.csv" ]]; then
            echo "[skip] Online eval exists: ${EVAL_DIR}/online_metrics.csv"
        else
            echo "[run]  Online eval: selector=$SELECTOR"
            python eval/eval_online_irl.py \
                --config "$CONFIG" \
                --student_dir "$STUDENT_DIR" \
                --output_dir "$EVAL_DIR" \
                --n_episodes "$ONLINE_EVAL_EPISODES" --seed "$SEED"
        fi
    done

    unset RANK_PATHS
done

echo ""
echo "═══════════════════════════════════════════════"
echo "seals/Ant-v1 AIRL state pipeline complete."
echo ""
echo "Results layout:"
echo "  outputs/datasets/seals_ant/seed{N}/dataset.npz"
echo "  outputs/students/seals_ant/seed{N}/full/full/model.pt   (full BC for SAGE/SHAP)"
echo "  outputs/rankings_{mci_nn,mci_hdc,sage,shap}/seals_ant/seed{N}/ranking.csv"
echo "  outputs/students_irl/seals_ant/seed{N}/{selector}/{k_label}/model.pt"
echo "  outputs/eval/online_irl/seals_ant/seed{N}/{selector}/online_metrics.csv"
echo ""
echo "Next: bash scripts/run_ant_latent_irl.sh   (latent-space variant)"
