#!/usr/bin/env bash
# Latent-space AIRL pipeline on seals/Ant-v1.
#
# Mirrors scripts/run_ant_irl.sh but ranks + trains on the first-layer latent
# of a full-feature BC student (instead of the raw 29-dim obs). All outputs
# go to *_latent paths so the state-space pipeline is untouched.
#
# Stages per seed (state pipeline must have run first):
#   1. Assume   dataset.npz + full BC student already exist (from run_ant_irl.sh).
#   2. Extract  latent dataset → outputs/datasets_latent/seals_ant/seed{N}/latent_dataset.npz
#   3. Rank     latents (5 methods, _latent suffix)
#   4. Train    AIRL students on latent subset (via --latent_mode flag)
#   5. Online   eval (re-uses eval/eval_online_irl.py, which honours ckpt[latent_mode]=True)
#
# Prerequisites:
#   - conda env `il` activated
#   - bash scripts/run_ant_irl.sh has run for the requested seeds
#   - Run from sverl_feature_distill/
#
# Usage:
#   bash scripts/run_ant_latent_irl.sh
#   bash scripts/run_ant_latent_irl.sh "0 1"

set -euo pipefail

CONFIG="configs/seals_ant.yaml"
SEEDS=(0 1 2)
# Selectors that have a meaningful latent analog (oracle/mi excluded — no
# ground-truth latent oracle, and MI on latents already action-trained is degenerate).
SELECTORS=(mci_nn mci_hdc sage kernel_shap random full)
LATENT_LAYER="pre_relu"

MCI_NN_PERMS=20
MCI_NN_EPOCHS=20
MCI_HDC_PERMS=100
MCI_HDC_RFF=32
MCI_HDC_BW=1.0
MCI_HDC_LAM=0.001
SAGE_PERMS=200
SHAP_BG=200
SHAP_EXP=200

QUICK_EVAL_EPISODES=10
ONLINE_EVAL_EPISODES=20

if [[ $# -ge 1 ]]; then
    IFS=' ' read -r -a SEEDS <<< "$1"
fi

echo "===== seals/Ant-v1 AIRL pipeline (latent space, layer=$LATENT_LAYER) ====="
echo "Config:    $CONFIG"
echo "Seeds:     ${SEEDS[*]}"
echo "Selectors: ${SELECTORS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  SEED $SEED"
    echo "═══════════════════════════════════════════════"

    STATE_DATASET="outputs/datasets/seals_ant/seed${SEED}/dataset.npz"
    FULL_BC_PT="outputs/students/seals_ant/seed${SEED}/full/full/model.pt"
    LATENT_DIR="outputs/datasets_latent/seals_ant/seed${SEED}"
    LATENT_NPZ="${LATENT_DIR}/latent_dataset.npz"

    if [[ ! -f "$STATE_DATASET" ]]; then
        echo "[err]  Missing state dataset: $STATE_DATASET — run run_ant_irl.sh first."
        continue
    fi
    if [[ ! -f "$FULL_BC_PT" ]]; then
        echo "[err]  Missing full BC student: $FULL_BC_PT — run run_ant_irl.sh first."
        continue
    fi

    # ── Stage 2: Extract latents ───────────────────────────────────────────
    if [[ -f "$LATENT_NPZ" ]]; then
        echo "[skip] Latent dataset exists: $LATENT_NPZ"
    else
        echo "[run]  Extract latents → $LATENT_NPZ"
        python explain/latent_extract.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$STATE_DATASET" \
            --full_student_path "$FULL_BC_PT" \
            --output_path "$LATENT_NPZ" \
            --latent_layer "$LATENT_LAYER"
    fi

    # ── Stage 2b: Latent-space full BC (powers SAGE/SHAP on latents) ───────
    # NOTE: we re-train a BC student on the latent inputs because SAGE/SHAP
    # need a model that maps latents → actions. The original full BC student
    # maps raw obs → actions, so it can't be SHAPed in the latent space.
    LATENT_FULL_BC_DIR="outputs/students_latent/seals_ant/seed${SEED}/full"
    LATENT_FULL_BC_PT="${LATENT_FULL_BC_DIR}/full/model.pt"
    if [[ -f "$LATENT_FULL_BC_PT" ]]; then
        echo "[skip] Latent full BC exists: $LATENT_FULL_BC_PT"
    else
        echo "[run]  Train full BC on latent inputs (SAGE/SHAP target)"
        python student/train_student_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$LATENT_NPZ" \
            --selector full \
            --output_dir "$LATENT_FULL_BC_DIR"
    fi

    # ── Stage 3: Latent rankings (5 methods) ───────────────────────────────
    RANK_MCI_NN_L="outputs/rankings_mci_nn_latent/seals_ant/seed${SEED}"
    RANK_MCI_HDC_L="outputs/rankings_mci_hdc_latent/seals_ant/seed${SEED}"
    RANK_SAGE_L="outputs/rankings_sage_latent/seals_ant/seed${SEED}"
    RANK_SHAP_L="outputs/rankings_shap_latent/seals_ant/seed${SEED}"

    if [[ -f "${RANK_MCI_NN_L}/ranking.csv" ]]; then
        echo "[skip] MCI-NN latent ranking exists"
    else
        echo "[run]  MCI-NN ranking on latents"
        python explain/mci_rank_nn.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$LATENT_NPZ" \
            --n_perms "$MCI_NN_PERMS" --epochs "$MCI_NN_EPOCHS" \
            --aggregation mean \
            --output_dir "$RANK_MCI_NN_L"
    fi

    if [[ -f "${RANK_MCI_HDC_L}/ranking.csv" ]]; then
        echo "[skip] MCI-HDC latent ranking exists"
    else
        echo "[run]  MCI-HDC ranking on latents"
        python explain/mci_rank.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$LATENT_NPZ" \
            --output_dir "$RANK_MCI_HDC_L" \
            --target action_multi \
            --rff_dim "$MCI_HDC_RFF" --bandwidth "$MCI_HDC_BW" \
            --lambda_ "$MCI_HDC_LAM" --n_perms "$MCI_HDC_PERMS"
    fi

    if [[ -f "${RANK_SAGE_L}/ranking.csv" ]]; then
        echo "[skip] SAGE latent ranking exists"
    else
        echo "[run]  SAGE ranking on latents"
        python explain/sage_rank_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$LATENT_NPZ" \
            --student_dir "$LATENT_FULL_BC_DIR" \
            --n_permutations "$SAGE_PERMS" \
            --output_dir "$RANK_SAGE_L"
    fi

    if [[ -f "${RANK_SHAP_L}/ranking.csv" ]]; then
        echo "[skip] KernelSHAP latent ranking exists"
    else
        echo "[run]  KernelSHAP ranking on latents"
        python explain/shap_rank_continuous.py \
            --config "$CONFIG" --seed "$SEED" \
            --dataset_path "$LATENT_NPZ" \
            --student_dir "$LATENT_FULL_BC_DIR" \
            --background_size "$SHAP_BG" --explain_size "$SHAP_EXP" \
            --output_dir "$RANK_SHAP_L"
    fi

    # ── Stage 4: AIRL students (latent_mode) ───────────────────────────────
    declare -A RANK_PATHS_L=(
        [mci_nn]="${RANK_MCI_NN_L}/ranking.csv"
        [mci_hdc]="${RANK_MCI_HDC_L}/ranking.csv"
        [sage]="${RANK_SAGE_L}/ranking.csv"
        [kernel_shap]="${RANK_SHAP_L}/ranking.csv"
    )

    for SELECTOR in "${SELECTORS[@]}"; do
        STUDENT_DIR="outputs/students_irl_latent/seals_ant/seed${SEED}/${SELECTOR}"
        SUMMARY_CSV="${STUDENT_DIR}/summary.csv"

        case "$SELECTOR" in
            mci_nn|mci_hdc)
                SEL_FLAG="mci"; RANK_PATH="${RANK_PATHS_L[$SELECTOR]}" ;;
            sage|kernel_shap)
                SEL_FLAG="shap"; RANK_PATH="${RANK_PATHS_L[$SELECTOR]}" ;;
            random|full)
                SEL_FLAG="$SELECTOR"; RANK_PATH="" ;;
            *)
                echo "Unknown selector: $SELECTOR" ; exit 1 ;;
        esac

        if [[ -f "$SUMMARY_CSV" ]]; then
            echo "[skip] Latent IRL student exists: $SUMMARY_CSV"
        else
            echo "[run]  Latent IRL student: selector=$SELECTOR (flag=$SEL_FLAG)"
            RANK_ARG=""
            if [[ -n "$RANK_PATH" ]]; then RANK_ARG="--ranking_path $RANK_PATH"; fi
            python student/train_student_irl.py \
                --config "$CONFIG" --seed "$SEED" \
                --dataset_path "$STATE_DATASET" \
                --selector "$SEL_FLAG" $RANK_ARG \
                --latent_mode \
                --full_student_path "$FULL_BC_PT" \
                --latent_layer "$LATENT_LAYER" \
                --quick_eval_episodes "$QUICK_EVAL_EPISODES" \
                --output_dir "$STUDENT_DIR"
        fi

        # ── Stage 5: Online eval ───────────────────────────────────────────
        EVAL_DIR="outputs/eval/online_irl_latent/seals_ant/seed${SEED}/${SELECTOR}"
        if [[ -f "${EVAL_DIR}/online_metrics.csv" ]]; then
            echo "[skip] Latent online eval exists: ${EVAL_DIR}/online_metrics.csv"
        else
            echo "[run]  Latent online eval: selector=$SELECTOR"
            python eval/eval_online_irl.py \
                --config "$CONFIG" \
                --student_dir "$STUDENT_DIR" \
                --output_dir "$EVAL_DIR" \
                --n_episodes "$ONLINE_EVAL_EPISODES" --seed "$SEED"
        fi
    done

    unset RANK_PATHS_L
done

echo ""
echo "═══════════════════════════════════════════════"
echo "seals/Ant-v1 AIRL latent pipeline complete."
echo ""
echo "Results layout:"
echo "  outputs/datasets_latent/seals_ant/seed{N}/latent_dataset.npz"
echo "  outputs/students_latent/seals_ant/seed{N}/full/full/model.pt   (BC on latents)"
echo "  outputs/rankings_{mci_nn,mci_hdc,sage,shap}_latent/seals_ant/seed{N}/ranking.csv"
echo "  outputs/students_irl_latent/seals_ant/seed{N}/{selector}/{k_label}/model.pt"
echo "  outputs/eval/online_irl_latent/seals_ant/seed{N}/{selector}/online_metrics.csv"
echo ""
echo "Compare state vs latent returns with eval/compare_state_vs_latent.py "
echo "(or write a small wrapper that reads online_metrics.csv instead of offline)."
