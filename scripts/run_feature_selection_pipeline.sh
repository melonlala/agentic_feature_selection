#!/usr/bin/env bash
# Full feature-selection analysis for seals_ant + kitchen_complete, over the
# bc / irl / pc tasks, comparing 5 ranking methods, each evaluated by a student.
#
#   collect dataset → rank features (5 methods) → train+eval student on test
#
# Methods: kernelshap, mci_kernel, mci_nn (per-subset student retraining),
#          random, mi.
# Tasks  : bc (X→actions), irl (X→reward), pc (preference over reward fragments).
#
# Layout:
#   outputs/feature_selection/{env}/{task}/rankings/{method}/ranking.csv
#   outputs/feature_selection/{env}/{task}/students/{method}/{summary.csv,metadata.json,k*/}
# Per-cell failures are logged and skipped so the whole sweep still completes.
#
# Run from sverl_feature_distill/ in the `il` env (designed for nohup):
#   nohup bash scripts/run_feature_selection_pipeline.sh > outputs/feature_selection/run.log 2>&1 &

set -uo pipefail

SEED="${SEED:-0}"
ROOT="outputs/feature_selection"
mkdir -p "$ROOT"
FAILLOG="$ROOT/failures.log"
: > "$FAILLOG"

TASKS=(bc irl pc)
METHODS=(kernelshap mci_kernel mci_nn random mi)

# env → config
declare -A CONFIG=(
    [seals_ant]="configs/seals_ant.yaml"
    [kitchen_complete]="configs/kitchen_complete.yaml"
)
# env → collector backend
declare -A BACKEND=(
    [seals_ant]="seals"
    [kitchen_complete]="d4rl"
)
ENVS=(seals_ant kitchen_complete)

run_step() {  # run_step "label" cmd...
    local label="$1"; shift
    echo ""; echo ">>>>> $label"
    if "$@"; then
        echo "<<<<< OK: $label"
    else
        echo "##### FAIL: $label" | tee -a "$FAILLOG"
    fi
}

# ── 1. Collect datasets (skip if present) ──────────────────────────────────
for ENV in "${ENVS[@]}"; do
    DS="outputs/datasets/${ENV}/seed${SEED}/dataset.npz"
    if [ -f "$DS" ]; then
        echo "[collect] ${ENV}: dataset exists, skipping."
        continue
    fi
    if [ "${BACKEND[$ENV]}" = "seals" ]; then
        run_step "collect ${ENV}" python teacher/collect_ant_expert_data.py \
            --config "${CONFIG[$ENV]}" --seed "$SEED" \
            --output_dir "outputs/datasets/${ENV}/seed${SEED}"
    else
        run_step "collect ${ENV}" python teacher/load_d4rl_dataset.py \
            --config "${CONFIG[$ENV]}" --seed "$SEED" \
            --output_dir "outputs/datasets/${ENV}/seed${SEED}"
    fi
done

# ── 2 + 3. Rank + student-evaluate every (env, task, method) ────────────────
for ENV in "${ENVS[@]}"; do
    CFG="${CONFIG[$ENV]}"
    DS="outputs/datasets/${ENV}/seed${SEED}/dataset.npz"
    [ -f "$DS" ] || { echo "##### FAIL: missing dataset $DS" | tee -a "$FAILLOG"; continue; }

    # mci_nn (per-subset retraining) cost knobs — lighter for high-D kitchen.
    if [ "$ENV" = "kitchen_complete" ]; then NN_PERMS=3; NN_EP=15; else NN_PERMS=4; NN_EP=20; fi

    for TASK in "${TASKS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            RANKDIR="$ROOT/$ENV/$TASK/rankings/$METHOD"
            STUDDIR="$ROOT/$ENV/$TASK/students/$METHOD"
            mkdir -p "$RANKDIR"

            case "$METHOD" in
                kernelshap)
                    run_step "rank $ENV/$TASK/$METHOD" python ranker/kernelshap.py \
                        --config "$CFG" --task "$TASK" --dataset_path "$DS" \
                        --output_dir "$RANKDIR" --seed "$SEED" ;;
                mci_kernel)
                    run_step "rank $ENV/$TASK/$METHOD" python ranker/mci_rank_kernel.py \
                        --evaluator_name "$TASK" --dataset_path "$DS" \
                        --output_dir "$RANKDIR" --seed "$SEED" ;;
                mci_nn)
                    run_step "rank $ENV/$TASK/$METHOD" python ranker/mci_subset_rank.py \
                        --config "$CFG" --task "$TASK" --dataset_path "$DS" \
                        --output_dir "$RANKDIR" --seed "$SEED" \
                        --n_perms "$NN_PERMS" --subset_epochs "$NN_EP" ;;
                random|mi)
                    run_step "rank $ENV/$TASK/$METHOD" python ranker/baseline_rank.py \
                        --method "$METHOD" --task "$TASK" --config "$CFG" \
                        --dataset_path "$DS" --output_dir "$RANKDIR" --seed "$SEED" ;;
            esac

            if [ -f "$RANKDIR/ranking.csv" ]; then
                run_step "student $ENV/$TASK/$METHOD" python student/train_student.py \
                    --config "$CFG" --task "$TASK" --dataset_path "$DS" \
                    --ranking_path "$RANKDIR/ranking.csv" \
                    --output_dir "$STUDDIR" --seed "$SEED"
            else
                echo "##### FAIL: no ranking.csv for $ENV/$TASK/$METHOD (skip student)" | tee -a "$FAILLOG"
            fi
        done
    done
done

# ── 4. Aggregate results ────────────────────────────────────────────────────
run_step "aggregate" python scripts/aggregate_feature_selection.py --root "$ROOT" --seed "$SEED"

echo ""
echo "================ PIPELINE COMPLETE ================"
if [ -s "$FAILLOG" ]; then
    echo "Some steps failed ($(wc -l < "$FAILLOG")):"; cat "$FAILLOG"
else
    echo "All steps succeeded."
fi
echo "Results under $ROOT/ ; summary: $ROOT/summary_seed${SEED}.csv"
