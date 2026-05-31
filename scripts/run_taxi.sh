#!/usr/bin/env bash
# Collect teacher-labeled datasets for seeds 0..4.
# Teacher was trained on clean (no-noise) observations; the collection env
# adds noise. collect_dataset.py auto-detects the mismatch and passes only
# the first 4 (clean) features to the teacher while recording the full
# noisy observation for student training.
# Run from the sverl_feature_distill/ directory.

set -euo pipefail

NOISE_CONFIG="configs/taxi_noise8.yaml"
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
    echo "===== Collecting dataset seed=$SEED ====="
    python teacher/collect_dataset.py \
        --config "$NOISE_CONFIG" \
        --seed "$SEED" \
        --teacher_ckpt "outputs/teachers/clean/seed${SEED}/model.zip" \
        --output_dir "outputs/datasets/taxi_noise8/seed${SEED}"
done

echo "All datasets collected."

#!/usr/bin/env bash
# Train DQN teachers WITHOUT noise for seeds 0..4.
# The teacher is trained on clean 4-feature observations (no noise injected).
# Run from the sverl_feature_distill/ directory.

set -euo pipefail

CONFIG="configs/taxi_clean.yaml"
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
    echo "===== Training teacher (clean) seed=$SEED ====="
    python teacher/train_teacher.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --output_dir "outputs/teachers/clean/seed${SEED}"
done

echo "All teachers trained."


#!/usr/bin/env bash
# HDC-encoded MCI feature ranking pipeline.
#
# Replaces the SHAP pipeline (run_shap.sh) with the HDC+MCI approach:
#   - No full-student pre-training required.
#   - Closed-form ridge regression on HDC-encoded features → much faster.
#   - Produces ranking.csv in the same schema → downstream scripts unchanged.
#
# Prerequisite: datasets must exist (run run_collect.sh first).
#
# Run from the sverl_feature_distill/ directory:
#   bash scripts/run_mci.sh

set -euo pipefail

CONFIG="configs/taxi_noise8.yaml"
SEEDS=(0 1 2 3 4)

# MCI hyperparameters (can be overridden by editing below)
RFF_DIM=64
BANDWIDTH=1.0
LAMBDA=0.001
N_PERMS=200

echo "===== MCI Feature Ranking Pipeline ====="
echo "Config: $CONFIG | Seeds: ${SEEDS[*]}"
echo "RFF_DIM=$RFF_DIM  BANDWIDTH=$BANDWIDTH  LAMBDA=$LAMBDA  N_PERMS=$N_PERMS"
echo ""

# --- Step 1: Compute MCI rankings per seed ---
COMPLETED_SEEDS=()
for SEED in "${SEEDS[@]}"; do
    DATASET="outputs/datasets/taxi_noise8/seed${SEED}/dataset.npz"
    if [[ ! -f "$DATASET" ]]; then
        echo "SKIP seed=$SEED — dataset not found: $DATASET"
        continue
    fi
    echo "===== MCI ranking seed=$SEED ====="
    python explain/mci_rank.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --dataset_path "$DATASET" \
        --output_dir "outputs/rankings_mci/taxi_noise8/seed${SEED}" \
        --rff_dim "$RFF_DIM" \
        --bandwidth "$BANDWIDTH" \
        --lambda_ "$LAMBDA" \
        --n_perms "$N_PERMS"
    COMPLETED_SEEDS+=("$SEED")
done

if [[ ${#COMPLETED_SEEDS[@]} -eq 0 ]]; then
    echo "ERROR: No datasets found for any seed. Run run_collect.sh first."
    exit 1
fi

# --- Step 2: Rank stability across seeds ---
echo "===== Rank stability across seeds ====="
RANKING_PATHS=()
for SEED in "${COMPLETED_SEEDS[@]}"; do
    RANKING_PATHS+=("outputs/rankings_mci/taxi_noise8/seed${SEED}/ranking.csv")
done

if [[ ${#COMPLETED_SEEDS[@]} -ge 2 ]]; then
    python explain/rank_stability.py \
        --ranking_paths "${RANKING_PATHS[@]}" \
        --output_dir "outputs/rankings_mci/taxi_noise8/stability"
else
    echo "SKIP rank stability — need ≥2 seeds, only ${#COMPLETED_SEEDS[@]} completed."
fi

echo ""
echo "MCI pipeline done."
echo ""
echo "To train selector students using MCI ranking:"
echo "  python student/train_student.py \\"
echo "      --config $CONFIG --seed 0 \\"
echo "      --dataset_path outputs/datasets/taxi_noise8/seed0/dataset.npz \\"
echo "      --selector shap \\"
echo "      --ranking_path outputs/rankings_mci/taxi_noise8/seed0/ranking.csv \\"
echo "      --output_dir outputs/students_mci/taxi_noise8/seed0/mci"



#!/usr/bin/env bash
# Step 1: Train a full-feature student (needed for SHAP).
# Step 2: Run SHAP explanation on the student policy.
# Step 3: Aggregate global feature rankings.
# Step 4: Compute rank stability across seeds.
# Run from the sverl_feature_distill/ directory.

set -euo pipefail

CONFIG="configs/taxi_noise8.yaml"
SEEDS=(0 1 2 3 4)

# --- Train full students first (SHAP is run on the student, not teacher) ---
for SEED in "${SEEDS[@]}"; do
    echo "===== Training full student seed=$SEED ====="
    python student/train_student.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --dataset_path "outputs/datasets/taxi_noise8/seed${SEED}/dataset.npz" \
        --selector full \
        --output_dir "outputs/students/taxi_noise8/seed${SEED}/full"
done

# --- SHAP on student + global ranking ---
for SEED in "${SEEDS[@]}"; do
    echo "===== SHAP explanation (student) seed=$SEED ====="
    python explain/shap_behavior.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --student_data "outputs/students/taxi_noise8/seed${SEED}/full" \
        --dataset_path "outputs/datasets/taxi_noise8/seed${SEED}/dataset.npz" \
        --output_dir "outputs/shap/taxi_noise8/seed${SEED}"

    echo "===== Global ranking seed=$SEED ====="
    python explain/global_rank.py \
        --input "outputs/shap/taxi_noise8/seed${SEED}/shap_values.npz" \
        --output_dir "outputs/rankings/taxi_noise8/seed${SEED}"
done

echo "===== Rank stability across seeds ====="
RANKING_PATHS=()
for SEED in "${SEEDS[@]}"; do
    RANKING_PATHS+=("outputs/rankings/taxi_noise8/seed${SEED}/ranking.csv")
done

python explain/rank_stability.py \
    --ranking_paths "${RANKING_PATHS[@]}" \
    --output_dir "outputs/rankings/taxi_noise8/stability"

echo "SHAP pipeline done."


#!/usr/bin/env bash
# Train students with all feature selectors (shap, random, oracle, mi, full)
# for seeds 0..4, then run offline and online evaluation.
# Requires SHAP rankings to exist (run run_shap.sh first).
# Run from the sverl_feature_distill/ directory.

set -euo pipefail

CONFIG="configs/taxi_noise8.yaml"
SEEDS=(0 1 2 3 4)
SELECTORS=(shap random oracle mi full)

for SEED in "${SEEDS[@]}"; do
    for SEL in "${SELECTORS[@]}"; do
        echo "===== Training student seed=$SEED selector=$SEL ====="

        RANKING_ARG=""
        if [ "$SEL" = "shap" ]; then
            RANKING_ARG="--ranking_path outputs/rankings/taxi_noise8/seed${SEED}/ranking.csv"
        fi

        python student/train_student.py \
            --config "$CONFIG" \
            --seed "$SEED" \
            --dataset_path "outputs/datasets/taxi_noise8/seed${SEED}/dataset.npz" \
            --selector "$SEL" \
            $RANKING_ARG \
            --output_dir "outputs/students/taxi_noise8/seed${SEED}/${SEL}"

        echo "===== Offline eval seed=$SEED selector=$SEL ====="
        python eval/eval_offline.py \
            --config "$CONFIG" \
            --dataset_path "outputs/datasets/taxi_noise8/seed${SEED}/dataset.npz" \
            --student_dir "outputs/students/taxi_noise8/seed${SEED}/${SEL}" \
            --output_dir "outputs/eval/offline/taxi_noise8/seed${SEED}/${SEL}"

        echo "===== Online eval seed=$SEED selector=$SEL ====="
        python eval/eval_online.py \
            --config "$CONFIG" \
            --seed "$SEED" \
            --student_dir "outputs/students/taxi_noise8/seed${SEED}/${SEL}" \
            --output_dir "outputs/eval/online/taxi_noise8/seed${SEED}/${SEL}"
    done
done

echo "All students trained and evaluated."
