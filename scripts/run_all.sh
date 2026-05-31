#!/usr/bin/env bash
# Full end-to-end pipeline (modified v2 order):
#   1. Train clean teachers (no noise)
#   2. Collect noisy datasets
#   3. Train full students + run SHAP on student + global ranking
#   4. Train selector students + evaluate
#   5. Plot
# Run from the sverl_feature_distill/ directory.

set -euo pipefail

echo "========================================="
echo " SVERL Feature Distillation — Full Pipeline"
echo "========================================="

bash scripts/run_teachers.sh
bash scripts/run_collect.sh
bash scripts/run_shap.sh
bash scripts/run_students.sh

echo "===== Generating plots ====="
python eval/make_plots.py \
    --input_root outputs \
    --output_dir outputs/plots

echo "========================================="
echo " Pipeline complete. Results in outputs/"
echo "========================================="
