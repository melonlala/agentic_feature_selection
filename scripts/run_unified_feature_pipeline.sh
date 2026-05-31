#!/usr/bin/env bash
# Unified feature-ranking + BC distillation pipeline.
#
# Runs the same six feature selectors — sage, mci_hdc, mci_nn, random, full,
# kernelshap — across multiple environments (kitchen_complete, seals_ant) in
# both state and latent feature spaces, with a single resumable driver.
#
# Combines scripts/run_d4rl_mci.sh + scripts/run_d4rl_latent_mci.sh +
# scripts/run_ant_irl.sh so we only have to maintain one entry point.
#
# Trainer: student/train_student_continuous.py (BC via imitation.bc.BC) is used
#          uniformly for both envs. Online RL eval is intentionally out of
#          scope — every selector's metrics come from the summary.csv that
#          train_student_continuous.py writes (val_mse, test_mse, cosine_sim).
#
# Output layout (each leaf directory's name is an "argument" — env, seed,
# space, selector, k — exactly as the user requested):
#
#   outputs/unified/{env}/seed{N}/
#     state/
#       dataset.npz                                       # raw observations
#       full_bc/full/model.pt                             # powers sage/kernelshap on state
#       rankings/{selector}/ranking.csv
#       students/{selector}/summary.csv  +  k{K}/model.pt
#     latent/
#       latent_dataset.npz                                # layer-1 latents of state full_bc
#       full_bc/full/model.pt                             # powers sage/kernelshap on latents
#       rankings/{selector}/ranking.csv
#       students/{selector}/summary.csv  +  k{K}/model.pt
#
# Usage:
#   bash scripts/run_unified_feature_pipeline.sh
#   bash scripts/run_unified_feature_pipeline.sh --envs "kitchen_complete"
#   bash scripts/run_unified_feature_pipeline.sh --seeds "0 1" \
#        --selectors "mci_hdc random full" --spaces "state"
#
# Resumability: every stage checks for its expected output file and skips if
# already present. Delete the output dir to force a re-run.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — overridable via CLI flags
# ─────────────────────────────────────────────────────────────────────────────
ENVS=(kitchen_complete seals_ant)
SEEDS=(0 1 2)
SPACES=(state latent)
SELECTORS=(mci_hdc mci_nn sage kernelshap random full)
LATENT_LAYER="pre_relu"

# Ranker hyperparameters — same defaults as the original per-env scripts.
MCI_HDC_RFF=64
MCI_HDC_BW=1.0
MCI_HDC_LAM=0.001
MCI_HDC_PERMS=200
MCI_HDC_MAX_SAMPLES=3000

# Latent space gets a narrower RFF block (latent dim is larger than raw obs).
MCI_HDC_RFF_LATENT=32

MCI_NN_PERMS=50
MCI_NN_EPOCHS=30
MCI_NN_AGG="mean"

SAGE_PERMS=200
SAGE_BG=200
SAGE_EXP=200

SHAP_BG=200
SHAP_EXP=200

# ─────────────────────────────────────────────────────────────────────────────
# Env registry
# ─────────────────────────────────────────────────────────────────────────────
declare -A ENV_CONFIG=(
    [kitchen_complete]="configs/kitchen_complete.yaml"
    [seals_ant]="configs/seals_ant.yaml"
)

# Build dataset.npz for a given env+seed at $OUT_DIR.
collect_kitchen_complete() {
    local seed="$1" out_dir="$2"
    python teacher/load_d4rl_dataset.py \
        --config "${ENV_CONFIG[kitchen_complete]}" \
        --seed "$seed" \
        --output_dir "$out_dir"
}
collect_seals_ant() {
    local seed="$1" out_dir="$2"
    python teacher/collect_ant_expert_data.py \
        --config "${ENV_CONFIG[seals_ant]}" \
        --seed "$seed" \
        --output_dir "$out_dir"
}

# Dispatcher: collect_$ENV.
collect_env_dataset() {
    local env="$1" seed="$2" out_dir="$3"
    local fn="collect_${env}"
    if ! declare -F "$fn" >/dev/null; then
        echo "ERROR: no dataset collector for env=$env (expected function ${fn})." >&2
        return 1
    fi
    "$fn" "$seed" "$out_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Selector registry
# ─────────────────────────────────────────────────────────────────────────────
# Each rank_<selector> function takes:
#   $1=config  $2=seed  $3=dataset_path  $4=ranking_out_dir
#   $5=full_bc_student_dir (used by sage / kernelshap; ignored otherwise)
#   $6=space   (state|latent; used to swap a few hyperparams for latent)
rank_mci_hdc() {
    local cfg="$1" seed="$2" ds="$3" out="$4" _full_bc="$5" space="$6"
    local rff="$MCI_HDC_RFF"
    [[ "$space" == "latent" ]] && rff="$MCI_HDC_RFF_LATENT"
    python explain/mci_rank.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$ds" --output_dir "$out" \
        --target action_multi \
        --rff_dim "$rff" --bandwidth "$MCI_HDC_BW" \
        --lambda_ "$MCI_HDC_LAM" --n_perms "$MCI_HDC_PERMS" \
        --max_samples "$MCI_HDC_MAX_SAMPLES"
}
rank_mci_nn() {
    local cfg="$1" seed="$2" ds="$3" out="$4" _full_bc="$5" _space="$6"
    python explain/mci_rank_nn.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$ds" --output_dir "$out" \
        --n_perms "$MCI_NN_PERMS" --epochs "$MCI_NN_EPOCHS" \
        --aggregation "$MCI_NN_AGG"
}
rank_sage() {
    local cfg="$1" seed="$2" ds="$3" out="$4" full_bc="$5" _space="$6"
    python explain/sage_rank_continuous.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$ds" --student_dir "$full_bc" \
        --output_dir "$out" \
        --background_size "$SAGE_BG" --explain_size "$SAGE_EXP" \
        --n_permutations "$SAGE_PERMS"
}
rank_kernelshap() {
    local cfg="$1" seed="$2" ds="$3" out="$4" full_bc="$5" _space="$6"
    python explain/shap_rank_continuous.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$ds" --student_dir "$full_bc" \
        --output_dir "$out" \
        --background_size "$SHAP_BG" --explain_size "$SHAP_EXP"
}

# Selectors that need a full-feature BC student as their explanation target.
selector_needs_full_bc() {
    case "$1" in
        sage|kernelshap) return 0 ;;
        *)               return 1 ;;
    esac
}

# Selectors that produce a ranking.csv (the others — random, full — don't).
selector_produces_ranking() {
    case "$1" in
        mci_hdc|mci_nn|sage|kernelshap) return 0 ;;
        *)                              return 1 ;;
    esac
}

# Map our 6-way selector → the per-selector training script under student/.
selector_train_script() {
    case "$1" in
        mci_hdc)    echo "student/train_student_mci_hdc.py" ;;
        mci_nn)     echo "student/train_student_mci_nn.py" ;;
        sage)       echo "student/train_student_sage.py" ;;
        kernelshap) echo "student/train_student_kernelshap.py" ;;
        random)     echo "student/train_student_random.py" ;;
        full)       echo "student/train_student_full.py" ;;
        *)          echo "Unknown selector: $1" >&2; return 1 ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────
ensure_state_dataset() {
    local env="$1" seed="$2" root="$3"
    local out_dir="${root}/state"
    local npz="${out_dir}/dataset.npz"
    if [[ -f "$npz" ]]; then
        echo "[skip] state dataset exists: $npz"
    else
        echo "[run]  collect state dataset → $npz"
        collect_env_dataset "$env" "$seed" "$out_dir"
    fi
}

ensure_full_bc() {
    # Train a full-feature BC student that powers sage/kernelshap on this space.
    local env="$1" seed="$2" cfg="$3" dataset_path="$4" space_root="$5"
    local out_dir="${space_root}/full_bc"
    local ckpt="${out_dir}/full/model.pt"
    if [[ -f "$ckpt" ]]; then
        echo "[skip] full BC exists: $ckpt"
        return
    fi
    echo "[run]  full-feature BC → $ckpt"
    python student/train_student_full.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$dataset_path" \
        --output_dir "$out_dir"
}

ensure_latent_dataset() {
    local env="$1" seed="$2" cfg="$3" root="$4"
    local state_npz="${root}/state/dataset.npz"
    local state_full_bc="${root}/state/full_bc/full/model.pt"
    local latent_dir="${root}/latent"
    local latent_npz="${latent_dir}/latent_dataset.npz"
    if [[ -f "$latent_npz" ]]; then
        echo "[skip] latent dataset exists: $latent_npz"
        return
    fi
    if [[ ! -f "$state_full_bc" ]]; then
        echo "[err]  cannot extract latents — missing state full BC: $state_full_bc"
        return 1
    fi
    echo "[run]  extract latents → $latent_npz"
    mkdir -p "$latent_dir"
    python explain/latent_extract.py \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$state_npz" \
        --full_student_path "$state_full_bc" \
        --output_path "$latent_npz" \
        --latent_layer "$LATENT_LAYER"
}

ensure_ranking() {
    # Produce ranking.csv for one (env, seed, space, selector) combination.
    local env="$1" seed="$2" cfg="$3" space="$4" selector="$5" root="$6"
    if ! selector_produces_ranking "$selector"; then
        return 0
    fi
    local space_root="${root}/${space}"
    local dataset_path
    case "$space" in
        state)  dataset_path="${space_root}/dataset.npz" ;;
        latent) dataset_path="${space_root}/latent_dataset.npz" ;;
        *)      echo "Unknown space: $space" >&2; return 1 ;;
    esac
    local full_bc_dir="${space_root}/full_bc"
    local rank_dir="${space_root}/rankings/${selector}"
    local rank_csv="${rank_dir}/ranking.csv"
    if [[ -f "$rank_csv" ]]; then
        echo "[skip] ranking exists ($space/$selector): $rank_csv"
        return
    fi
    if selector_needs_full_bc "$selector" && [[ ! -f "${full_bc_dir}/full/model.pt" ]]; then
        echo "[err]  selector=$selector needs full BC at ${full_bc_dir}/full/model.pt"
        return 1
    fi
    echo "[run]  rank: env=$env seed=$seed space=$space selector=$selector"
    local fn="rank_${selector}"
    "$fn" "$cfg" "$seed" "$dataset_path" "$rank_dir" "$full_bc_dir" "$space"
}

train_one_selector_student() {
    # Train a BC student for one (env, seed, space, selector) — one summary.csv
    # covers all topk values (the trainer iterates topk_list internally).
    local env="$1" seed="$2" cfg="$3" space="$4" selector="$5" root="$6"
    local space_root="${root}/${space}"
    local dataset_path
    case "$space" in
        state)  dataset_path="${space_root}/dataset.npz" ;;
        latent) dataset_path="${space_root}/latent_dataset.npz" ;;
    esac
    local out_dir="${space_root}/students/${selector}"
    local summary="${out_dir}/summary.csv"
    if [[ -f "$summary" ]]; then
        echo "[skip] student exists ($space/$selector): $summary"
        return
    fi

    local script
    script="$(selector_train_script "$selector")"

    local ranking_arg=""
    if selector_produces_ranking "$selector"; then
        local rank_csv="${space_root}/rankings/${selector}/ranking.csv"
        if [[ ! -f "$rank_csv" ]]; then
            echo "[err]  selector=$selector but ranking missing: $rank_csv"
            return 1
        fi
        ranking_arg="--ranking_path $rank_csv"
    fi

    echo "[run]  student: env=$env seed=$seed space=$space selector=$selector script=$script"
    # shellcheck disable=SC2086
    python "$script" \
        --config "$cfg" --seed "$seed" \
        --dataset_path "$dataset_path" \
        $ranking_arg \
        --output_dir "$out_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────────────────────────────────────
print_help() {
    cat <<EOF
Usage: bash scripts/run_unified_feature_pipeline.sh [options]
Options:
  --envs       "env1 env2 ..."         (default: ${ENVS[*]})
  --seeds      "s1 s2 ..."             (default: ${SEEDS[*]})
  --spaces     "state latent"          (default: ${SPACES[*]})
  --selectors  "mci_hdc mci_nn ..."    (default: ${SELECTORS[*]})
  --output_root DIR                    (default: outputs/unified)
  -h | --help                          show this help

Supported envs:      ${!ENV_CONFIG[*]}
Supported selectors: mci_hdc mci_nn sage kernelshap random full
EOF
}

OUTPUT_ROOT="outputs/unified"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --envs)       IFS=' ' read -r -a ENVS      <<< "$2"; shift 2 ;;
        --seeds)      IFS=' ' read -r -a SEEDS     <<< "$2"; shift 2 ;;
        --spaces)     IFS=' ' read -r -a SPACES    <<< "$2"; shift 2 ;;
        --selectors)  IFS=' ' read -r -a SELECTORS <<< "$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help)    print_help; exit 0 ;;
        *)            echo "Unknown flag: $1"; print_help; exit 1 ;;
    esac
done

# Validate selections.
for env in "${ENVS[@]}"; do
    if [[ -z "${ENV_CONFIG[$env]+_}" ]]; then
        echo "Unsupported env: $env (known: ${!ENV_CONFIG[*]})" >&2
        exit 1
    fi
done
for sel in "${SELECTORS[@]}"; do
    case "$sel" in
        mci_hdc|mci_nn|sage|kernelshap|random|full) ;;
        *) echo "Unsupported selector: $sel" >&2; exit 1 ;;
    esac
done
for sp in "${SPACES[@]}"; do
    case "$sp" in state|latent) ;;
                  *) echo "Unsupported space: $sp" >&2; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════════"
echo " Unified feature-ranking + BC pipeline"
echo "════════════════════════════════════════════════════════════"
echo " Envs:        ${ENVS[*]}"
echo " Seeds:       ${SEEDS[*]}"
echo " Spaces:      ${SPACES[*]}"
echo " Selectors:   ${SELECTORS[*]}"
echo " Output root: ${OUTPUT_ROOT}"
echo

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
need_latent=false
for sp in "${SPACES[@]}"; do
    [[ "$sp" == "latent" ]] && need_latent=true
done

# Did the user pick at least one selector that needs the full BC?
need_full_bc=false
for sel in "${SELECTORS[@]}"; do
    if selector_needs_full_bc "$sel"; then need_full_bc=true; fi
done

for env in "${ENVS[@]}"; do
    cfg="${ENV_CONFIG[$env]}"
    for seed in "${SEEDS[@]}"; do
        root="${OUTPUT_ROOT}/${env}/seed${seed}"
        mkdir -p "$root"

        echo
        echo "──────────────────────────────────────────────"
        echo "  env=${env}   seed=${seed}   cfg=${cfg}"
        echo "──────────────────────────────────────────────"

        # ── State pipeline ───────────────────────────────────────────────────
        if [[ " ${SPACES[*]} " == *" state "* ]] || [[ "$need_latent" == "true" ]]; then
            # State dataset is needed for state runs AND as a prerequisite for
            # latent extraction (the layer-1 latents come from a state full BC).
            ensure_state_dataset "$env" "$seed" "$root"
        fi

        state_npz="${root}/state/dataset.npz"
        if [[ "$need_full_bc" == "true" ]] || [[ "$need_latent" == "true" ]]; then
            ensure_full_bc "$env" "$seed" "$cfg" "$state_npz" "${root}/state"
        fi

        # ── Latent pipeline prep ─────────────────────────────────────────────
        if [[ "$need_latent" == "true" ]]; then
            ensure_latent_dataset "$env" "$seed" "$cfg" "$root"
            if [[ "$need_full_bc" == "true" ]]; then
                ensure_full_bc "$env" "$seed" "$cfg" \
                    "${root}/latent/latent_dataset.npz" "${root}/latent"
            fi
        fi

        # ── Rankings (per requested space × selector) ────────────────────────
        for space in "${SPACES[@]}"; do
            for sel in "${SELECTORS[@]}"; do
                ensure_ranking "$env" "$seed" "$cfg" "$space" "$sel" "$root"
            done
        done

        # ── Students + their inline offline eval (summary.csv) ───────────────
        for space in "${SPACES[@]}"; do
            for sel in "${SELECTORS[@]}"; do
                train_one_selector_student "$env" "$seed" "$cfg" "$space" "$sel" "$root"
            done
        done
    done
done

echo
echo "════════════════════════════════════════════════════════════"
echo " Unified pipeline complete."
echo
echo " Inspect results under: ${OUTPUT_ROOT}/{env}/seed{N}/{space}/"
echo "   rankings/{selector}/ranking.csv"
echo "   students/{selector}/summary.csv         (val_mse, test_mse, cosine_sim)"
echo "════════════════════════════════════════════════════════════"
