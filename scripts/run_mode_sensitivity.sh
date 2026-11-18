#!/usr/bin/env bash
# Run from any directory. Every new run has its own output directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dataset=${1:?Usage: bash scripts/run_mode_sensitivity.sh DATASET stage1|train|test|summary [SETTING|all] [SEED|all]}
phase=${2:?Missing phase}
setting_filter=${3:-all}
seed_filter=${4:-all}
case "$dataset" in cholectrack20|phakir) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
case "$setting_filter" in all|k2|k6|weight05|weight2) ;; *) echo "Unknown setting: $setting_filter" >&2; exit 2 ;; esac
case "$seed_filter" in all|42|123|3407) ;; *) echo "Unknown seed: $seed_filter" >&2; exit 2 ;; esac

if [[ "$phase" == stage1 ]]; then
    case "$setting_filter" in all|k2|k6) ;; *) echo "Stage 1 setting must be k2 or k6" >&2; exit 2 ;; esac
    for k in 2 6; do
        [[ "$setting_filter" == all || "$setting_filter" == "k${k}" ]] || continue
        config="configs/sensitivity/${dataset}_stage1_k${k}.yaml"
        out="outputs/${dataset}_stage1_geometry_k${k}"
        [[ ! -e "$out" ]] || { echo "Refusing to reuse Stage 1 output: $out" >&2; exit 1; }
        python scripts/prepare_data.py --config "$config"
        python stage1/fit_geometry.py --config "$config"
        python stage1/build_qgt_geometry.py --config "$config"
    done
    exit 0
fi

if [[ "$phase" == summary ]]; then
    python analysis/summarize_mode_sensitivity.py --dataset "$dataset" \
        --output-dir "outputs/mode_sensitivity_summary_${dataset}"
    exit 0
fi

case "$phase" in train|test) ;; *) echo "Unknown phase: $phase" >&2; exit 2 ;; esac
settings=(k2 k6 weight05 weight2)
seeds=(42 123 3407)
for setting in "${settings[@]}"; do
    [[ "$setting_filter" == all || "$setting_filter" == "$setting" ]] || continue
    config="configs/sensitivity/${dataset}_full_${setting}.yaml"
    if [[ "$setting" == k2 || "$setting" == k6 ]]; then
        stage1_dir="outputs/${dataset}_stage1_geometry_${setting}"
    else
        stage1_dir="outputs/${dataset}_stage1_geometry"
    fi
    [[ -f "$stage1_dir/stage1/prototypes.npz" && -f "$stage1_dir/stage1/qgt_train.npz" ]] || {
        echo "Stage 1 artifacts are missing: $stage1_dir" >&2; exit 1;
    }
    for seed in "${seeds[@]}"; do
        [[ "$seed_filter" == all || "$seed_filter" == "$seed" ]] || continue
        run="outputs/${dataset}_full_sensitivity_${setting}_seed${seed}"
        # All CholecTrack20 sensitivity seeds run for 20 epochs; all PhaKIR
        # sensitivity seeds run for 67 epochs. Selection remains val meanADE.
        if [[ "$dataset" == cholectrack20 ]]; then
            epochs=20
        else
            epochs=67
        fi
        if [[ "$phase" == train ]]; then
            [[ ! -e "$run" ]] || { echo "Refusing to reuse training output: $run" >&2; exit 1; }
            python train.py --config "$config" \
                --set "training.seed=$seed" --set "experiment.output_dir=$run" \
                --set "training.epochs=$epochs" --set "training.checkpoint_every_epochs=$epochs"
        else
            checkpoint="$run/checkpoints/best_meanADE.pt"
            [[ -f "$checkpoint" ]] || { echo "Missing meanADE-selected checkpoint: $checkpoint" >&2; exit 1; }
            [[ ! -e "$run/evaluation_test" ]] || { echo "Refusing to overwrite test output: $run/evaluation_test" >&2; exit 1; }
            python evaluate.py --config "$config" \
                --set "training.seed=$seed" --set "experiment.output_dir=$run" \
                --set "training.epochs=$epochs" --set "training.checkpoint_every_epochs=$epochs" \
                --checkpoint "$checkpoint" --split test --output "$run/evaluation_test"
        fi
    done
done
