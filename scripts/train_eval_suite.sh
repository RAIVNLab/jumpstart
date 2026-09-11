#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

name=$(basename "$0" .sh)
mkdir -p "results/$name"
csv="results/$name.csv"
echo 'dataset,algorithm,mean_return,status' > "$csv"

datasets=(D4RL/antmaze/umaze-v1 atari/pong/expert-v0 mujoco/hopper/medium-v0)
continuous_algos=(act bc bcp bcq cql diffusion dt iql rebrac vqbet)
atari_algos=(bc bcp bcq cql dt)
failed=0

for dataset in "${datasets[@]}"; do
    algos=("${continuous_algos[@]}")
    if [[ "$dataset" == atari/* ]]; then
        algos=("${atari_algos[@]}")
    fi

    for algo in "${algos[@]}"; do
        # Short training run; increase these budgets as needed.
        case "$algo" in
            bcq|cql|iql|rebrac) train_args=(--n-steps 1000) ;;
            *) train_args=(--epochs 1) ;;
        esac
        if [[ "$algo" == bcp ]]; then
            train_args+=(--train-percentile 90)
        fi

        log_dir="results/$name/${dataset//\//_}_$algo"
        mkdir -p "$log_dir"
        echo "Training and evaluating $algo on $dataset"
        status=ok
        if ! uv run python main.py --algorithm "$algo" --dataset "$dataset" \
            "${train_args[@]}" --device cuda --seed 0 --no-compile-graph \
            --eval-with-env --eval-episodes 10 --eval-num-envs 1 \
            --project-name "$log_dir/training" 2>&1 | tee "$log_dir/run.log"; then
            status=failed
        fi
        metric=$(awk '/^Final evaluation metric:/ {metric=$NF}
            /Warning: Could not recover environment/ {recovery_failed=1}
            END {if (!recovery_failed) print metric}' "$log_dir/run.log")
        if [[ "$status" == failed || -z "$metric" ]]; then
            status=failed
            metric=""
            failed=1
        fi
        printf '%s,%s,%s,%s\n' "$dataset" "$algo" "$metric" "$status" >> "$csv"
    done
done

echo "Results: $csv"
exit "$failed"
