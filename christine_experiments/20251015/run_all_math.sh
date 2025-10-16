#!/bin/bash

set -e
set -o pipefail

cleanup() {
    echo ""
    echo "Caught interrupt signal, cleaning up..."
    if [ -n "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
    $VLLM_UTILS_DIR/stop_vllm.sh
    exit 1
}

trap cleanup INT TERM
export HF_HOME=/scr/biggest/cye/.cache/huggingface
export HOME=$SCR

# Format: "model_path:tensor_parallel"
MODELS=(
#"Qwen/Qwen2.5-0.5B-Instruct:1"
#"Qwen/Qwen2.5-1.5B-Instruct:1"
"Qwen/Qwen2.5-7B-Instruct:1"
"Qwen/Qwen2.5-3B-Instruct:1"
"Qwen/Qwen2.5-14B-Instruct:2"
"Qwen/Qwen2.5-32B-Instruct:4")

N_DEVICES=4
MAX_CONNECTIONS=64
HINT_FRACTIONS=(1.0 0.0 0.2 0.4 0.6 0.8)
FEWSHOTS=(0)
BASE_PORT=5000
VLLM_UTILS_DIR="$SPHINX/emergent-doordash/src/utils"
CODE_DIR="$SPHINX/emergent-doordash/christine_experiments/20251015"
EXPERIMENTS_DIR="$SPHINX/emergent-doordash/christine_experiments/20251015/results"
LIMIT=5000
EPOCHS=5

for MODEL_SPEC in "${MODELS[@]}"; do
    MODEL="${MODEL_SPEC%%:*}"
    TP="${MODEL_SPEC##*:}"
    MODEL_NAME="${MODEL##*/}"
    MAX_WAIT=1200

    echo "Starting vLLM server for $MODEL_NAME on port $BASE_PORT..."
    $VLLM_UTILS_DIR/start_vllm.sh $MODEL $TP $MODEL_NAME $N_DEVICES $BASE_PORT &
    VLLM_PID=$!

    ELAPSED=0
    while ! curl -s http://localhost:$BASE_PORT/health >/dev/null 2>&1; do
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server failed to start within ${MAX_WAIT}s"
            kill $VLLM_PID 2>/dev/null || true
            $VLLM_UTILS_DIR/stop_vllm.sh
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo "  Waiting... (${ELAPSED}s elapsed)"
    done

    echo "Running experiments for $MODEL_NAME..."

    for FEWSHOT in "${FEWSHOTS[@]}"; do
        LOG_DIR="$EXPERIMENTS_DIR/math/${FEWSHOT}shot/$MODEL_NAME"

        for HINT_FRACTION in "${HINT_FRACTIONS[@]}"; do
            echo "  Running with fewshot=$FEWSHOT, hint_fraction=$HINT_FRACTION"
            cd $CODE_DIR
            python math_hint_eval.py \
                --model vllm/$MODEL_NAME \
                --hint_fraction $HINT_FRACTION \
                --fewshot $FEWSHOT \
                --max_connections $MAX_CONNECTIONS \
                --log_dir $LOG_DIR \
                --limit $LIMIT \
                --base_port $BASE_PORT \
                --epochs $EPOCHS
        done
    done

    echo "Stopping vLLM server for $MODEL_NAME..."
    kill $VLLM_PID 2>/dev/null || true
    $VLLM_UTILS_DIR/stop_vllm.sh

    # Wait for ports to be fully released before starting next model
    echo "Waiting for ports to be released..."
    sleep 10
done

echo "All experiments completed!"
