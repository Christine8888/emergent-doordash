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

export HF_HOME="/scr/biggest/cye/.cache/huggingface"

# Format: "model_path:tensor_parallel"
MODELS=("Qwen/Qwen2.5-1.5B-Instruct:1"
"Qwen/Qwen2.5-14B-Instruct:2"
"Qwen/Qwen2.5-32B-Instruct:2")

N_DEVICES=4
MAX_CONNECTIONS=32
HINT_FRACTIONS=(0.0 0.2 0.4 0.6 0.8)
VLLM_UTILS_DIR="$NLP/emergent-doordash/src/utils"
EXPERIMENTS_DIR="$NLP/emergent-doordash/christine_experiments/20251007"

for MODEL_SPEC in "${MODELS[@]}"; do
    MODEL="${MODEL_SPEC%%:*}"
    TP="${MODEL_SPEC##*:}"
    MODEL_NAME="${MODEL##*/}"
    LOG_DIR="$EXPERIMENTS_DIR/gpqa/$MODEL_NAME"
    MAX_WAIT=1200

    echo "Starting vLLM server for $MODEL_NAME..."
    $VLLM_UTILS_DIR/start_vllm.sh $MODEL $TP $MODEL_NAME $N_DEVICES &
    VLLM_PID=$!

    ELAPSED=0
    while ! curl -s http://localhost:9000/health >/dev/null 2>&1; do
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
    for HINT_FRACTION in "${HINT_FRACTIONS[@]}"; do
        echo "  Running with hint_fraction=$HINT_FRACTION"
        cd $EXPERIMENTS_DIR
        python get_gpqa_traces_hint.py \
            --model vllm/$MODEL_NAME \
            --hint_fraction $HINT_FRACTION \
            --max_connections $MAX_CONNECTIONS \
            --log_dir $LOG_DIR
    done

    echo "Stopping vLLM server for $MODEL_NAME..."
    kill $VLLM_PID 2>/dev/null || true
    $VLLM_UTILS_DIR/stop_vllm.sh

    # Wait for ports to be fully released before starting next model
    echo "Waiting for ports to be released..."
    sleep 10
done

echo "All experiments completed!"
