#!/bin/bash

set -e
set -o pipefail

if [ $# -lt 1 ] || [ $# -gt 7 ]; then
    echo "Usage: $0 <model_path> [tensor_parallel] [model_name] [n_devices] [base_port] [max_length] [chat_template]"
    echo "Examples:"
    echo "  $0 /workspace/model"
    echo "  $0 /workspace/model 2 my-model 4 9000 12800"
    echo "  $0 /workspace/model 2 my-model 4 9000 12800 'simple'"
    exit 1
fi

MODEL_PATH="$1"
TP="${2:-4}"
MODEL_NAME="${3:-}"
N_DEVICES="${4:-$TP}"
BASE_PORT="${5:-9000}"
MAX_LENGTH="${6:-12800}"
CHAT_TEMPLATE="${7:-}"

VLLM_PID=""
SHUTTING_DOWN=false

cleanup() {
    SHUTTING_DOWN=true
    echo "Shutting down server..."

    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true

    sleep 5
    pkill -f "vllm serve" 2>/dev/null || true

    lsof -ti:$BASE_PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true

    echo "Shutdown complete"
    exit 0
}

trap cleanup EXIT INT TERM

# Check if port is already in use and kill if needed
echo "Checking if port $BASE_PORT is free..."
if lsof -ti:$BASE_PORT >/dev/null 2>&1; then
    echo "Port $BASE_PORT is in use, killing existing process..."
    lsof -ti:$BASE_PORT | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# Kill any stray vllm processes
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

if [ $((N_DEVICES % TP)) -ne 0 ]; then
    echo "Error: n_devices ($N_DEVICES) must be divisible by tensor_parallel ($TP)"
    exit 1
fi

# Calculate data parallel size
DP=$((N_DEVICES / TP))

VLLM_ARGS=(
    --dtype auto
    --max-model-len $MAX_LENGTH
    --tensor-parallel-size $TP
    --data-parallel-size $DP
    --enable-prefix-caching
    --max-num-seqs 16
    --max-num-batched-tokens 65536
    --limit-mm-per-prompt.image 4
    --enable-chunked-prefill
    --gpu-memory-utilization 0.9
    --kv-cache-dtype auto
    --max-parallel-loading-workers 2
    --port $BASE_PORT
)

[ -n "$MODEL_NAME" ] && VLLM_ARGS+=(--served-model-name "$MODEL_NAME")

# Add chat template if provided
if [ -n "$CHAT_TEMPLATE" ]; then
    VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

echo "Starting vLLM with native data parallelism"
echo "Model: $MODEL_PATH"
echo "TP: $TP | DP: $DP | Total GPUs: $N_DEVICES"
echo "Port: $BASE_PORT"

CUDA_VISIBLE_DEVICES=$(seq 0 $((N_DEVICES - 1)) | tr '\n' ',' | sed 's/,$//')

echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for server..."
while ! curl -s http://localhost:$BASE_PORT/health >/dev/null 2>&1; do
    [ "$SHUTTING_DOWN" = true ] && exit 0
    sleep 2
done

echo ""
echo "============================================"
echo "vLLM server started successfully!"
echo "============================================"
echo "Server: http://localhost:$BASE_PORT"
echo "Tensor Parallel: $TP"
echo "Data Parallel: $DP"
echo "Total GPUs: $N_DEVICES"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"

wait
