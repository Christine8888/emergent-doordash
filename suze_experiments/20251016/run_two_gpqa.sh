#!/bin/bash
# chmod +x suze_experiments/20251016/run_two_gpqa.sh
# bash suze_experiments/20251016/run_two_gpqa.sh
set -e
set -o pipefail

export PYTHONPATH="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/src:/afs/cs.stanford.edu/u/suzeva/emergent-doordash:$PYTHONPATH"

cleanup() {
    echo ""
    if [ -n "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
    $VLLM_UTILS_DIR/stop_vllm.sh
    exit 1
}

trap cleanup INT TERM

MODEL_A="allenai/OLMo-2-0425-1B" # read from huggingface
TP_A=1 # TP_A + TP_B = num gpus

MODEL_B="/sphinx/u/suzeva/emergent-doordash/test_20251015_212536/checkpoint-14245"
TP_B=1

# Parallelism per launch (use TP per model for simplicity)
N_DEVICES_DEFAULT=4

MAX_CONNECTIONS=32
HINT_FRACTIONS=(0.0 0.2 0.4 0.6 0.8 1.0)
FEWSHOTS=(0)
VLLM_PORT=5000
EPOCHS=5

VLLM_UTILS_DIR="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/src/utils"
CODE_DIR="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/christine_experiments/20251015"
EXPERIMENTS_DIR="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/suze_experiments/20251016/results"

# Build MODELS list from A and B
MODELS=(
"${MODEL_A}:${TP_A}"
"${MODEL_B}:${TP_B}"
)
# ========================================================

# Pre-clean ports to avoid EADDRINUSE (load balancer + up to 4 backends)
echo "Initial cleanup..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "load_balancer" 2>/dev/null || true
sleep 5

for p in $VLLM_PORT $((VLLM_PORT+1)) $((VLLM_PORT+2)) $((VLLM_PORT+3)) $((VLLM_PORT+4)); do
    lsof -ti:$p | xargs -r kill -9 2>/dev/null || true
done
sleep 5
echo "Initial cleanup complete."

for MODEL_SPEC in "${MODELS[@]}"; do
    MODEL="${MODEL_SPEC%%:*}"
    TP="${MODEL_SPEC##*:}"
    MODEL_NAME="${MODEL##*/}"
    MAX_WAIT=1200

    # Choose number of devices; simplest is equal to TP
    N_DEVICES=$TP
    if [ "$N_DEVICES" -lt 1 ]; then
        N_DEVICES=$N_DEVICES_DEFAULT
    fi

    # Ensure ports are free before each model launch
    echo "Cleaning up ports and processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "load_balancer" 2>/dev/null || true
    sleep 5
    
    for p in $VLLM_PORT $((VLLM_PORT+1)) $((VLLM_PORT+2)) $((VLLM_PORT+3)) $((VLLM_PORT+4)); do
        lsof -ti:$p | xargs -r kill -9 2>/dev/null || true
    done
    sleep 5
    
    # Verify ports are actually free
    for p in $VLLM_PORT $((VLLM_PORT+1)) $((VLLM_PORT+2)) $((VLLM_PORT+3)) $((VLLM_PORT+4)); do
        if lsof -ti:$p >/dev/null 2>&1; then
            echo "WARNING: Port $p still in use, killing again..."
            lsof -ti:$p | xargs -r kill -9 2>/dev/null || true
            sleep 3
        fi
    done

    echo "Starting vLLM server for $MODEL_NAME... on port $VLLM_PORT"
    $VLLM_UTILS_DIR/start_vllm.sh $MODEL $TP $MODEL_NAME $N_DEVICES $VLLM_PORT 4096 &
    VLLM_PID=$!


    ELAPSED=0
    while ! curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1; do
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
        LOG_DIR="$EXPERIMENTS_DIR/gpqa/${FEWSHOT}shot/$MODEL_NAME"

        for HINT_FRACTION in "${HINT_FRACTIONS[@]}"; do
            echo "  Running with fewshot=$FEWSHOT, hint_fraction=$HINT_FRACTION"
            cd $CODE_DIR
            python gpqa_hint_eval.py \
                --model vllm/$MODEL_NAME \
                --fewshot $FEWSHOT \
                --hint_fraction $HINT_FRACTION \
                --max_connections $MAX_CONNECTIONS \
                --log_dir $LOG_DIR \
                --base_port $VLLM_PORT \
                --epochs $EPOCHS
        done
    done

    echo "Stopping vLLM server for $MODEL_NAME..."
    kill $VLLM_PID 2>/dev/null || true
    $VLLM_UTILS_DIR/stop_vllm.sh
    
    # Give extra time for cleanup before next model
    echo "Waiting for complete cleanup..."
    sleep 15
done

echo "All experiments completed!"


