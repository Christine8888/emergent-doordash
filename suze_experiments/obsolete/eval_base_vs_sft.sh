#!/bin/bash
# bash suze_experiments/20251016/eval_base_vs_sft.sh

set -e
set -o pipefail

# ================= CONSTANTS =================
BASE_PORT=5000
TP=1
N_DEVICES=1
EPOCHS=5
FEWSHOT=0
MAX_CONNECTIONS=20
HINTS=(0.0 0.2 0.4 0.6 0.8 1.0)
RESULTS_ROOT="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/suze_experiments/20251015/results/gpqa"
EVAL_SCRIPT="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/christine_experiments/20251015/gpqa_hint_eval.py"
START_VLLM="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/src/utils/start_vllm.sh"
STOP_VLLM="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/src/utils/stop_vllm.sh"

# Base model (non-SFT)
BASE_MODEL="allenai/OLMo-2-0425-1B"
BASE_NAME="OLMo-2-0425-1B-base"

# SFT checkpoint (directory path) and served name
SFT_CHECKPOINT="/sphinx/u/suzeva/emergent-doordash/test_20251015_212536/checkpoint-14245"
SFT_NAME="OLMo-2-0425-1B-sft"
# ============================================

wait_for_vllm() {
  local port=$1
  local elapsed=0
  local max_wait=1200
  # Poll the first backend (BASE_PORT+1); LB on BASE_PORT starts after backends are ready
  local backend_port=$((port+1))
  while ! curl -s "http://localhost:${backend_port}/health" >/dev/null 2>&1; do
    if [ $elapsed -ge $max_wait ]; then
      echo "Error: vLLM server failed to start within ${max_wait}s" >&2
      exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "  Waiting for vLLM... (${elapsed}s elapsed)"
  done
}

run_eval() {
  local model_path="$1"
  local served_name="$2"

  echo "Starting vLLM for ${served_name} on port ${BASE_PORT}..."
  "$START_VLLM" "$model_path" $TP "$served_name" $N_DEVICES $BASE_PORT &
  wait_for_vllm $BASE_PORT

  local log_dir="${RESULTS_ROOT}/${FEWSHOT}shot/${served_name}"
  mkdir -p "$log_dir"

  echo "Running GPQA for ${served_name} (fewshot=${FEWSHOT})..."
  for H in "${HINTS[@]}"; do
    echo "  hint_fraction=${H}"
    python "$EVAL_SCRIPT" \
      --model vllm/${served_name} \
      --fewshot ${FEWSHOT} \
      --hint_fraction ${H} \
      --max_connections ${MAX_CONNECTIONS} \
      --log_dir "$log_dir" \
      --base_port ${BASE_PORT} \
      --epochs ${EPOCHS}
  done

  echo "Stopping vLLM for ${served_name}..."
  "$STOP_VLLM"
}

# Run base and SFT evaluations
run_eval "$BASE_MODEL" "$BASE_NAME"
run_eval "$SFT_CHECKPOINT" "$SFT_NAME"