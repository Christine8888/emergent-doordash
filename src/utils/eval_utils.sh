#!/bin/bash
# Common utilities for running evaluations with vLLM

set -e
set -o pipefail

# Global variables that will be set
VLLM_PID=""

# Check if all output files exist for a given model and configuration
# Args:
#   $1: LOG_DIR_TEMPLATE (e.g., "$RESULTS_DIR/arc/{fewshot}shot/$MODEL_NAME")
#   $2: EVAL_NAME (e.g., "arc")
#   $3: SOLVER_NAME (e.g., "prefill", "solution", "baseline")
#   $4: MODEL_NAME (e.g., "Qwen2.5-0.5B-Instruct")
#   $5: FEWSHOTS array name (e.g., "FEWSHOTS")
#   $6: HINT_FRACTIONS array name (e.g., "HINT_FRACTIONS")
# Returns:
#   0 if all outputs exist, 1 otherwise
check_all_outputs_exist() {
    local LOG_DIR_TEMPLATE="$1"
    local EVAL_NAME="$2"
    local SOLVER_NAME="$3"
    local MODEL_NAME="$4"
    local -n fewshots_ref="$5"
    local -n fractions_ref="$6"

    local existing_files=()
    local missing_files=()

    for fewshot in "${fewshots_ref[@]}"; do
        for hint_fraction in "${fractions_ref[@]}"; do
            # Replace {fewshot} in template
            local log_dir="${LOG_DIR_TEMPLATE//\{fewshot\}/$fewshot}"

            # Format filename based on whether solver_name is empty
            if [ -z "$SOLVER_NAME" ]; then
                local filename="$log_dir/${EVAL_NAME}_${fewshot}shot_${hint_fraction}.json"
            else
                local filename="$log_dir/${EVAL_NAME}_${SOLVER_NAME}_${fewshot}shot_${hint_fraction}.json"
            fi

            if [ -f "$filename" ]; then
                existing_files+=("$filename")
            else
                missing_files+=("$filename")
            fi
        done
    done

    # Log what we found
    if [ ${#existing_files[@]} -eq 0 ]; then
        echo "  Found 0 existing output files for $MODEL_NAME"
    else
        echo "  Found ${#existing_files[@]} existing output file(s) for $MODEL_NAME:"
        for file in "${existing_files[@]}"; do
            echo "    ✓ $(basename "$file")"
        done
    fi

    # Return 0 if all exist, 1 otherwise
    [ ${#missing_files[@]} -eq 0 ]
}

# Cleanup function to stop vLLM
cleanup() {
    echo ""
    if [ -n "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
    "$ROOT/src/utils/stop_vllm.sh"
    exit 1
}

# Start vLLM server and wait for it to be ready
# Args:
#   $1: MODEL (e.g., Qwen/Qwen2.5-0.5B-Instruct)
#   $2: TP (tensor parallel size)
#   $3: MODEL_NAME (e.g., Qwen2.5-0.5B-Instruct)
#   $4: N_DEVICES (total number of devices)
#   $5: VLLM_PORT (port for vLLM)
#   $6: MAX_LENGTH (max sequence length)
#   $7: MAX_WAIT (optional, default 1200s)
start_vllm_and_wait() {
    local MODEL="$1"
    local TP="$2"
    local MODEL_NAME="$3"
    local N_DEVICES="$4"
    local VLLM_PORT="$5"
    local MAX_LENGTH="$6"
    local MAX_WAIT="${7:-1200}"

    echo "Starting vLLM server for $MODEL_NAME on port $VLLM_PORT..."
    "$ROOT/src/utils/start_vllm.sh" "$MODEL" "$TP" "$MODEL_NAME" "$N_DEVICES" "$VLLM_PORT" "$MAX_LENGTH" &
    VLLM_PID=$!

    local ELAPSED=0
    while ! curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1; do
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server failed to start within ${MAX_WAIT}s"
            kill $VLLM_PID 2>/dev/null || true
            "$ROOT/src/utils/stop_vllm.sh"
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo "  Waiting... (${ELAPSED}s elapsed)"
    done

    echo "vLLM server ready!"
}

# Stop vLLM server
stop_vllm() {
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    "$ROOT/src/utils/stop_vllm.sh"
    sleep 10
}

# Run a Python evaluation script
# Args:
#   $1: SCRIPT_PATH (full path to script, e.g., $ROOT/christine_experiments/20251030/arc_hint_eval.py)
#   $@: All remaining arguments are passed to the Python script
run_eval_script() {
    local SCRIPT_PATH="$1"
    shift  # Remove first argument, rest will be passed to python

    local SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

    cd "$SCRIPT_DIR"
    python "$(basename "$SCRIPT_PATH")" "$@"
}

# Run a full model sweep over fewshots and hint fractions
# This is the main entry point that orchestrates everything
# Args:
#   $1: SCRIPT_PATH (full path to Python eval script)
#   $2: EVAL_NAME (e.g., "arc", "gpqa")
#   $3: SOLVER_NAME (e.g., "prefill", "solution", "baseline")
#   $4: LOG_DIR_TEMPLATE (e.g., "$RESULTS_DIR/arc/{fewshot}shot/$MODEL_NAME")
#   $5: MODELS array name (e.g., "MODELS")
#   $6: FEWSHOTS array name (e.g., "FEWSHOTS")
#   $7: HINT_FRACTIONS array name (e.g., "HINT_FRACTIONS")
#   $8: N_DEVICES
#   $9: VLLM_PORT
#   $10: MAX_LENGTH
#   $11: MAX_CONNECTIONS
#   $12: EPOCHS
run_model_sweep() {
    local SCRIPT_PATH="$1"
    local EVAL_NAME="$2"
    local SOLVER_NAME="$3"
    local LOG_DIR_TEMPLATE="$4"
    local -n models_ref="$5"
    local -n fewshots_ref="$6"
    local -n fractions_ref="$7"
    local N_DEVICES="$8"
    local VLLM_PORT="$9"
    local MAX_LENGTH="${10}"
    local MAX_CONNECTIONS="${11}"
    local EPOCHS="${12}"

    for MODEL_SPEC in "${models_ref[@]}"; do
        local MODEL="${MODEL_SPEC%%:*}"
        local TP="${MODEL_SPEC##*:}"
        local MODEL_NAME="${MODEL##*/}"

        # Replace MODEL_NAME in template
        local LOG_DIR_FOR_MODEL="${LOG_DIR_TEMPLATE//\$MODEL_NAME/$MODEL_NAME}"

        # Check if all outputs already exist for this model
        if check_all_outputs_exist "$LOG_DIR_FOR_MODEL" "$EVAL_NAME" "$SOLVER_NAME" "$MODEL_NAME" "$6" "$7"; then
            echo "Skipping $MODEL_NAME - all outputs already exist"
            continue
        fi

        # Start vLLM and wait for it to be ready
        start_vllm_and_wait "$MODEL" "$TP" "$MODEL_NAME" "$N_DEVICES" "$VLLM_PORT" "$MAX_LENGTH"

        echo "Running experiments for $MODEL_NAME..."
        for fewshot in "${fewshots_ref[@]}"; do
            # Replace {fewshot} in template
            local LOG_DIR="${LOG_DIR_FOR_MODEL//\{fewshot\}/$fewshot}"

            for hint_fraction in "${fractions_ref[@]}"; do
                echo "  Running with fewshot=$fewshot, hint_fraction=$hint_fraction"

                run_eval_script "$SCRIPT_PATH" \
                    --model "vllm/$MODEL_NAME" \
                    --fewshot "$fewshot" \
                    --hint_fraction "$hint_fraction" \
                    --max_connections "$MAX_CONNECTIONS" \
                    --log_dir "$LOG_DIR" \
                    --base_port "$VLLM_PORT" \
                    --epochs "$EPOCHS"
            done
        done

        # Stop vLLM server
        stop_vllm
    done
}
