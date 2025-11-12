#!/bin/bash
#SBATCH --job-name=gpqa_cot
#SBATCH --output=gpqa_cot_intext.out
#SBATCH --error=gpqa_cot_intext.err
#SBATCH --time=20:00:00
#SBATCH --partition=sphinx
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --account=nlp
#SBATCH --ntasks-per-node=1

export SPHINX=/sphinx/u/cye
source /scr-ssd/cye/.venv/bin/activate
export HF_TOKEN=$(cat $SPHINX/emergent-doordash/hf.tok)
cd $SPHINX/emergent-doordash/christine_experiments/20251105

ROOT="$SPHINX/emergent-doordash"
export HF_HOME="/scr/biggest/cye/.cache/huggingface"
export HOME="/scr-ssd/cye"

source "$ROOT/src/utils/eval_utils.sh"

trap cleanup INT TERM

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct:1"
    "Qwen/Qwen2.5-1.5B-Instruct:1"
    "Qwen/Qwen2.5-3B-Instruct:1"
    "Qwen/Qwen2.5-7B-Instruct:1"
    "Qwen/Qwen2.5-14B-Instruct:2"
    "Qwen/Qwen2.5-32B-Instruct:4"
)

FEWSHOTS=(0)
HINT_FRACTIONS=(1.0 0.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

N_DEVICES=4
VLLM_PORT=3000
MAX_LENGTH=16384
MAX_CONNECTIONS=32
EPOCHS=10

EVAL_NAME="gpqa"
SOLVER_NAME="cot_intext" 
SCRIPT_PATH="$ROOT/christine_experiments/20251105/gpqa_cot_intext.py"
RESULTS_DIR="$ROOT/christine_experiments/20251105/results"

LOG_DIR_TEMPLATE="$RESULTS_DIR/$EVAL_NAME/$SOLVER_NAME/{fewshot}shot/\$MODEL_NAME"

run_model_sweep \
    "$SCRIPT_PATH" \
    "$EVAL_NAME" \
    "$SOLVER_NAME" \
    "$LOG_DIR_TEMPLATE" \
    MODELS \
    FEWSHOTS \
    HINT_FRACTIONS \
    "$N_DEVICES" \
    "$VLLM_PORT" \
    "$MAX_LENGTH" \
    "$MAX_CONNECTIONS" \
    "$EPOCHS"