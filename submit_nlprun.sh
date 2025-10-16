#!/usr/bin/env bash

# Simple nlprun sweep launcher for sft.py
# Usage:
#   ./submit_nlprun.sh key1=val1 key2=val2
# Any key=val pairs are forwarded as OmegaConf overrides to sft.py for all jobs.

set -euo pipefail

# Resources
GPUS=4
MEM=64G
CPUS=16
MACHINE="sphinx11"
PARTITION="standard"
NPROC_PER_NODE=4

# Environment
CONDA_BASE="/nlp/scr/suzeva/miniconda3"
CONDA_ENV="olmo-env2"

# Paths
PROJECT_DIR="/afs/cs.stanford.edu/u/suzeva/emergent-doordash"
SCRIPT_PATH="${PROJECT_DIR}/suze_experiments/20251014/sft.py"
CONFIG_PATH="${PROJECT_DIR}/suze_experiments/20251014/default_config.yaml"

# Naming
EXP_BASE="sft_sweep"

# Sweep pairs: "model_id|revision"
SWEEP_PAIRS=(
  "allenai/OLMo-2-0425-1B|main"
  "allenai/OLMo-2-0425-1B|stage1-step0-tokens0B"
)

OVERRIDES=("$@")

submit_job() {
  local model_id="$1"
  local revision="$2"
  local tag_model="${model_id//\//-}"
  local tag_rev="${revision//\//-}"
  local exp_name="${EXP_BASE}_$(date +%F_%H%M%S)_${tag_model}_${tag_rev}"
  local log_name="${exp_name}.log"

  local CMD="bash -c \"source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \\
python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} ${SCRIPT_PATH} config=${CONFIG_PATH} \\
experiment_name=${exp_name} model_id=${model_id} revision=${revision} ${OVERRIDES[*]}\""

  set -x
  nlprun -g ${GPUS} -m ${MACHINE} -r ${MEM} -p ${PARTITION} -c ${CPUS} -o ${log_name} "${CMD}"
  set +x
}

for pair in "${SWEEP_PAIRS[@]}"; do
  IFS='|' read -r MID REV <<<"${pair}"
  submit_job "${MID}" "${REV}"
done
