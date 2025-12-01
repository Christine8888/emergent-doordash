#!/bin/bash

# Parameters
#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --error=/sphinx/u/cye/emergent-doordash/christine_experiments/20251113/submitit_logs/%j_0_log.err
#SBATCH --exclude=sphinx2
#SBATCH --gpus-per-node=1
#SBATCH --job-name=baseline_mmlu_0_shot_Qwen2.5-0.5B-Instruct
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/sphinx/u/cye/emergent-doordash/christine_experiments/20251113/submitit_logs/%j_0_log.out
#SBATCH --partition=sphinx
#SBATCH --signal=USR2@90
#SBATCH --time=600
#SBATCH --wckey=submitit

# setup
source /scr-ssd/cye/.venv/bin/activate
export HF_HOME=/sphinx/u/cye/.cache/huggingface

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/submitit_logs/%j_%t_log.out --error /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/submitit_logs/%j_%t_log.err --cpu-bind=none /sphinx/u/cye/.venv/bin/python -u -m submitit.core._submit /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/submitit_logs
