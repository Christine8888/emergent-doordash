# New Experiment System - Usage Guide

## Overview

The new experiment system uses **submitit** for job submission, with **one job per (model, fewshot, hint_fraction) combination**. This replaces the old bash orchestration system.

### Key Benefits:
- ✓ Parallel execution of all experiments
- ✓ Auto-resubmission of failed jobs
- ✓ No bash scripts needed
- ✓ Minimal boilerplate per experiment
- ✓ vLLM lifecycle managed inside each job
- ✓ SLURM handles GPU allocation (no CUDA_VISIBLE_DEVICES hacks)

---

## Quick Start

### 1. Define your experiment

Create `christine_experiments/YYYYMMDD/my_experiment.py`:

```python
from experiments.base_experiment import Experiment
from environments.gpqa.gpqa import gpqa_diamond, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, generate


class MyExperiment(Experiment):
    """Your experiment description."""

    name = "my_experiment"  # Used in filenames
    eval_name = "gpqa"  # Dataset name
    data_path = "/path/to/hints.jsonl"  # Path to hint data

    def build_task(self, hint_fraction: float, sample_ids: set[str]):
        """Build the Inspect task for this configuration."""
        prefill_config = PrefillConfig(
            path=self.data_path,
            fraction=hint_fraction
        )

        solver = [
            instructions(DEFAULT_INSTRUCTIONS),
            intext(prefill_config, prefix="Hint:\n"),
            generate(timeout=self.timeout)
        ]

        return gpqa_diamond(sample_ids=sample_ids, solver=solver)
```

### 2. Create launch script

Create `christine_experiments/YYYYMMDD/launch_my_experiment.py`:

```python
from utils.submitit_utils import launch_experiment, SubmitConfig
from my_experiment import MyExperiment  # Import from same directory

# Define the sweep
MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", 1),  # (model_path, tensor_parallel)
    ("Qwen/Qwen2.5-14B-Instruct", 2),
]

HINT_FRACTIONS = [1.0, 0.5, 0.0]
EPOCHS = 10

# Optional: override defaults
config = SubmitConfig(
    time_hours=24,
    mem_gb=128,
)

if __name__ == "__main__":
    jobs = launch_experiment(
        experiment_class=MyExperiment,
        models=MODELS,
        hint_fractions=HINT_FRACTIONS,
        epochs=EPOCHS,
        results_dir="./results",
        config=config,
    )

    print(f"Submitted {len(jobs)} jobs")
```

### 3. Launch!

```bash
cd christine_experiments/YYYYMMDD
python launch_my_experiment.py
```

---

## Configuration

### Global Defaults

Edit `src/utils/submitit_defaults.py` to change cluster-wide defaults:

```python
@dataclass
class SubmitConfig:
    partition: str = "sphinx"
    account: str = "nlp"
    cpus_per_task: int = 16
    mem_gb: int = 64
    time_hours: int = 20
    max_model_len: int = 16384
    max_connections: int = 32
    gpu_memory_utilization: float = 0.9
    timeout: int = 600
    max_retries: int = 3
    # ...
```

### Per-Experiment Overrides

Override in your launch script:

```python
config = SubmitConfig(
    partition="sphinx",      # Which SLURM partition
    time_hours=48,          # Longer time limit
    mem_gb=128,             # More memory
    max_model_len=32768,    # Longer context
)
```

### Per-Model GPUs

GPUs are automatically set from `tensor_parallel_size`:

```python
MODELS = [
    ("model-7B", 1),   # 1 GPU per job
    ("model-32B", 4),  # 4 GPUs per job
]
```

---

## Job Management

### Check Job Status

```python
from utils.submitit_utils import check_job_status

# After launching
jobs = launch_experiment(...)

# Check status
status_map = check_job_status(jobs)
# Prints: PENDING: 5, RUNNING: 10, DONE: 3, FAILED: 1
```

### Resubmit Failed Jobs

```python
from utils.submitit_utils import resubmit_failed_jobs

# Resubmit any FAILED or TIMEOUT jobs
new_jobs = resubmit_failed_jobs(jobs, max_retries=3)
```

### Wait for Completion

```python
from utils.submitit_utils import wait_for_jobs

# Wait with periodic status updates
wait_for_jobs(jobs, check_interval=300)  # Check every 5 minutes
```

### Full Monitoring Script

```python
if __name__ == "__main__":
    # Launch
    jobs = launch_experiment(...)

    # Wait and auto-resubmit failures
    wait_for_jobs(jobs)
    resubmit_failed_jobs(jobs)
```

---

## How It Works

### Job Execution Flow

Each submitit job:
1. Gets GPU allocation from SLURM (via `CUDA_VISIBLE_DEVICES`)
2. Starts vLLM server on allocated GPUs
3. Runs experiment (Inspect eval)
4. Shuts down vLLM server
5. Saves results to JSON

### Output Files

Results are saved to:
```
results_dir/
  {eval_name}/
    {experiment_name}/
      {fewshot}shot/
        {model_name}/
          {eval}_{experiment}_{fewshot}shot_{fraction}.json
```

Example:
```
results/gpqa/cot_intext/0shot/Qwen2.5-7B-Instruct/gpqa_cot_intext_0shot_0.5.json
```

### Skip Existing

By default, jobs are skipped if output file already exists:

```python
launch_experiment(..., skip_existing=True)  # Default
```

---

## Migration from Old System

### Old (Bash):
```bash
#!/bin/bash
MODELS=("Qwen/Qwen2.5-7B-Instruct:1")
HINT_FRACTIONS=(1.0 0.5 0.0)
run_model_sweep ...  # 50 lines of bash
```

### New (Python):
```python
from utils.submitit_utils import launch_experiment
from my_experiment import MyExperiment

MODELS = [("Qwen/Qwen2.5-7B-Instruct", 1)]
HINT_FRACTIONS = [1.0, 0.5, 0.0]

launch_experiment(
    experiment_class=MyExperiment,
    models=MODELS,
    hint_fractions=HINT_FRACTIONS,
    epochs=10,
    results_dir="./results",
)
```

**That's it!** No bash, no manual vLLM management, no orchestration.

---

## Troubleshooting

### Import Errors

Make sure you've installed the package:
```bash
cd /path/to/emergent-doordash
uv pip install -e .
```

### vLLM Won't Start

Check SLURM GPU allocation:
- Make sure `tensor_parallel_size` matches requested GPUs
- Logs are in `submitit_logs/`

### Jobs Failing Silently

Check submitit logs:
```bash
ls -lrt submitit_logs/
cat submitit_logs/{job_id}_0_log.err
```

### Port Conflicts

vLLM auto-finds free ports, but if issues occur, check:
```python
# In vllm_server.py logs
logger.info(f"Starting vLLM server on port {self.port}")
```

---

## Example: Full Launch Script with Monitoring

```python
"""Complete example with monitoring and resubmission."""

from utils.submitit_utils import (
    launch_experiment,
    wait_for_jobs,
    check_job_status,
    resubmit_failed_jobs,
    SubmitConfig
)
from my_experiment import MyExperiment

if __name__ == "__main__":
    # Configuration
    MODELS = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 1),
        ("Qwen/Qwen2.5-7B-Instruct", 1),
        ("Qwen/Qwen2.5-32B-Instruct", 4),
    ]
    HINT_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    config = SubmitConfig(time_hours=20)

    # Launch
    print("Launching experiments...")
    jobs = launch_experiment(
        experiment_class=MyExperiment,
        models=MODELS,
        hint_fractions=HINT_FRACTIONS,
        epochs=10,
        results_dir="./results",
        config=config,
    )

    print(f"\nSubmitted {len(jobs)} jobs")

    # Monitor
    print("\nWaiting for completion (checking every 5min)...")
    wait_for_jobs(jobs, check_interval=300)

    # Check final status
    print("\nFinal status:")
    status_map = check_job_status(jobs)

    # Resubmit failures
    if status_map.get('FAILED') or status_map.get('TIMEOUT'):
        print("\nResubmitting failed jobs...")
        new_jobs = resubmit_failed_jobs(jobs, max_retries=2)
        if new_jobs:
            wait_for_jobs(new_jobs)

    print("\n✓ All experiments complete!")
```

---

## Files Created

### Core Infrastructure
- `src/utils/vllm_server.py` - vLLM server management
- `src/utils/submitit_defaults.py` - Default configuration
- `src/utils/submitit_utils.py` - Job launching and monitoring
- `src/experiments/base_experiment.py` - Base experiment class

### Example Experiment
- `christine_experiments/20251105/gpqa_cot_intext_v2.py` - Experiment definition
- `christine_experiments/20251105/launch_gpqa_cot_intext.py` - Launch script

---

## Next Steps

1. Test the system with a small experiment
2. Migrate other experiments to new format
3. Remove old bash scripts once confirmed working
