"""Submit a tiny 1-GPU vLLM inference smoke test."""

import os
import sys
from pathlib import Path

# Defaults/constants (keep at top)
_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_PROMPT = "Reply with exactly: OK"
_DEFAULT_MAX_TOKENS = 8
_DEFAULT_PARTITION = "sphinx"
_DEFAULT_MEM_GB = 16
_DEFAULT_TIME_HOURS = 1

# Add src/ to path so imports work from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.model_config import ModelSpec
from utils.submitit_defaults import DEFAULT_CONFIG
from utils.submitit_utils import launch_smoke_inference
from utils.setup import setup_logging

logger = setup_logging()

# Load HF token based on current user (so vLLM can download models)
import getpass
_HF_TOKEN_PATHS = {
    "cye": "/sphinx/u/cye/emergent-doordash/hf.tok",
    "suzeva": "/afs/cs.stanford.edu/u/suzeva/hf.tok",
}
_token_path = _HF_TOKEN_PATHS.get(getpass.getuser(), _HF_TOKEN_PATHS["cye"])
with open(_token_path, "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit 1-GPU vLLM inference smoke test")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="HF model path")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    parser.add_argument("--max_tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--partition", type=str, default=_DEFAULT_PARTITION)
    parser.add_argument("--nodelist", type=str, default=None, help="Optional SLURM nodelist (leave empty for any node)")
    parser.add_argument("--mem_gb", type=int, default=_DEFAULT_MEM_GB)
    parser.add_argument("--time_hours", type=int, default=_DEFAULT_TIME_HOURS)
    parser.add_argument("--submitit_folder", type=str, default="./submitit_logs")
    parser.add_argument("--wait", action="store_true", help="Wait for job to finish and surface exception")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.override(
        partition=args.partition,
        nodelist=args.nodelist,
        mem_gb=args.mem_gb,
        time_hours=args.time_hours,
        submitit_folder=args.submitit_folder,
        gpus_per_job=1,
    )

    model = ModelSpec(args.model, tp=1)
    job = launch_smoke_inference(
        model=model,
        config=cfg,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        wait=args.wait,
    )
    logger.info(f"submitted smoke job_id={job.job_id}")


"""
python /afs/cs.stanford.edu/u/suzeva/emergent-doordash/suze_experiments/20251113/smoke_inference.py \
  --partition sphinx \
  --nodelist 'sphinx[1-11]'
"""