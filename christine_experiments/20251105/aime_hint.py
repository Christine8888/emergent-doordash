"""AIME evaluation with prefill hints.

Example usage:
    python aime_hint_eval.py --model vllm/Qwen2.5-0.5B-Instruct --hint_fraction 0.8
"""

from utils.eval_utils import create_base_parser, setup_vllm_env, check_output_exists, run_eval
from utils.setup import setup_logging
from environments.aime.aime import aime
from environments.math.math import DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import format_prompt, add_prefill, generate_with_continuation

logger = setup_logging()

BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/20251030/aime_samples_filtered.jsonl"

if __name__ == "__main__":
    # Parse arguments
    parser = create_base_parser(default_log_dir="./aime")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/aime_{args.hint_fraction}.json"
    check_output_exists(filename)

    # Configure prefill
    prefill_config = PrefillConfig(path=DATA_PATH, fraction=args.hint_fraction)
    sample_ids = prefill_config.get_available_ids()
    logger.info(f"Running on {len(sample_ids)} samples with {args.hint_fraction} hint fraction")

    # Compose solver
    solver = [
        format_prompt(instruction_template=DEFAULT_INSTRUCTIONS),
        add_prefill(prefill_config),
        generate_with_continuation(timeout=args.timeout)
    ]

    # Create task
    task = aime(sample_ids=sample_ids, solver=solver)

    # Run evaluation
    run_eval(
        task=task,
        args=args,
        output_filename=filename,
        scorer_name='aime_scorer',
        extra_metadata={"hint_fraction": args.hint_fraction, "data_path": DATA_PATH}
    )
