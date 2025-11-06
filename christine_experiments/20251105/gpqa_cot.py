"""GPQA evaluation with prefill hints.

Example usage:
    python gpqa_hint_eval.py --model vllm/Qwen2.5-0.5B-Instruct --hint_fraction 0.8
"""

from utils.eval_utils import create_base_parser, setup_vllm_env, check_output_exists, run_eval, get_valid_problem_ids
from utils.setup import setup_logging
from environments.gpqa.gpqa import gpqa_diamond, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, prefill, generate

logger = setup_logging()

BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/20251030/gpqa_diamond_samples_filtered.jsonl"

if __name__ == "__main__":
    # Parse arguments
    parser = create_base_parser(default_log_dir="./gpqa")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/gpqa_diamond_{args.hint_fraction}.json"
    check_output_exists(filename)

    # Get valid problem IDs (intersection of all data files if multiple exist)
    sample_ids = get_valid_problem_ids([DATA_PATH])
    logger.info(f"Running on {len(sample_ids)} samples with {args.hint_fraction} hint fraction")

    # Configure prefill
    prefill_config = PrefillConfig(path=DATA_PATH, fraction=args.hint_fraction)

    # Compose solver
    solver = [
        instructions(DEFAULT_INSTRUCTIONS),
        prefill(prefill_config),
        generate(timeout=args.timeout)
    ]

    # Create task
    task = gpqa_diamond(sample_ids=sample_ids, solver=solver)

    # Run evaluation
    run_eval(
        task=task,
        args=args,
        output_filename=filename,
        scorer_name='gpqa_scorer',
        extra_metadata={"hint_fraction": args.hint_fraction, "data_path": DATA_PATH}
    )
