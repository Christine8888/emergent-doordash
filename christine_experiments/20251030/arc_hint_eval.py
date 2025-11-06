"""ARC evaluation with prefill hints."""

from utils.eval_utils import create_base_parser, setup_vllm_env, check_output_exists, run_eval
from utils.setup import setup_logging
from environments.arc.arc import arc_task, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import format_prompt, add_prefill, generate_with_continuation

logger = setup_logging()

solver_name = "prefill"
eval_name = "arc"
BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/20251030/arc_filtered_long.jsonl"

if __name__ == "__main__":
    parser = create_base_parser(default_log_dir=f"./{eval_name}")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    parser.add_argument("--fewshot", type=int, default=0)
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/{eval_name}_{solver_name}_{args.fewshot}shot_{args.hint_fraction}.json"
    check_output_exists(filename)

    prefill_config = PrefillConfig(path=DATA_PATH, fraction=args.hint_fraction)
    sample_ids = prefill_config.get_available_ids()
    logger.info(f"Running on {len(sample_ids)} samples with {args.hint_fraction} hint fraction")

    solver = [
        format_prompt(instruction_template=DEFAULT_INSTRUCTIONS),
        add_prefill(prefill_config),
        generate_with_continuation(timeout=args.timeout)
    ]

    task = arc_task(sample_ids=sample_ids, solver=solver)

    run_eval(
        task=task,
        args=args,
        output_filename=filename,
        scorer_name=f'{eval_name}_scorer',
        extra_metadata={"hint_fraction": args.hint_fraction, "data_path": DATA_PATH, "solver_name": solver_name}
    )
