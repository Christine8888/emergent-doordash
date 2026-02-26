from utils.eval_utils import create_base_parser, setup_vllm_env, check_output_exists, run_eval, get_valid_problem_ids
from utils.setup import setup_logging
from environments.gpqa.gpqa import gpqa_diamond, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, generate
from inspect_ai.solver import solver as inspect_solver

logger = setup_logging()

solver_name = "solution_intext_mask"
eval_name = "gpqa"
BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/data/solution/{eval_name}.jsonl"

if __name__ == "__main__":
    parser = create_base_parser(default_log_dir=f"./{solver_name}/{eval_name}")
    parser.add_argument(
        "--print_example_prompt",
        action="store_true",
        help="Print one runtime user prompt after intext hint injection, then exit.",
    )
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/{eval_name}_{solver_name}_0shot_{args.hint_fraction}.json"
    if not args.print_example_prompt:
        check_output_exists(filename)

    valid_samples = get_valid_problem_ids([DATA_PATH])
    sample_ids = set(valid_samples.keys())
    logger.info(f"Running on {len(sample_ids)} samples with {args.hint_fraction} hint fraction")

    prefill_config = PrefillConfig(path=DATA_PATH, fraction=args.hint_fraction, mode = "masked")

    @inspect_solver
    def print_prompt_and_exit():
        async def solve(state, generate):
            print("=" * 80)
            print(
                f"Runtime prompt before generation "
                f"(sample_id={state.sample_id}, epoch={state.epoch}, hint_fraction={args.hint_fraction})"
            )
            print("=" * 80)
            print(state.user_prompt.text)
            print("=" * 80)
            raise SystemExit(0)
        return solve

    solver = [
        instructions(DEFAULT_INSTRUCTIONS),
        intext(prefill_config, prefix="Here is part of a hint that may be helpful to your solution:\n"),
        print_prompt_and_exit() if args.print_example_prompt else generate(timeout=args.timeout)
    ]

    if args.print_example_prompt:
        args.limit = 1

    task = gpqa_diamond(sample_ids=sample_ids, solver=solver)

    run_eval(
        task=task,
        args=args,
        output_filename=filename,
        scorer_name=f'{eval_name}_scorer',
        extra_metadata={"hint_fraction": args.hint_fraction, "data_path": DATA_PATH, "solver_name": solver_name}
    )

"""PYTHONPATH=src python christine_experiments/20251105/gpqa_solution_intext_mask.py --print_example_prompt --hint_fraction 1 --limit 1
"""