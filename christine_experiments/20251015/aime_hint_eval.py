from hint_eval_utils import (
    create_base_parser, setup_vllm_env, check_output_exists,
    create_configs, run_eval_and_save
)
from environments.aime.aime import aime

DATA_PATH = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251010_data/aime_filtered.jsonl"

if __name__ == "__main__":
    parser = create_base_parser(
        default_log_dir="./aime",
        default_prefill_path=DATA_PATH,
        default_fewshot_path=DATA_PATH
    )
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/aime_{args.fewshot}shot_{args.hint_fraction}.json"
    check_output_exists(filename)

    prefill_config, fewshot_config = create_configs(args)

    task_kwargs = {
        "fewshot_config": fewshot_config,
        "prefill_config": prefill_config,
        "split": "test"
    }

    run_eval_and_save(
        task_fn=aime,
        task_kwargs=task_kwargs,
        args=args,
        output_filename=filename,
        bootstrap_metric='expression_exact_match_sympy'
    )
