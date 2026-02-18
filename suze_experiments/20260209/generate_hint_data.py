"""Generate CoT/solution hint JSONLs for external baselines.

This is a thin driver around the existing sampling scripts:
- `src/hints/cot.py`
- `src/hints/solution.py`

It runs them for a list of evals and writes JSONLs under:
  {output_root}/{cot,solution}/{eval}.jsonl

Defaults intentionally match prior usage:
- model: anthropic/claude-sonnet-4-5-20250929 (kept as the sampling default)
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_EVALS = [
    "hellaswag",
    "piqa",
    "mmlu_0_shot",
    "bbh",
    "arc_challenge",
    "winogrande",
    "math_level_5",
]

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data"


def _parse_csv(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _run_one(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script] + args
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hint JSONLs (cot/solution) for baselines.")
    parser.add_argument(
        "--evals",
        type=str,
        default=",".join(DEFAULT_EVALS),
        help="Comma-separated eval names (default: the ECI baseline set)",
    )
    parser.add_argument(
        "--hint_type",
        choices=["cot", "solution", "all"],
        default="all",
        help="Which hint type(s) to generate",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for outputs (default: suze_experiments/data/hints)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model ID")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--max_concurrent", type=int, default=None, help="Override max concurrent requests")
    parser.add_argument("--max_retries", type=int, default=None, help="Override max retries per problem")
    parser.add_argument("--n_per_question", type=int, default=None, help="Override number of correct samples per question")
    parser.add_argument("--rationalize", action="store_true", help="Pass --rationalize to sampling scripts")
    parser.add_argument("--prompt_suffix", type=str, default=None, help="Append text to each prompt")
    parser.add_argument(
        "--math_split",
        type=str,
        default=None,
        help="For math/math_level_5 only: dataset split (train/test/validation).",
    )
    parser.add_argument(
        "--debug-first-problem",
        action="store_true",
        help="Print the first problem's prompt, target, and graded solution logs.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N problems (for debugging)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every API request, response, and grading result")
    args = parser.parse_args()

    evals = _parse_csv(args.evals)
    output_root = Path(args.output_root).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent.parent
    cot_script = str(project_root / "src" / "hints" / "cot.py")
    solution_script = str(project_root / "src" / "hints" / "solution.py")

    for eval_name in evals:
        common_cli = ["--eval", eval_name]

        if args.model is not None:
            common_cli += ["--model", args.model]
        if args.temperature is not None:
            common_cli += ["--temperature", str(args.temperature)]
        if args.max_tokens is not None:
            common_cli += ["--max-tokens", str(args.max_tokens)]
        if args.max_concurrent is not None:
            common_cli += ["--max-concurrent", str(args.max_concurrent)]
        if args.max_retries is not None:
            common_cli += ["--max-retries", str(args.max_retries)]
        if args.n_per_question is not None:
            common_cli += ["--n-per-question", str(args.n_per_question)]
        if args.rationalize:
            common_cli += ["--rationalize"]
        if args.prompt_suffix is not None:
            common_cli += ["--prompt-suffix", args.prompt_suffix]

        if args.math_split is not None and eval_name in ("math", "math_level_5"):
            common_cli += ["--split", args.math_split]
        if args.debug_first_problem:
            common_cli += ["--debug-first-problem"]
        if args.limit is not None:
            common_cli += ["--limit", str(args.limit)]
        if args.verbose:
            common_cli += ["--verbose"]

        if args.hint_type in ("cot", "all"):
            out = output_root / "cot" / f"{eval_name}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            _run_one(cot_script, common_cli + ["--output-file", str(out)])

        if args.hint_type in ("solution", "all"):
            out = output_root / "solution" / f"{eval_name}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            _run_one(solution_script, common_cli + ["--output-file", str(out)])


if __name__ == "__main__":
    main()



"""
python suze_experiments/20260209/generate_hint_data.py --hint_type solution --max_retries 5

python suze_experiments/20260209/generate_hint_data.py --evals hellaswag --hint_type solution --debug-first-problem --limit 10 --verbose --max_retries 5



so far working:
- mmlu_5_shot_cot
- hellaswag
- piqa
- math_level_5
- bbh
- arc_challenge
- winogrande
"""
