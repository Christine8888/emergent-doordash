from __future__ import annotations

import argparse
import json

from src.batch_hinting import (
    build_jsonl,
    download,
    estimate,
    process_results,
    status,
    submit,
    wait,
)


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def _add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="aime2025_2026")
    parser.add_argument("--hint-type", default="answer_not_revealed")
    parser.add_argument("--fractioner", default="mask_word")
    parser.add_argument("--hint-fractions", nargs="*", default=["0.0"])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Limit requests written to the batch JSONL. Omit to include all eligible requests.",
    )
    parser.add_argument(
        "--token-estimate-method",
        choices=["chars", "gemini-count-tokens"],
        default="chars",
        help=(
            "Use chars for offline estimates, or gemini-count-tokens for Gemini's "
            "count_tokens endpoint before writing the manifest."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run hinted inference through staged Gemini Batch API jobs. "
            "Start with build-jsonl, then submit, status/wait, download, and process-results."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-jsonl")
    _add_common_build_args(build_parser)
    build_parser.set_defaults(func=build_jsonl)

    estimate_parser = subparsers.add_parser("estimate")
    estimate_parser.add_argument("manifest")
    estimate_parser.add_argument("--assumed-output-tokens", type=int, default=None)
    estimate_parser.set_defaults(func=estimate)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("manifest")
    submit_parser.add_argument("--display-name", default=None)
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("manifest")
    status_parser.set_defaults(func=status)

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("manifest")
    wait_parser.add_argument("--poll-interval-seconds", type=int, default=30)
    wait_parser.set_defaults(func=wait)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("manifest")
    download_parser.add_argument("--output-path", default=None)
    download_parser.set_defaults(func=download)

    process_parser = subparsers.add_parser("process-results")
    process_parser.add_argument("manifest")
    process_parser.add_argument("--results-jsonl", default=None)
    process_parser.set_defaults(func=process_results)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = args.func(args)
    _print_payload(payload)
    if args.command == "build-jsonl":
        print(f"MANIFEST {payload['manifest_tag']}", flush=True)
    elif args.command == "submit":
        print(f"MANIFEST {payload['submitted_manifest_tag']}", flush=True)


if __name__ == "__main__":
    main()

"""
python -m runs.generate_hinted_batch build-jsonl \
  --benchmark hle \
  --hint-type answer_not_revealed \
  --fractioner mask_word \
  --model gemini-3.1-pro-preview \
  --hint-fractions 0.0

python -m runs.generate_hinted_batch submit hle/answer_not_revealed/mask_word/gemini-3.1-pro-preview/20260512_052817

python -m runs.generate_hinted_batch wait l0fldo6kmmf139yehdekfovl2u7gmy9hd91k

python -m runs.generate_hinted_batch download <submitted-manifest-tag>

python -m runs.generate_hinted_batch process-results <submitted-manifest-tag>
"""
