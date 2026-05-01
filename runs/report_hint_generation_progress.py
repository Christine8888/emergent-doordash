from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from src.datasets import get_dataset_spec
from src.hint_types import HintType
from src.storage import build_hint_generation_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def _print_counter(title: str, counts: Counter[str], *, denominator: int) -> None:
    print(title)
    if not counts:
        print("  none")
        return
    for value, count in counts.most_common():
        print(f"  {value}: {count} ({_percent(count, denominator)})")


def _stop_reason_from_row(row: dict[str, Any]) -> str:
    stop_reason = row.get("stop_reason")
    if stop_reason is None:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            stop_reason = metadata.get("stop_reason")
    if stop_reason is None:
        return "missing"
    return str(stop_reason)


def _failed_reason_from_row(row: dict[str, Any]) -> str:
    failure_type = str(row.get("failure_type", "unknown"))
    stop_reason = _stop_reason_from_row(row)
    return f"{failure_type} (stop_reason={stop_reason})"


def _task_key_from_row(row: dict[str, Any]) -> tuple[str, int] | None:
    rollout_id = row.get("rollout_id")
    if not isinstance(rollout_id, int):
        return None
    problem_id = row.get("problem_id")
    if problem_id is None:
        return None
    return str(problem_id), rollout_id


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _field_from_row_or_metadata(row: dict[str, Any], field: str) -> Any:
    if field in row:
        return row.get(field)
    return _row_metadata(row).get(field)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _thinking_bucket(row: dict[str, Any]) -> str | None:
    thinking_enabled = _as_bool(_field_from_row_or_metadata(row, "thinking_enabled"))
    if not thinking_enabled:
        return "non-thinking"

    reasoning_tokens = _maybe_float(_field_from_row_or_metadata(row, "reasoning_token_count"))
    thinking_tokens = _maybe_float(_field_from_row_or_metadata(row, "thinking_token_count"))
    if reasoning_tokens is None and thinking_tokens is None:
        return None

    effort = _field_from_row_or_metadata(row, "effort")
    return f"thinking_{effort}" if effort is not None else "thinking_unknown"


def _token_value(row: dict[str, Any], field: str) -> float | None:
    return _maybe_float(_field_from_row_or_metadata(row, field))


def _is_graded_correct(row: dict[str, Any]) -> bool | None:
    if "failure_type" not in row:
        return True
    if row.get("failure_type") == "grader_rejected":
        return False
    return None


def _print_token_averages(rows: list[dict[str, Any]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_enabled_missing_reasoning = 0
    for row in rows:
        bucket = _thinking_bucket(row)
        if bucket is None:
            excluded_enabled_missing_reasoning += 1
            continue
        buckets[bucket].append(row)

    print("Token averages by thinking level")
    if not buckets:
        print("  none")
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket]
        output_values = [
            value
            for row in bucket_rows
            if (value := _token_value(row, "output_token_count")) is not None
        ]
        reasoning_values = [
            value
            for row in bucket_rows
            if (value := _token_value(row, "reasoning_token_count")) is not None
        ]
        thinking_values = [
            value
            for row in bucket_rows
            if (value := _token_value(row, "thinking_token_count")) is not None
        ]
        visible_values = [
            value
            for row in bucket_rows
            if (value := _token_value(row, "visible_output_token_count")) is not None
        ]
        avg_output = sum(output_values) / len(output_values) if output_values else 0.0
        avg_reasoning = sum(reasoning_values) / len(reasoning_values) if reasoning_values else None
        avg_thinking = sum(thinking_values) / len(thinking_values) if thinking_values else None
        avg_visible = sum(visible_values) / len(visible_values) if visible_values else None
        graded_values = [
            is_correct
            for row in bucket_rows
            if (is_correct := _is_graded_correct(row)) is not None
        ]
        correct_count = sum(1 for is_correct in graded_values if is_correct)
        accuracy = correct_count / len(graded_values) if graded_values else None
        parts = [
            f"n={len(bucket_rows)}",
            f"avg_output_tokens={avg_output:.1f}",
        ]
        if accuracy is not None:
            parts.append(f"graded_n={len(graded_values)}")
            parts.append(f"correct={correct_count}")
            parts.append(f"accuracy={100.0 * accuracy:.1f}%")
        if avg_reasoning is not None:
            parts.append(f"avg_reasoning_tokens={avg_reasoning:.1f}")
        if avg_thinking is not None:
            parts.append(f"avg_thinking_tokens={avg_thinking:.1f}")
        if avg_visible is not None:
            parts.append(f"avg_visible_output_tokens={avg_visible:.1f}")
        print(f"  {bucket}: " + ", ".join(parts))
    if excluded_enabled_missing_reasoning:
        print(
            "  excluded_enabled_thinking_missing_reasoning_token_count: "
            f"{excluded_enabled_missing_reasoning}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report hint generation progress and stop reasons.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[hint.value for hint in HintType], required=True)
    parser.add_argument("--num-rollouts", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_rollouts < 1:
        raise SystemExit("--num-rollouts must be >= 1")

    dataset_spec = get_dataset_spec(args.benchmark)
    problems = dataset_spec.load_problems()
    dataset_problem_ids = {problem.problem_id for problem in problems}

    success_path = build_hint_generation_path(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        data_root=args.data_root,
    )
    failed_path = success_path.with_name(f"{success_path.stem}_failed.jsonl")
    success_rows = _read_jsonl(success_path)
    failed_rows_all = _read_jsonl(failed_path)

    expected_tasks = len(problems) * args.num_rollouts
    successful_tasks = {key for row in success_rows if (key := _task_key_from_row(row)) is not None}
    accepted_tasks = {
        key
        for key in successful_tasks
        if key[0] in dataset_problem_ids and 0 <= key[1] < args.num_rollouts
    }
    failed_attempt_tasks = {
        key
        for row in failed_rows_all
        if (key := _task_key_from_row(row)) is not None
        if key[0] in dataset_problem_ids and 0 <= key[1] < args.num_rollouts
    }
    failed_tasks_without_success = failed_attempt_tasks - accepted_tasks
    attempted_tasks = accepted_tasks | failed_attempt_tasks
    extra_success_rows = [
        row
        for row in success_rows
        if _task_key_from_row(row) not in accepted_tasks
    ]
    extra_failed_rows = [
        row
        for row in failed_rows_all
        if _task_key_from_row(row) not in failed_attempt_tasks
    ]

    accepted_by_problem: dict[str, int] = defaultdict(int)
    for problem_id, _rollout_id in accepted_tasks:
        accepted_by_problem[problem_id] += 1
    accepted_complete_problem_count = sum(
        1 for count in accepted_by_problem.values() if count >= args.num_rollouts
    )
    attempted_problem_count = len({problem_id for problem_id, _rollout_id in attempted_tasks})

    print(f"benchmark={args.benchmark}")
    print(f"hint_type={args.hint_type}")
    print(f"num_rollouts={args.num_rollouts}")
    print(f"success_path={success_path}")
    print(f"failed_path={failed_path}")
    print()
    print("Progress")
    print(f"  dataset_problems: {len(problems)}")
    print(f"  expected_tasks: {expected_tasks}")
    print(f"  attempted_tasks: {len(attempted_tasks)} ({_percent(len(attempted_tasks), expected_tasks)})")
    print(f"  accepted_tasks: {len(accepted_tasks)} ({_percent(len(accepted_tasks), expected_tasks)})")
    print(
        "  failed_tasks_without_success: "
        f"{len(failed_tasks_without_success)} ({_percent(len(failed_tasks_without_success), expected_tasks)})"
    )
    print(
        "  accepted_complete_problems: "
        f"{accepted_complete_problem_count} ({_percent(accepted_complete_problem_count, len(problems))})"
    )
    print(
        f"  attempted_problems: {attempted_problem_count} ({_percent(attempted_problem_count, len(problems))})"
    )
    print(f"  unattempted_tasks: {max(0, expected_tasks - len(attempted_tasks))}")
    print(f"  successful_rows_total: {len(success_rows)}")
    print(f"  successful_rows_outside_requested_rollouts_or_dataset: {len(extra_success_rows)}")
    print(f"  failed_rows_total: {len(failed_rows_all)}")
    print(f"  failed_rows_outside_requested_rollouts_or_dataset: {len(extra_failed_rows)}")
    print()
    accepted_stop_reason_counts = Counter(_stop_reason_from_row(row) for row in success_rows)
    failed_reason_counts = Counter(_failed_reason_from_row(row) for row in failed_rows_all)
    _print_counter(
        "Accepted stop reasons",
        accepted_stop_reason_counts,
        denominator=len(success_rows),
    )
    print()
    _print_counter(
        "Failed reasons",
        failed_reason_counts,
        denominator=len(failed_rows_all),
    )
    print()
    _print_token_averages(success_rows + failed_rows_all)


if __name__ == "__main__":
    main()

"""
python -m runs.report_hint_generation_progress --benchmark hle --hint-type basic_hint_hle --num-rollouts 1

"""
