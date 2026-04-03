from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets import get_dataset_spec
from src.storage import append_jsonl, build_hint_generation_path, make_stable_id, read_jsonl
from src.types import HintGenerationRecord

def _parse_anthropic_message_text(message) -> str:
    texts: list[str] = []
    for block in message.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts).strip()


def query_claude_hint(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _parse_anthropic_message_text(response)
    usage = {
        "input_token_count": int(response.usage.input_tokens),
        "output_token_count": int(response.usage.output_tokens),
        "stop_reason": getattr(response, "stop_reason", None),
    }
    return text, usage


def _existing_hint_ids(path: str | Path) -> set[str]:
    existing = set()
    for row in read_jsonl(path, model_cls=None):
        hint_id = row.get("hint_id") if isinstance(row, dict) else None
        if isinstance(hint_id, str):
            existing.add(hint_id)
    return existing


def generate_hints(
    *,
    benchmark_name: str,
    hint_type: str,
    generator_model: str,
    num_rollouts: int,
    limit: int,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    resume: bool,
) -> str:
    """Generate hint records and append them to a JSONL file."""
    spec = get_dataset_spec(benchmark_name)
    problems = spec.load_problems()[:limit]
    prompt_version = spec.prompt_version(hint_type)

    out_path = str(
        build_hint_generation_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            data_root="data",
        )
    )

    existing = _existing_hint_ids(out_path) if resume else set()
    written = 0
    would_write = 0
    skipped = 0

    for problem in problems:
        for rollout_id in range(num_rollouts):
            hint_id = make_stable_id(
                problem.problem_id,
                hint_type,
                rollout_id,
                generator_model,
                length=16,
            )
            if hint_id in existing:
                skipped += 1
                continue

            prompt = spec.build_hint_prompt(problem, hint_type)
            if dry_run:
                would_write += 1
                continue

            hint_text, usage = query_claude_hint(
                prompt=prompt,
                model=generator_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            record = HintGenerationRecord(
                hint_id=hint_id,
                problem_id=problem.problem_id,
                benchmark_name=benchmark_name,
                hint_type=hint_type,
                rollout_id=rollout_id,
                generator_model=generator_model,
                question=problem.question,
                answer=problem.answer,
                full_hint=hint_text,
                input_token_count=usage["input_token_count"],
                output_token_count=usage["output_token_count"],
                metadata={
                    "prompt": prompt,
                    "prompt_version": prompt_version,
                    "dataset_spec": spec.name,
                    "problem_source": problem.source,
                    "temperature": temperature,
                    **usage,
                },
            )
            append_jsonl(out_path, record)
            existing.add(hint_id)
            written += 1

    if dry_run:
        print(
            f"[hint_generation] dry_run benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} would_write={would_write} skipped={skipped} "
            f"output={out_path}"
        )
    else:
        print(
            f"[hint_generation] done benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} written={written} skipped={skipped} "
            f"output={out_path}"
        )
    return out_path
