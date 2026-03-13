from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_ROOTS = {
    "suzeva": PROJECT_ROOT / "christine_experiments/20251113",
    "christine": Path("/sphinx/u/cye/emergent-doordash/christine_experiments/20251113"),
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "consolidated_jsonl"

STATE_DIRNAME = "_state"
STATE_MANIFEST_FILENAME = "eval_manifest_state.jsonl"
PARSE_ERRORS_FILENAME = "parse_errors.jsonl"
RUN_SUMMARY_FILENAME = "run_summary_consolidate.json"
COUNTS_SUMMARY_FILENAME = "counts_by_run_benchmark_model_hint.jsonl"

RUN_TYPES = {"baseline", "results"}


@dataclass(frozen=True)
class ConsolidateConfig:
    eval_roots: dict[str, Path]
    output_dir: Path
    max_eval_files: int | None = None
    progress_every: int = 100


@dataclass(frozen=True)
class GroupInfo:
    run_type: str
    benchmark: str
    group_key: str
    model_path: str | None
    path_hint_level: str
    path_hint_segments: list[str]


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jsonl_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str) + "\n"


def normalize_score_value(value: Any) -> str:
    if value is None:
        return "U"

    if isinstance(value, bool):
        return "C" if value else "I"

    if isinstance(value, (int, float)):
        if value == 1:
            return "C"
        if value == 0:
            return "I"
        return "U"

    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"C", "CORRECT", "TRUE", "T", "YES", "Y"}:
            return "C"
        if v in {"I", "INCORRECT", "FALSE", "F", "NO", "N"}:
            return "I"
        return "U"

    return "U"


def score_extracted_answer(score_payload: dict[str, Any]) -> str | None:
    answer = score_payload.get("answer")
    if answer is not None and str(answer).strip() != "":
        return str(answer)

    metadata = score_payload.get("metadata")
    if isinstance(metadata, dict):
        extracted = metadata.get("extracted_answer")
        if extracted is not None and str(extracted).strip() != "":
            return str(extracted)
    return None


def extraction_status(extracted_answer: str | None) -> str:
    return "ok" if extracted_answer is not None and extracted_answer.strip() != "" else "failed"


def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", item)))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def extract_prompt_text(sample: dict[str, Any]) -> str | None:
    messages = sample.get("messages")
    if isinstance(messages, list) and len(messages) > 0:
        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content")
            return as_text(content) if content is not None else None
        return as_text(first)

    if sample.get("input") is not None:
        return as_text(sample.get("input"))

    return None


def extract_output_text(sample: dict[str, Any]) -> str | None:
    output = sample.get("output")
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content is not None:
                        return as_text(content)
        completion = output.get("completion")
        if completion is not None:
            return as_text(completion)

    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = str(msg.get("role") or "").lower()
                if role == "assistant":
                    content = msg.get("content")
                    if content is not None:
                        return as_text(content)
    return None


def extract_scorer_names(eval_obj: dict[str, Any]) -> list[str]:
    scorers = eval_obj.get("scorers")
    names: list[str] = []
    if isinstance(scorers, list):
        for scorer in scorers:
            if isinstance(scorer, str):
                names.append(scorer)
            elif isinstance(scorer, dict):
                name = scorer.get("name") or scorer.get("scorer") or scorer.get("id")
                if name is not None:
                    names.append(str(name))
    return names


def eval_rel_path(
    *,
    eval_path: Path,
    owner: str,
    owner_root: Path,
    project_root: Path,
) -> str:
    try:
        return f"{owner}:{eval_path.relative_to(owner_root)}"
    except ValueError:
        pass
    try:
        return str(eval_path.relative_to(project_root))
    except ValueError:
        return str(eval_path)


def make_eval_id(source_owner: str, eval_rel_path_value: str) -> str:
    digest = hashlib.sha1(f"{source_owner}\0{eval_rel_path_value}".encode("utf-8")).hexdigest()
    return f"eval_{digest}"


def make_rollout_id(
    *,
    eval_id: str,
    rollout_ordinal: int,
    sample_id: Any,
    epoch: Any,
    sample_idx: Any,
    sample_file: str,
) -> str:
    payload = f"{eval_id}\0{rollout_ordinal}\0{sample_id}\0{epoch}\0{sample_idx}\0{sample_file}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"rollout_{digest}"


def discover_eval_files(config: ConsolidateConfig) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    owner_totals: dict[str, int] = {}
    for owner, root in config.eval_roots.items():
        if not root.exists():
            print(f"[{ts_now()}] skip missing root: owner={owner} root={root}", flush=True)
            owner_totals[owner] = 0
            continue
        print(f"[{ts_now()}] scanning root: owner={owner} root={root}", flush=True)
        owner_count = 0
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".eval"):
                    found.append((owner, Path(dirpath) / filename))
                    owner_count += 1
                    if owner_count % 5000 == 0:
                        print(
                            f"[{ts_now()}] discovered {owner_count} eval files for owner={owner}",
                            flush=True,
                        )
        owner_totals[owner] = owner_count
        print(f"[{ts_now()}] finished scan: owner={owner} eval_files={owner_count}", flush=True)

    found.sort(key=lambda x: (x[0], str(x[1])))

    print(f"[{ts_now()}] discovery summary:", flush=True)
    for owner in sorted(owner_totals.keys()):
        print(f"  {owner}: {owner_totals[owner]}", flush=True)
    print(f"  total: {sum(owner_totals.values())}", flush=True)

    if config.max_eval_files is not None:
        found = found[: config.max_eval_files]
        print(
            f"[{ts_now()}] applying max_eval_files={config.max_eval_files} -> processing {len(found)}",
            flush=True,
        )
    return found


def derive_group_info(eval_path: Path, owner_root: Path) -> GroupInfo | None:
    try:
        rel = eval_path.relative_to(owner_root)
    except ValueError:
        return None

    parts = rel.parts
    run_idx = -1
    for i, part in enumerate(parts):
        if part in RUN_TYPES:
            run_idx = i
            break
    if run_idx < 0 or len(parts) <= run_idx + 1:
        return None

    run_type = parts[run_idx]
    benchmark = parts[run_idx + 1]
    if benchmark.strip() == "":
        return None

    model_path = parts[-2] if len(parts) >= 2 else None
    hint_segments = list(parts[run_idx + 2 : -2]) if len(parts) > run_idx + 3 else []
    path_hint_level = "/".join(hint_segments) if hint_segments else "__none__"
    return GroupInfo(
        run_type=run_type,
        benchmark=benchmark,
        group_key=f"{run_type}__{benchmark}",
        model_path=model_path,
        path_hint_level=path_hint_level,
        path_hint_segments=hint_segments,
    )


def load_state_index(state_manifest_path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    if not state_manifest_path.exists():
        return state

    with state_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eval_id = row.get("eval_id")
            if isinstance(eval_id, str) and eval_id:
                state[eval_id] = row
    return state


def parse_eval(
    *,
    owner: str,
    eval_path: Path,
    eval_rel_path_value: str,
    eval_id: str,
    group: GroupInfo,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with ZipFile(eval_path) as zf:
        try:
            start_payload = json.loads(zf.read("_journal/start.json"))
        except KeyError as exc:
            raise RuntimeError("missing _journal/start.json") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid _journal/start.json: {exc}") from exc

        eval_obj = start_payload.get("eval", {}) if isinstance(start_payload, dict) else {}
        metadata = eval_obj.get("metadata") if isinstance(eval_obj.get("metadata"), dict) else {}
        configured_scorers = extract_scorer_names(eval_obj)
        hint_fraction = metadata.get("hint_fraction")
        solver_name = metadata.get("solver_name")

        sample_names = [
            name
            for name in zf.namelist()
            if name.startswith("samples/") and name.endswith(".json")
        ]
        sample_names.sort()

        rollouts: list[dict[str, Any]] = []
        num_questions_scored = 0
        scorer_score_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"C": 0, "I": 0, "U": 0}
        )

        for rollout_ordinal, sample_name in enumerate(sample_names):
            try:
                sample = json.loads(zf.read(sample_name))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid sample JSON {sample_name}: {exc}") from exc

            sample_id = sample.get("id")
            epoch = sample.get("epoch")
            sample_idx = sample.get("sample_idx")
            target = str(sample.get("target") or "")
            prompt_text = extract_prompt_text(sample)
            output_text = extract_output_text(sample)

            raw_scores = sample.get("scores")
            scores_json: dict[str, Any] = raw_scores if isinstance(raw_scores, dict) else {}
            if scores_json:
                num_questions_scored += 1

            scorer_outcomes: dict[str, dict[str, Any]] = {}
            for scorer_name, scorer_payload in scores_json.items():
                payload = scorer_payload if isinstance(scorer_payload, dict) else {"value": scorer_payload}
                score_raw_value = payload.get("value")
                score_normalized = normalize_score_value(score_raw_value)
                extracted_answer = score_extracted_answer(payload)
                extract_status = extraction_status(extracted_answer)

                is_correct: bool | None = None
                if score_normalized == "C":
                    is_correct = True
                elif score_normalized == "I":
                    is_correct = False

                scorer_outcomes[str(scorer_name)] = {
                    "score_raw_value": score_raw_value,
                    "score_normalized": score_normalized,
                    "is_correct": is_correct,
                    "extracted_answer": extracted_answer,
                    "extraction_status": extract_status,
                    "explanation": payload.get("explanation"),
                    "metadata_json": payload.get("metadata"),
                }
                scorer_score_counts[str(scorer_name)][score_normalized] += 1

            rollout_row = {
                "rollout_id": make_rollout_id(
                    eval_id=eval_id,
                    rollout_ordinal=rollout_ordinal,
                    sample_id=sample_id,
                    epoch=epoch,
                    sample_idx=sample_idx,
                    sample_file=sample_name,
                ),
                "eval_id": eval_id,
                "source_owner": owner,
                "eval_path": str(eval_path),
                "eval_rel_path": eval_rel_path_value,
                "run_type": group.run_type,
                "benchmark": group.benchmark,
                "group_key": group.group_key,
                "task_name": eval_obj.get("task"),
                "task_id": eval_obj.get("task_id"),
                "task_display_name": eval_obj.get("task_display_name"),
                "model": eval_obj.get("model"),
                "model_path": group.model_path,
                "run_id": eval_obj.get("run_id"),
                "created": eval_obj.get("created"),
                "hint_fraction": hint_fraction,
                "path_hint_level": group.path_hint_level,
                "path_hint_segments": group.path_hint_segments,
                "solver_name": solver_name,
                "configured_scorers": configured_scorers,
                "sample_file": sample_name,
                "rollout_ordinal": rollout_ordinal,
                "sample_id": sample_id,
                "sample_idx": sample_idx,
                "epoch": epoch,
                "target": target,
                "prompt_text": prompt_text,
                "output_text": output_text,
                "scores_json": scores_json,
                "scorer_outcomes": scorer_outcomes,
                "num_scorers_logged": len(scores_json),
                "questions_scored_for_rollout": 1 if scores_json else 0,
            }
            rollouts.append(rollout_row)

        eval_summary = {
            "eval_id": eval_id,
            "source_owner": owner,
            "eval_path": str(eval_path),
            "eval_rel_path": eval_rel_path_value,
            "run_type": group.run_type,
            "benchmark": group.benchmark,
            "group_key": group.group_key,
            "task_name": eval_obj.get("task"),
            "task_id": eval_obj.get("task_id"),
            "task_display_name": eval_obj.get("task_display_name"),
            "model": eval_obj.get("model"),
            "model_path": group.model_path,
            "run_id": eval_obj.get("run_id"),
            "created": eval_obj.get("created"),
            "hint_fraction": hint_fraction,
            "path_hint_level": group.path_hint_level,
            "path_hint_segments": group.path_hint_segments,
            "solver_name": solver_name,
            "configured_scorers": configured_scorers,
            "metadata_json": metadata,
            "dataset_json": eval_obj.get("dataset"),
            "parse_status": "ok",
            "error": None,
            "num_rollouts": len(rollouts),
            "num_questions_scored": num_questions_scored,
            "scorer_score_counts": scorer_score_counts,
        }
        return rollouts, eval_summary


class GroupWriterPool:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.handles: dict[str, TextIO] = {}
        self.output_paths: dict[str, Path] = {}
        self.rows_written_by_group: dict[str, int] = defaultdict(int)

    def _path_for_group(self, group: GroupInfo) -> Path:
        filename = f"{group.run_type}__{group.benchmark}.jsonl"
        return self.output_dir / filename

    def write_rollout(self, group: GroupInfo, row: dict[str, Any]) -> None:
        key = group.group_key
        handle = self.handles.get(key)
        if handle is None:
            out_path = self._path_for_group(group)
            handle = out_path.open("a", encoding="utf-8")
            self.handles[key] = handle
            self.output_paths[key] = out_path
        handle.write(jsonl_line(row))
        self.rows_written_by_group[key] += 1

    def flush_group(self, group: GroupInfo) -> None:
        handle = self.handles.get(group.group_key)
        if handle is not None:
            handle.flush()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def build_counts_summary_rows(
    state_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in state_index.values():
        if row.get("parse_status") != "ok":
            continue
        key = (
            row.get("run_type"),
            row.get("benchmark"),
            row.get("model"),
            row.get("path_hint_level"),
            row.get("hint_fraction"),
        )
        if key not in grouped:
            grouped[key] = {
                "run_type": row.get("run_type"),
                "benchmark": row.get("benchmark"),
                "model": row.get("model"),
                "path_hint_level": row.get("path_hint_level"),
                "hint_fraction": row.get("hint_fraction"),
                "num_eval_files": 0,
                "num_rollouts": 0,
                "num_questions_scored": 0,
                "score_counts_by_scorer": defaultdict(lambda: {"C": 0, "I": 0, "U": 0}),
            }
        out = grouped[key]
        out["num_eval_files"] += 1
        out["num_rollouts"] += int(row.get("num_rollouts") or 0)
        out["num_questions_scored"] += int(row.get("num_questions_scored") or 0)

        scorer_counts = row.get("scorer_score_counts")
        if isinstance(scorer_counts, dict):
            for scorer_name, score_counts in scorer_counts.items():
                if not isinstance(score_counts, dict):
                    continue
                for label in ("C", "I", "U"):
                    out["score_counts_by_scorer"][str(scorer_name)][label] += int(
                        score_counts.get(label) or 0
                    )

    rows = list(grouped.values())
    for row in rows:
        score_counts = row["score_counts_by_scorer"]
        row["score_counts_by_scorer"] = {
            scorer: counts for scorer, counts in sorted(score_counts.items())
        }
    rows.sort(
        key=lambda r: (
            str(r.get("run_type")),
            str(r.get("benchmark")),
            str(r.get("model")),
            str(r.get("path_hint_level")),
            str(r.get("hint_fraction")),
        )
    )
    return rows


def preflight_resume_status(
    *,
    eval_files: list[tuple[str, Path]],
    config: ConsolidateConfig,
    state_index: dict[str, dict[str, Any]],
) -> dict[str, int]:
    stats = {
        "total_discovered": len(eval_files),
        "already_processed_ok": 0,
        "already_seen_error": 0,
        "left_to_process": 0,
        "immutability_mismatches": 0,
    }

    for owner, eval_path in eval_files:
        owner_root = config.eval_roots[owner]
        rel_path_value = eval_rel_path(
            eval_path=eval_path,
            owner=owner,
            owner_root=owner_root,
            project_root=PROJECT_ROOT,
        )
        eval_id = make_eval_id(owner, rel_path_value)
        prior = state_index.get(eval_id)

        if prior is None:
            stats["left_to_process"] += 1
            continue

        stat = eval_path.stat()
        same_file = (
            prior.get("mtime_ns") == stat.st_mtime_ns
            and prior.get("file_size") == stat.st_size
        )
        if not same_file:
            stats["immutability_mismatches"] += 1
            continue

        if prior.get("parse_status") == "ok":
            stats["already_processed_ok"] += 1
        elif prior.get("parse_status") == "error":
            stats["already_seen_error"] += 1
        else:
            stats["left_to_process"] += 1

    return stats


def consolidate(config: ConsolidateConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = config.output_dir / STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)

    state_manifest_path = state_dir / STATE_MANIFEST_FILENAME
    parse_errors_path = state_dir / PARSE_ERRORS_FILENAME
    run_summary_path = config.output_dir / RUN_SUMMARY_FILENAME
    counts_summary_path = config.output_dir / COUNTS_SUMMARY_FILENAME

    state_index = load_state_index(state_manifest_path)
    print(
        f"[{ts_now()}] loaded prior state rows={len(state_index)} from {state_manifest_path}",
        flush=True,
    )

    eval_files = discover_eval_files(config)
    print(f"[{ts_now()}] total eval files to scan: {len(eval_files)}", flush=True)
    preflight = preflight_resume_status(
        eval_files=eval_files,
        config=config,
        state_index=state_index,
    )
    print(
        f"[{ts_now()}] preflight resume status: "
        f"already_processed_ok={preflight['already_processed_ok']} "
        f"already_seen_error={preflight['already_seen_error']} "
        f"left_to_process={preflight['left_to_process']} "
        f"total={preflight['total_discovered']}",
        flush=True,
    )
    if preflight["immutability_mismatches"] > 0:
        print(
            f"[{ts_now()}] preflight warning: "
            f"immutability_mismatches={preflight['immutability_mismatches']} "
            f"(run will error when first mismatch is encountered)",
            flush=True,
        )

    counts: dict[str, int] = {
        "eval_files_discovered": len(eval_files),
        "eval_files_processed": 0,
        "eval_files_failed": 0,
        "eval_files_skipped_unchanged": 0,
        "eval_files_skipped_non_target_path": 0,
        "rollouts_written": 0,
        "questions_scored_total": 0,
    }

    started = time.time()
    writer_pool = GroupWriterPool(config.output_dir)
    state_manifest_f = state_manifest_path.open("a", encoding="utf-8")
    parse_errors_f = parse_errors_path.open("a", encoding="utf-8")

    try:
        for idx, (owner, eval_path) in enumerate(eval_files, start=1):
            owner_root = config.eval_roots[owner]
            group = derive_group_info(eval_path, owner_root)
            if group is None:
                counts["eval_files_skipped_non_target_path"] += 1
                if idx % config.progress_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"[{ts_now()}] [{idx}/{len(eval_files)}] skipped_non_target="
                        f"{counts['eval_files_skipped_non_target_path']} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                continue

            rel_path_value = eval_rel_path(
                eval_path=eval_path,
                owner=owner,
                owner_root=owner_root,
                project_root=PROJECT_ROOT,
            )
            eval_id = make_eval_id(owner, rel_path_value)

            stat = eval_path.stat()
            mtime_ns = stat.st_mtime_ns
            file_size = stat.st_size

            prior = state_index.get(eval_id)
            if prior is not None:
                prior_mtime_ns = prior.get("mtime_ns")
                prior_file_size = prior.get("file_size")
                if prior_mtime_ns == mtime_ns and prior_file_size == file_size:
                    counts["eval_files_skipped_unchanged"] += 1
                    if idx % config.progress_every == 0:
                        elapsed = time.time() - started
                        print(
                            f"[{ts_now()}] [{idx}/{len(eval_files)}] skipped_unchanged="
                            f"{counts['eval_files_skipped_unchanged']} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                    continue
                raise RuntimeError(
                    "immutability violation: previously-ingested eval changed: "
                    f"eval_id={eval_id} path={eval_path} "
                    f"old_mtime_ns={prior_mtime_ns} new_mtime_ns={mtime_ns} "
                    f"old_file_size={prior_file_size} new_file_size={file_size}"
                )

            try:
                rollouts, eval_summary = parse_eval(
                    owner=owner,
                    eval_path=eval_path,
                    eval_rel_path_value=rel_path_value,
                    eval_id=eval_id,
                    group=group,
                )
            except Exception as exc:  # noqa: BLE001
                error_row = {
                    "ts": ts_now(),
                    "eval_id": eval_id,
                    "source_owner": owner,
                    "eval_path": str(eval_path),
                    "eval_rel_path": rel_path_value,
                    "run_type": group.run_type,
                    "benchmark": group.benchmark,
                    "group_key": group.group_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                parse_errors_f.write(jsonl_line(error_row))
                parse_errors_f.flush()

                state_row = {
                    "ts": ts_now(),
                    "eval_id": eval_id,
                    "source_owner": owner,
                    "eval_path": str(eval_path),
                    "eval_rel_path": rel_path_value,
                    "run_type": group.run_type,
                    "benchmark": group.benchmark,
                    "group_key": group.group_key,
                    "model": None,
                    "model_path": group.model_path,
                    "hint_fraction": None,
                    "path_hint_level": group.path_hint_level,
                    "path_hint_segments": group.path_hint_segments,
                    "mtime_ns": mtime_ns,
                    "file_size": file_size,
                    "parse_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "num_rollouts": 0,
                    "num_questions_scored": 0,
                    "scorer_score_counts": {},
                }
                state_manifest_f.write(jsonl_line(state_row))
                state_manifest_f.flush()
                state_index[eval_id] = state_row

                counts["eval_files_failed"] += 1
                print(
                    f"[{ts_now()}] [{idx}/{len(eval_files)}] parse error: {eval_path} -> "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            for rollout_row in rollouts:
                writer_pool.write_rollout(group, rollout_row)
            # Flush per eval so resume state and group JSONL stay aligned on interruptions.
            writer_pool.flush_group(group)
            counts["rollouts_written"] += len(rollouts)
            counts["questions_scored_total"] += int(eval_summary["num_questions_scored"])
            counts["eval_files_processed"] += 1

            state_row = {
                "ts": ts_now(),
                **eval_summary,
                "mtime_ns": mtime_ns,
                "file_size": file_size,
            }
            state_manifest_f.write(jsonl_line(state_row))
            state_manifest_f.flush()
            state_index[eval_id] = state_row

            if idx % config.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"[{ts_now()}] [{idx}/{len(eval_files)}] processed={counts['eval_files_processed']} "
                    f"failed={counts['eval_files_failed']} "
                    f"skipped={counts['eval_files_skipped_unchanged']} "
                    f"rollouts_written={counts['rollouts_written']} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        writer_pool.close()
        state_manifest_f.close()
        parse_errors_f.close()

    counts_summary_rows = build_counts_summary_rows(state_index)
    with counts_summary_path.open("w", encoding="utf-8") as f:
        for row in counts_summary_rows:
            f.write(jsonl_line(row))

    elapsed = time.time() - started
    summary = {
        **counts,
        "elapsed_sec": elapsed,
        "output_dir": str(config.output_dir),
        "state_dir": str(state_dir),
        "state_manifest_jsonl": str(state_manifest_path),
        "parse_errors_jsonl": str(parse_errors_path),
        "counts_summary_jsonl": str(counts_summary_path),
        "groups_written": {
            key: {
                "output_path": str(writer_pool.output_paths[key]),
                "rows_written_this_run": writer_pool.rows_written_by_group.get(key, 0),
            }
            for key in sorted(writer_pool.output_paths.keys())
        },
        "state_rows_total": len(state_index),
        "counts_rows_total": len(counts_summary_rows),
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate Inspect .eval files into per-group JSONL outputs, where each group is "
            "baseline__<benchmark> or results__<benchmark>."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-eval-files",
        type=int,
        default=None,
        help="Optional max number of .eval files to process after sorting.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N discovered eval files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConsolidateConfig(
        eval_roots=DEFAULT_EVAL_ROOTS,
        output_dir=args.output_dir,
        max_eval_files=args.max_eval_files,
        progress_every=args.progress_every,
    )
    print(
        f"[{ts_now()}] starting consolidate output_dir={config.output_dir} "
        f"max_eval_files={config.max_eval_files}",
        flush=True,
    )
    summary = consolidate(config)
    print("\nConsolidation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # python suze_experiments/20260313/consolidate_eval_jsonl.py 
    main()
