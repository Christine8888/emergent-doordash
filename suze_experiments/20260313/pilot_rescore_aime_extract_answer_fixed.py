from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Simple pilot settings (edit these if needed).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
INPUT_FILE = Path(__file__).resolve().parent / "consolidated_jsonl" / "results__aime.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parent / "consolidated_jsonl" / "results__aime.pilot_extract_answer_fixed.scorers.jsonl"
COUNTS_FILE = INPUT_FILE.parent / "counts_by_run_benchmark_model_hint.jsonl"
INDEX_SUFFIX = ".rollout_index.sqlite3"
LIMIT: int | None = 4
SKIP_ELIGIBLE = 0
DISTINCT_SAMPLE_ID = False
OLD_SCORER_NAME = "aime_scorer"
NEW_SCORER_NAME = "aime_scorer_extract_answer_fixed"
FILTER_MODEL: str | None = "Qwen3-14B"
FILTER_HINT_FRACTION: float | None = 1  # set to a value in [0, 1] to filter
FILTER_OLD_SCORE: str | None = None  # one of: "I", "C", "U"
# Optional direct lookup path (fast): restrict to these rollout_ids via index query.
# Example: FILTER_ROLLOUT_IDS = ["rollout_abc", "rollout_def"]
FILTER_ROLLOUT_IDS: list[str] | None = [
    "rollout_c8400076a1ad44536687f4bbe766837d5b9879c8", "rollout_3b5fb9bc322390645b91484613ca94b9e4433044",
    "rollout_b8dea38a2ba82744248c0de64afae905bca69f37",
]
# Optional deterministic reconstruction path. Each item must contain:
# eval_id, rollout_ordinal, sample_id, epoch, sample_idx, sample_file
# Example:
# RECONSTRUCT_ROLLOUT_KEYS = [
#   {
#     "eval_id": "eval_xxx",
#     "rollout_ordinal": 0,
#     "sample_id": "2003-I-14",
#     "epoch": 1,
#     "sample_idx": 0,
#     "sample_file": "samples/000000.json",
#   }
# ]
RECONSTRUCT_ROLLOUT_KEYS: list[dict[str, Any]] | None = None

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environments.math.utils import extract_answer_fixed, grade_math_answer


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


def extraction_status(extracted_answer: str | None) -> str:
    return "ok" if extracted_answer is not None and extracted_answer.strip() else "failed"


def old_extracted_answer(old_payload: dict[str, Any]) -> str | None:
    answer = old_payload.get("answer")
    if answer is not None and str(answer).strip() != "":
        return str(answer)
    metadata = old_payload.get("metadata_json")
    if isinstance(metadata, dict):
        extracted = metadata.get("extracted_answer")
        if extracted is not None and str(extracted).strip() != "":
            return str(extracted)
    extracted2 = old_payload.get("extracted_answer")
    if extracted2 is not None and str(extracted2).strip() != "":
        return str(extracted2)
    return None


def _matches_model_filter(row_model: str, filter_model: str | None) -> bool:
    if filter_model is None:
        return True
    m = filter_model.strip()
    return row_model == m or row_model.endswith(f"/{m}")


def make_rollout_id(
    *,
    eval_id: str,
    rollout_ordinal: Any,
    sample_id: Any,
    epoch: Any,
    sample_idx: Any,
    sample_file: str,
) -> str:
    payload = f"{eval_id}\0{rollout_ordinal}\0{sample_id}\0{epoch}\0{sample_idx}\0{sample_file}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"rollout_{digest}"


def resolve_requested_rollout_ids() -> list[str]:
    if FILTER_ROLLOUT_IDS and RECONSTRUCT_ROLLOUT_KEYS:
        raise ValueError("Set only one of FILTER_ROLLOUT_IDS or RECONSTRUCT_ROLLOUT_KEYS.")
    if FILTER_ROLLOUT_IDS:
        return list(dict.fromkeys([str(x) for x in FILTER_ROLLOUT_IDS if str(x).strip()]))
    if RECONSTRUCT_ROLLOUT_KEYS:
        out: list[str] = []
        for i, item in enumerate(RECONSTRUCT_ROLLOUT_KEYS):
            try:
                rid = make_rollout_id(
                    eval_id=str(item["eval_id"]),
                    rollout_ordinal=item["rollout_ordinal"],
                    sample_id=item["sample_id"],
                    epoch=item["epoch"],
                    sample_idx=item["sample_idx"],
                    sample_file=str(item["sample_file"]),
                )
            except KeyError as exc:
                raise KeyError(
                    f"RECONSTRUCT_ROLLOUT_KEYS[{i}] missing required key: {exc}"
                ) from exc
            out.append(rid)
        return list(dict.fromkeys(out))
    raise ValueError(
        "Strict index lookup mode: you must set FILTER_ROLLOUT_IDS or RECONSTRUCT_ROLLOUT_KEYS."
    )


def _resolve_model_filter_from_counts(filter_model: str | None) -> str | None:
    if filter_model is None:
        return None
    if "/" in filter_model:
        return filter_model
    if not COUNTS_FILE.exists():
        return filter_model

    candidates: set[str] = set()
    with COUNTS_FILE.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("run_type") != "results" or row.get("benchmark") != "aime":
                continue
            model = str(row.get("model") or "")
            if model.endswith(f"/{filter_model}") or model == filter_model:
                candidates.add(model)

    if len(candidates) == 1:
        return next(iter(candidates))
    return filter_model


def _count_matches_from_counts(
    *,
    model_filter: str | None,
    hint_fraction_filter: float | None,
) -> int | None:
    if not COUNTS_FILE.exists():
        return None
    total = 0
    found_any = False
    with COUNTS_FILE.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("run_type") != "results" or row.get("benchmark") != "aime":
                continue
            model = str(row.get("model") or "")
            hint = row.get("hint_fraction")
            if not _matches_model_filter(model, model_filter):
                continue
            if hint_fraction_filter is not None:
                try:
                    hint_f = float(hint)
                except (TypeError, ValueError):
                    continue
                if abs(hint_f - hint_fraction_filter) > 1e-9:
                    continue
            found_any = True
            total += int(row.get("num_rollouts") or 0)
    if not found_any:
        return 0
    return total


def index_path_for_source(source_path: Path) -> Path:
    return source_path.with_name(source_path.name + INDEX_SUFFIX)


def load_offsets_from_index(
    source_path: Path,
    *,
    rollout_ids: list[str] | None = None,
) -> list[tuple[str, int]]:
    index_path = index_path_for_source(source_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Required rollout index not found: {index_path}\n"
            f"Build it first with:\n"
            f"python suze_experiments/20260313/build_rollout_offset_index.py --files {source_path.name}"
        )

    con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        if rollout_ids:
            unique_ids = list(dict.fromkeys(rollout_ids))
            placeholders = ",".join("?" for _ in unique_ids)
            rows = con.execute(
                f"SELECT rollout_id, byte_offset FROM offsets WHERE rollout_id IN ({placeholders}) ORDER BY byte_offset",
                unique_ids,
            ).fetchall()
            found_ids = {str(r[0]) for r in rows}
            missing = [rid for rid in unique_ids if rid not in found_ids]
            if missing:
                raise KeyError(f"rollout_id(s) not found in index: {missing}")
        else:
            rows = con.execute(
                "SELECT rollout_id, byte_offset FROM offsets ORDER BY byte_offset"
            ).fetchall()
    finally:
        con.close()
    return [(str(rollout_id), int(byte_offset)) for rollout_id, byte_offset in rows]


async def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"missing input file: {INPUT_FILE}")
    if FILTER_HINT_FRACTION is not None and not (0.0 <= FILTER_HINT_FRACTION <= 1.0):
        raise ValueError(f"FILTER_HINT_FRACTION must be in [0, 1], got {FILTER_HINT_FRACTION!r}")
    if FILTER_OLD_SCORE is not None and FILTER_OLD_SCORE.strip().upper() not in {"I", "C", "U"}:
        raise ValueError(f"FILTER_OLD_SCORE must be one of 'I', 'C', 'U', or None; got {FILTER_OLD_SCORE!r}")
    resolved_model_filter = _resolve_model_filter_from_counts(FILTER_MODEL)
    estimated_matches = _count_matches_from_counts(
        model_filter=resolved_model_filter,
        hint_fraction_filter=FILTER_HINT_FRACTION,
    )
    if estimated_matches == 0 and not FILTER_ROLLOUT_IDS and not RECONSTRUCT_ROLLOUT_KEYS:
        raise ValueError(
            "No AIME/results rows match current model+hint filters. "
            f"model={FILTER_MODEL!r} resolved_model={resolved_model_filter!r} "
            f"hint_fraction={FILTER_HINT_FRACTION!r}"
        )
    requested_rollout_ids = resolve_requested_rollout_ids()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Starting pilot rescoring", flush=True)
    print(f"Input: {INPUT_FILE}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)
    limit_text = "all" if LIMIT is None else str(LIMIT)
    print(
        f"Settings: limit={limit_text} skip_eligible={SKIP_ELIGIBLE} "
        f"distinct_sample_id={DISTINCT_SAMPLE_ID}",
        flush=True,
    )
    print(
        f"Filters: model={FILTER_MODEL!r} hint_fraction={FILTER_HINT_FRACTION!r} "
        f"old_score={FILTER_OLD_SCORE!r}",
        flush=True,
    )
    print(f"Resolved model filter: {resolved_model_filter!r}", flush=True)
    if estimated_matches is not None:
        print(f"Estimated matching rows from counts file: {estimated_matches:,}", flush=True)
    print(f"Rollout filter: {len(requested_rollout_ids)} requested rollout_id(s)", flush=True)

    started = time.perf_counter()
    eligible_seen = 0
    scored = 0
    changed_score = 0
    changed_extract = 0
    seen_sample_ids: set[str] = set()
    row_times: list[float] = []
    out_rows: list[dict[str, Any]] = []
    offsets = load_offsets_from_index(INPUT_FILE, rollout_ids=requested_rollout_ids)
    print(
        f"Loaded {len(offsets):,} indexed offsets from {index_path_for_source(INPUT_FILE)}",
        flush=True,
    )

    with INPUT_FILE.open("rb") as f:
        for rollout_id_from_index, byte_offset in offsets:
            f.seek(byte_offset)
            raw_line = f.readline()
            if not raw_line:
                continue
            row = json.loads(raw_line.decode("utf-8", errors="replace"))

            rollout_id = str(row.get("rollout_id") or "")
            if rollout_id_from_index and rollout_id and rollout_id_from_index != rollout_id:
                raise RuntimeError(
                    "Index/source mismatch detected: "
                    f"index rollout_id={rollout_id_from_index!r} file rollout_id={rollout_id!r}"
                )

            if LIMIT is not None and scored >= LIMIT:
                break

            sample_id = str(row.get("sample_id") or "")
            if DISTINCT_SAMPLE_ID and sample_id in seen_sample_ids:
                continue

            if row.get("output_text") is None or row.get("target") is None:
                continue
            row_model = str(row.get("model") or "")
            row_hint = row.get("hint_fraction")
            if not _matches_model_filter(row_model, resolved_model_filter):
                continue
            if FILTER_HINT_FRACTION is not None:
                try:
                    row_hint_float = float(row_hint)
                except (TypeError, ValueError):
                    continue
                if abs(row_hint_float - FILTER_HINT_FRACTION) > 1e-9:
                    continue

            scorer_outcomes = row.get("scorer_outcomes") or {}
            scorer_outcomes = scorer_outcomes if isinstance(scorer_outcomes, dict) else {}
            old_payload_raw = scorer_outcomes.get(OLD_SCORER_NAME)
            old_payload = old_payload_raw if isinstance(old_payload_raw, dict) else {}
            old_score = normalize_score_value(old_payload.get("score_normalized"))
            if FILTER_OLD_SCORE is not None and old_score != FILTER_OLD_SCORE.strip().upper():
                continue

            eligible_seen += 1
            if eligible_seen <= SKIP_ELIGIBLE:
                continue

            output_text = str(row.get("output_text") or "")
            target = str(row.get("target") or "")
            old_answer = old_extracted_answer(old_payload)

            t0 = time.perf_counter()
            new_answer = extract_answer_fixed(output_text)
            is_correct = await grade_math_answer(
                answer=new_answer,
                target=target,
                exact_match=True,
                use_sympy=True,
            )
            dt = time.perf_counter() - t0
            row_times.append(dt)

            new_score = normalize_score_value(is_correct)
            score_changed = old_score != new_score
            extract_changed = (old_answer or "") != (new_answer or "")
            if score_changed:
                changed_score += 1
            if extract_changed:
                changed_extract += 1

            scored += 1
            seen_sample_ids.add(sample_id)
            print(
                f"[{scored}/{limit_text}] Example",
                flush=True,
            )
            print(
                f"  sample_id={sample_id} epoch={row.get('epoch')} rollout_id={rollout_id}",
                flush=True,
            )
            print(f"  model={row_model!r} hint_fraction={row_hint!r}", flush=True)
            print(f"  target_answer={target!r}", flush=True)
            print(f"  old_extracted_answer={old_answer!r}", flush=True)
            print(f"  new_extracted_answer={new_answer!r}", flush=True)
            print(f"  old_score={old_score} new_score={new_score}", flush=True)
            print(
                f"  changed_score={score_changed} changed_extract={extract_changed} rescore_time={dt:.3f}s",
                flush=True,
            )

            out_rows.append(
                {
                    "rollout_id": rollout_id,
                    "sample_id": sample_id,
                    "epoch": row.get("epoch"),
                    "target": target,
                    "source_file": str(INPUT_FILE),
                    "source_row_line": None,
                    "scorer_name": NEW_SCORER_NAME,
                    "score_raw_value": bool(is_correct),
                    "score_normalized": new_score,
                    "is_correct": bool(is_correct),
                    "extracted_answer": new_answer,
                    "extraction_status": extraction_status(new_answer),
                    "explanation": None,
                    "metadata_json": {
                        "method": "extract_answer_fixed + grade_math_answer(exact_match=True,use_sympy=True)",
                        "old_scorer_name": OLD_SCORER_NAME,
                        "old_score_normalized": old_score,
                        "old_extracted_answer": old_answer,
                        "changed_score": score_changed,
                        "changed_extracted_answer": extract_changed,
                        "rescore_elapsed_sec": dt,
                        "source": "rollout_offset_index",
                    },
                }
            )

    if scored == 0:
        print("No rows scored. Adjust LIMIT/SKIP_ELIGIBLE or verify input.", flush=True)
        return

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for row in out_rows:
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    elapsed = time.perf_counter() - started
    rows_per_sec = scored / max(elapsed, 1e-9)
    avg_row_sec = sum(row_times) / max(len(row_times), 1)
    print(
        f"Done: scored={scored} elapsed={elapsed:.2f}s rows_per_sec={rows_per_sec:.2f} "
        f"avg_rescore_sec_per_row={avg_row_sec:.3f} changed_score={changed_score} changed_extract={changed_extract}",
        flush=True,
    )
    print(f"Wrote pilot sidecar: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    # python suze_experiments/20260313/pilot_rescore_aime_extract_answer_fixed.py
    asyncio.run(main())
