from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_INPUT = Path(__file__).resolve().parent / "consolidated_jsonl" / "results__aime.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "consolidated_jsonl" / "results__aime.extract_answer_fixed.scorers.jsonl"
DEFAULT_STATE = Path(__file__).resolve().parent / "consolidated_jsonl" / "_state" / "results__aime.extract_answer_fixed.state.json"
DEFAULT_NEW_SCORER = "aime_scorer_extract_answer_fixed"
DEFAULT_OLD_SCORER = "aime_scorer"
STATE_VERSION = 1


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def extraction_status(answer: str | None) -> str:
    return "ok" if answer is not None and answer.strip() else "failed"


def normalize_answer_for_compare(answer: str | None) -> str:
    if answer is None:
        return ""
    return str(answer).strip()


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rescore consolidated AIME JSONL using extract_answer_fixed + canonical math grader. "
            "Writes scorer-only sidecar JSONL; never mutates the input file."
        )
    )
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    p.add_argument("--new-scorer-name", type=str, default=DEFAULT_NEW_SCORER)
    p.add_argument("--old-scorer-name", type=str, default=DEFAULT_OLD_SCORER)
    p.add_argument("--checkpoint-every", type=int, default=100000, help="Write checkpoint every N scored rows.")
    p.add_argument("--log-every", type=int, default=100000, help="Progress log every N scored rows.")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of scored rows.")
    p.add_argument("--restart", action="store_true", help="Ignore existing checkpoint/output and start from scratch.")
    return p.parse_args()


def suppress_noisy_warnings() -> None:
    """Hide known noisy non-fatal warnings from sympy/parser stack."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"sympy\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Using non-Expr arguments in Pow is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*antlr4\.error\.ErrorListener module is not installed.*",
        category=UserWarning,
    )
    try:
        from sympy.utilities.exceptions import SymPyDeprecationWarning

        warnings.filterwarnings(
            "ignore",
            category=SymPyDeprecationWarning,
        )
    except Exception:
        # If sympy isn't importable in this environment, do nothing here.
        pass


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def make_initial_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at": ts_now(),
        "updated_at": ts_now(),
        "completed": False,
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "new_scorer_name": args.new_scorer_name,
        "old_scorer_name": args.old_scorer_name,
        "next_byte_offset": 0,
        "lines_seen": 0,
        "rows_scored": 0,
        "rows_reused_old_score": 0,
        "rows_regraded": 0,
        "rows_skipped_missing_fields": 0,
        "score_counts": {"C": 0, "I": 0, "U": 0},
        "changed_from_old": {
            "score_changed": 0,
            "extracted_answer_changed": 0,
        },
    }


def validate_resume_state(state: dict[str, Any], args: argparse.Namespace) -> None:
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"state version mismatch: {state.get('version')} != {STATE_VERSION}")
    checks = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "new_scorer_name": args.new_scorer_name,
        "old_scorer_name": args.old_scorer_name,
    }
    for k, expected in checks.items():
        actual = state.get(k)
        if actual != expected:
            raise ValueError(f"state mismatch {k}: {actual!r} != {expected!r}")


async def run(args: argparse.Namespace) -> int:
    from environments.math.utils import extract_answer_fixed, grade_math_answer

    args.input_file = args.input_file.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    if not args.input_file.exists():
        raise FileNotFoundError(f"input file not found: {args.input_file}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)

    if args.restart:
        if args.output_file.exists():
            args.output_file.unlink()
        if args.state_file.exists():
            args.state_file.unlink()

    state = load_state(args.state_file)
    if state is None:
        state = make_initial_state(args)
        atomic_write_json(args.state_file, state)
    else:
        validate_resume_state(state, args)

    next_offset = int(state.get("next_byte_offset") or 0)
    lines_seen = int(state.get("lines_seen") or 0)
    rows_scored = int(state.get("rows_scored") or 0)
    rows_reused_old_score = int(state.get("rows_reused_old_score") or 0)
    rows_regraded = int(state.get("rows_regraded") or 0)
    rows_skipped = int(state.get("rows_skipped_missing_fields") or 0)
    score_counts = state.get("score_counts") or {"C": 0, "I": 0, "U": 0}
    changed = state.get("changed_from_old") or {"score_changed": 0, "extracted_answer_changed": 0}

    started = time.perf_counter()
    file_size = args.input_file.stat().st_size
    print(
        f"[{ts_now()}] START rescoring input={args.input_file} output={args.output_file} "
        f"state={args.state_file} resume_offset={next_offset:,}",
        flush=True,
    )

    scored_since_ckpt = 0
    reached_eof = False
    limit_hit = False

    with args.input_file.open("rb") as src, args.output_file.open("a", encoding="utf-8") as out:
        src.seek(next_offset)

        while True:
            line_start = src.tell()
            raw = src.readline()
            if not raw:
                reached_eof = True
                break

            lines_seen += 1
            if args.limit is not None and rows_scored >= int(args.limit):
                limit_hit = True
                break

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                next_offset = src.tell()
                continue

            try:
                row = json.loads(line)
            except Exception:
                next_offset = src.tell()
                continue

            rollout_id = str(row.get("rollout_id") or "")
            output_text = row.get("output_text")
            target = row.get("target")
            sample_id = str(row.get("sample_id") or "")
            epoch = row.get("epoch")

            if not rollout_id or output_text is None or target is None:
                rows_skipped += 1
                next_offset = src.tell()
                continue

            new_answer = extract_answer_fixed(str(output_text))

            old_payload = (row.get("scorer_outcomes") or {}).get(args.old_scorer_name, {})
            old_payload = old_payload if isinstance(old_payload, dict) else {}
            old_score = normalize_score_value(old_payload.get("score_normalized"))
            old_answer = old_extracted_answer(old_payload)
            answer_changed = normalize_answer_for_compare(old_answer) != normalize_answer_for_compare(new_answer)

            if (not answer_changed) and old_score in {"C", "I"}:
                new_score = old_score
                is_correct = old_score == "C"
                rows_reused_old_score += 1
                score_source = "reused_old_score_same_extracted_answer"
            else:
                if not normalize_answer_for_compare(new_answer):
                    is_correct = False
                    new_score = "I"
                else:
                    is_correct = await grade_math_answer(
                        answer=new_answer,
                        target=str(target),
                        exact_match=True,
                        use_sympy=True,
                    )
                    new_score = normalize_score_value(is_correct)
                rows_regraded += 1
                score_source = "regraded_with_extract_answer_fixed"

            score_counts[new_score] = int(score_counts.get(new_score, 0)) + 1
            score_changed = old_score != new_score
            if score_changed:
                changed["score_changed"] = int(changed.get("score_changed", 0)) + 1
            if answer_changed:
                changed["extracted_answer_changed"] = int(changed.get("extracted_answer_changed", 0)) + 1

            sidecar_row = {
                "rollout_id": rollout_id,
                "source_file": str(args.input_file),
                "source_row_byte_offset": line_start,
                "sample_id": sample_id,
                "epoch": epoch,
                "scorer_name": args.new_scorer_name,
                "score_raw_value": bool(is_correct),
                "score_normalized": new_score,
                "is_correct": bool(is_correct),
                "extracted_answer": new_answer,
                "extraction_status": extraction_status(new_answer),
                "explanation": None,
                "metadata_json": {
                    "method": "extract_answer_fixed + conditional grade_math_answer(exact_match=True,use_sympy=True)",
                    "score_source": score_source,
                    "old_scorer_name": args.old_scorer_name,
                    "old_score_normalized": old_score,
                    "old_extracted_answer": old_answer,
                    "changed_score": score_changed,
                    "changed_extracted_answer": answer_changed,
                },
            }
            out.write(json.dumps(sidecar_row, ensure_ascii=False, default=str) + "\n")

            rows_scored += 1
            scored_since_ckpt += 1
            next_offset = src.tell()

            if rows_scored % int(args.log_every) == 0:
                elapsed = time.perf_counter() - started
                pct = 100.0 * next_offset / max(file_size, 1)
                rps = rows_scored / max(elapsed, 1e-9)
                print(
                    f"[{ts_now()}] progress={pct:.2f}% rows_scored={rows_scored:,} lines_seen={lines_seen:,} "
                    f"rows/s={rps:.2f} reused={rows_reused_old_score:,} regraded={rows_regraded:,} "
                    f"score_counts={score_counts}",
                    flush=True,
                )

            if scored_since_ckpt >= int(args.checkpoint_every):
                out.flush()
                os.fsync(out.fileno())
                state.update(
                    {
                        "updated_at": ts_now(),
                        "completed": False,
                        "next_byte_offset": next_offset,
                        "lines_seen": lines_seen,
                        "rows_scored": rows_scored,
                        "rows_reused_old_score": rows_reused_old_score,
                        "rows_regraded": rows_regraded,
                        "rows_skipped_missing_fields": rows_skipped,
                        "score_counts": score_counts,
                        "changed_from_old": changed,
                    }
                )
                atomic_write_json(args.state_file, state)
                scored_since_ckpt = 0

        out.flush()
        os.fsync(out.fileno())

    state.update(
        {
            "updated_at": ts_now(),
            "completed": bool(reached_eof and not limit_hit),
            "next_byte_offset": next_offset,
            "lines_seen": lines_seen,
            "rows_scored": rows_scored,
            "rows_reused_old_score": rows_reused_old_score,
            "rows_regraded": rows_regraded,
            "rows_skipped_missing_fields": rows_skipped,
            "score_counts": score_counts,
            "changed_from_old": changed,
        }
    )
    atomic_write_json(args.state_file, state)

    elapsed = time.perf_counter() - started
    print(
        f"[{ts_now()}] DONE rows_scored={rows_scored:,} elapsed={elapsed/60:.1f}m "
        f"reused={rows_reused_old_score:,} regraded={rows_regraded:,} "
        f"score_counts={score_counts} changed={changed}",
        flush=True,
    )
    return 0


def main() -> None:
    suppress_noisy_warnings()
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
