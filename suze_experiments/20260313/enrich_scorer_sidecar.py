from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "consolidated_jsonl"
    / "results__aime.extract_answer_fixed.scorers.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "consolidated_jsonl"
    / "results__aime.extract_answer_fixed.scorers.enriched.jsonl"
)
DEFAULT_STATE = (
    Path(__file__).resolve().parent
    / "consolidated_jsonl"
    / "_state"
    / "results__aime.extract_answer_fixed.scorers.enriched.state.json"
)
DEFAULT_SOURCE_FILE = (
    Path(__file__).resolve().parent
    / "consolidated_jsonl"
    / "results__aime.jsonl"
)
STATE_VERSION = 1

# Extra fields copied from the original consolidated rollout row.
ENRICH_FIELDS = [
    "eval_id",
    "source_owner",
    "run_type",
    "benchmark",
    "group_key",
    "task_name",
    "model",
    "model_path",
    "solver_name",
    "hint_fraction",
    "path_hint_level",
    "sample_idx",
    "target",
    "eval_path",
    "eval_rel_path",
    "created",
]


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Enrich scorer sidecar rows with rollout metadata from consolidated JSONL. "
            "Does not rescore and does not modify source files."
        )
    )
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    p.add_argument(
        "--default-source-file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Fallback source JSONL if a sidecar row has no source_file.",
    )
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=100000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--restart", action="store_true")
    return p.parse_args()


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
        "default_source_file": str(args.default_source_file),
        "next_input_byte_offset": 0,
        "lines_seen": 0,
        "rows_written": 0,
        "bad_sidecar_json_lines": 0,
        "bad_source_json_lines": 0,
        "missing_source_pointer": 0,
        "missing_source_file": 0,
        "source_lookup_failures": 0,
        "rollout_id_mismatch": 0,
    }


def validate_resume_state(state: dict[str, Any], args: argparse.Namespace) -> None:
    if int(state.get("version", -1)) != STATE_VERSION:
        raise ValueError(
            f"state version mismatch: {state.get('version')} != {STATE_VERSION}"
        )
    checks = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "default_source_file": str(args.default_source_file),
    }
    for k, expected in checks.items():
        if state.get(k) != expected:
            raise ValueError(f"state mismatch {k}: {state.get(k)!r} != {expected!r}")


def _parse_source_offset(value: Any) -> int | None:
    if value is None:
        return None
    try:
        offset = int(value)
    except Exception:
        return None
    if offset < 0:
        return None
    return offset


def run(args: argparse.Namespace) -> int:
    args.input_file = args.input_file.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    args.default_source_file = args.default_source_file.expanduser().resolve()

    if not args.input_file.exists():
        raise FileNotFoundError(f"input sidecar not found: {args.input_file}")

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

    next_input_offset = int(state.get("next_input_byte_offset") or 0)
    lines_seen = int(state.get("lines_seen") or 0)
    rows_written = int(state.get("rows_written") or 0)
    bad_sidecar_json_lines = int(state.get("bad_sidecar_json_lines") or 0)
    bad_source_json_lines = int(state.get("bad_source_json_lines") or 0)
    missing_source_pointer = int(state.get("missing_source_pointer") or 0)
    missing_source_file = int(state.get("missing_source_file") or 0)
    source_lookup_failures = int(state.get("source_lookup_failures") or 0)
    rollout_id_mismatch = int(state.get("rollout_id_mismatch") or 0)

    input_size = args.input_file.stat().st_size
    started = time.perf_counter()
    print(
        f"[{ts_now()}] START enrich_sidecar input={args.input_file} output={args.output_file} "
        f"state={args.state_file} resume_offset={next_input_offset:,}",
        flush=True,
    )

    source_handles: dict[str, Any] = {}
    last_lookup_key: tuple[str, int] | None = None
    last_source_row: dict[str, Any] | None = None
    since_ckpt = 0
    reached_eof = False
    limit_hit = False

    def get_source_handle(path_text: str) -> Any | None:
        nonlocal missing_source_file
        if path_text in source_handles:
            return source_handles[path_text]
        p = Path(path_text)
        if not p.exists():
            missing_source_file += 1
            return None
        fh = p.open("rb")
        source_handles[path_text] = fh
        return fh

    with args.input_file.open("rb") as src_sidecar, args.output_file.open(
        "a", encoding="utf-8"
    ) as out:
        src_sidecar.seek(next_input_offset)

        while True:
            raw = src_sidecar.readline()
            if not raw:
                reached_eof = True
                break

            lines_seen += 1
            if args.limit is not None and rows_written >= int(args.limit):
                limit_hit = True
                break

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                next_input_offset = src_sidecar.tell()
                continue

            try:
                row = json.loads(line)
            except Exception:
                bad_sidecar_json_lines += 1
                next_input_offset = src_sidecar.tell()
                continue

            enriched = dict(row)
            source_file_text = str(
                row.get("source_file") or str(args.default_source_file)
            )
            source_offset = _parse_source_offset(row.get("source_row_byte_offset"))

            source_row: dict[str, Any] | None = None
            lookup_ok = False
            if source_offset is None:
                missing_source_pointer += 1
            else:
                key = (source_file_text, source_offset)
                if key == last_lookup_key and last_source_row is not None:
                    source_row = last_source_row
                    lookup_ok = True
                else:
                    fh = get_source_handle(source_file_text)
                    if fh is None:
                        source_row = None
                    else:
                        try:
                            fh.seek(source_offset)
                            src_raw = fh.readline()
                            src_line = src_raw.decode("utf-8", errors="replace").strip()
                            source_row = json.loads(src_line) if src_line else None
                            lookup_ok = source_row is not None
                            if lookup_ok:
                                last_lookup_key = key
                                last_source_row = source_row
                        except Exception:
                            bad_source_json_lines += 1
                            source_row = None

            if source_row is None:
                source_lookup_failures += 1
                enriched["enrichment_status"] = "source_lookup_failed"
            else:
                for k in ENRICH_FIELDS:
                    enriched[k] = source_row.get(k)
                enriched["enrichment_status"] = "ok"
                enriched["enriched_from_source_file"] = source_file_text

                src_rollout_id = str(source_row.get("rollout_id") or "")
                side_rollout_id = str(row.get("rollout_id") or "")
                if src_rollout_id and side_rollout_id and src_rollout_id != side_rollout_id:
                    rollout_id_mismatch += 1
                    enriched["enrichment_status"] = "rollout_id_mismatch"

            out.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
            rows_written += 1
            since_ckpt += 1
            next_input_offset = src_sidecar.tell()

            if rows_written % int(args.log_every) == 0:
                elapsed = time.perf_counter() - started
                pct = 100.0 * next_input_offset / max(input_size, 1)
                rps = rows_written / max(elapsed, 1e-9)
                print(
                    f"[{ts_now()}] progress={pct:.2f}% rows_written={rows_written:,} lines_seen={lines_seen:,} "
                    f"rows/s={rps:.2f} missing_ptr={missing_source_pointer:,} "
                    f"lookup_fail={source_lookup_failures:,} sidecar_bad_json={bad_sidecar_json_lines:,}",
                    flush=True,
                )

            if since_ckpt >= int(args.checkpoint_every):
                out.flush()
                os.fsync(out.fileno())
                state.update(
                    {
                        "updated_at": ts_now(),
                        "completed": False,
                        "next_input_byte_offset": next_input_offset,
                        "lines_seen": lines_seen,
                        "rows_written": rows_written,
                        "bad_sidecar_json_lines": bad_sidecar_json_lines,
                        "bad_source_json_lines": bad_source_json_lines,
                        "missing_source_pointer": missing_source_pointer,
                        "missing_source_file": missing_source_file,
                        "source_lookup_failures": source_lookup_failures,
                        "rollout_id_mismatch": rollout_id_mismatch,
                    }
                )
                atomic_write_json(args.state_file, state)
                since_ckpt = 0

        out.flush()
        os.fsync(out.fileno())

    for fh in source_handles.values():
        try:
            fh.close()
        except Exception:
            pass

    state.update(
        {
            "updated_at": ts_now(),
            "completed": bool(reached_eof and not limit_hit),
            "next_input_byte_offset": next_input_offset,
            "lines_seen": lines_seen,
            "rows_written": rows_written,
            "bad_sidecar_json_lines": bad_sidecar_json_lines,
            "bad_source_json_lines": bad_source_json_lines,
            "missing_source_pointer": missing_source_pointer,
            "missing_source_file": missing_source_file,
            "source_lookup_failures": source_lookup_failures,
            "rollout_id_mismatch": rollout_id_mismatch,
        }
    )
    atomic_write_json(args.state_file, state)

    elapsed = time.perf_counter() - started
    print(
        f"[{ts_now()}] DONE rows_written={rows_written:,} elapsed={elapsed/60:.1f}m "
        f"sidecar_bad_json={bad_sidecar_json_lines:,} source_bad_json={bad_source_json_lines:,} "
        f"missing_ptr={missing_source_pointer:,} missing_source_file={missing_source_file:,} "
        f"lookup_fail={source_lookup_failures:,} rollout_id_mismatch={rollout_id_mismatch:,}",
        flush=True,
    )
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    # python suze_experiments/20260313/enrich_scorer_sidecar.py
    main()

