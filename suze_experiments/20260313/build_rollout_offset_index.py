from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "consolidated_jsonl"
INDEX_SUFFIX = ".rollout_index.sqlite3"
ROLLOUT_ID_RE = re.compile(rb'"rollout_id"\s*:\s*"([^"]+)"')
# Optional default file filter when --files is not passed.
# Examples:
#   DEFAULT_FILES = None  # index all JSONL files
#   DEFAULT_FILES = ["results__aime.jsonl"]
#   DEFAULT_FILES = ["results__aime.jsonl", "results__gpqa.jsonl"]
DEFAULT_FILES: list[str] | None = ["results__aime.jsonl"]


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_size(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == "TB":
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} TB"


def index_path_for_source(source_path: Path) -> Path:
    return source_path.with_name(source_path.name + INDEX_SUFFIX)


def _read_meta(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    try:
        with sqlite3.connect(str(index_path)) as con:
            rows = con.execute("SELECT key, value FROM meta").fetchall()
        return {str(k): str(v) for k, v in rows}
    except Exception:
        return {}


def is_fresh(source_path: Path, index_path: Path) -> bool:
    if not index_path.exists():
        return False
    st = source_path.stat()
    meta = _read_meta(index_path)
    return (
        meta.get("size_bytes") == str(int(st.st_size))
        and meta.get("mtime_ns") == str(int(st.st_mtime_ns))
    )


def create_schema(con: sqlite3.Connection) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS offsets (rollout_id TEXT PRIMARY KEY, byte_offset INTEGER NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def build_index_for_file(
    source_path: Path,
    *,
    force: bool,
    batch_size: int,
    log_every_sec: float,
) -> tuple[bool, int]:
    index_path = index_path_for_source(source_path)
    if not force and is_fresh(source_path, index_path):
        print(f"[{ts_now()}] SKIP fresh index exists: {index_path.name}", flush=True)
        return False, 0

    if index_path.exists():
        index_path.unlink()

    st = source_path.stat()
    print(
        f"[{ts_now()}] START {source_path.name} size={human_size(st.st_size)} -> {index_path.name}",
        flush=True,
    )

    con = sqlite3.connect(str(index_path))
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        create_schema(con)

        started = time.time()
        last_log = started
        rows = 0
        bytes_seen = 0
        batch: list[tuple[str, int]] = []

        with source_path.open("rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                bytes_seen += len(line)

                m = ROLLOUT_ID_RE.search(line)
                if m is None:
                    # Fallback for any atypical formatting.
                    try:
                        row = json.loads(line.decode("utf-8", errors="replace"))
                        rid = str(row.get("rollout_id") or "")
                    except Exception:
                        rid = ""
                else:
                    rid = m.group(1).decode("utf-8", errors="replace")

                if rid:
                    batch.append((rid, off))

                if len(batch) >= batch_size:
                    con.executemany("INSERT OR REPLACE INTO offsets(rollout_id, byte_offset) VALUES (?, ?)", batch)
                    rows += len(batch)
                    batch.clear()

                now = time.time()
                if now - last_log >= log_every_sec:
                    elapsed = now - started
                    speed = bytes_seen / max(elapsed, 1e-9)
                    pct = 100.0 * bytes_seen / max(st.st_size, 1)
                    print(
                        f"[{ts_now()}] {source_path.name} progress={pct:.2f}% "
                        f"rows={rows:,} read={human_size(bytes_seen)} speed={human_size(speed)}/s",
                        flush=True,
                    )
                    last_log = now

        if batch:
            con.executemany("INSERT OR REPLACE INTO offsets(rollout_id, byte_offset) VALUES (?, ?)", batch)
            rows += len(batch)

        con.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            [
                ("size_bytes", str(int(st.st_size))),
                ("mtime_ns", str(int(st.st_mtime_ns))),
                ("built_at", ts_now()),
                ("rows_indexed", str(int(rows))),
                ("source_file", str(source_path)),
            ],
        )
        con.commit()
    finally:
        con.close()

    elapsed = time.time() - started
    avg_speed = st.st_size / max(elapsed, 1e-9)
    print(
        f"[{ts_now()}] DONE {source_path.name} rows={rows:,} elapsed={elapsed/60:.1f}m "
        f"avg_speed={human_size(avg_speed)}/s",
        flush=True,
    )
    return True, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rollout_id->byte_offset indexes for consolidated JSONL files.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--files", nargs="*", default=None, help="Optional specific filenames, e.g. results__aime.jsonl")
    p.add_argument("--force", action="store_true", help="Rebuild even when index appears fresh.")
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--log-every-sec", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")

    files = sorted(data_dir.glob("*.jsonl"))
    files = [p for p in files if not p.name.startswith("_")]
    selected_files = args.files if args.files else DEFAULT_FILES
    if selected_files:
        keep = set(selected_files)
        files = [p for p in files if p.name in keep]
    if not files:
        print(f"[{ts_now()}] no matching files in {data_dir}", flush=True)
        return

    print(
        f"[{ts_now()}] building rollout offset indexes for {len(files)} file(s) "
        f"(selected={selected_files if selected_files else 'all'})",
        flush=True,
    )
    rebuilt = 0
    total_rows = 0
    for p in files:
        did_build, rows = build_index_for_file(
            p,
            force=bool(args.force),
            batch_size=int(args.batch_size),
            log_every_sec=float(args.log_every_sec),
        )
        if did_build:
            rebuilt += 1
            total_rows += rows

    print(f"[{ts_now()}] COMPLETE rebuilt={rebuilt} indexed_rows={total_rows:,}", flush=True)


if __name__ == "__main__":
    # python suze_experiments/20260313/build_rollout_offset_index.py
    main()
