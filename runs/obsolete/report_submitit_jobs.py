from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("data/submitit_logs/hinted")
N = 20
HEAD_BYTES = 64 * 1024
TAIL_BYTES = 256 * 1024
COUNT_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class JobReport:
    job_id: str
    submitted_at: str | None
    status: str
    job_name: str | None
    model: str | None
    benchmark: str | None
    hint_type: str | None
    fractioner: str | None
    gpus: str | None
    partition: str | None
    account: str | None
    time_limit: str | None
    stdout_size: int
    stderr_size: int
    result_exists: bool
    vllm_started: bool
    vllm_healthy: bool
    inference_started: bool
    final_error_count: int
    retry_count: int
    clip_count: int
    latest_progress: str | None
    last_error: str | None
    stdout_path: str | None
    stderr_path: str | None


def _read_head(path: Path, limit: int = HEAD_BYTES) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


def _read_tail(path: Path, limit: int = TAIL_BYTES) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > limit:
            f.seek(size - limit)
        return f.read(limit).decode("utf-8", errors="replace")


def _count_bytes(path: Path, needle: bytes) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(COUNT_CHUNK_BYTES)
            if not chunk:
                break
            count += chunk.count(needle)
    return count


def _contains_bytes(path: Path, needle: bytes) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as f:
        while True:
            chunk = f.read(COUNT_CHUNK_BYTES)
            if not chunk:
                return False
            if needle in chunk:
                return True


def _clean_line(line: str) -> str:
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    line = line.replace("\r", "\n").splitlines()[-1] if "\r" in line else line
    return re.sub(r"\s+", " ", line).strip()


def _parse_sbatch(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in _read_head(path).splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH"):
            continue
        body = line.removeprefix("#SBATCH").strip()
        if body.startswith("--") and "=" in body:
            key, value = body[2:].split("=", 1)
            fields[key.strip()] = value.strip()
        else:
            parts = body.split(maxsplit=1)
            if parts and parts[0].startswith("--"):
                fields[parts[0][2:]] = parts[1].strip() if len(parts) > 1 else "true"
    return fields


def _submission_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*_submission.sh"))


def _job_id_from_submission(path: Path) -> str | None:
    match = re.fullmatch(r"(\d+)_submission\.sh", path.name)
    return match.group(1) if match else None


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _extract_model(*, job_name: str | None, head_text: str, tail_text: str) -> str | None:
    text = head_text + "\n" + tail_text
    match = re.search(r"\[vllm\] starting: .*?vllm serve ([^\s]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"\[hinted_inference\] start model=([^\s]+)", text)
    if match:
        return match.group(1)
    if job_name and job_name.startswith("hinted_"):
        return job_name.removeprefix("hinted_")
    return None


def _extract_combo(text: str) -> tuple[str | None, str | None, str | None]:
    match = re.search(
        r"expanded_hinted_prompts/([^/\s]+)/([^/\s]+)/([^/\s]+)/fraction_", text
    )
    if match:
        return match.group(1), match.group(2), match.group(3)

    match = re.search(
        r"expanded prompts built fraction=.*?path=.*?/([^/\s]+)/([^/\s]+)/([^/\s]+)/fraction_",
        text,
    )
    if match:
        return match.group(1), match.group(2), match.group(3)

    return None, None, None


def _last_matching_line(text: str, patterns: list[str]) -> str | None:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for line in reversed(text.splitlines()):
        clean = _clean_line(line)
        if clean and any(pattern.search(clean) for pattern in compiled):
            return clean
    return None


def _latest_progress(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        clean = _clean_line(line)
        if "hinted:" in clean or "[hinted_inference] checkpoint" in clean:
            return clean[-220:]
    return None


def _status(
    *,
    stdout_head: str,
    stdout_tail: str,
    stderr_tail: str,
    result_exists: bool,
    stdout_exists: bool,
    stderr_exists: bool,
    final_error_count: int,
) -> str:
    text = "\n".join([stdout_head, stdout_tail, stderr_tail])
    if "Job completed successfully" in text or "Exiting after successful completion" in text:
        if final_error_count:
            return "finished_success_internal_errors"
        return "finished_success"
    if re.search(
        r"(submitit ERROR|Traceback \(most recent call last\)|RuntimeError:|TimeoutError:|"
        r"vLLM exited early|Killed|OutOfMemory|CUDA out of memory|OOM)",
        text,
        flags=re.IGNORECASE,
    ):
        if result_exists:
            return "finished_error"
        return "running_or_failed"
    if result_exists:
        return "finished_unknown"
    if stdout_exists or stderr_exists:
        return "running_or_stuck"
    return "submitted_no_logs"


def _build_report(log_dir: Path, submission_path: Path) -> JobReport:
    job_id = _job_id_from_submission(submission_path)
    if job_id is None:
        raise ValueError(f"Unexpected submission filename: {submission_path}")

    stdout_path = log_dir / f"{job_id}_0_log.out"
    stderr_path = log_dir / f"{job_id}_0_log.err"
    result_path = log_dir / f"{job_id}_0_result.pkl"

    stdout_head = _read_head(stdout_path)
    stdout_tail = _read_tail(stdout_path)
    stderr_head = _read_head(stderr_path)
    stderr_tail = _read_tail(stderr_path)
    sbatch = _parse_sbatch(submission_path)
    job_name = sbatch.get("job-name")
    combined_head = "\n".join([stdout_head, stderr_head])
    combined_tail = "\n".join([stdout_tail, stderr_tail])
    benchmark, hint_type, fractioner = _extract_combo(combined_head + "\n" + combined_tail)

    final_error_count = _count_bytes(stderr_path, b"[hinted_inference] final_error")
    final_error_count += _count_bytes(stdout_path, b"[hinted_inference] final_error")
    retry_count = _count_bytes(stdout_path, b"[hinted_inference] retry")
    retry_count += _count_bytes(stderr_path, b"[hinted_inference] retry")
    clip_count = _count_bytes(stdout_path, b"clip_max_tokens")
    clip_count += _count_bytes(stderr_path, b"clip_max_tokens")
    vllm_started = _contains_bytes(stderr_path, b"[vllm] starting:")
    vllm_healthy = _contains_bytes(stderr_path, b"[vllm] healthy")
    inference_started = _contains_bytes(stdout_path, b"[hinted_inference] start")

    last_error = _last_matching_line(
        combined_tail,
        [
            r"final_error",
            r"Traceback",
            r"RuntimeError:",
            r"TimeoutError:",
            r"vLLMValidationError",
            r"CUDA out of memory",
            r"OutOfMemory",
            r"vLLM exited early",
            r"did not become healthy",
            r"Killed",
        ],
    )
    latest_progress = _latest_progress(stdout_tail)

    stdout_exists = stdout_path.exists()
    stderr_exists = stderr_path.exists()
    status = _status(
        stdout_head=stdout_head,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        result_exists=result_path.exists(),
        stdout_exists=stdout_exists,
        stderr_exists=stderr_exists,
        final_error_count=final_error_count,
    )

    return JobReport(
        job_id=job_id,
        submitted_at=_mtime_iso(submission_path),
        status=status,
        job_name=job_name,
        model=_extract_model(
            job_name=job_name,
            head_text=combined_head,
            tail_text=combined_tail,
        ),
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        gpus=sbatch.get("gpus-per-node"),
        partition=sbatch.get("partition"),
        account=sbatch.get("account"),
        time_limit=sbatch.get("time"),
        stdout_size=stdout_path.stat().st_size if stdout_path.exists() else 0,
        stderr_size=stderr_path.stat().st_size if stderr_path.exists() else 0,
        result_exists=result_path.exists(),
        vllm_started=vllm_started,
        vllm_healthy=vllm_healthy,
        inference_started=inference_started,
        final_error_count=final_error_count,
        retry_count=retry_count,
        clip_count=clip_count,
        latest_progress=latest_progress,
        last_error=last_error,
        stdout_path=str(stdout_path) if stdout_path.exists() else None,
        stderr_path=str(stderr_path) if stderr_path.exists() else None,
    )


def _format_size(size: int) -> str:
    value = float(size)
    for suffix in ["B", "K", "M", "G"]:
        if value < 1024 or suffix == "G":
            return f"{value:.1f}{suffix}" if suffix != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def _short(text: str | None, width: int) -> str:
    if not text:
        return "-"
    text = _clean_line(text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _print_table(reports: list[JobReport]) -> None:
    columns = [
        ("job_id", 8),
        ("submitted", 19),
        ("status", 32),
        ("model", 34),
        ("frac", 13),
        ("gpu", 3),
        ("vllm", 7),
        ("final", 5),
        ("retry", 5),
        ("clip", 5),
        ("last_error/progress", 100),
    ]

    def cell(text: str | None, width: int) -> str:
        return _short(text, width).ljust(width)

    print("  ".join(cell(header, width) for header, width in columns))
    print("  ".join("-" * width for _, width in columns))
    for report in reports:
        vllm = "healthy" if report.vllm_healthy else ("started" if report.vllm_started else "-")
        detail = report.last_error or report.latest_progress
        values = [
            report.job_id,
            (report.submitted_at or "-").replace("T", " "),
            report.status,
            report.model,
            report.fractioner or "-",
            report.gpus or "-",
            vllm,
            str(report.final_error_count),
            str(report.retry_count),
            str(report.clip_count),
            detail,
        ]
        print("  ".join(cell(value, width) for value, (_, width) in zip(values, columns)))


def main() -> None:
    submissions = sorted(
        _submission_files(LOG_DIR),
        key=lambda path: int(_job_id_from_submission(path) or 0),
        reverse=True,
    )
    reports = [_build_report(LOG_DIR, path) for path in submissions[:N]]
    _print_table(reports)


if __name__ == "__main__":
    # python -m runs.report_submitit_jobs
    main()
