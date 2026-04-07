from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class VLLMServerConfig:
    model_path: str
    served_model_name: str
    tensor_parallel_size: int
    data_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int | None
    dtype: str


class VLLMServer:
    def __init__(self, config: VLLMServerConfig):
        self.config = config
        self.port = _find_free_port()
        self.process: subprocess.Popen[str] | None = None

    def _cmd(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.config.model_path,
            "--port",
            str(self.port),
            "--served-model-name",
            self.config.served_model_name,
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--data-parallel-size",
            str(self.config.data_parallel_size),
            "--max-model-len",
            str(self.config.max_model_len),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--dtype",
            self.config.dtype,
        ]
        cmd.append("--enable-prefix-caching")
        cmd.append("--enable-chunked-prefill")
        if self.config.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)])
        return cmd

    def start(self, *, health_timeout: int = 1200) -> None:
        if self.process is not None:
            return

        cmd = self._cmd()
        print(f"[vllm] starting: {' '.join(cmd)}", flush=True, file=sys.stderr)
        self.process = subprocess.Popen(
            cmd,
            # Route all vLLM output to stderr.
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        self._wait_for_health(timeout=health_timeout)

    def _wait_for_health(self, *, timeout: int) -> None:
        if self.process is None:
            raise RuntimeError("vLLM process is not started")

        url = f"http://localhost:{self.port}/health"
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited early with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if int(resp.status) == 200:
                        print(f"[vllm] healthy on port={self.port}", flush=True, file=sys.stderr)
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM did not become healthy within {timeout}s")

    def shutdown(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None
        print(f"[vllm] stopped port={self.port}", flush=True, file=sys.stderr)

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        self.shutdown()
        return False
