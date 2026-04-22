from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_HEALTH_TIMEOUT_SECONDS = 3600
HF_HUB_ETAG_TIMEOUT_SECONDS = 300
HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 1800


def _timestamp() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True, file=sys.stderr)


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

    def start(self, *, health_timeout: int = DEFAULT_HEALTH_TIMEOUT_SECONDS) -> None:
        if self.process is not None:
            return

        cmd = self._cmd()
        _log(f"[vllm] starting: {' '.join(cmd)}")
        env = os.environ.copy()
        # vLLM resolves model shards through huggingface_hub, whose default 10s HEAD
        # timeout is too short for large model manifests on a slow or loaded connection.
        env.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_SECONDS))
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS))
        # Keep vLLM's internal engine startup timeout aligned with the outer health
        # wait so large models do not fail inside vLLM before this wrapper gives up.
        env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(health_timeout))
        self.process = subprocess.Popen(
            cmd,
            env=env,
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
                        _log(f"[vllm] healthy on port={self.port}")
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
        _log(f"[vllm] stopped port={self.port}")

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        self.shutdown()
        return False
