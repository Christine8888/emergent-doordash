"""vLLM server management for experiments.

Adapted from fuzzy-tasks/src/model.py with modifications:
- Removed CUDA_VISIBLE_DEVICES logic (SLURM handles GPU allocation)
- Added context manager support
- Simplified configuration
"""

import os
import subprocess
import time
import socket
import logging
import signal
import threading
from typing import Optional
import requests

logger = logging.getLogger(__name__)


def _force_exit(signum=None, frame=None):
    """Force immediate exit without cleanup (avoids async hangs)."""
    code = 128 + signum if signum else 1
    logger.info(f"Force exiting with code {code}")
    os._exit(code)


def get_served_name(model_path: str) -> str:
    """Extract served model name from path."""
    import re

    # Handle checkpoint paths
    if "checkpoint" in model_path or "final-model" in model_path:
        model_path = re.sub(r"/checkpoint-\d+", "", model_path)
        model_path = re.sub(r"/final-model", "", model_path)

    return os.path.basename(model_path)


class vLLMServer:
    """Manages a vLLM server subprocess.

    IMPORTANT: Does NOT set CUDA_VISIBLE_DEVICES - SLURM handles GPU allocation.
    """

    def __init__(
        self,
        model_path: str,
        port: Optional[int] = None,
        served_model_name: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        n_gpus: Optional[int] = None,
        verbose: bool = True,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        dtype: str = "auto",
        max_num_batched_tokens: Optional[int] = None,
        **vllm_kwargs
    ):
        """Initialize vLLM server configuration.

        Args:
            model_path: Path to model
            port: Port to serve on (auto-find if None)
            served_model_name: Name to serve model as (auto-detect if None)
            tensor_parallel_size: Tensor parallelism size
            max_model_len: Max sequence length
            gpu_memory_utilization: GPU memory utilization
            n_gpus: Total GPUs available (for data parallelism calculation)
            verbose: Whether to log stderr
            enable_prefix_caching: Enable vLLM prefix caching
            enable_chunked_prefill: Enable vLLM chunked prefill
            dtype: Model dtype (e.g. "auto", "float16", "bfloat16")
            max_num_batched_tokens: Max tokens per batch during prefill.
                    When chunked prefill is enabled vLLM defaults to 2048;
                    None means use vLLM's default.
            **vllm_kwargs: Additional vLLM arguments
        """
        self.model_path = model_path
        self.port = port or self._find_free_port()
        self.served_model_name = served_model_name or get_served_name(model_path)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.verbose = verbose
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.dtype = dtype
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vllm_kwargs = vllm_kwargs

        # Calculate data parallelism
        if n_gpus is not None:
            self.data_parallel_size = n_gpus // tensor_parallel_size
            logger.info(
                f"n_gpus={n_gpus}: data_parallel_size={self.data_parallel_size} "
                f"(n_gpus // tensor_parallel_size={tensor_parallel_size})"
            )
        else:
            self.data_parallel_size = 1

        self.process: Optional[subprocess.Popen] = None
        self._stop_monitor = threading.Event()

    def _find_free_port(self) -> int:
        """Find a free port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    def _monitor_process(self):
        """Monitor server process, force exit if server dies."""
        while not self._stop_monitor.is_set():
            if self.process and self.process.poll() is not None:
                logger.error(f"vLLM server died with exit code {self.process.returncode}")
                _force_exit()
            self._stop_monitor.wait(5)

    def start(self, health_timeout: int = 1200):
        """Start the vLLM server and wait for health check.

        Args:
            health_timeout: Seconds to wait for health check
        """
        import threading

        # Log GPU environment (set by SLURM)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set (will use all)')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(self.port),
            "--served-model-name", self.served_model_name,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--data-parallel-size", str(self.data_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
        ]
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])

        # Add any extra vLLM arguments
        for key, value in self.vllm_kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        logger.info(f"Starting vLLM server on port {self.port}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Served as: {self.served_model_name}")
        logger.info(f"TP: {self.tensor_parallel_size}, DP: {self.data_parallel_size}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def stream_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if line:
                    logger.info(f"[vLLM {prefix}] {line.rstrip()}")

        # Stream stdout
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(self.process.stdout, "stdout"),
            daemon=True
        )
        stdout_thread.start()

        # Stream stderr if verbose
        if self.verbose:
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(self.process.stderr, "stderr"),
                daemon=True
            )
            stderr_thread.start()

        self._wait_for_health(timeout=health_timeout)

        # Register signal handlers for clean forced exit
        signal.signal(signal.SIGTERM, _force_exit)
        signal.signal(signal.SIGINT, _force_exit)

        # Start health monitor
        threading.Thread(target=self._monitor_process, daemon=True).start()

    def _wait_for_health(self, timeout: int = 300):
        """Wait for vLLM server to become healthy."""
        health_url = f"http://localhost:{self.port}/health"
        start_time = time.time()

        logger.info(f"Waiting for health check at {health_url}")

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                raise RuntimeError("vLLM server process died during startup")

            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready on port {self.port}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass

            time.sleep(2)

        raise TimeoutError(f"vLLM server did not become healthy within {timeout}s")

    def shutdown(self):
        """Shutdown the vLLM server."""
        self._stop_monitor.set()

        if self.process is None:
            return

        logger.info(f"Shutting down vLLM server on port {self.port}")
        self.process.terminate()

        try:
            self.process.wait(timeout=10)
            logger.info("vLLM server terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Force killing vLLM server")
            self.process.kill()
            self.process.wait()

        self.process = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
