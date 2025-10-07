import logging
import os
import subprocess
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class VLLMServer:
    def __init__(
        self,
        model_path: str,
        port: int = 9000,
        tensor_parallel: int = 1,
        gpu_ids: list[int] | None = None,
        model_name: str | None = None,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.port = port
        self.tensor_parallel = tensor_parallel
        self.gpu_ids = gpu_ids or list(range(tensor_parallel))
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.process = None

    def start(self, wait_for_ready: bool = True, timeout: int = 300):
        """Start vLLM server"""
        if self.process is not None:
            logger.warning("Server already running")
            return

        cmd = [
            "vllm",
            "serve",
            self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", "auto",
            "--enable-prefix-caching",
            "--max-num-seqs", "32",
            "--enable-chunked-prefill",
            "--kv-cache-dtype", "auto",
        ]

        if self.model_name:
            cmd.extend(["--served-model-name", self.model_name])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        logger.info(f"Starting vLLM server on port {self.port} with GPUs {self.gpu_ids}")
        self.process = subprocess.Popen(cmd, env=env)

        if wait_for_ready:
            self.wait_for_ready(timeout)

    def wait_for_ready(self, timeout: int = 300):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                logger.info(f"Server ready on port {self.port}")
                return
            time.sleep(2)
        raise TimeoutError(f"Server not ready after {timeout}s")

    def is_ready(self) -> bool:
        """Check if server is ready"""
        try:
            response = requests.get(f"{self.url}/health", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def stop(self):
        """Stop server"""
        if self.process is None:
            return
        logger.info(f"Stopping server on port {self.port}")
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None

    @property
    def url(self) -> str:
        """Get server URL"""
        return f"http://localhost:{self.port}"

    @property
    def served_model_name(self) -> str:
        """Get model name"""
        if self.model_name:
            return self.model_name
        return Path(self.model_path).name

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        response = requests.post(
            f"{self.url}/v1/completions",
            json={"model": self.served_model_name, "prompt": prompt, **kwargs},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Chat completion"""
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json={"model": self.served_model_name, "messages": messages, **kwargs},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def start_multi_instance(
    model_path: str,
    n_devices: int = 4,
    tensor_parallel: int = 1,
    base_port: int = 9001,
    model_name: str | None = None,
) -> list[VLLMServer]:
    """Start multiple vLLM instances"""
    if n_devices % tensor_parallel != 0:
        raise ValueError(f"n_devices ({n_devices}) must be divisible by tensor_parallel ({tensor_parallel})")

    num_instances = n_devices // tensor_parallel
    servers = []

    for i in range(num_instances):
        gpu_start = i * tensor_parallel
        gpu_ids = list(range(gpu_start, gpu_start + tensor_parallel))
        port = base_port + i

        server = VLLMServer(
            model_path=model_path,
            port=port,
            tensor_parallel=tensor_parallel,
            gpu_ids=gpu_ids,
            model_name=model_name,
        )
        server.start(wait_for_ready=True)
        servers.append(server)

    return servers


def stop_all_servers(servers: list[VLLMServer]):
    """Stop all servers"""
    for server in servers:
        server.stop()
