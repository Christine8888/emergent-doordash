"""Global submitit configuration defaults."""

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class SubmitConfig:
    """Submitit configuration with sensible defaults.

    Can be overridden per-experiment or per-job.
    """

    # Cluster config
    partition: str = "sphinx"
    account: str = "nlp"
    job_name_prefix: str = "exp"

    # Resources
    # Note: gpus_per_job will be auto-set from tensor_parallel_size if not specified
    gpus_per_job: Optional[int] = None
    cpus_per_task: int = 16
    mem_gb: int = 64
    time_hours: int = 20

    # vLLM config
    max_model_len: int = 16384
    max_connections: int = 32
    gpu_memory_utilization: float = 0.9

    # Experiment config
    timeout: int = 600  # Timeout per eval task
    max_retries: int = 3  # Auto-resubmit failed jobs

    # Submitit config
    submitit_folder: str = "./submitit_logs"

    def override(self, **kwargs) -> "SubmitConfig":
        """Create new config with overrides.

        Args:
            **kwargs: Fields to override

        Returns:
            New SubmitConfig with overrides applied
        """
        return replace(self, **kwargs)

    def with_gpus(self, n_gpus: int) -> "SubmitConfig":
        """Set number of GPUs for this job.

        Args:
            n_gpus: Number of GPUs (typically equals tensor_parallel_size)

        Returns:
            New SubmitConfig with gpus_per_job set
        """
        return self.override(gpus_per_job=n_gpus)


# Global default instance
DEFAULT_CONFIG = SubmitConfig()
