"""Global submitit configuration defaults."""

import getpass
from dataclasses import dataclass, field, replace

# User-specific setup scripts
_SETUP_SCRIPTS = {
    "cye": "/sphinx/u/cye/emergent-doordash/scripts/setup_env.sh",
    "suzeva": "/afs/cs.stanford.edu/u/suzeva/emergent-doordash/scripts/setup_env_suze.sh",
}

def _get_setup_commands() -> list:
    """Get setup commands for current user."""
    user = getpass.getuser()
    if user not in _SETUP_SCRIPTS:
        raise RuntimeError(
            f"Unknown user '{user}' for submitit setup script. "
            f"Add your script path to _SETUP_SCRIPTS in {__file__}."
        )
    script = _SETUP_SCRIPTS[user]
    return [f"source {script}"]


@dataclass
class SubmitConfig:
    """Submitit configuration with sensible defaults."""

    # Cluster config
    partition: str = "sphinx,miso,jag-standard"
    qos: str | None = None
    account: str = "nlp"
    job_name_prefix: str = "exp"
    exclude_nodes: str = ""
    nodelist: str | None = None

    # GPU/resource config
    gpus_per_job: int | None = None
    cpus_per_task: int = 4
    mem_gb: int = 64
    time_hours: int = 48

    # vLLM config
    max_model_len: int = 32768
    max_connections: int = 64
    gpu_memory_utilization: float = 0.85

    # Experiment config
    timeout: int = 1200
    max_retries: int = 3

    # Submitit config
    submitit_folder: str = "./submitit_logs"
    setup_commands: list = field(default_factory=_get_setup_commands)

    def override(self, **kwargs) -> "SubmitConfig":
        """Create new config with overrides."""
        return replace(self, **kwargs)

    def with_gpus(self, n_gpus: int) -> "SubmitConfig":
        """Set number of GPUs for this job."""
        return self.override(gpus_per_job=n_gpus)


DEFAULT_CONFIG = SubmitConfig()
