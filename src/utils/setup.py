import os
import sys
import logging
from pathlib import Path

import dotenv

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_env():
    dotenv.load_dotenv(_project_root / ".env")


def setup_inspect_logging(level: str = "warning", log_file: str | None = None):
    """Configure Inspect AI logging level.

    Args:
        level: Log level - "debug", "trace", "http", "info", "warning" (default)
               Use "http" to see all HTTP requests/retries (useful for debugging hangs)
        log_file: Optional path to write Python logs to file
    """
    os.environ["INSPECT_LOG_LEVEL"] = level
    os.environ["INSPECT_LOG_LEVEL_TRANSCRIPT"] = level

    if log_file:
        os.environ["INSPECT_PY_LOGGER_FILE"] = log_file
        os.environ["INSPECT_PY_LOGGER_LEVEL"] = level


def setup_openai_retry_debug_logging(enabled: bool = True):
    """Enable more verbose logging for openai-python retries.

    This helps explain lines like:
      "Retrying request to /chat/completions in X seconds"

    Notes:
    - This does not change retry behaviour; it only increases visibility.
    - We intentionally avoid enabling very noisy `httpx`/`httpcore` debug logs.
    """
    if not enabled:
        return

    # openai-python uses module logger names like "openai._base_client"
    for name in ("openai._base_client", "openai._client", "openai._exceptions", "openai"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    # Some versions also respect this env var for richer logs.
    os.environ.setdefault("OPENAI_LOG", "debug")


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(level=logging.INFO):
    """Configure consistent logging across all scripts with immediate flushing."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = FlushingStreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root.addHandler(handler)
    return logging.getLogger(__name__)