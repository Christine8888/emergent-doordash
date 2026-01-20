import sys
import logging
from pathlib import Path

import dotenv

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_env():
    dotenv.load_dotenv(_project_root / ".env")


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