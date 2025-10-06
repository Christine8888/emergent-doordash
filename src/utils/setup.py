import sys
from pathlib import Path

import dotenv

# Auto-add project root to sys.path on import
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_env():
    dotenv.load_dotenv(_project_root / ".env")