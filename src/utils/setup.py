import sys
from pathlib import Path

import dotenv

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_env():
    dotenv.load_dotenv(_project_root / ".env")