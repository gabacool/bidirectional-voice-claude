"""Make the sibling `client/` modules importable as top-level names.

`voice_api` imports its peers as `from tts_client import ...` (no package
prefix), so the `client/` directory must be on sys.path when the tests run.
"""

import sys
from pathlib import Path

CLIENT_DIR = Path(__file__).resolve().parent.parent
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))
