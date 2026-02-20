"""Conftest for integration/api tests — ensure src/ is on sys.path."""

import os
import sys

# Add src/ to path for api.* imports
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
