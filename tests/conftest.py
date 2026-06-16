# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Pytest configuration and shared fixtures for the rotator_library test suite.

This conftest adds the source directories to sys.path so that both
standalone modules (request_sanitizer, session_tracking, etc.) and
package-qualified modules (rotator_library.core.utils) can be imported
without requiring a full editable install of every dependency.
"""

import sys
from pathlib import Path

# Resolve the repository root (tests/ -> parent)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
_LIB_ROOT = _SRC_ROOT / "rotator_library"

# Prepend source paths so that:
#   - `import rotator_library.X` works  (via _SRC_ROOT)
#   - `from request_sanitizer import ...` works  (via _LIB_ROOT)
for _path in (_SRC_ROOT, _LIB_ROOT):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)
