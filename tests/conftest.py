from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Transaction archives land in a per-session tmp dir: the suite must
# never pollute the operator's logs/transactions with test records.
_TRANSACTION_TMP = Path(tempfile.mkdtemp(prefix="proxy-txn-tests-"))
import os as _os

_os.environ.setdefault("TRANSACTION_LOG_DIR", str(_TRANSACTION_TMP))
