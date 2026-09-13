from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Transaction archives land in a per-session tmp dir: the suite must
# never pollute the operator's logs/transactions with test records.
_TRANSACTION_TMP = Path(tempfile.mkdtemp(prefix="proxy-txn-tests-"))
import os as _os

_os.environ.setdefault("TRANSACTION_LOG_DIR", str(_TRANSACTION_TMP))


@pytest.fixture(autouse=True)
def _isolate_storage_engines(tmp_path, monkeypatch):
    """Give every test a fresh set of storage engines rooted in tmp_path.

    One engine per data family is process-global by design; without this
    reset, rows written by one test would leak into the next.
    """

    from rotator_library.storage import engine as engine_mod
    from rotator_library.utils import paths as paths_mod

    monkeypatch.setattr(paths_mod, "get_default_root", lambda: tmp_path)
    engine_mod.close_all_engines()
    try:
        yield
    finally:
        engine_mod.close_all_engines()
