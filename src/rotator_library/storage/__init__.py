# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Storage engine package (G17): SQLite-backed KV stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .engine import StorageEngine, get_engine, close_all_engines

__all__ = ["StorageEngine", "get_engine", "close_all_engines"]


def __getattr__(name: str):  # lazy: importing the package stays cheap
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
