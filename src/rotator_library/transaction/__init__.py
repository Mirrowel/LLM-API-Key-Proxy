"""G10 transaction record package: accumulator, archive format, writer.

Lazy exports keep the startup path fast (root ``rotator_library`` import
must stay ~0.5s — heavy imports belong at call sites).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .archive import (
        archive_filename,
        compress_envelope,
        decompress_archive,
        read_archive,
        transactions_dir,
    )
    from .record import ChangeEvent, TransactionRecord
    from .writer import TransactionWriter


def __getattr__(name: str):  # noqa: D103
    if name in {
        "TransactionRecord",
        "ChangeEvent",
    }:
        from . import record

        return getattr(record, name)
    if name in {"TransactionWriter"}:
        from . import writer

        return getattr(writer, name)
    if name in {
        "archive_filename",
        "compress_envelope",
        "decompress_archive",
        "read_archive",
        "transactions_dir",
        "prune_archives",
        "iter_archives",
        "iter_archives",
    }:
        from . import archive

        return getattr(archive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
