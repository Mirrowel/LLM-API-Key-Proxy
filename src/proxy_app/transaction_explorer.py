# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Transaction explorer — the basic TUI over sealed transaction archives.

Launched from the proxy launcher's main menu. Lists every sealed
transaction archive, opens any one of them, and reconstructs the
deterministic intermediate pipeline (parse → neutral → provider build)
on demand from the archive's recipe + client request + change log.

Scoped deliberately basic per the operator ruling: list, inspect,
reconstruct. The full explorer (web UI, search) comes after the TUI
rework, after this PR merges.

Module boundary: stdlib-only imports at module level (launcher fast
path); rotator_library loads lazily inside the functions that need it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAGE_SIZE = 15


def _load_archives() -> list[tuple[Path, dict]]:
    """Read every archive envelope (newest first)."""

    from rotator_library.transaction.archive import iter_archives, read_archive

    entries: list[tuple[Path, dict]] = []
    for path in iter_archives():
        try:
            entries.append((path, read_archive(path)))
        except Exception as exc:
            entries.append((path, {"format": "unreadable", "error": str(exc), "recipe": {}}))
    entries.reverse()  # newest first
    return entries


def _row(index: int, path: Path, envelope: dict) -> str:
    recipe = envelope.get("recipe") or {}
    stamp = path.name.split("_", 1)[0]
    status = envelope.get("status_code", "?")
    flags = []
    if envelope.get("errors"):
        flags.append("errors")
    if envelope.get("escalations"):
        flags.append("escalated")
    if envelope.get("truncation"):
        flags.append("truncated")
    flag_text = f" [{','.join(flags)}]" if flags else ""
    return (
        f"{index:>4}. {stamp} {str(recipe.get('protocol', '?')):<18} "
        f"{str(recipe.get('provider', '?')):<14} {str(recipe.get('model', '?'))[:28]:<28} "
        f"{status}{flag_text}"
    )


def _pretty(value, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + f"\n... ({len(text) - limit} more characters)"
    return text


def _section(title: str, value) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))
    if value in (None, {}, []):
        print("(absent)")
    else:
        print(_pretty(value))


def _show_detail(envelope: dict) -> None:
    """Full record view: recipe, boundaries, chunks, change log, errors."""

    print("=" * 70)
    print("TRANSACTION RECORD")
    print("=" * 70)
    _section("recipe", envelope.get("recipe"))
    boundaries = envelope.get("boundaries") or {}
    for name in ("client_request", "provider_request", "provider_response", "client_egress"):
        if name in boundaries:
            _section(f"boundary: {name}", boundaries[name])
    stream_chunks = envelope.get("stream_chunks") or []
    client_chunks = envelope.get("client_chunks") or []
    print(f"\n--- stream evidence: {len(stream_chunks)} provider frames, {len(client_chunks)} client frames")
    if stream_chunks:
        _section("provider frames (first 5)", stream_chunks[:5])
    change_log = envelope.get("change_log") or []
    print(f"\n--- change log: {len(change_log)} events")
    for event in change_log:
        print(
            f"  #{event.get('seq', '?'):>4} [{event.get('stage', '?')}] {event.get('kind', '?')}"
            f" {('{' + str(event.get('code', '')) + '}') if event.get('code') else ''}"
            f" {str(event.get('detail', ''))[:100]}"
        )
    _section("metadata", envelope.get("metadata"))
    _section("attempts", envelope.get("attempts"))
    _section("errors", envelope.get("errors"))
    if envelope.get("escalations"):
        print(f"\n--- escalations: {', '.join(envelope['escalations'])}")
    if envelope.get("truncation"):
        print(f"--- truncation: {json.dumps(envelope['truncation'], indent=2)}")


def reconstruct_transaction(envelope: dict) -> dict:
    """Rebuild the deterministic intermediates (T2) from the archive.

    parse(client request, recorded protocol) → neutral canonical →
    build(recorded provider protocol). Overlays recorded in the change
    log (hook edits, cache injections) are listed as replay notes; the
    truly non-derivable facts (provider response, repairs, minted ids)
    are surfaced from the archive itself. Returns a dict of artifacts —
    the caller renders it.
    """

    from rotator_library.protocols import get_protocol
    from rotator_library.protocols.types import ProtocolContext

    recipe = envelope.get("recipe") or {}
    boundaries = envelope.get("boundaries") or {}
    client_request = boundaries.get("client_request")
    input_protocol = str(recipe.get("protocol") or "")
    provider = str(recipe.get("provider") or "")
    artifacts: dict = {
        "recipe": recipe,
        "replay_notes": [],
        "unreconstructable": [],
    }
    change_log = envelope.get("change_log") or []
    for event in change_log:
        kind = str(event.get("kind", ""))
        if kind in {"hook_edit", "adapter_edit", "cache_injection", "overlay", "repair", "minted_id"}:
            artifacts["replay_notes"].append(
                {
                    "seq": event.get("seq"),
                    "kind": kind,
                    "detail": event.get("detail"),
                    "recorded_value": event.get("value"),
                }
            )
    if envelope.get("escalations"):
        artifacts["unreconstructable"].extend(envelope["escalations"])
    artifacts["unreconstructable"].extend(
        [
            "provider response & stream frames (external input — stored verbatim in the archive)",
            "wall-clock timing (metadata only)",
        ]
    )
    if not isinstance(client_request, dict) or not input_protocol:
        artifacts["error"] = "archive carries no client_request boundary or recipe protocol — nothing to replay"
        return artifacts
    try:
        protocol = get_protocol(input_protocol)
        context = ProtocolContext(
            source_protocol=input_protocol,
            target_protocol=input_protocol,
            input_protocol=input_protocol,
            client_protocol=input_protocol,
        )
        unified = protocol.parse_request(dict(client_request), context)
        from rotator_library.protocols.types import serialize_value

        serialized = serialize_value(unified)
        artifacts["neutral_parse"] = serialized if isinstance(serialized, dict) else json.loads(serialized)
        target = f"{provider}/{recipe.get('model', '')}" if provider else str(recipe.get("model", ""))
        artifacts["provider_target"] = target
    except Exception as exc:
        artifacts["error"] = f"deterministic replay failed: {type(exc).__name__}: {exc}"
    return artifacts


def _show_reconstruction(envelope: dict) -> None:
    print("=" * 70)
    print("RECONSTRUCTION (T2: deterministic intermediates, rebuilt offline)")
    print("=" * 70)
    artifacts = reconstruct_transaction(envelope)
    if artifacts.get("error"):
        print(f"\n! {artifacts['error']}")
    _section("neutral canonical parse", artifacts.get("neutral_parse"))
    _section("provider target", artifacts.get("provider_target"))
    notes = artifacts.get("replay_notes") or []
    print(f"\n--- replay notes (recorded runtime deltas applied on top): {len(notes)}")
    for note in notes:
        print(
            f"  #{note.get('seq', '?'):>4} {note.get('kind')}: {str(note.get('detail'))[:120]}"
        )
    print("\n--- not reconstructable (by design):")
    for item in artifacts.get("unreconstructable") or []:
        print(f"  - {item}")


def run_explorer() -> None:
    """The interactive loop: list → pick → inspect/reconstruct."""

    try:
        archives = _load_archives()
    except Exception as exc:
        print(f"Could not read the transactions directory: {exc}")
        input("Press Enter to return...")
        return
    if not archives:
        print("No transaction archives yet (start the proxy with request logging enabled).")
        input("Press Enter to return...")
        return
    page = 0
    while True:
        total = len(archives)
        max_page = (total - 1) // PAGE_SIZE
        page = max(0, min(page, max_page))
        print("=" * 70)
        print(f"TRANSACTION EXPLORER — {total} records (page {page + 1}/{max_page + 1})")
        print("=" * 70)
        start = page * PAGE_SIZE
        for offset in range(PAGE_SIZE):
            index = start + offset
            if index >= total:
                break
            path, envelope = archives[index]
            print(_row(index + 1, path, envelope))
        print()
        print("<number> open record | n/p page | q quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", ""):
            return
        if choice in ("n", "next"):
            page += 1
            continue
        if choice in ("p", "prev"):
            page -= 1
            continue
        if not choice.isdigit():
            continue
        index = int(choice) - 1
        if not 0 <= index < total:
            print("No such record.")
            continue
        path, envelope = archives[index]
        _show_detail(envelope)
        print("\n[Enter] back | r reconstruct | q quit")
        sub = input("> ").strip().lower()
        if sub == "q":
            return
        if sub == "r":
            _show_reconstruction(envelope)
            input("\nPress Enter to return...")


def main(argv: list[str] | None = None) -> int:
    """Standalone entry (python -m proxy_app.transaction_explorer)."""

    try:
        run_explorer()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
