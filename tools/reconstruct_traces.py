#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel
"""Reconstruct transaction intermediates from L1 boundary artifacts.

Quick-access offline tool for the leveled transaction logging (D15): given
a transaction directory containing the L1 boundaries (request.json and
metadata.json), it replays the deterministic payload pipeline —
parse -> neutral canonical -> provider build — and regenerates a candidate
L2-style trace report WITHOUT contacting any provider.

Live-state decisions (session scores, cooldowns, stream timing) are not
reconstructable; they remain in metadata.json. Field-cache injections and
adapter edits depend on runtime state and are marked as such.

Usage:
    python tools/reconstruct_traces.py logs/transactions/<transaction_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rotator_library.protocols import get_protocol  # noqa: E402
from rotator_library.utils import zstd_io  # noqa: E402


def _load(path: Path, name: str):
    try:
        return zstd_io.read_json_any(path / name)
    except FileNotFoundError:
        return None


def reconstruct(transaction_dir: Path) -> int:
    request_payload = _load(transaction_dir, "request.json")
    metadata = _load(transaction_dir, "metadata.json")
    if request_payload is None:
        print(f"error: {transaction_dir}/request.json(.zst) not found — L1 boundaries required", file=sys.stderr)
        return 2
    metadata = metadata or {}
    raw_request = request_payload.get("data", request_payload)

    client_protocol_name = (metadata.get("extra") or {}).get("input_protocol") or "openai_chat"
    provider_protocol_name = (metadata.get("extra") or {}).get("upstream_protocol") or client_protocol_name
    client_protocol = get_protocol(client_protocol_name)
    provider_protocol = get_protocol(provider_protocol_name)

    report: dict[str, object] = {
        "transaction": str(transaction_dir),
        "input_protocol": client_protocol_name,
        "upstream_protocol": provider_protocol_name,
        "deterministic_replay": True,
        "notes": [
            "payload transforms only; live-state decisions stay in metadata.json",
            "field-cache injections and adapter edits depend on runtime state and are not replayed",
        ],
        "stages": [],
    }
    stages = report["stages"]

    stages.append({"stage": "parse_client_request", "protocol": client_protocol_name})
    unified = client_protocol.parse_request(raw_request)
    stages.append({
        "stage": "neutral_canonical",
        "messages": len(unified.messages),
        "instructions": len(unified.system) if unified.system else 0,
        "tools": len(unified.tools) if unified.tools else 0,
        "model": unified.model,
    })

    stages.append({"stage": "build_provider_request", "protocol": provider_protocol_name})
    provider_payload = provider_protocol.build_request(unified)
    stages.append({
        "stage": "provider_request_built",
        "keys": sorted(provider_payload.keys()),
        "model": provider_payload.get("model"),
    })

    output = transaction_dir / "reconstructed_report.json"
    zstd_io.write_json(output, report)
    print(f"reconstructed pipeline report -> {output} ({'zstd' if zstd_io.compression_available() else 'plain'})")
    for stage in stages:
        print(f"  - {json.dumps(stage, ensure_ascii=False)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transaction_dir", type=Path, help="transaction directory with L1 boundaries")
    args = parser.parse_args()
    return reconstruct(args.transaction_dir)


if __name__ == "__main__":
    raise SystemExit(main())
