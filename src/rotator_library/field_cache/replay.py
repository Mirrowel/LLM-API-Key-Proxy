# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Declarative cache-and-replay configuration (W13/D14).

Operators declare, per provider, what provider state to cache, how long to
keep it, and how to send it back — as JSON (env ``<NAME>_CACHE_REPLAY`` or
the JSON provider config) or as a class attribute on code providers. The
declaration compiles to ordinary FieldCacheRules: one surface, one engine.

Declaration schema (list of entries):

``name``           unique rule name
``source``         request | response | stream_event | unified_*
``path``           extraction path
``keep``           last | all | turn | turns:N | per_tool_call  (mode mapping)
``inject.path``    restore path
``inject.if``      auto (default — add only when absent) | always (overwrite;
                   an operator choice, honored for every field class)
``inject.target``  request | unified_request | metadata (default request)
``compatibility``  bound | portable (default bound)
``transform``      registered transform name (portable only)
``scope``          scope dimensions (default provider+model, credential and
                   session optional refinements per D11)
``ttl_seconds``    retention window

The operator is the trust boundary: explicit per-field choices (including
``always`` overwrites of bound fields) are honored without second-guessing;
automatic runtime behavior never overwrites bound state (D8).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from .types import (
    DEFAULT_SCOPE,
    FieldCacheInjection,
    FieldCacheMode,
    FieldCacheRule,
    FieldCacheScope,
    FieldCacheSource,
)

_MODE_MAP: dict[str, FieldCacheMode] = {
    "last": "last",
    "all": "all",
    "turn": "last_user_turn",
    "per_tool_call": "per_tool_call",
}


def _parse_keep(keep: Any) -> tuple[FieldCacheMode, Optional[int]]:
    """Map the keep vocabulary to engine modes.

    ``turns:N`` compiles to ``all`` with bounded history (the last N kept
    values restore together); ``turn`` is the last user-turn value.
    """

    if keep is None:
        return "last", None
    text = str(keep).strip().lower()
    if text.startswith("turns:"):
        try:
            count = int(text.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"cache_replay keep 'turns:N' needs an integer N, got {keep!r}") from exc
        if count <= 0:
            raise ValueError(f"cache_replay keep 'turns:N' needs a positive N, got {keep!r}")
        return "all", count
    if text not in _MODE_MAP:
        raise ValueError(
            f"cache_replay keep must be one of last | all | turn | turns:N | per_tool_call, got {keep!r}"
        )
    return _MODE_MAP[text], None


def compile_cache_replay(entries: Iterable[Any], *, provider: str) -> tuple[FieldCacheRule, ...]:
    """Compile declarative cache_replay entries into FieldCacheRules."""

    rules: list[FieldCacheRule] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"cache_replay entry {index} for {provider} must be an object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"cache_replay entry {index} for {provider} needs a name")
        source = entry.get("source") or "response"
        path = entry.get("path")
        if not path:
            raise ValueError(f"cache_replay rule {name!r} needs a path")
        mode, turns_cap = _parse_keep(entry.get("keep"))
        inject_entry = entry.get("inject")
        injection: Optional[FieldCacheInjection] = None
        if isinstance(inject_entry, dict) and inject_entry.get("path"):
            inject_if = str(inject_entry.get("if", "auto")).strip().lower()
            if inject_if not in {"auto", "always"}:
                raise ValueError(
                    f"cache_replay rule {name!r} inject.if must be auto or always, got {inject_entry.get('if')!r}"
                )
            injection = FieldCacheInjection(
                target=str(inject_entry.get("target", "request")),
                path=str(inject_entry["path"]),
                when_missing_only=inject_if == "auto",
                insert=bool(inject_entry.get("insert", False)),
                as_list=bool(inject_entry.get("as_list", False)),
            )
        compatibility = entry.get("compatibility")
        if compatibility is not None and str(compatibility) not in {"bound", "portable"}:
            raise ValueError(f"cache_replay rule {name!r} compatibility must be bound or portable")
        transform = entry.get("transform")
        if transform and str(compatibility or "") != "portable":
            raise ValueError(f"cache_replay rule {name!r} transform requires compatibility=portable")
        if transform:
            # Fail at compile time (startup/config), never mid-request.
            from ..protocols.transforms import get_transform

            get_transform(str(transform))
        scope_entry = entry.get("scope")
        scope: tuple[FieldCacheScope, ...]
        if isinstance(scope_entry, list) and scope_entry:
            scope = tuple(str(dimension) for dimension in scope_entry)  # type: ignore[assignment]
        else:
            scope = DEFAULT_SCOPE
        metadata: dict[str, Any] = {"cache_replay": True, "provider": provider}
        if compatibility:
            metadata["compatibility"] = str(compatibility)
        if transform:
            metadata["transform"] = str(transform)
        if entry.get("tool_call_id_path"):
            metadata["tool_call_id_path"] = str(entry["tool_call_id_path"])
        rules.append(
            FieldCacheRule(
                name=name,
                source=source,  # type: ignore[arg-type]
                path=str(path),
                mode=mode,
                scope=scope,
                inject=injection,
                enabled=bool(entry.get("enabled", True)),
                ttl_seconds=int(entry["ttl_seconds"]) if entry.get("ttl_seconds") is not None else None,
                metadata=metadata,
                allow_missing_session=bool(entry.get("allow_missing_session", True)),
                # FieldCacheRule's uniform value bound applies when no
                # turns cap is declared, keeping replay rules identical in
                # shape to raw/JSON-configured rules (the weakening guard
                # compares this field).
                max_values=turns_cap if turns_cap is not None else 1024,
            )
        )
    return tuple(rules)


def parse_cache_replay_config(raw: Any, *, provider: str) -> tuple[FieldCacheRule, ...]:
    """Parse JSON text (env var) or an already-decoded list."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        entries = json.loads(text)
    else:
        entries = raw
    if not isinstance(entries, list):
        raise ValueError(f"cache_replay config for {provider} must be a JSON list of rule entries")
    return compile_cache_replay(entries, provider=provider)
