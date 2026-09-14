# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Reasoning-effort emission wiring for the native executor (G8).

The ladder math lives in ``protocols.effort``; the protocol formatters
live in ``protocols.canonical``. This module is the ONE seam that knows
provider + model + wire together:

- the canonical reasoning control's effort word is normalized ONCE,
  before any per-protocol build, against the declared accepted
  vocabulary for (provider, model) — the word the builders receive is
  already provider-legal;
- the raw chat fast path normalizes its flat ``reasoning_effort`` key the
  same way (it carries provider + model context);
- a provider declaring the OFF control on a thinking toggle emits that
  toggle on the chat wire: OFF becomes ``{"thinking": {"type":
  "disabled"}}`` and the effort word drops; an ON word rides next to
  ``{"thinking": {"type": "enabled"}}``.

Every transformation is recorded on the request's conversion-warning
channel (``reasoning_effort_normalized``) — nothing silent. Providers
without a declaration keep their protocol's existing emission; only the
openai_chat wire has the toggle construct (Responses keeps its native
``reasoning.effort``, Anthropic its protocol toggle).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from ..protocols.canonical import add_conversion_warning, family_wire_name
from ..protocols.effort import (
    OFF_WORDS,
    normalize_effort,
    resolve_accepted_effort,
    resolve_effort_toggle,
)

_NOTE_CODE = "reasoning_effort_normalized"
_NOTE_FIELD = "reasoning.effort"


def _declared_accepted(
    provider_plugin: Any,
    model: str,
    runtime_config: Optional[Mapping[str, Any]],
    protocol_name: str,
) -> Optional[Tuple[str, ...]]:
    """The declared accepted vocabulary, or None when nothing declares one.

    The protocol base is the chain's floor, not a declaration: a provider
    that never opted into the effort system keeps its protocol's existing
    vocabulary (and warnings) untouched.
    """

    accepted, source = resolve_accepted_effort(
        provider_plugin,
        model,
        runtime_config=dict(runtime_config) if isinstance(runtime_config, Mapping) else runtime_config,
        protocol_family=family_wire_name(protocol_name),
    )
    if source == "protocol_base" or not accepted:
        return None
    return accepted


def _record_note(unified_request: Any, note: str, protocol_name: str) -> None:
    if unified_request is None or not note:
        return
    add_conversion_warning(
        unified_request,
        code=_NOTE_CODE,
        message=note,
        field=_NOTE_FIELD,
        target_protocol=protocol_name,
    )


def normalize_request_effort(
    unified_request: Any,
    *,
    provider_plugin: Any = None,
    model: str = "",
    protocol_name: str = "",
    runtime_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Normalize the canonical reasoning control's effort word in place.

    The word is folded ONCE against the provider-declared accepted set
    before any per-protocol builder runs; the transformation (or the
    drop) is recorded on the request's conversion-warning channel.
    """

    params = getattr(unified_request, "generation_params", None)
    if not isinstance(params, dict):
        return
    reasoning = params.get("reasoning")
    if not isinstance(reasoning, dict):
        return
    word = reasoning.get("effort")
    if word is None:
        return
    accepted = _declared_accepted(provider_plugin, model, runtime_config, protocol_name)
    if accepted is None:
        return
    normalized, note = normalize_effort(str(word), accepted)
    _record_note(unified_request, note, protocol_name)
    if normalized is None:
        reasoning.pop("effort", None)
        if not reasoning:
            params.pop("reasoning", None)
    else:
        reasoning["effort"] = normalized


def normalize_wire_effort(
    payload: Any,
    *,
    unified_request: Any = None,
    provider_plugin: Any = None,
    model: str = "",
    protocol_name: str = "",
    runtime_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Normalize a flat wire ``reasoning_effort`` against the declaration.

    Applied to every provider-built payload (idempotent over an already
    normalized word). The canonical OFF word never overwrites the wire's
    own off spelling — the protocol formatter (or the raw client payload)
    owns that — so only ON words are folded here; unknown words drop with
    a recorded note.
    """

    if not isinstance(payload, dict) or "reasoning_effort" not in payload:
        return
    word = payload.get("reasoning_effort")
    if word is None:
        return
    accepted = _declared_accepted(provider_plugin, model, runtime_config, protocol_name)
    if accepted is None:
        return
    normalized, note = normalize_effort(str(word), accepted)
    _record_note(unified_request, note, protocol_name)
    if normalized is None:
        payload.pop("reasoning_effort", None)
    elif normalized == "off":
        # Keep the payload's own off spelling ("none" on the Chat wire);
        # the canonical ladder word is not a wire value.
        return
    else:
        payload["reasoning_effort"] = normalized


def apply_effort_toggle(
    payload: Any,
    *,
    provider_plugin: Any = None,
    model: str = "",
    protocol_name: str = "",
    runtime_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Emit the declared thinking toggle on the chat wire.

    OFF drops the effort word and emits ``{"thinking": {"type":
    "disabled"}}``; an ON word stays and rides next to ``{"thinking":
    {"type": "enabled"}}``. Only the openai_chat wire is touched, and only
    when the resolved declaration says the OFF control rides the toggle.
    Returns whether the payload was modified.
    """

    if family_wire_name(protocol_name) != "openai_chat":
        return False
    if not isinstance(payload, dict) or "reasoning_effort" not in payload:
        return False
    value = payload.get("reasoning_effort")
    if value is None:
        # A null control is an absence, not an OFF request.
        return False
    if not resolve_effort_toggle(
        provider_plugin,
        model,
        runtime_config=dict(runtime_config) if isinstance(runtime_config, Mapping) else runtime_config,
    ):
        return False
    word = str(value).strip().lower()
    if word in OFF_WORDS:
        payload.pop("reasoning_effort", None)
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "enabled"}
    return True
