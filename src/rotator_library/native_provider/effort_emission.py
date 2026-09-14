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


def _set_dotted(payload: dict, path: str, value: Any) -> None:
    """Write ``value`` at a dotted path, creating intermediate objects.

    ``chat_template_kwargs.thinking`` materializes the ctk dict before
    writing the leaf. A non-dict intermediate is never silently coerced:
    the write is skipped with a warning (a provider's payload shape wins
    over a declaration mistake).
    """

    parts = [part for part in str(path).split(".") if part]
    if not parts:
        return
    cursor: Any = payload
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            if nxt is not None:
                logger.warning(
                    "reasoning emission: path %r collides with a non-object at %r; write skipped",
                    path,
                    part,
                )
                return
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


# The legacy ``toggle: True`` declaration means the DeepSeek-style object
# toggle; expressed as the generalized preset it now is.
_OBJECT_TOGGLE_PRESET = {
    "toggle_field": "thinking",
    "toggle_on": {"type": "enabled"},
    "toggle_off": {"type": "disabled"},
}


def resolve_reasoning_targets(
    provider_plugin: Any = None,
    model: str = "",
    *,
    runtime_config: Any = None,
) -> Dict[str, Any]:
    """Resolve the emission targets for (provider, model).

    Keys: ``effort_field`` (where the folded effort word lands; default
    top-level ``reasoning_effort``), ``toggle_field`` / ``toggle_on`` /
    ``toggle_off`` (the thinking-toggle target and its on/off values —
    booleans for the vLLM families, the DeepSeek object pair via the
    legacy preset). Declared in ``model_rules`` rows (model-level beats
    provider-wide attrs ``reasoning_toggle_field`` etc., which beat the
    legacy ``reasoning_effort_toggle``/``toggle: True`` preset).
    """

    targets: Dict[str, Any] = {}
    if provider_plugin is not None:
        row_declared_toggle = False
        for key, attr in (
            ("effort_field", "reasoning_effort_field"),
            ("toggle_field", "reasoning_toggle_field"),
            ("toggle_on", "reasoning_toggle_on"),
            ("toggle_off", "reasoning_toggle_off"),
        ):
            value = getattr(provider_plugin, attr, None)
            if value is not None:
                targets[key] = value
        if resolve_effort_toggle(provider_plugin, model, runtime_config=runtime_config):
            # The legacy object preset (DeepSeek): the effort word KEEPS
            # riding top-level next to the toggle object.
            targets.setdefault("toggle_field", _OBJECT_TOGGLE_PRESET["toggle_field"])
            targets.setdefault("toggle_on", _OBJECT_TOGGLE_PRESET["toggle_on"])
            targets.setdefault("toggle_off", _OBJECT_TOGGLE_PRESET["toggle_off"])
            targets["effort_rides"] = True
        from ..adapters.param_rules import _model_match_candidates, _row_matches
        from ..protocols.effort import _model_rule_rows

        for row in _model_rule_rows(provider_plugin, model):
            if not _row_matches(row, _model_match_candidates(model)):
                continue
            if "toggle_field" in row and "effort_field" not in row:
                row_declared_toggle = True
            for key in ("effort_field", "toggle_field", "toggle_on", "toggle_off"):
                if key in row:
                    targets[key] = row[key]
            if row.get("toggle") is True:
                targets.setdefault("toggle_field", _OBJECT_TOGGLE_PRESET["toggle_field"])
                targets.setdefault("toggle_on", _OBJECT_TOGGLE_PRESET["toggle_on"])
                targets.setdefault("toggle_off", _OBJECT_TOGGLE_PRESET["toggle_off"])
        if row_declared_toggle:
            targets["toggle_only"] = True
    if isinstance(runtime_config, Mapping):
        for key in ("effort_field", "toggle_field", "toggle_on", "toggle_off"):
            if key in runtime_config:
                targets[key] = runtime_config[key]
    return targets


def apply_reasoning_emission(
    payload: Any,
    *,
    provider_plugin: Any = None,
    model: str = "",
    protocol_name: str = "",
    runtime_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Emit the declared reasoning controls on the chat wire.

    The effort word (already ladder-normalized by
    :func:`normalize_wire_effort`) lands at the declared
    ``effort_field`` — top-level by default, nested for the
    chat-template families — and a declared toggle target receives its
    on value alongside. OFF words drop the effort write and set the
    toggle's off value (an object, a boolean, whatever the family's
    wire shape is). Only the openai_chat wire is touched; providers
    without declarations keep the plain top-level emission. Returns
    whether the payload was modified.
    """

    if family_wire_name(protocol_name) != "openai_chat":
        return False
    if not isinstance(payload, dict):
        return False
    value = payload.get("reasoning_effort")
    if value is None:
        # A null control is an absence, not an OFF request.
        return False
    targets = resolve_reasoning_targets(
        provider_plugin, model, runtime_config=runtime_config
    )
    if not targets:
        return False
    toggle_field = targets.get("toggle_field")
    word = str(value).strip().lower()
    effort_field = targets.get("effort_field") or "reasoning_effort"
    if toggle_field and targets.get("toggle_only"):
        # Boolean-toggle family without a declared effort target: the
        # family's wire takes the toggle ONLY (kimi-k2's thinking bool,
        # enable_thinking) — the top-level word never rides.
        payload.pop("reasoning_effort", None)
        if word in OFF_WORDS:
            _set_dotted(payload, toggle_field, targets.get("toggle_off", False))
        else:
            _set_dotted(payload, toggle_field, targets.get("toggle_on", True))
        return True
    if word in OFF_WORDS:
        payload.pop("reasoning_effort", None)
        if toggle_field:
            _set_dotted(payload, toggle_field, targets.get("toggle_off", False))
            return True
        if effort_field == "reasoning_effort":
            # No toggle target: the wire keeps its own off spelling
            # (e.g. ``reasoning_effort: "none"`` on surfaces that
            # accept it).
            payload["reasoning_effort"] = value
        return True
    if effort_field == "reasoning_effort":
        payload["reasoning_effort"] = value
    else:
        payload.pop("reasoning_effort", None)
        _set_dotted(payload, effort_field, value)
    if toggle_field:
        _set_dotted(payload, toggle_field, targets.get("toggle_on", True))
    return True
