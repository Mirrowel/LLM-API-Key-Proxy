# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Declared request-parameter rules (G8): one generic, reusable adapter.

Providers and models declare what the wire accepts — strip lists, clamps,
value maps — and this adapter enforces the declarations on the raw
provider-bound payload. No provider hardcodes parameter hygiene in its
own code anymore; the declarations carry it.

Rule sources (later wins): provider class ``param_rules`` declaration →
JSON runtime config ``param_rules`` → per-model overrides via the ordered
``model_rules`` capability table, or the legacy ``model_param_rules``
mapping (capability data on the provider, config in JSON). Every edit is
recorded through the adapter-chain trace like any wire change.

Declaration shape (dict)::

    {
      "strip": ["logit_bias", "logprobs"],
      "clamp": {"temperature": [0.0, 1.0]},
      "map": {"reasoning_effort": {"medium": "high", "xhigh": "high"}},
      "rename": {"max_completion_tokens": "max_tokens"}
    }

    ``map`` values that are absent from the table pass through unchanged
    (declaring an exhaustive table is the provider's choice, not ours).

``model_param_rules`` entries may carry ``strip_override`` (a list):
when present on a model it REPLACES the provider strip list for that
model — the escape hatch for "provider strips X globally, this model
allows it". It is terminal: no deep-merge with any inherited strip
list (scoped or provider-level), and it resolves to the ordinary
``strip`` table so already-resolved configurations never carry it.
``model_param_rules`` is SUPERSEDED by ``model_rules`` (kept working as
a bridge); when both declare the same knob, the capability table wins.

The capability table (``model_rules``) is an ORDERED tuple of rows, a
top-to-bottom rule cascade::

    model_rules = (
        {"match": "*", "strip": ["logit_bias"], "effort_accept": ["off", "low", "high"]},
        {"match": "reasoner-*", "strip": ["logprobs"], "effort_accept": ["off", "high"], "toggle": True},
        {"match": "gpt-5-mini", "allow": ["openai_chat"], "deny": ["responses"]},
    )

    ``match``      fnmatch wildcard on the model id, case-insensitive;
                   ``*`` is the provider-default row.
    row content    the param_rules vocabulary inline (strip, clamp, map,
                   rename, strip_override) plus ``effort_accept`` (the
                   accepted reasoning-effort vocabulary the ladder folds
                   into), ``toggle`` (the OFF control rides the chat
                   wire's thinking toggle) and ``allow``/``deny``
                   (per-model face limiting, enforced by the provider's
                   protocol resolution — not a param rule).
    resolution     rows matching the model apply in order; LATER rows
                   override conflicting keys, non-conflicting keys inherit
                   (CSS cascade). JSON runtime ``model_rules`` rows append
                   after the class rows, so config overrides code.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from fnmatch import fnmatchcase
from typing import Any, Dict, Mapping, Optional

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")

# Row keys that never compile into param-rule tables (capability
# declarations consumed by the effort system and the face limiter).
_ROW_STRUCTURE_KEYS = frozenset(
    {
        "match",
        "allow",
        "deny",
        "effort_accept",
        "toggle",
        # Capability vocabulary (the gemini split, G8): per-model
        # knowledge the protocols consume through the resolved
        # capability record instead of hardcoding.
        "thinking_dialect",  # "level" | "budget" — which thinking knob the model speaks
        "thinking_budget_range",  # [min, max] — documented budget bounds
        "tool_call_ids",  # functionCall.id emitted + echoed (3.x family)
        "requires_thought_signatures",  # missing signature on a call part is a 400
        "output_modalities",  # e.g. ["text"] / ["image", "text"] / ["audio"]
        "hosted_tools",  # available hosted tool names (googleSearch, ...)
        "max_candidates",  # candidateCount ceiling (per-model 400 otherwise)
    }
)
_ROW_TABLE_KEYS = frozenset({"strip", "clamp", "map", "rename", "strip_override"})


class ModelRulesFaceError(ValueError):
    """A model_rules row refuses the executing protocol face."""


def _model_match_candidates(model: str, provider: str = "") -> list[str]:
    """Case-folded match candidates for a model id.

    A provider-prefixed id (``deepseek/deepseek-chat``) also matches as its
    stripped form; nested ids (``openrouter/meta/llama``) keep their slashes
    and simply also try the first segment stripped — the wildcard decides.
    """

    text = str(model or "")
    if not text:
        return []
    candidates = [text.lower()]
    stripped = _canonical_stripped_model(text, provider)
    if stripped and stripped.lower() not in candidates:
        candidates.append(stripped.lower())
    return candidates


def _canonical_stripped_model(model: str, provider: str = "") -> str:
    if "/" not in model:
        return ""
    prefix = f"{provider}/" if provider else ""
    if prefix and model.startswith(prefix) and len(model) > len(prefix):
        return model[len(prefix):]
    return model.split("/", 1)[1]


def _row_matches(row: Mapping[str, Any], candidates: list[str]) -> bool:
    pattern = str(row.get("match", "")).strip().lower()
    if not pattern:
        raise ValueError('model_rules rows require a non-empty "match" wildcard')
    return any(fnmatchcase(candidate, pattern) for candidate in candidates)


def resolve_model_rules(rows: Any, model: str, provider: str = "") -> Dict[str, Any]:
    """Resolve the capability table for one model (CSS cascade).

    Rows matching the model id apply top-to-bottom: later rows override
    conflicting keys, non-conflicting keys inherit. Capability keys
    (``effort_accept``, ``toggle``, ``allow``/``deny``) resolve in the
    same cascade; param-rule tables consume what remains.
    """

    if not rows or not model:
        return {}
    merged: Dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("model_rules rows must be objects")
        if not _row_matches(row, _model_match_candidates(model, provider)):
            continue
        for key, value in row.items():
            if key == "match":
                continue
            merged[key] = value
    return merged


def model_rules_face_restriction(
    rows: Any,
    model: str,
    provider: str = "",
) -> tuple[Optional[tuple[str, ...]], Optional[tuple[str, ...]], Optional[str], Optional[str]]:
    """Return (allow, deny, allow_row, deny_row) for a model.

    The cascade decides each face list independently: the LAST matching
    row declaring ``allow`` (resp. ``deny``) wins, and the returned row
    pattern names the deciding declaration in refusal errors.
    """

    if not rows or not model:
        return None, None, None, None
    candidates = _model_match_candidates(model, provider)
    allow: Optional[tuple[str, ...]] = None
    deny: Optional[tuple[str, ...]] = None
    allow_row: Optional[str] = None
    deny_row: Optional[str] = None
    for row in rows:
        if not isinstance(row, Mapping) or not _row_matches(row, candidates):
            continue
        if "allow" in row:
            allow = tuple(str(protocol) for protocol in row["allow"])
            allow_row = str(row.get("match", "")).strip()
        if "deny" in row:
            deny = tuple(str(protocol) for protocol in row["deny"])
            deny_row = str(row.get("match", "")).strip()
    return allow, deny, allow_row, deny_row


def enforce_model_rules_faces(rows: Any, model: str, protocol: str, provider: str = "") -> None:
    """Refuse a face outside a model's allow set (or inside its deny set).

    Protocol names match exactly or by wire family (a ``responses`` entry
    governs the sibling variants). The error names the deciding row so the
    operator can find the declaration that refused the face.
    """

    allow, deny, allow_row, deny_row = model_rules_face_restriction(rows, model, provider)
    if allow is None and deny is None:
        return
    from ..protocols.defaults import protocol_family

    names = {str(protocol), protocol_family(str(protocol))}
    if deny and names & set(deny):
        raise ModelRulesFaceError(
            f"model_rules row {deny_row!r} denies protocol face {protocol!r} "
            f"for model {model!r} on provider {provider!r}"
        )
    if allow is not None and not (names & set(allow)):
        raise ModelRulesFaceError(
            f"model_rules row {allow_row!r} limits model {model!r} on provider "
            f"{provider!r} to faces {list(allow)}; {protocol!r} is outside the allowed set"
        )


def _deep_merge(base: Any, override: Any, kind: str = "") -> Any:
    """Merge override into base by rule kind.

    Mappings recurse (scoped tables extend, model overrides win per key).
    Strip lists union (a scoped list adds knobs, never narrows). Clamp
    bounds replace per parameter — a range is indivisible, and the scoped
    face narrows it. Maps and renames merge per key with override winning.
    """

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value, kind) if key in merged else value
        return merged
    if isinstance(base, list) and isinstance(override, list) and kind == "strip":
        return list(base) + [item for item in override if item not in base]
    return override


_TABLE_KEYS = ("strip", "clamp", "map", "rename")


def _apply_model_content(resolved: Dict[str, Any], content: Mapping[str, Any]) -> Dict[str, Any]:
    """Overlay one model-level rule content onto resolved tables.

    Shared by the legacy ``model_param_rules`` entries and the capability
    table's cascade output: table keys deep-merge (model wins per key),
    ``strip_override`` stays the terminal strip replacement.
    """

    for key, value in content.items():
        if key in _ROW_STRUCTURE_KEYS:
            continue
        if key == "strip_override":
            # Terminal model-level escape hatch: the override list
            # REPLACES whatever strip list the provider/scoped tables
            # produced for this model. It never merges — a model that
            # allows X must not silently re-inherit a global strip
            # of X added later.
            if isinstance(value, (list, tuple)):
                resolved["strip"] = list(value)
            continue
        resolved[key] = _deep_merge(resolved.get(key), value, key) if key in resolved else value
    return resolved


def _resolve_rules(provider: str, model: str, config: Mapping[str, Any], *, protocol: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """Merge provider-level rules with model-level overrides (model wins).

    Protocol- and profile-scoped tables overlay the flat base when their
    key matches the executing transport (``by_protocol``/``by_profile``) —
    the same strip/clamp/map/rename vocabulary, applied only on that face.

    Already-resolved flat tables (what ``get_adapter_config`` stores for
    the declared ``param_rules`` adapter) pass through unchanged — the
    resolution pass is idempotent over them.
    """

    resolved: Dict[str, Any] = {}
    wrapper_keys = ("param_rules", "model_param_rules", "model_rules", "by_protocol", "by_profile")
    is_flat_tables = bool(config) and not any(key in config for key in wrapper_keys) and all(key in _TABLE_KEYS for key in config)
    provider_rules = config if is_flat_tables else config.get("param_rules")
    if isinstance(provider_rules, Mapping):
        resolved.update(provider_rules)
    for scope_key, scope_value in (("by_protocol", protocol), ("by_profile", profile)):
        scoped = config.get(scope_key)
        if scope_value and isinstance(scoped, Mapping):
            section = scoped.get(scope_value)
            if isinstance(section, Mapping):
                for key, value in section.items():
                    resolved[key] = _deep_merge(resolved.get(key), value, key) if key in resolved else value
    model_rules = config.get("model_param_rules")
    if isinstance(model_rules, Mapping):
        per_model = model_rules.get(model)
        if isinstance(per_model, Mapping):
            _apply_model_content(resolved, per_model)
    # The capability table (G8): ordered cascade over matching rows. It
    # applies AFTER the legacy per-model mapping — model_param_rules is
    # the superseded bridge, the table is the successor surface.
    capability_rows = config.get("model_rules")
    if capability_rows:
        content = resolve_model_rules(capability_rows, model, provider)
        if content:
            _apply_model_content(resolved, content)
    return resolved


class ParamRulesAdapter(PayloadAdapter):
    """Apply declared strip/clamp/map/rename rules to a request payload.

    The enforcement arm of the capability cascade: consumes the tables
    resolved by :func:`declared_param_rules` (protocol base → database
    seam → provider code → model rows → config) and rewrites the
    provider-bound payload accordingly. Wired as an always-on pipeline
    stage — no provider declares it by name; no resolved rules means an
    identity no-op.
    """

    name = "param_rules"
    supported_stages: tuple[str, ...] = ("request",)
    # Chain entries that build on this engine (e.g. the mistral adapter)
    # set this so the interface's config fill feeds them their resolved
    # tables under their own key — matched by consumption, not by name.
    consumes_param_rules = True

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        config: Mapping[str, Any] = {}
        if context is not None:
            config = context.config_for(self.name) or context.metadata.get("param_rules_config") or {}
        rules = _resolve_rules(
            context.provider if context else "",
            context.model if context else "",
            config,
            protocol=getattr(context, "protocol", None) if context else None,
            profile=getattr(context, "profile", None) if context else None,
        )
        if not rules:
            return payload
        updated = deepcopy(payload)
        changed = False

        for key in rules.get("strip") or ():
            if key in updated:
                updated.pop(key)
                changed = True

        clamps = rules.get("clamp")
        if isinstance(clamps, Mapping):
            for key, bounds in clamps.items():
                value = updated.get(key)
                if isinstance(value, (int, float)) and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    low, high = bounds
                    clamped = min(max(float(value), float(low)), float(high))
                    if clamped != value:
                        updated[key] = type(value)(clamped) if isinstance(value, int) and float(clamped).is_integer() else clamped
                        changed = True

        maps = rules.get("map")
        if isinstance(maps, Mapping):
            for key, table in maps.items():
                value = updated.get(key)
                if value is not None and isinstance(table, Mapping) and str(value) in table:
                    mapped = table[str(value)]
                    if mapped != value:
                        updated[key] = mapped
                        logger.info(
                            "param_rules: %s/%s maps %s=%r -> %r",
                            context.provider,
                            context.model,
                            key,
                            value,
                            mapped,
                        )
                        changed = True

        renames = rules.get("rename")
        if isinstance(renames, Mapping):
            for old, new in renames.items():
                if old in updated and new not in updated:
                    updated[new] = updated.pop(old)
                    changed = True

        return updated if changed else payload


def declared_param_rules(provider_plugin: Any, model: str = "", runtime_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the effective param rules for a provider+model.

    Class declaration ``param_rules`` (code) is the base; JSON runtime
    config extends it; the ordered ``model_rules`` capability table (class
    rows first, JSON rows appended after — config overrides code) and the
    legacy ``model_param_rules`` override per model.
    """

    config: Dict[str, Any] = {}
    class_rules = getattr(provider_plugin, "param_rules", None)
    if isinstance(class_rules, Mapping):
        config["param_rules"] = dict(class_rules)
    class_model_rules = getattr(provider_plugin, "model_param_rules", None)
    if isinstance(class_model_rules, Mapping):
        config["model_param_rules"] = dict(class_model_rules)
    capability_rows: list[Any] = list(getattr(provider_plugin, "model_rules", None) or ())
    if isinstance(runtime_config, Mapping):
        for key in ("param_rules", "model_param_rules"):
            value = runtime_config.get(key)
            if isinstance(value, Mapping):
                config[key] = {**config.get(key, {}), **value}
        runtime_rows = runtime_config.get("model_rules")
        if isinstance(runtime_rows, (list, tuple)):
            capability_rows.extend(runtime_rows)
    if capability_rows:
        config["model_rules"] = capability_rows
    return _resolve_rules(getattr(provider_plugin, "provider_env_name", "") or "", model, config)
