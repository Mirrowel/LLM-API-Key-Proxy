# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Declared request-parameter rules (G8): one generic, reusable adapter.

Providers and models declare what the wire accepts — strip lists, clamps,
value maps — and this adapter enforces the declarations on the raw
provider-bound payload. No provider hardcodes parameter hygiene in its
own code anymore; the declarations carry it.

Rule sources (later wins): provider class ``param_rules`` declaration →
JSON runtime config ``param_rules`` → per-model overrides via
``model_param_rules`` (capability data on the provider, config in JSON).
Every edit is recorded through the adapter-chain trace like any wire
change.

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
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")


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
    wrapper_keys = ("param_rules", "model_param_rules", "by_protocol", "by_profile")
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
            for key, value in per_model.items():
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


class ParamRulesAdapter(PayloadAdapter):
    """Apply declared strip/clamp/map/rename rules to a request payload."""

    name = "param_rules"
    supported_stages: tuple[str, ...] = ("request",)

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
    config extends it; ``model_param_rules`` overrides per model.
    """

    config: Dict[str, Any] = {}
    class_rules = getattr(provider_plugin, "param_rules", None)
    if isinstance(class_rules, Mapping):
        config["param_rules"] = dict(class_rules)
    class_model_rules = getattr(provider_plugin, "model_param_rules", None)
    if isinstance(class_model_rules, Mapping):
        config["model_param_rules"] = dict(class_model_rules)
    if isinstance(runtime_config, Mapping):
        for key in ("param_rules", "model_param_rules"):
            value = runtime_config.get(key)
            if isinstance(value, Mapping):
                config[key] = {**config.get(key, {}), **value}
    return _resolve_rules(getattr(provider_plugin, "provider_env_name", "") or "", model, config)
