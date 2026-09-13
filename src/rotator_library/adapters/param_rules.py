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
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")


def _deep_merge(base: Any, override: Any) -> Any:
    """Merge override into base, recursing through dict values so a
    model-level table extends (not replaces) the provider's."""

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return override


def _resolve_rules(provider: str, model: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge provider-level rules with model-level overrides (model wins)."""

    resolved: Dict[str, Any] = {}
    provider_rules = config.get("param_rules")
    if isinstance(provider_rules, Mapping):
        resolved.update(provider_rules)
    model_rules = config.get("model_param_rules")
    if isinstance(model_rules, Mapping):
        per_model = model_rules.get(model)
        if isinstance(per_model, Mapping):
            for key, value in per_model.items():
                resolved[key] = _deep_merge(resolved.get(key), value) if key in resolved else value
    return resolved


class ParamRulesAdapter(PayloadAdapter):
    """Apply declared strip/clamp/map/rename rules to a request payload."""

    name = "param_rules"
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        config: Mapping[str, Any] = {}
        if context is not None:
            config = context.config_for(self.name) or context.metadata.get("param_rules_config") or {}
        rules = _resolve_rules(context.provider if context else "", context.model if context else "", config)
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
