# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""DeepSeek — a first-class declared provider (G8 remake).

Identity plus declarations, no execution code: the native runtime owns the
calls (chat face by default, Responses and Anthropic-compatibility faces
via ``deepseek:responses/...`` and ``deepseek:anthropic/...`` addressing),
the generic ``param_rules`` adapter owns parameter hygiene, and the
field-cache engine owns reasoning-content preservation.

Everything this provider needs is a declaration:

- ``param_rules`` renames ``max_completion_tokens`` → ``max_tokens``.
- ``model_param_rules`` carries the official V4 reasoning-effort mapping
  as MODEL capability data (per-model tables, exact model ids).
- ``field_cache_rules`` replaces the retired hand-rolled reasoning cache:
  response and stream siblings share one store (``cache_key``), correlate
  per occurrence by tool-call id first and content sha second, and inject
  ``reasoning_content`` back into every in-scope assistant message —
  DeepSeek is the one provider where ALL history is the default scope
  (tools present demands reasoning on every turn). A miss injects the
  documented placeholder with an engine warning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


HARDCODED_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
    "deepseek-flash",
]

REASONING_PLACEHOLDER = "Reasoning content unavailable."

# Reasoning-content retention window (7 days, matching the retired disk
# cache's TTL for the same state).
REASONING_TTL_SECONDS = 604800

# Models speaking the official {low, high, max} effort vocabulary. Kept as
# a class constant so the capability list is one edit; model_param_rules
# keys are exact model ids.
V4_EFFORT_MODELS = (
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
)

# Official mapping from the proxy's canonical effort vocabulary. Values
# absent from a model's table pass through unchanged; when the client sends
# NOTHING, nothing is injected — the server default (on/high) applies.
V4_EFFORT_MAP = {
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
}

# One injection declaration shared by the response and stream siblings
# (shared-cache-key rules must agree on mode, scope, TTL, injection, and
# correlation behavior). The path is the turn-relative message path; the
# engine resolves it per message occurrence.
_REASONING_INJECTION = FieldCacheInjection(
    target="request",
    path="messages.*.reasoning_content",
    when_missing_only=True,
)

# Occurrence correlation: the response/stream message's tool_calls ids
# (primary), with the engine's automatic content-sha fallback (secondary).
_REASONING_RULE_METADATA: Dict[str, Any] = {
    "tool_call_id_path": "tool_calls.*.id",
}


class DeepseekProvider(ProviderInterface):
    """First-party DeepSeek API — three transport faces, one identity."""

    provider_env_name = "deepseek"
    skip_cost_calculation = True

    default_rotation_mode: str = "sequential"
    default_max_concurrent_per_key: int = -1

    # -- transport ------------------------------------------------------
    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://api.deepseek.com"
    default_profile = "chat"
    transport_profiles = {
        "chat": {
            "protocol": "openai_chat",
            "endpoint_paths": {
                "chat": "/chat/completions",
                "models": "/models",
            },
        },
        "responses": {
            "protocol": "responses",
            "endpoint_paths": {
                "responses": "/responses",
            },
        },
        "anthropic": {
            "protocol": "anthropic_messages",
            "endpoint_paths": {
                "messages": "/anthropic/v1/messages",
                "count_tokens": "/anthropic/v1/messages/count_tokens",
            },
        },
    }

    # -- payload shaping --------------------------------------------------
    # The generic param_rules adapter enforces the declarations below; no
    # deepseek-specific adapter code exists (nothing remained that a
    # declaration could not express).
    adapter_names = ("param_rules",)

    param_rules = {
        "rename": {"max_completion_tokens": "max_tokens"},
    }

    model_param_rules = {
        model: {"map": {"reasoning_effort": dict(V4_EFFORT_MAP)}}
        for model in V4_EFFORT_MODELS
    }

    # -- state preservation ------------------------------------------------
    field_cache_rules = (
        FieldCacheRule(
            name="reasoning",
            cache_key="deepseek_reasoning",
            source="response",
            path="choices.0.message.reasoning_content",
            mode="all",
            inject=_REASONING_INJECTION,
            placeholder=REASONING_PLACEHOLDER,
            ttl_seconds=REASONING_TTL_SECONDS,
            metadata=dict(_REASONING_RULE_METADATA),
        ),
        FieldCacheRule(
            name="reasoning_stream",
            cache_key="deepseek_reasoning",
            source="stream_event",
            # Stream events reach the engine as serialized neutral events;
            # the provider chunk rides under ``raw``.
            path="raw.choices.0.delta.reasoning_content",
            mode="all",
            inject=_REASONING_INJECTION,
            placeholder=REASONING_PLACEHOLDER,
            ttl_seconds=REASONING_TTL_SECONDS,
            metadata=dict(_REASONING_RULE_METADATA),
        ),
    )

    # -- discovery -----------------------------------------------------------

    def _models_url(self) -> str:
        return self.get_native_endpoint(operation="models")

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available DeepSeek models, with a conservative fallback list."""
        try:
            response = await client.get(
                self._models_url(),
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
            if models:
                return [f"deepseek/{model}" for model in models]
        except Exception as e:
            lib_logger.debug(f"Failed to fetch DeepSeek models, using fallback list: {e}")

        return [f"deepseek/{model}" for model in HARDCODED_MODELS]
