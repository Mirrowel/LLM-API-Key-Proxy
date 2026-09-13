# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""DeepSeek — the provider-envelope showcase (G8 final).

Identity plus declarations, no execution code: ``speaks`` names the three
transport faces (chat by default, Responses and Anthropic-compatibility
as the second and third entries — profile names are protocol names on
the envelope), the ``model_rules`` capability cascade owns parameter
hygiene, the shared interface implementation owns model listing, and ONE
field-addressed field-cache rule owns reasoning-content preservation.

Everything this provider needs is a declaration:

- ``model_rules``: the ``*`` row renames ``max_completion_tokens`` →
  ``max_tokens`` for every model; the V4 wildcard rows carry the official
  reasoning-effort mapping. The wildcards intentionally cover the dated
  snapshot ids (``deepseek-v4-pro-0813``, ...) the exact-id
  ``model_param_rules`` table could not — a capability declared once per
  family, not once per release.
- ``field_cache_rules``: one rule addressing ``field="reasoning"``. The
  engine resolves extraction (response + stream siblings via
  ``sources``), injection, and tool-call-id correlation from the protocol
  registry for whatever face executes (``FieldCacheContext.
  protocol_family``) — the same rule would serve the Responses face the
  day it extracts reasoning there. DeepSeek is the one provider where
  ALL history is the default scope (mode ``all`` — tools present demands
  reasoning on every turn); a miss injects the documented placeholder
  with an engine warning.
"""

from __future__ import annotations

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

REASONING_PLACEHOLDER = "Reasoning content unavailable."

# Reasoning-content retention window (7 days, matching the retired disk
# cache's TTL for the same state).
REASONING_TTL_SECONDS = 604800


class DeepseekProvider(ProviderInterface):
    """First-party DeepSeek API — three transport faces, one identity."""

    provider_env_name = "deepseek"

    # NOTE(for-removal): dies with the cost phase.
    skip_cost_calculation = True

    # -- transport (the envelope) ---------------------------------------
    # Endpoints, auth, and listing inherit from the protocol registry;
    # the anthropic-compatibility face overrides only its diverging
    # routes. The first entry is the default face.
    speaks = (
        "openai_chat",
        "responses",
        (
            "anthropic_messages",
            {
                "endpoint_paths": {
                    "messages": "/anthropic/v1/messages",
                    "count_tokens": "/anthropic/v1/messages/count_tokens",
                }
            },
        ),
    )
    native_streaming_supported = True
    default_api_base = "https://api.deepseek.com"

    # -- payload shaping --------------------------------------------------
    # The generic param_rules adapter enforces the capability cascade
    # below; no deepseek-specific adapter code exists (nothing remained
    # that a declaration could not express).
    adapter_names = ("param_rules",)

    # Official {low, high, max} effort vocabulary for the V4 family.
    # ``deepseek-flash`` is a V4 alias and speaks the same table. Values
    # absent from a model's table pass through unchanged; when the client
    # sends NOTHING, nothing is injected — the server default (on/high)
    # applies. The wildcards also cover dated snapshot ids (-0813,
    # -0731, ...) — an intentional improvement over the exact-id table.
    model_rules = (
        {
            "match": "*",
            "rename": {"max_completion_tokens": "max_tokens"},
        },
        {
            "match": "deepseek-v4*",
            "effort_map": {
                "low": "low",
                "medium": "high",
                "high": "high",
                "xhigh": "high",
                "max": "max",
            },
        },
        {
            "match": "deepseek-flash",
            "effort_map": {
                "low": "low",
                "medium": "high",
                "high": "high",
                "xhigh": "high",
                "max": "max",
            },
        },
    )

    # -- state preservation ------------------------------------------------
    # One field-addressed rule: response + stream siblings (expanded by
    # the engine from ``sources``) share one store, correlate per
    # occurrence by tool-call id first and content sha second (paths from
    # the protocol registry), and re-inject only where history is missing
    # the field. A miss injects the documented placeholder.
    field_cache_rules = (
        FieldCacheRule(
            name="reasoning",
            field="reasoning",
            sources=("response", "stream_event"),
            cache_key="deepseek_reasoning",
            mode="all",
            placeholder=REASONING_PLACEHOLDER,
            ttl_seconds=REASONING_TTL_SECONDS,
            inject=FieldCacheInjection(target="request", path="", when_missing_only=True),
        ),
    )

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation
    # (openai_chat face, bearer auth inherited, /models route); a failed
    # listing is an honest empty — the hardcoded fallback list is dead.
