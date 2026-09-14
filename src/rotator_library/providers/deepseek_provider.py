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
  ``max_tokens`` for every model; the provider-level
  ``reasoning_effort_accept`` declares the full accepted vocabulary and
  the old V4 snapshot rows (``-0813``-era) shrink it to the official
  {off, low, high, max} set — the ladder then folds ``medium`` up to
  ``high`` for them. The OFF control rides the chat wire's thinking
  toggle (``toggle``), declared provider-wide and on the old rows.
- ``field_cache_rules``: one rule addressing ``field="reasoning"``. The
  engine resolves extraction (response + stream siblings via
  ``sources``), injection, and tool-call-id correlation from the protocol
  registry for whatever face executes (``FieldCacheContext.
  protocol_family``) — the same rule would serve the Responses face the
  day it extracts reasoning there. DeepSeek is the one provider where
  ALL history is the default scope (mode ``all`` — tools present demands
  reasoning on every turn); a miss injects the documented placeholder
  with an engine warning. The cache key derives from provider+field and
  the store's 3-day inactivity default owns retention (no rule TTL).
"""

from __future__ import annotations

from ..field_cache import FieldCacheRule
from .provider_interface import ProviderInterface

REASONING_PLACEHOLDER = "Reasoning content unavailable."


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

    # Provider-level accepted reasoning-effort vocabulary (off is the
    # thinking toggle on the chat wire). When the client sends NOTHING,
    # nothing is injected — the server default applies. The old V4
    # snapshot rows (``-0813``-era) shrink the set to the official
    # {off, low, high, max}; the ladder folds ``medium`` up to ``high``
    # there.
    reasoning_effort_accept = ("off", "low", "medium", "high", "max")
    reasoning_effort_toggle = True

    model_rules = (
        {
            "match": "*",
            "rename": {"max_completion_tokens": "max_tokens"},
        },
        {
            "match": "deepseek-v4-pro-08*",
            "effort_accept": ["off", "low", "high", "max"],
            "toggle": True,
        },
        {
            "match": "deepseek-v4-flash-08*",
            "effort_accept": ["off", "low", "high", "max"],
            "toggle": True,
        },
    )

    # -- state preservation ------------------------------------------------
    # One field-addressed rule: response + stream siblings (expanded by
    # the engine from ``sources``) share one store, correlate per
    # occurrence by tool-call id first and content sha second (paths from
    # the protocol registry), and re-inject only where history is missing
    # the field. A miss injects the documented placeholder. The cache key
    # auto-derives (provider:field); retention is the store's 3-day
    # inactivity default — no rule TTL.
    field_cache_rules = (
        FieldCacheRule(
            name="reasoning",
            field="reasoning",
            sources=("response", "stream_event"),
            mode="all",
            placeholder=REASONING_PLACEHOLDER,
            inject="auto",
        ),
    )

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation
    # (openai_chat face, bearer auth inherited, /models route); a failed
    # listing is an honest empty — the hardcoded fallback list is dead.
