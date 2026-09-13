# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Mistral — a first-class declared provider (G8 remake).

Identity plus declarations; the one execution surface that remains is the
``mistral`` adapter, and even that is an EXTENSION of the generic
``param_rules`` engine (one chain entry applies both). Everything else
is declaration:

- ``param_rules`` strips provider-wide rejects — ``reasoning_effort``
  lives here because only the declared reasoning models accept it —
  clamps ``temperature`` to Mistral's documented 0..1 window, pins ``n``
  to 1, renames ``max_completion_tokens`` → ``max_tokens``, and maps
  ``tool_choice: "required"`` → ``"any"`` (Mistral's spelling).
- ``model_param_rules`` carries the four current reasoning models:
  ``strip_override`` REPLACES the provider strip list there (re-admitting
  ``reasoning_effort`` — terminal, never a union), and the effort table
  folds the wider official vocabulary onto the spec enum ``high``
  (``none`` passes through). When the client sends NOTHING, nothing is
  injected — the server default (reasoning off) applies.
- ``field_cache_rules`` preserves reasoning across turns: the response
  rule reads the assembled ``reasoning_content`` AFTER the adapter folded
  Mistral's think-chunks; the stream sibling reads the neutral delta's
  reasoning text (same ``cache_key`` — the shared-store contract). Mode
  is intentionally UNDECLARED (the global default ``turn``): Mistral
  documents no reasoning-replay requirement beyond the current turn.
  Injection is auto (``when_missing_only``) onto the message-level
  ``reasoning_content``; NO placeholder — Mistral has no documented
  400-on-missing contract, so a miss leaves the message clean instead of
  fabricating text.
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

# The current reasoning-capable models (exact upstream ids). Kept at module
# level so the class-body comprehension below can see it; mirrored as the
# MISTRAL_REASONING_MODELS class constant so the capability list is one edit.
MISTRAL_REASONING_MODELS = (
    "mistral-small-latest",
    "mistral-small-2603",
    "mistral-medium-3-5",
    "mistral-medium-2604",
)

# Effort table for the reasoning models: the spec enum is high|none; the
# wider official words fold to high. ``none`` is absent from the table and
# therefore passes through unchanged.
_MISTRAL_REASONING_EFFORT_MAP = {
    "minimal": "high",
    "low": "high",
    "medium": "high",
    "xhigh": "high",
}

# One injection declaration shared by the response and stream siblings
# (shared-cache_key rules must agree on mode, scope, TTL, injection, and
# correlation behavior — see field_cache/engine.py).
_REASONING_INJECTION = FieldCacheInjection(
    target="request",
    path="messages.*.reasoning_content",
    when_missing_only=True,
)


class MistralProvider(ProviderInterface):
    """First-party Mistral API (La Plateforme) over the chat-completions face."""

    provider_env_name = "mistral"

    # -- transport ------------------------------------------------------
    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://api.mistral.ai/v1"

    # -- payload shaping --------------------------------------------------
    # The mistral adapter extends the param_rules engine, so the single
    # chain entry applies the declared tables below AND the provider-
    # specific wire handling (think-chunk folding, history-reasoning
    # strip, nested seed rename) that lives in adapters/mistral.py.
    adapter_names = ("mistral",)

    MISTRAL_REASONING_MODELS = MISTRAL_REASONING_MODELS

    param_rules = {
        # reasoning_effort is stripped provider-wide: only the declared
        # reasoning models accept it, and their strip_override re-admits it.
        "strip": ["reasoning_effort", "logit_bias", "logprobs", "top_logprobs"],
        "clamp": {"temperature": [0.0, 1.0], "n": [1, 1]},
        "rename": {"max_completion_tokens": "max_tokens"},
        "map": {"tool_choice": {"required": "any"}},
    }

    model_param_rules = {
        model: {
            # Terminal: REPLACES the provider strip list for this model —
            # reasoning_effort is allowed exactly here.
            "strip_override": ["logit_bias", "logprobs", "top_logprobs"],
            "map": {"reasoning_effort": dict(_MISTRAL_REASONING_EFFORT_MAP)},
        }
        for model in MISTRAL_REASONING_MODELS
    }

    # -- state preservation ------------------------------------------------
    field_cache_rules = (
        FieldCacheRule(
            name="reasoning",
            cache_key="mistral_reasoning",
            source="response",
            # The assembled message reasoning AFTER the mistral adapter
            # converted think-chunks (the adapter chain runs before
            # extraction on the raw provider response).
            path="choices.0.message.reasoning_content",
            # mode intentionally UNDECLARED: the global default "turn" is
            # the declared behavior — no doc requirement beyond the
            # current turn, so the narrowest scope applies.
            inject=_REASONING_INJECTION,
            # No placeholder: Mistral has no 400-on-missing contract for
            # absent reasoning_content — a miss leaves the message clean.
        ),
        FieldCacheRule(
            name="reasoning_stream",
            cache_key="mistral_reasoning",
            source="stream_event",
            # Serialized neutral stream event: the provider chunk rides
            # under ``raw``, but the adapter has already moved think-chunk
            # text into the delta's reasoning blocks, so the neutral delta
            # is the extraction surface.
            path="delta.reasoning.0.text",
            inject=_REASONING_INJECTION,
        ),
    )

    # -- discovery -----------------------------------------------------------

    def get_adapter_config(self, model: str = "") -> Dict[str, Dict[str, Any]]:
        """Expose the resolved param-rule tables to the ``mistral`` adapter.

        The generic hook fills the ``param_rules`` config key only for chains
        that declare that adapter BY NAME; the mistral adapter extends the
        engine instead, so its resolved provider+model tables ride under the
        adapter's own key.
        """

        config = super().get_adapter_config(model)
        if "mistral" in self.get_adapter_names(model) and "mistral" not in config:
            from ..adapters.param_rules import declared_param_rules

            rules = declared_param_rules(self, model)
            if rules:
                config["mistral"] = rules
        return config

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Mistral API.
        """
        try:
            response = await client.get(
                f"{self.get_provider_api_base()}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"mistral/{model['id']}"
                for model in response.json().get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Mistral models: {e}")
            return []
