# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Mistral — a first-class declared provider on the envelope (G8 final).

Identity plus declarations; the one execution surface that remains is the
``mistral`` adapter, and even that is an EXTENSION of the generic
``param_rules`` engine (one chain entry applies both). Everything else
is declaration:

- ``speaks`` the chat-completions face; endpoints, bearer auth, and the
  /models listing all inherit from the protocol registry.
- ``model_rules`` carries the whole parameter story as a cascade: the
  ``*`` row strips provider-wide rejects (``reasoning_effort`` lives
  there because only the declared reasoning families accept it), clamps
  ``temperature`` to Mistral's documented 0..1 window, pins ``n`` to 1,
  renames ``max_completion_tokens`` → ``max_tokens``, and maps
  ``tool_choice: "required"`` → ``"any"`` (Mistral's spelling). The
  ``mistral-small*``/``mistral-medium*`` rows carry the reasoning
  capability: ``strip_override`` REPLACES the provider strip list there
  (re-admitting ``reasoning_effort`` — terminal, never a union), and the
  effort table folds the wider official vocabulary onto the spec enum
  ``high`` (``none`` passes through). When the client sends NOTHING,
  nothing is injected — the server default (reasoning off) applies. The
  wildcards cover every dated variant of the two reasoning families —
  the exact-id constant list is dead.
- ``field_cache_rules`` preserves reasoning across turns with ONE
  field-addressed rule: response and stream siblings (expanded from
  ``sources``) share one store, and paths/injection/correlation resolve
  from the protocol registry for the executing face. Mode is
  intentionally UNDECLARED (the global default ``turn``): Mistral
  documents no reasoning-replay requirement beyond the current turn.
  Injection is auto (``when_missing_only``) onto the message-level
  ``reasoning_content``; NO placeholder — Mistral has no documented
  400-on-missing contract, so a miss leaves the message clean instead of
  fabricating text.
"""

from __future__ import annotations

from typing import Any, Dict

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

# Effort table for the reasoning families: the spec enum is high|none; the
# wider official words fold to high. ``none`` is absent from the table and
# therefore passes through unchanged.
_MISTRAL_REASONING_EFFORT_MAP = {
    "minimal": "high",
    "low": "high",
    "medium": "high",
    "xhigh": "high",
}


class MistralProvider(ProviderInterface):
    """First-party Mistral API (La Plateforme) over the chat-completions face."""

    provider_env_name = "mistral"

    # -- transport (the envelope) ---------------------------------------
    speaks = ("openai_chat",)
    native_streaming_supported = True
    default_api_base = "https://api.mistral.ai/v1"

    # -- payload shaping --------------------------------------------------
    # The mistral adapter extends the param_rules engine, so the single
    # chain entry applies the capability cascade below AND the provider-
    # specific wire handling (think-chunk folding, history-reasoning
    # strip, nested seed rename) that lives in adapters/mistral.py.
    adapter_names = ("mistral",)

    model_rules = (
        {
            # reasoning_effort is stripped provider-wide: only the
            # reasoning families accept it, and their strip_override
            # re-admits it.
            "match": "*",
            "strip": ["reasoning_effort", "logit_bias", "logprobs", "top_logprobs"],
            "clamp": {"temperature": [0.0, 1.0], "n": [1, 1]},
            "rename": {"max_completion_tokens": "max_tokens"},
            "map": {"tool_choice": {"required": "any"}},
        },
        {
            "match": "mistral-small*",
            # Terminal: REPLACES the provider strip list for this family —
            # reasoning_effort is allowed exactly here.
            "strip_override": ["logit_bias", "logprobs", "top_logprobs"],
            "effort_map": dict(_MISTRAL_REASONING_EFFORT_MAP),
        },
        {
            "match": "mistral-medium*",
            "strip_override": ["logit_bias", "logprobs", "top_logprobs"],
            "effort_map": dict(_MISTRAL_REASONING_EFFORT_MAP),
        },
    )

    # -- state preservation ------------------------------------------------
    field_cache_rules = (
        FieldCacheRule(
            name="reasoning",
            field="reasoning",
            sources=("response", "stream_event"),
            cache_key="mistral_reasoning",
            # mode intentionally UNDECLARED: the global default "turn" is
            # the declared behavior — no doc requirement beyond the
            # current turn, so the narrowest scope applies.
            # No placeholder: Mistral has no 400-on-missing contract for
            # absent reasoning_content — a miss leaves the message clean.
            inject=FieldCacheInjection(target="request", path="", when_missing_only=True),
        ),
    )

    # -- discovery -----------------------------------------------------------

    def get_adapter_config(self, model: str = "") -> Dict[str, Dict[str, Any]]:
        """Expose the resolved param-rule tables to the ``mistral`` adapter.

        The generic hook fills the ``param_rules`` config key only for chains
        that declare that adapter BY NAME; the mistral adapter extends the
        engine instead, so its resolved capability cascade rides under the
        adapter's own key.
        """

        config = super().get_adapter_config(model)
        if "mistral" in self.get_adapter_names(model) and "mistral" not in config:
            from ..adapters.param_rules import declared_param_rules

            rules = declared_param_rules(self, model)
            if rules:
                config["mistral"] = rules
        return config

    # Model listing is the shared, protocol-aware interface implementation
    # (openai_chat face, bearer auth inherited, /models route); a failed
    # listing is an honest empty.
