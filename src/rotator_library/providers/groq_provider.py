# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Groq — OpenAI-compatible chat on GroqLPUs (G8 final).

Identity plus declarations: ``speaks`` the chat-completions face, and the
``model_rules`` capability cascade owns the parameter hygiene Groq's
documented surface demands — the knobs Groq hard-rejects ("not yet
supported by any of our models" in its API reference: frequency/presence
penalties, the logit-bias and logprob families), ``temperature`` clamped
to Groq's accepted 0..2 window with the float32-safe epsilon floor zeros
get rewritten to, and ``n`` pinned to 1 (Groq serves one candidate).

The ``groq`` adapter keeps ONLY what a flat row cannot express — the
conditional wire surgery: ``reasoning_format: parsed`` is forced when
tools or JSON output are present (raw thinking plus tools is a documented
400, and ``include_reasoning`` is mutually exclusive with
``reasoning_format``, so it cannot ride along), and Groq's ``reasoning``
field is renamed to the chat-family ``reasoning_content`` spelling on
responses and stream deltas (with the ``x_groq.usage`` payload lifted
into the standard usage slot).

Model listing is the shared, protocol-aware interface implementation
(bearer auth inherited, ``/models`` route, ``data[].id`` shape); a failed
listing is an honest empty.
"""

from __future__ import annotations

from .provider_interface import ProviderInterface


class GroqProvider(ProviderInterface):
    """First-party Groq API over the chat-completions face."""

    provider_env_name = "groq"

    # -- transport (the envelope) ---------------------------------------
    speaks = ("openai_chat",)
    native_streaming_supported = True
    default_api_base = "https://api.groq.com/openai/v1"

    # -- payload shaping --------------------------------------------------
    # The groq adapter extends the pipeline after the always-on param
    # engine (its stage handles the conditional reasoning_format work and
    # the response-side reasoning rename).
    adapter_names = ("groq",)

    model_rules = (
        {
            "match": "*",
            # Groq's API reference marks these "not yet supported by any of
            # our models" — they hard-fail the request, so they strip
            # provider-wide.
            "strip": [
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "logprobs",
                "top_logprobs",
            ],
            # Documented sampling window is 0..2; the epsilon floor is the
            # float32-safe value Groq requires instead of exact 0.
            "clamp": {"temperature": [1e-8, 2.0], "n": [1, 1]},
        },
    )

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation.
