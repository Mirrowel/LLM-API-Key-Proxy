# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenRouter — three native faces over one credential pool (G8 final).

``speaks`` names all three transport faces; everything inherits from the
protocol registry except the anthropic-compatibility face's real diffs:

- ``chat`` (default) and ``responses``: OpenRouter serves the conventional
  chat-completions and Responses routes on ``openrouter.ai/api/v1`` —
  endpoints, bearer auth, and the OpenAI-shaped ``/models`` listing all
  inherit.
- ``anthropic``: OpenRouter's Anthropic-compatible surface lives at
  ``/api/v1/messages`` on this base, not the protocol default's
  ``/v1/messages`` — the face overrides the message routes only. Model
  ids keep their colons (``:free`` / ``:nitro`` variants ride the profile
  grammar untouched).
- Model listing is the shared, protocol-aware interface implementation
  (the openai_chat face wins the listing priority); a failed listing is
  an honest empty.
"""

from __future__ import annotations

from .provider_interface import ProviderInterface


class OpenRouterProvider(ProviderInterface):
    """First-party OpenRouter API — chat, Responses, and Anthropic faces."""

    provider_env_name = "openrouter"

    # -- transport (the envelope) ---------------------------------------
    # First entry is the default face. The anthropic face overrides only
    # the routes that diverge from the protocol default; the profile name
    # stays the historical ``anthropic`` for addressing stability.
    speaks = (
        ("chat", "openai_chat", {}),
        "responses",
        (
            "anthropic",
            "anthropic_messages",
            {
                "endpoint_paths": {
                    "messages": "/messages",
                    "count_tokens": "/messages/count_tokens",
                },
                # OpenRouter authenticates every face with a bearer token,
                # not the Anthropic protocol's conventional x-api-key.
                "auth_mode": "bearer",
            },
        ),
    )
    native_streaming_supported = True
    default_api_base = "https://openrouter.ai/api/v1"

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation.
