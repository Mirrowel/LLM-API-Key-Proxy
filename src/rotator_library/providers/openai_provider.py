# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI — the reference provider, Responses-first (G8 final).

Identity plus declarations: the ``speaks`` envelope names both transport
faces, everything else inherits from the protocol registry.

- Responses is the default face (the first ``speaks`` entry): its routes
  inherit from the Responses protocol defaults, with native token
  counting as the ONE real diff — OpenAI serves
  ``/responses/input_tokens`` where the protocol default says
  ``/messages/count_tokens``. Responses carries reasoning,
  encrypted-content replay, and the primary API shape; bare
  ``openai/model`` still resolves to the client's own protocol through
  profile matching, so only conversion cases steer here.
- The chat face stays first-class for multi-candidate traffic and
  chat-native clients, addressed explicitly as ``openai:chat/model``.
  Endpoints (``/chat/completions``), bearer auth, and the ``/models``
  listing all inherit — nothing to override.
- Model listing is the shared, protocol-aware interface implementation
  (openai_chat face, bearer auth inherited, ``/models`` route); a failed
  listing is an honest empty — the hand-rolled fetch is dead.
"""

from __future__ import annotations

from .provider_interface import ProviderInterface


class OpenAIProvider(ProviderInterface):
    """First-party OpenAI API — Responses and chat-completions faces."""

    provider_env_name = "openai"

    # -- transport (the envelope) ---------------------------------------
    # First entry is the default face. The Responses face overrides only
    # its diverging token-count route; the chat face is named explicitly
    # so the historical ``openai:chat/model`` addressing keeps working.
    speaks = (
        ("responses", {"endpoint_paths": {"count_tokens": "/responses/input_tokens"}}),
        ("chat", "openai_chat", {}),
    )
    native_streaming_supported = True
    default_api_base = "https://api.openai.com/v1"

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation.
