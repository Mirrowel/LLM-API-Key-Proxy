# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Cohere — the OpenAI-compatible face is the declared wire (G8 final).

Cohere's native v2 API is not one of the proxy's protocols; the
compatibility surface (``api.cohere.ai/compatibility/v1``) carries tools,
strict json_schema, and ``reasoning_effort`` (none|high). ``speaks``
declares that one chat face; the base URL, bearer auth, ``/models``
listing, and chat route all resolve from the protocol registry.

The ``cohere`` adapter is the genuinely custom wire logic the declaration
cannot express: the canonical effort vocabulary narrows to Cohere's
``none | high`` thinking control (every ON rung folds to ``high``, off
spellings normalize to ``none``, a null control drops) with a recorded
log line when a word is narrowed.

Model listing is the shared, protocol-aware interface implementation
(``data[].id`` — the compat face's shape); a failed listing is an honest
empty.
"""

from __future__ import annotations

from .provider_interface import ProviderInterface


class CohereProvider(ProviderInterface):
    """First-party Cohere API over the compatibility face."""

    provider_env_name = "cohere"

    # -- transport (the envelope) ---------------------------------------
    speaks = ("openai_chat",)
    native_streaming_supported = True
    default_api_base = "https://api.cohere.ai/compatibility/v1"

    # -- payload shaping --------------------------------------------------
    # The cohere adapter owns the none|high effort narrowing (see the
    # module docstring); everything else is protocol-owned.
    adapter_names = ("cohere",)

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation.
