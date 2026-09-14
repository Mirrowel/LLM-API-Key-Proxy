# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Cohere — the OpenAI-compatible face is the declared wire (G8 final).

Cohere's native v2 API is not one of the proxy's protocols; the
compatibility surface (``api.cohere.ai/compatibility/v1``) carries tools,
strict json_schema, and ``reasoning_effort``. ``speaks`` declares that one
chat face; the base URL, bearer auth, ``/models`` listing, and chat route
all resolve from the protocol registry.

Reasoning control is declared, not coded: the compat surface accepts only
``none | high`` on ``reasoning_effort`` (docs; ``medium``/``low`` are
rejected outright), so the capability cascade declares the accepted
vocabulary and the effort system folds every other word with a note —
``off`` rides the wire's ``none`` spelling, ON words land on ``high``.
There is no adapter: the declaration replaced it.

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

    # -- payload shaping (declarations only) ------------------------------
    # One wildcard row: the vocabulary restriction is the compat FACE's,
    # not a model family's, so there is no family split to express.
    model_rules = (
        {
            "match": "*",
            # The compat face accepts the off-group and ``high`` only; the
            # ladder folds low/medium (and forward vocabulary) to ``high``
            # with recorded notes, and an off request rides the
            # openai_chat wire's ``none`` spelling.
            "effort_accept": ["off", "high"],
            # Cohere's compat surface does not document OpenAI's
            # ``required`` tool_choice word (native uses uppercase enum
            # spellings); fold the openai-compat convention instead of
            # riding an unverified word to a 400.
            "map": {"tool_choice": {"required": "any"}},
        },
    )

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation.
