# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Gemini — two transport faces over one identity (G8 final).

``speaks`` declares both faces and everything else inherits from the
protocol registry:

- ``native`` (the default face, first entry): the Gemini wire on
  ``/v1beta/models/{model}:generateContent`` (and its
  streamGenerateContent / countTokens siblings), authenticated with
  ``x-goog-api-key``. The profile name ``native`` is preserved for
  addressing stability (``gemini:native/model``).
- ``openai``: Google's OpenAI-compatibility surface on
  ``/v1beta/openai/...`` with Bearer auth — the only real overrides are
  the compat routes and the auth mode, both declared on the face.

The one piece of genuinely custom transport logic that remains is the
BASE normalization: ``GEMINI_API_BASE`` is commonly configured WITH the
version path (``.../v1beta``), while the endpoints own the version
segment themselves — the override strips a trailing version so paths
never double-append. Model listing is the shared, protocol-aware
interface implementation (``models[].name`` shape, ``models/`` prefix
stripped, ``/v1beta/models`` on the normalized base); a failed listing is
an honest empty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False  # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

_GEMINI_API_VERSION_SUFFIXES = ("/v1beta", "/v1")


def _strip_gemini_api_version(base: str) -> str:
    """Normalize a configured Gemini base so paths never double-append.

    ``GEMINI_API_BASE`` is commonly configured WITH the version path
    (``.../v1beta``); the endpoint builder appends ``/v1beta/...`` itself, so
    a trailing version suffix is stripped — otherwise every request becomes
    ``/v1beta/v1beta/...``.
    """

    trimmed = str(base or "").rstrip("/")
    for suffix in _GEMINI_API_VERSION_SUFFIXES:
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)].rstrip("/")
    return trimmed


class GeminiProvider(ProviderInterface):
    """
    Provider implementation for the Google Gemini API.

    G11: two transport faces share one provider identity — ``native``
    (the Gemini format on /v1beta/models/..., x-goog-api-key) and
    ``openai`` (Google's OpenAI-compat surface on /v1beta/openai/...,
    Bearer). Default face: native. The gemini FORMAT itself stays a
    normal declarable protocol (Google's AI Studio + Vertex + the rare
    third-party native surface).
    """

    provider_env_name = "gemini"

    # -- transport (the envelope) ---------------------------------------
    # First entry is the default face. The native face inherits the
    # x-goog auth + the /v1beta :generateContent/:streamGenerateContent/
    # :countTokens templates from the protocol defaults; the compat face
    # overrides only its real routes and auth mode.
    speaks = (
        ("native", "gemini", {}),
        (
            "openai",
            "openai_chat",
            {
                "endpoint_paths": {
                    "chat": "/v1beta/openai/chat/completions",
                    "models": "/v1beta/openai/models",
                },
                "auth_mode": "bearer",
            },
        ),
    )
    native_streaming_supported = True
    default_api_base = "https://generativelanguage.googleapis.com"
    # Listing runs on the NATIVE face: its descriptor parses
    # ``models[].name`` and strips the ``models/`` prefix, while the compat
    # face's OpenAI-shaped ids would keep the prefix. The hint names the
    # profile; the shared implementation resolves it to the protocol.
    listing_profile = "native"

    def get_provider_api_base(self) -> Optional[str]:
        """Version-normalized transport base (see ``_strip_gemini_api_version``).

        The inherited endpoint builder composes base + registry path; the
        registry paths carry ``/v1beta``, so a base configured WITH the
        version (the common GEMINI_API_BASE spelling) is stripped here —
        both spellings now resolve to the same upstream URL.
        """

        base = super().get_provider_api_base()
        return _strip_gemini_api_version(base) if base else base

    # =========================================================================
    # SAFETY SETTINGS (REMOVED)
    # =========================================================================
    #
    # Previously, the proxy auto-injected default Gemini safety settings for every
    # request. This caused 400 errors on models that don't support those categories
    # (e.g. Gemma models reject harassment, hate_speech, sexually_explicit,
    # dangerous_content, civic_integrity). The safety settings system has been
    # removed from the transform pipeline. Safety settings are now passed through
    # unchanged if the caller provides them.
    #
    # Previous defaults that were injected:
    #
    #   Generic form (dict):
    #     {
    #         "harassment": "OFF",
    #         "hate_speech": "OFF",
    #         "sexually_explicit": "OFF",
    #         "dangerous_content": "OFF",
    #         "civic_integrity": "BLOCK_NONE",
    #     }
    #
    #   Gemini-native form (list):
    #     [
    #         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    #         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    #         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    #         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    #         {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    #     ]
    #
    # Removed from:
    #   - ProviderTransforms._transform_gemini_safety  (transforms.py)
    #   - ProviderTransforms.convert_safety_settings   (transforms.py)
    #   - ProviderInterface.convert_safety_settings    (provider_interface.py)
    #   - GeminiProvider.convert_safety_settings       (this file)
    # =========================================================================

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Handles reasoning parameters for Gemini models, with three distinct paths:
        1. Applies a non-standard, high-value token budget if 'custom_reasoning_budget' is true.
        2. Leaves the 'reasoning_effort' parameter alone for LiteLLM to handle if it's present
           without the custom flag.
        3. Applies a default 'thinking' value for specific models if no other reasoning
           parameters are provided, ensuring they 'think' by default.
        """
        # Set default temperature to 1 if not provided
        if "temperature" not in payload:
            payload["temperature"] = 1

        custom_reasoning_budget = payload.get("custom_reasoning_budget", False)
        reasoning_effort = payload.get("reasoning_effort")

        # If 'thinking' is already explicitly set, do nothing to avoid overriding it.
        if "thinking" in payload:
            return

        # Path 1: Custom budget is explicitly requested.
        if custom_reasoning_budget:
            # Case 1a: Both params are present, so we can apply the custom budget.
            if reasoning_effort:
                if "gemini-2.5-pro" in model:
                    budgets = {"low": 8192, "medium": 16384, "high": 32768}
                elif "gemini-2.5-flash" in model:
                    budgets = {"low": 6144, "medium": 12288, "high": 24576}
                else:  # Fallback for other models if the custom flag is still used
                    budgets = {"low": 1024, "medium": 2048, "high": 4096}

                budget = budgets.get(reasoning_effort)
                if budget is not None:
                    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                elif reasoning_effort == "disable":
                    payload["thinking"] = {"type": "enabled", "budget_tokens": 0}

                # Clean up the handled 'reasoning_effort' parameter.
                payload.pop("reasoning_effort", None)

            # Case 1b: In all cases where the custom flag was present, remove it
            # as it's not a standard LiteLLM parameter.
            payload.pop("custom_reasoning_budget", None)
            return

        # Path 2: No custom budget. Now check for standard or default behavior.
        # If 'reasoning_effort' is present, we do nothing, allowing LiteLLM to handle it.
        # If 'reasoning_effort' is NOT present, then we apply the default thinking behavior.
        if not reasoning_effort:
            if "gemini-2.5-pro" in model or "gemini-2.5-flash" in model:
                payload["thinking"] = {"type": "enabled", "budget_tokens": -1}
