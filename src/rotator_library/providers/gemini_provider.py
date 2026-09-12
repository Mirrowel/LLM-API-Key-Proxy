# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface, declared_endpoint_path, render_endpoint_path

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
    """

    protocol_name = "gemini"
    native_streaming_supported = True
    default_api_base = "https://generativelanguage.googleapis.com"

    def get_native_operation(self, model: str = "", request=None, stream: bool = False) -> str:
        return "stream_generate" if stream else "generate"

    def get_native_endpoint(self, model: str = "", operation: str = "chat", profile: Optional[str] = None) -> str:
        base = _strip_gemini_api_version(self.get_provider_api_base() or self.default_api_base or "")
        # File-based transport profiles may declare their own endpoint path
        # (D13); a declared path wins over the conventional model-ridden ones.
        entry: Optional[Any] = None
        if profile and self.transport_profiles:
            entry = self.transport_profiles.get(profile)
        path = declared_endpoint_path(entry, operation)
        if not path:
            path = declared_endpoint_path(self._get_runtime_config(model), operation)
        if path:
            rendered = render_endpoint_path(
                path,
                model=self.normalize_native_model(model),
                operation=operation,
                provider=self._provider_config_key() or "",
            )
            if rendered.startswith(("http://", "https://")):
                return rendered
            return f"{base}/{rendered.lstrip('/')}"
        if operation == "count_tokens":
            # Token counting is its own action (never :generateContent).
            return f"{base}/v1beta/models/{model}:countTokens"
        action = "streamGenerateContent?alt=sse" if operation == "stream_generate" else "generateContent"
        return f"{base}/v1beta/models/{model}:{action}"

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "chat") -> Dict[str, str]:
        return {"x-goog-api-key": credential_identifier}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Google Gemini API.

        Uses the provider's configured base (never the hardcoded public URL) so
        proxies/mirrors are honored, and paginates via ``nextPageToken``.
        """
        base = _strip_gemini_api_version(self.get_provider_api_base() or self.default_api_base or "")
        try:
            models: List[str] = []
            page_token: Optional[str] = None
            while True:
                params: Dict[str, Any] = {"pageSize": 1000}
                if page_token:
                    params["pageToken"] = page_token
                response = await client.get(
                    f"{base}/v1beta/models",
                    headers={"x-goog-api-key": api_key},
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
                for model in payload.get("models", []) or []:
                    name = model.get("name") if isinstance(model, dict) else model
                    normalized = str(name or "").replace("models/", "")
                    if normalized:
                        models.append(f"gemini/{normalized}")
                page_token = payload.get("nextPageToken") if isinstance(payload, dict) else None
                if not page_token:
                    break
            return models
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Gemini models: {e}")
            return []

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
