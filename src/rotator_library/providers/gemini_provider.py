# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from .utilities.gemini_quota_utils import parse_google_quota_error
from ..core.types import RequestCompleteResult

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False  # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

RATE_LIMIT_COOLDOWN = 15.0


class GeminiProvider(ProviderInterface):
    """
    Provider implementation for the Google Gemini API.
    """

    @staticmethod
    def parse_quota_error(error: Exception, error_body: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return parse_google_quota_error(error, error_body)

    def on_request_complete(
        self,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        """
        Apply per-key cooldown after rate-limit / quota errors.

        Google's free-tier API keys share per-project RPM limits (typically
        15 RPM).  A short cooldown (15s) keeps the key out of rotation
        briefly, but short enough that the 30s request deadline can wait
        for it to expire and retry — avoiding both wasteful retry storms
        and premature "all credentials exhausted" failures.
        """
        if success or error is None:
            return None

        error_type = getattr(error, "error_type", "")
        if error_type not in ("rate_limit", "quota_exceeded"):
            return None

        retry_after = getattr(error, "retry_after", None)
        cooldown = float(retry_after) if retry_after and retry_after > 0 else RATE_LIMIT_COOLDOWN

        return RequestCompleteResult(cooldown_override=cooldown)

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Google Gemini API.
        """
        try:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
            )
            response.raise_for_status()
            return [
                f"google/{model['name'].replace('models/', '')}"
                for model in response.json().get("models", [])
            ]
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
