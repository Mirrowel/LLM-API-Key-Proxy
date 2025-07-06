import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class GeminiProvider(ProviderInterface):
    """
    Provider implementation for the Google Gemini API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Google Gemini API.
        """
        try:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key}
            )
            response.raise_for_status()
            return [f"gemini/{model['name'].replace('models/', '')}" for model in response.json().get("models", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Gemini models: {e}")
            return []

    def convert_safety_settings(self, settings: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Converts generic safety settings to the Gemini-specific format.
        """
        if not settings:
            return []

        gemini_settings = []
        category_map = {
            "harassment": "HARM_CATEGORY_HARASSMENT",
            "hate_speech": "HARM_CATEGORY_HATE_SPEECH",
            "sexually_explicit": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "dangerous_content": "HARM_CATEGORY_DANGEROUS_CONTENT",
        }

        for generic_category, threshold in settings.items():
            if generic_category in category_map:
                gemini_settings.append({
                    "category": category_map[generic_category],
                    "threshold": threshold.upper()
                })
        
        return gemini_settings

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Handles the 'reasoning_effort' parameter and translates it to the Gemini-specific
        'thinking' parameter with a token budget. Supports a 'custom_reasoning_budget'
        flag to enable higher, model-specific token budgets.
        """
        custom_reasoning_budget = payload.pop("custom_reasoning_budget", False)
        reasoning_effort = payload.pop("reasoning_effort", None)

        if "thinking" in payload:
            return  # Do nothing if 'thinking' is already explicitly set

        if custom_reasoning_budget and reasoning_effort:
            if "gemini-2.5-pro" in model:
                budgets = {"low": 8192, "medium": 16384, "high": 32768}
            elif "gemini-2.5-flash" in model:
                budgets = {"low": 6144, "medium": 12288, "high": 24576}
            else:
                # Fallback to LiteLLM defaults for other models
                budgets = {"low": 1024, "medium": 2048, "high": 4096}
            
            budget = budgets.get(reasoning_effort)
            if budget is not None:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            elif reasoning_effort == "disable":
                payload["thinking"] = {"type": "enabled", "budget_tokens": 0}

        elif "gemini-2.5-pro" in model or "gemini-2.5-flash" in model:
            # Default behavior if no reasoning_effort is specified
            payload["thinking"] = {"type": "enabled", "budget_tokens": -1}
