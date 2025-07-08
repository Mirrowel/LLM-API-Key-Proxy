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
        Handles reasoning parameters for Gemini models, with three distinct paths:
        1. Applies a non-standard, high-value token budget if 'custom_reasoning_budget' is true.
        2. Leaves the 'reasoning_effort' parameter alone for LiteLLM to handle if it's present
           without the custom flag.
        3. Applies a default 'thinking' value for specific models if no other reasoning
           parameters are provided, ensuring they 'think' by default.
        """
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
                else: # Fallback for other models if the custom flag is still used
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
