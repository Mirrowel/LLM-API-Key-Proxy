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
        Adds a default thinking parameter for specific Gemini models if not already present.
        """
        if model in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"] and "thinking" not in payload and "reasoning_effort" not in payload:
            payload["thinking"] = {"type": "enabled", "budget_tokens": -1}
