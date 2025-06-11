import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class GeminiProvider(ProviderInterface):
    """
    Provider implementation for the Google Gemini API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the Google Gemini API.
        """
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key}
            )
            response.raise_for_status()
            return [f"gemini/{model['name'].replace('models/', '')}" for model in response.json().get("models", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch Gemini models: {e}")
            return []
