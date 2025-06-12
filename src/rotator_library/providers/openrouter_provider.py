import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class OpenRouterProvider(ProviderInterface):
    """
    Provider implementation for the OpenRouter API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the OpenRouter API.
        """
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"openrouter/{model['id']}" for model in response.json().get("data", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch OpenRouter models: {e}")
            return []
