import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class AnthropicProvider(ProviderInterface):
    """
    Provider implementation for the Anthropic API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the Anthropic API.
        """
        try:
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            response.raise_for_status()
            return [f"anthropic/{model['id']}" for model in response.json().get("data", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch Anthropic models: {e}")
            return []
