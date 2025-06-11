import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class OpenAIProvider(ProviderInterface):
    """
    Provider implementation for the OpenAI API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the OpenAI API.
        """
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"openai/{model['id']}" for model in response.json().get("data", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch OpenAI models: {e}")
            return []
