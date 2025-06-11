import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class CohereProvider(ProviderInterface):
    """
    Provider implementation for the Cohere API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the Cohere API.
        """
        try:
            response = requests.get(
                "https://api.cohere.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"cohere/{model['name']}" for model in response.json().get("models", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch Cohere models: {e}")
            return []
