import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class MistralProvider(ProviderInterface):
    """
    Provider implementation for the Mistral API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the Mistral API.
        """
        try:
            response = requests.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"mistral/{model['id']}" for model in response.json().get("data", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch Mistral models: {e}")
            return []
