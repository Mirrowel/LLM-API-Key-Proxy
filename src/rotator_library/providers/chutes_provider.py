import requests
import logging
from typing import List
from .provider_interface import ProviderInterface

class ChutesProvider(ProviderInterface):
    """
    Provider implementation for the chutes.ai API.
    """
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available models from the chutes.ai API.
        """
        try:
            response = requests.get(
                "https://llm.chutes.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"chutes/{model['id']}" for model in response.json().get("data", [])]
        except requests.RequestException as e:
            logging.error(f"Failed to fetch chutes.ai models: {e}")
            return []
