import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class ChutesProvider(ProviderInterface):
    """
    Provider implementation for the chutes.ai API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the chutes.ai API.
        """
        try:
            response = await client.get(
                "https://llm.chutes.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"chutes/{model['id']}" for model in response.json().get("data", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch chutes.ai models: {e}")
            return []