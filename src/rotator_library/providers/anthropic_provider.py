import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class AnthropicProvider(ProviderInterface):
    """
    Provider implementation for the Anthropic API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Anthropic API.
        """
        try:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            response.raise_for_status()
            return [f"anthropic/{model['id']}" for model in response.json().get("data", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Anthropic models: {e}")
            return []
