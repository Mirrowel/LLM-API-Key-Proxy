import httpx
import logging
from typing import List
import litellm
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class NvidiaProvider(ProviderInterface):
    skip_cost_calculation = True
    """
    Provider implementation for the NVIDIA API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the NVIDIA API.
        """
        try:
            response = await client.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            models = [f"nvidia_nim/{model['id']}" for model in response.json().get("data", [])]
            return models
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch NVIDIA models: {e}")
            return []
