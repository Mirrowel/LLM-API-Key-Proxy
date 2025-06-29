from abc import ABC, abstractmethod
from typing import List
import httpx

class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, primarily for discovering
    available models.
    """

    @abstractmethod
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available model names from the provider's API.

        Args:
            api_key: The API key required for authentication.
            client: An httpx.AsyncClient instance for making requests.

        Returns:
            A list of model name strings.
        """
        pass
