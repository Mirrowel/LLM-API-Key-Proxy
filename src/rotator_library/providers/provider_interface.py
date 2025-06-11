from abc import ABC, abstractmethod
from typing import List, Any

class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, primarily for discovering
    available models.
    """

    @abstractmethod
    async def get_models(self, api_key: str) -> List[str]:
        """
        Fetches the list of available model names from the provider's API.

        Args:
            api_key: The API key required for authentication.

        Returns:
            A list of model name strings.
        """
        pass
