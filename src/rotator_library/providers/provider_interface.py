from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import httpx
import litellm

class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, including model
    discovery and custom API call handling for non-standard providers.
    """
    skip_cost_calculation: bool = False
    
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

    # [NEW] Add methods for providers that need to bypass litellm
    def has_custom_logic(self) -> bool:
        """
        Returns True if the provider implements its own acompletion/aembedding logic,
        bypassing the standard litellm call.
        """
        return False

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handles the entire completion call for non-standard providers.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement custom acompletion.")

    async def aembedding(self, client: httpx.AsyncClient, **kwargs) -> litellm.EmbeddingResponse:
        """Handles the entire embedding call for non-standard providers."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement custom aembedding.")
    
    def convert_safety_settings(self, settings: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        """
        Converts a generic safety settings dictionary to the provider-specific format.
        
        Args:
            settings: A dictionary with generic harm categories and thresholds.
            
        Returns:
            A list of provider-specific safety setting objects or None.
        """
        return None
    
    # [NEW] Add new methods for OAuth providers
    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        For OAuth providers, this method returns the Authorization header.
        For API key providers, this can be a no-op or raise NotImplementedError.
        """
        raise NotImplementedError("This provider does not support OAuth.")

    async def proactively_refresh(self, credential_path: str):
        """
        Proactively refreshes a token if it's nearing expiry.
        """
        pass