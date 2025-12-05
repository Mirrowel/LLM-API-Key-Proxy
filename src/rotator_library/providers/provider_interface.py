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
    
    # [NEW] Credential Prioritization System
    def get_credential_priority(self, credential: str) -> Optional[int]:
        """
        Returns the priority level for a credential.
        Lower numbers = higher priority (1 is highest).
        Returns None if provider doesn't use priorities.
        
        This allows providers to auto-detect credential tiers (e.g., paid vs free)
        and ensure higher-tier credentials are always tried first.
        
        Args:
            credential: The credential identifier (API key or path)
        
        Returns:
            Priority level (1-10) or None if no priority system
            
        Example:
            For Gemini CLI:
            - Paid tier credentials: priority 1 (highest)
            - Free tier credentials: priority 2
            - Unknown tier: priority 10 (lowest)
        """
        return None
    
    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        Returns the minimum priority tier required for a model.
        If a model requires priority 1, only credentials with priority <= 1 can use it.
        
        This allows providers to restrict certain models to specific credential tiers.
        For example, Gemini 3 models require paid-tier credentials.
        
        Args:
            model: The model name (with or without provider prefix)
        
        Returns:
            Minimum required priority level or None if no restrictions
            
        Example:
            For Gemini CLI:
            - gemini-3-*: requires priority 1 (paid tier only)
            - gemini-2.5-*: no restriction (None)
        """
        return None