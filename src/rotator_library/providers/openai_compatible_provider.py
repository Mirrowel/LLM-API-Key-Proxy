import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpenAICompatibleProvider(ProviderInterface):
    """
    Generic provider implementation for any OpenAI-compatible API.
    This provider can be configured via environment variables to support
    custom OpenAI-compatible endpoints without requiring code changes.
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment
        self.api_base = os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for OpenAI-compatible provider"
            )

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the OpenAI-compatible API.
        """
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [
                f"{self.provider_name}/{model['id']}"
                for model in response.json().get("data", [])
            ]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch models for {self.provider_name}: {e}")
            return []
        except Exception as e:
            lib_logger.error(
                f"Unexpected error fetching models for {self.provider_name}: {e}"
            )
            return []

    def has_custom_logic(self) -> bool:
        """
        Returns False since we want to use the standard litellm flow
        with just custom API base configuration.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns the standard Bearer token header for API key authentication.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
