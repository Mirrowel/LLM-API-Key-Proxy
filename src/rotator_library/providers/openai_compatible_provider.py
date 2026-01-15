import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpenAICompatibleProvider(ProviderInterface):
    """
    Generic provider implementation for any OpenAI-compatible API.
    This provider can be configured via environment variables to support
    custom OpenAI-compatible endpoints without requiring code changes.
    Supports both dynamic model discovery and static model definitions.

    Environment variable pattern:
        <NAME>_CUSTOM_API_BASE - The API base URL (required)
        <NAME>_API_KEY         - The API key (optional for some providers)

    Example:
        MYSERVER_CUSTOM_API_BASE=http://localhost:8000/v1
        MYSERVER_API_KEY=sk-xxx
    """

    skip_cost_calculation: bool = True  # Skip cost calculation for custom providers

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment (using _CUSTOM_API_BASE pattern)
        self.api_base = os.getenv(f"{provider_name.upper()}_CUSTOM_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_CUSTOM_API_BASE is required for custom OpenAI-compatible provider"
            )

        # Initialize model definitions loader
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the OpenAI-compatible API.
        Combines dynamic discovery with static model definitions.
        """
        models = []

        # First, try to get static model definitions
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        if static_models:
            models.extend(static_models)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for {self.provider_name}"
            )

        # Then, try dynamic discovery to get additional models
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()

            dynamic_models = [
                f"{self.provider_name}/{model['id']}"
                for model in response.json().get("data", [])
                if model["id"] not in [m.split("/")[-1] for m in static_models]
            ]

            if dynamic_models:
                models.extend(dynamic_models)
                lib_logger.debug(
                    f"Discovered {len(dynamic_models)} additional models for {self.provider_name}"
                )

        except httpx.RequestError:
            # Silently ignore dynamic discovery errors
            pass
        except Exception:
            # Silently ignore dynamic discovery errors
            pass

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific model from static definitions or environment variables.

        Args:
            model_name: Model name (without provider prefix)

        Returns:
            Dictionary of model options
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

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
