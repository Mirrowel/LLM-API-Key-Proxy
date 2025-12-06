from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import os
import httpx
import litellm


class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, including model
    discovery and custom API call handling for non-standard providers.
    """

    skip_cost_calculation: bool = False

    # Default rotation mode for this provider ("balanced" or "sequential")
    # - "balanced": Rotate credentials to distribute load evenly
    # - "sequential": Use one credential until exhausted, then switch to next
    default_rotation_mode: str = "balanced"

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

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handles the entire completion call for non-standard providers.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom acompletion."
        )

    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """Handles the entire embedding call for non-standard providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom aembedding."
        )

    def convert_safety_settings(
        self, settings: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
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

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Called at startup to initialize provider with all available credentials.

        Providers can override this to load cached tier data, discover priorities,
        or perform any other initialization needed before the first API request.

        This is called once during startup by the BackgroundRefresher before
        the main refresh loop begins.

        Args:
            credential_paths: List of credential file paths for this provider
        """
        pass

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the human-readable tier name for a credential.

        This is used for logging purposes to show which plan tier a credential belongs to.

        Args:
            credential: The credential identifier (API key or path)

        Returns:
            Tier name string (e.g., "free-tier", "paid-tier") or None if unknown
        """
        return None

    # =========================================================================
    # Sequential Rotation Support
    # =========================================================================

    @classmethod
    def get_rotation_mode(cls, provider_name: str) -> str:
        """
        Get the rotation mode for this provider.

        Checks ROTATION_MODE_{PROVIDER} environment variable first,
        then falls back to the class's default_rotation_mode.

        Args:
            provider_name: The provider name (e.g., "antigravity", "gemini_cli")

        Returns:
            "balanced" or "sequential"
        """
        env_key = f"ROTATION_MODE_{provider_name.upper()}"
        return os.getenv(env_key, cls.default_rotation_mode)

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a quota/rate-limit error and extract structured information.

        Providers should override this method to handle their specific error formats.
        This allows the error_handler to use provider-specific parsing when available,
        falling back to generic parsing otherwise.

        Args:
            error: The caught exception
            error_body: Optional raw response body string

        Returns:
            None if not a parseable quota error, otherwise:
            {
                "retry_after": int,  # seconds until quota resets
                "reason": str,       # e.g., "QUOTA_EXHAUSTED", "RATE_LIMITED"
                "reset_timestamp": str | None,  # ISO timestamp if available
            }
        """
        return None  # Default: no provider-specific parsing

    # TODO: Implement provider-specific quota reset schedules
    # Different providers have different quota reset periods:
    # - Most providers: Daily reset at a specific time
    # - Antigravity free tier: Weekly reset
    # - Antigravity paid tier: 5-hour rolling window
    #
    # Future implementation should add:
    # @classmethod
    # def get_quota_reset_behavior(cls) -> Dict[str, Any]:
    #     """
    #     Get provider-specific quota reset behavior.
    #     Returns:
    #         {
    #             "type": "daily" | "weekly" | "rolling",
    #             "reset_time_utc": "03:00",  # For daily/weekly
    #             "rolling_hours": 5,          # For rolling
    #         }
    #     """
    #     return {"type": "daily", "reset_time_utc": "03:00"}
