import asyncio
import httpx
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .provider_interface import ProviderInterface
from .utilities.chutes_quota_tracker import ChutesQuotaTracker

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

# Create a local logger for this module
import logging

lib_logger = logging.getLogger("rotator_library")

# Concurrency limit for parallel quota fetches
QUOTA_FETCH_CONCURRENCY = 5


class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
    """
    Provider implementation for the chutes.ai API with quota tracking.
    """

    # Quota groups for tracking daily limits
    # Uses a virtual model "_quota" for credential-level quota tracking
    model_quota_groups = {
        "chutes_global": ["_quota"],
    }

    def __init__(self, *args, **kwargs):
        """Initialize ChutesProvider with quota tracking."""
        super().__init__(*args, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            os.environ.get("CHUTES_QUOTA_REFRESH_INTERVAL", "300")
        )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Chutes models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "chutes_global"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models in a quota group.

        For Chutes, we use a virtual model "_quota" to track the
        credential-level daily quota.

        Args:
            group: Quota group name

        Returns:
            List of model names in the group
        """
        if group == "chutes_global":
            return ["_quota"]
        return []

    def get_quota_groups(self) -> List[str]:
        """
        Get the list of quota groups for this provider.

        Returns:
            List of quota group names
        """
        return ["chutes_global"]

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Chutes credentials.

        Chutes uses per_model mode to track usage at the model level,
        with daily quotas managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode
        """
        return {
            "mode": "per_model",
            "window_seconds": 86400,  # 24 hours (daily quota reset)
            "field_name": "daily",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Chutes API.

        Args:
            api_key: Chutes API key
            client: HTTP client

        Returns:
            List of model names prefixed with 'chutes/'
        """
        try:
            response = await client.get(
                "https://llm.chutes.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"chutes/{model['id']}" for model in response.json().get("data", [])
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch chutes.ai models: {e}")
            return []

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota usage refresh.

        Returns:
            Background job configuration for quota refresh
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "chutes_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def refresh_single_credential(
            api_key: str, client: httpx.AsyncClient
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        # Update quota cache
                        self._quota_cache[api_key] = usage_data

                        # Calculate values for usage manager
                        remaining_fraction = usage_data.get("remaining_fraction", 0.0)
                        quota = usage_data.get("quota", 0)
                        reset_ts = usage_data.get("reset_at")

                        # Store baseline in usage manager
                        # Since Chutes uses credential-level quota, we use a virtual model name
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "chutes/_quota",  # Virtual model for credential-level tracking
                            remaining_fraction,
                            max_requests=quota,  # Max requests = quota (1 request = 1 credit)
                            reset_timestamp=reset_ts,
                        )

                        lib_logger.debug(
                            f"Updated Chutes quota baseline for credential: "
                            f"{usage_data['remaining']:.0f}/{quota} remaining "
                            f"({remaining_fraction * 100:.0f}%)"
                        )

                except Exception as e:
                    lib_logger.warning(f"Failed to refresh Chutes quota usage: {e}")

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
