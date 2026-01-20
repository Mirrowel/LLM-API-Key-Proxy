"""
Firmware.ai Provider with Quota Tracking

Provider implementation for the Firmware.ai API with 5-hour rolling window quota tracking.
Uses the FirmwareQuotaTracker mixin to fetch quota usage from their API.

Environment variables:
    FIRMWARE_API_BASE: API base URL (default: https://app.firmware.ai/api/v1)
    FIRMWARE_API_KEY: API key for authentication
    FIRMWARE_QUOTA_REFRESH_INTERVAL: Quota refresh interval in seconds (default: 300)
"""

import asyncio
import httpx
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .provider_interface import ProviderInterface
from .utilities.firmware_quota_tracker import FirmwareQuotaTracker

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

import logging

lib_logger = logging.getLogger("rotator_library")

# Concurrency limit for parallel quota fetches
QUOTA_FETCH_CONCURRENCY = 5


class FirmwareProvider(FirmwareQuotaTracker, ProviderInterface):
    """
    Provider implementation for the Firmware.ai API with quota tracking.
    """

    # Quota groups for tracking 5-hour rolling window limits
    # Uses a virtual model "_quota" for credential-level quota tracking
    model_quota_groups = {
        "firmware_global": ["_quota"],
    }

    def __init__(self, *args, **kwargs):
        """Initialize FirmwareProvider with quota tracking."""
        super().__init__(*args, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        try:
            self._quota_refresh_interval = int(
                os.environ.get("FIRMWARE_QUOTA_REFRESH_INTERVAL", "300")
            )
        except ValueError:
            lib_logger.warning(
                "Invalid FIRMWARE_QUOTA_REFRESH_INTERVAL value, using default 300"
            )
            self._quota_refresh_interval = 300

        # API base URL (default to Firmware.ai)
        self.api_base = os.environ.get(
            "FIRMWARE_API_BASE", "https://app.firmware.ai/api/v1"
        )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Firmware.ai models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "firmware_global"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models in a quota group.

        For Firmware.ai, we use a virtual model "_quota" to track the
        credential-level 5-hour rolling window quota.

        Args:
            group: Quota group name

        Returns:
            List of model names in the group
        """
        if group == "firmware_global":
            return ["_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Firmware.ai credentials.

        Firmware.ai uses per_model mode to track usage at the model level,
        with 5-hour rolling window quotas managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode and 5-hour window
        """
        return {
            "mode": "per_model",
            "window_seconds": 18000,  # 5 hours (5-hour rolling window)
            "field_name": "models",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Firmware.ai API.

        Args:
            api_key: Firmware.ai API key
            client: HTTP client

        Returns:
            List of model names prefixed with 'firmware/'
        """
        try:
            response = await client.get(
                f"{self.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"firmware/{model['id']}" for model in response.json().get("data", [])
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Firmware.ai models: {e}")
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
            "name": "firmware_quota_refresh",
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
                        reset_ts = usage_data.get("reset_at")

                        # Store baseline in usage manager
                        # Since Firmware.ai uses credential-level quota, we use a virtual model name
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "firmware/_quota",  # Virtual model for credential-level tracking
                            remaining_fraction,
                            # No max_requests - Firmware.ai doesn't expose this
                            reset_timestamp=reset_ts,
                        )

                        lib_logger.debug(
                            f"Updated Firmware.ai quota baseline: "
                            f"{remaining_fraction * 100:.1f}% remaining, "
                            f"active_window={usage_data.get('has_active_window', False)}"
                        )

                except Exception as e:
                    lib_logger.warning(f"Failed to refresh Firmware.ai quota usage: {e}")

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
