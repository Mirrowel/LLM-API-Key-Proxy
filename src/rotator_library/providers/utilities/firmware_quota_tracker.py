"""
Firmware.ai Quota Tracking Mixin

Provides quota tracking for the Firmware.ai provider using their quota usage API.
Firmware.ai uses a 5-hour rolling window quota system where:
- `used` is already a ratio (0 to 1) indicating quota utilization
- `reset` is an ISO 8601 UTC timestamp, or null when no active window

API Details:
- Endpoint: GET https://app.firmware.ai/api/v1/quota
- Auth: Authorization: Bearer <api_key>
- Response: { used: float, reset: string|null }

Required from provider:
    - self.api_base: str (API base URL)
    - self._quota_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")


class FirmwareQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Firmware.ai provider.

    This mixin adds the following capabilities:
    - Fetch quota usage from the Firmware.ai API
    - Track 5-hour rolling window quota limits
    - Parse ISO 8601 reset timestamps

    Usage:
        class FirmwareProvider(FirmwareQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self.api_base: str = "https://app.firmware.ai/api/v1"
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    api_base: str
    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    def _get_quota_url(self) -> str:
        """Get the quota API URL based on configured api_base."""
        return f"{self.api_base.rstrip('/')}/quota"

    # =========================================================================
    # QUOTA USAGE API
    # =========================================================================

    async def fetch_quota_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the Firmware.ai API.

        Args:
            api_key: Firmware.ai API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "used": float,  # 0.0 to 1.0 (from API directly)
                "remaining_fraction": float,  # 1.0 - used
                "reset_at": float | None,  # Unix timestamp (seconds)
                "has_active_window": bool,  # True if reset is not null
                "fetched_at": float,
            }
        """
        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            quota_url = self._get_quota_url()

            if client is not None:
                response = await client.get(
                    quota_url, headers=headers, timeout=30
                )
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        quota_url, headers=headers, timeout=30
                    )
            response.raise_for_status()
            data = response.json()

            # Parse response - API returns ratio directly
            used_raw = data.get("used")
            # Validate used is numeric
            if not isinstance(used_raw, (int, float)):
                lib_logger.warning(
                    f"Firmware.ai quota API returned non-numeric 'used' value: {used_raw}"
                )
                used = 0.0
            else:
                used = float(used_raw)
            reset_iso = data.get("reset")

            # Calculate remaining (inverse of used), clamped to 0.0-1.0
            remaining_fraction = max(0.0, min(1.0, 1.0 - used))

            # Parse ISO 8601 reset timestamp
            reset_at = None
            if reset_iso is not None:
                reset_at = self._parse_iso_timestamp(reset_iso)
            # Only mark active window if we successfully parsed the timestamp
            has_active_window = reset_at is not None

            return {
                "status": "success",
                "error": None,
                "used": used,
                "remaining_fraction": remaining_fraction,
                "reset_at": reset_at,
                "has_active_window": has_active_window,
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            # Only log status code - error body may contain sensitive data
            error_msg = f"HTTP {e.response.status_code}"
            lib_logger.warning(f"Failed to fetch Firmware.ai quota: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "used": None,
                "remaining_fraction": None,  # None preserves cached value
                "reset_at": None,
                "has_active_window": False,
                "fetched_at": time.time(),
            }
        except Exception as e:
            # Log exception type only - message may contain sensitive data
            lib_logger.warning(f"Failed to fetch Firmware.ai quota: {type(e).__name__}")
            return {
                "status": "error",
                "error": type(e).__name__,
                "used": None,
                "remaining_fraction": None,  # None preserves cached value
                "reset_at": None,
                "has_active_window": False,
                "fetched_at": time.time(),
            }

    def _parse_iso_timestamp(self, iso_string: str) -> Optional[float]:
        """
        Parse ISO 8601 timestamp to Unix timestamp.

        Args:
            iso_string: ISO 8601 formatted timestamp (e.g., "2026-01-20T18:12:03.000Z")

        Returns:
            Unix timestamp in seconds, or None if parsing fails
        """
        try:
            # Handle 'Z' suffix by replacing with UTC offset
            if iso_string.endswith("Z"):
                iso_string = iso_string.replace("Z", "+00:00")

            dt = datetime.fromisoformat(iso_string)
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception as e:
            lib_logger.warning(f"Failed to parse ISO timestamp '{iso_string}': {e}")
            return None

    def get_remaining_fraction(self, usage_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction from usage data.

        Args:
            usage_data: Response from fetch_quota_usage()

        Returns:
            Remaining fraction (0.0 to 1.0)
        """
        return usage_data.get("remaining_fraction", 0.0)

    def get_reset_timestamp(self, usage_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the next reset timestamp from usage data.

        Args:
            usage_data: Response from fetch_quota_usage()

        Returns:
            Unix timestamp when quota resets, or None if no active window
        """
        return usage_data.get("reset_at")

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_quota_usage(
        self,
        api_key: str,
        credential_identifier: str,
    ) -> Dict[str, Any]:
        """
        Refresh and cache quota usage for a credential.

        Args:
            api_key: Firmware.ai API key
            credential_identifier: Identifier for caching

        Returns:
            Usage data from fetch_quota_usage()
        """
        usage_data = await self.fetch_quota_usage(api_key)

        if usage_data.get("status") == "success":
            self._quota_cache[credential_identifier] = usage_data

            lib_logger.debug(
                f"Firmware.ai quota for {credential_identifier}: "
                f"{usage_data['remaining_fraction'] * 100:.1f}% remaining, "
                f"active_window={usage_data['has_active_window']}"
            )

        return usage_data

    def get_cached_usage(self, credential_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get cached quota usage for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Copy of cached usage data or None
        """
        cached = self._quota_cache.get(credential_identifier)
        return dict(cached) if cached else None
