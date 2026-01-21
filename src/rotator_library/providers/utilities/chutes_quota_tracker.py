# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Chutes Quota Tracking Mixin

Provides quota tracking for the Chutes provider using their quota usage API.
Chutes tracks credit-based quotas at the credential level with daily limits:
- 1 request = 1 credit consumed
- Daily quota reset at 00:00 UTC

API Details:
- Endpoint: GET https://api.chutes.ai/users/me/quota_usage/me
- Auth: Authorization: Bearer <api_key>
- Response: { quota: int, used: float }

Required from provider:
    - self._get_api_key(credential_path) -> str
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# Chutes API endpoint
CHUTES_QUOTA_API_URL = "https://api.chutes.ai/users/me/quota_usage/me"


class ChutesQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Chutes provider.

    This mixin adds the following capabilities:
    - Fetch quota usage from the Chutes API
    - Track daily credit limits
    - Determine subscription tier from quota value

    Usage:
        class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # Tier thresholds
    TIER_THRESHOLDS = {200: "legacy", 300: "base", 2000: "plus", 5000: "pro"}

    # =========================================================================
    # QUOTA USAGE API
    # =========================================================================

    async def fetch_quota_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the Chutes API.

        Args:
            api_key: Chutes API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "quota": int,  # Total daily quota
                "used": float,  # Credits consumed today
                "remaining": float,  # Credits remaining
                "remaining_fraction": float,  # 0.0 to 1.0
                "tier": str,  # legacy/base/plus/pro
                "reset_at": float,  # Unix timestamp (seconds)
                "fetched_at": float,
            }
        """
        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            if client is not None:
                response = await client.get(
                    CHUTES_QUOTA_API_URL, headers=headers, timeout=30
                )
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        CHUTES_QUOTA_API_URL, headers=headers, timeout=30
                    )
            response.raise_for_status()
            data = response.json()

            # Parse response with null safety
            quota = data.get("quota") or 0
            used = data.get("used") or 0.0
            remaining = max(0.0, quota - used)
            remaining_fraction = (remaining / quota) if quota > 0 else 0.0

            # Detect tier from quota value
            tier = self._get_tier_from_quota(quota)

            # Calculate next reset (00:00 UTC)
            reset_at = self._calculate_next_reset()

            return {
                "status": "success",
                "error": None,
                "quota": quota,
                "used": used,
                "remaining": remaining,
                "remaining_fraction": remaining_fraction,
                "tier": tier,
                "reset_at": reset_at,
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.text
                if error_body:
                    error_msg = f"{error_msg}: {error_body[:200]}"
            except Exception:
                pass
            lib_logger.warning(f"Failed to fetch Chutes quota: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "quota": 0,
                "used": 0.0,
                "remaining": 0.0,
                "remaining_fraction": 0.0,
                "tier": "base",
                "reset_at": 0,
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Chutes quota: {e}")
            return {
                "status": "error",
                "error": str(e),
                "quota": 0,
                "used": 0.0,
                "remaining": 0.0,
                "remaining_fraction": 0.0,
                "tier": "base",
                "reset_at": 0,
                "fetched_at": time.time(),
            }

    def _get_tier_from_quota(self, quota: int) -> str:
        """
        Map Chutes quota value to tier name.

        Args:
            quota: Daily quota value (200, 300, 2000, or 5000)

        Returns:
            Tier name (legacy, base, plus, or pro)
        """
        tier = self.TIER_THRESHOLDS.get(quota)
        if tier is None:
            lib_logger.warning(
                f"Unknown Chutes quota value {quota}, defaulting to 'base' tier. "
                f"Known values: {list(self.TIER_THRESHOLDS.keys())}"
            )
            return "base"
        return tier

    def _calculate_next_reset(self) -> float:
        """
        Calculate next 00:00 UTC reset timestamp.

        Returns:
            Unix timestamp when quota resets
        """
        now = datetime.now(timezone.utc)
        next_reset = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_reset.timestamp()

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
            Unix timestamp when quota resets, or None
        """
        reset_at = usage_data.get("reset_at", 0)
        return reset_at if reset_at > 0 else None

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
            api_key: Chutes API key
            credential_identifier: Identifier for caching

        Returns:
            Usage data from fetch_quota_usage()
        """
        usage_data = await self.fetch_quota_usage(api_key)

        if usage_data.get("status") == "success":
            self._quota_cache[credential_identifier] = usage_data

            lib_logger.debug(
                f"Chutes quota for {credential_identifier}: "
                f"{usage_data['remaining']:.1f}/{usage_data['quota']} remaining "
                f"({usage_data['remaining_fraction'] * 100:.1f}%), "
                f"tier={usage_data['tier']}"
            )

        return usage_data

    def get_cached_usage(self, credential_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get cached quota usage for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Cached usage data or None
        """
        return self._quota_cache.get(credential_identifier)

    async def get_all_quota_info(
        self,
        api_keys: List[Tuple[str, str]],  # List of (identifier, api_key) tuples
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        Args:
            api_keys: List of (identifier, api_key) tuples

        Returns:
            {
                "credentials": {
                    "identifier": {
                        "identifier": str,
                        "tier": str,
                        "status": "success" | "error",
                        "error": str | None,
                        "quota": int,
                        "used": float,
                        "remaining": float,
                        "remaining_fraction": float,
                    }
                },
                "summary": {
                    "total_credentials": int,
                    "total_quota": int,
                    "total_used": float,
                    "total_remaining": float,
                },
                "timestamp": float,
            }
        """
        results = {}
        total_quota = 0
        total_used = 0.0
        total_remaining = 0.0

        # Fetch quota for all credentials in parallel with shared client
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(
            identifier: str, api_key: str, client: httpx.AsyncClient
        ):
            async with semaphore:
                return identifier, await self.fetch_quota_usage(api_key, client)

        async with httpx.AsyncClient() as client:
            tasks = [
                fetch_with_semaphore(ident, key, client) for ident, key in api_keys
            ]
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Quota fetch failed: {result}")
                continue

            identifier, usage_data = result

            if usage_data.get("status") == "success":
                total_quota += usage_data.get("quota", 0)
                total_used += usage_data.get("used", 0.0)
                total_remaining += usage_data.get("remaining", 0.0)

            results[identifier] = {
                "identifier": identifier,
                "tier": usage_data.get("tier"),
                "status": usage_data.get("status", "error"),
                "error": usage_data.get("error"),
                "quota": usage_data.get("quota"),
                "used": usage_data.get("used"),
                "remaining": usage_data.get("remaining"),
                "remaining_fraction": usage_data.get("remaining_fraction"),
                "reset_at": usage_data.get("reset_at"),
                "fetched_at": usage_data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(api_keys),
                "total_quota": total_quota,
                "total_used": total_used,
                "total_remaining": total_remaining,
            },
            "timestamp": time.time(),
        }
