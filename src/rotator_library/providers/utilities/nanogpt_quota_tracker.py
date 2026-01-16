"""
NanoGPT Quota Tracking Mixin

Provides quota tracking for the NanoGPT provider using their subscription usage API.
Unlike Gemini/Antigravity which track per-model quotas, NanoGPT tracks "usage units"
(successful operations) at the credential level with daily/monthly limits.

API Details (from https://docs.nano-gpt.com/api-reference/endpoint/subscription-usage):
- Endpoint: GET https://nano-gpt.com/api/subscription/v1/usage
- Auth: Authorization: Bearer <api_key> or x-api-key: <api_key>
- Response: { active, limits, daily, monthly, state, ... }

Required from provider:
    - self._get_api_key(credential_path) -> str
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# NanoGPT API base URL
NANOGPT_API_BASE = "https://nano-gpt.com"


class NanoGptQuotaTracker:
    """
    Mixin class providing quota tracking functionality for NanoGPT provider.

    This mixin adds the following capabilities:
    - Fetch subscription usage from the NanoGPT API
    - Track daily/monthly usage limits
    - Determine subscription tier from state field

    Usage:
        class NanoGptProvider(NanoGptQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._subscription_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    _subscription_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # SUBSCRIPTION USAGE API
    # =========================================================================

    async def fetch_subscription_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch subscription usage from the NanoGPT API.

        Args:
            api_key: NanoGPT API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "active": bool,
                "state": str,  # "active" | "grace" | "inactive"
                "limits": {"daily": int, "monthly": int},
                "daily": {
                    "used": int,
                    "remaining": int,
                    "percent_used": float,
                    "reset_at": float,  # Unix timestamp (seconds)
                },
                "monthly": {
                    "used": int,
                    "remaining": int,
                    "percent_used": float,
                    "reset_at": float,
                },
                "fetched_at": float,
            }
        """
        try:
            url = f"{NANOGPT_API_BASE}/api/subscription/v1/usage"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }

            # Use provided client or create a new one
            if client is not None:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                async with httpx.AsyncClient(timeout=30.0) as new_client:
                    response = await new_client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()

            # Parse response
            daily = data.get("daily", {})
            monthly = data.get("monthly", {})
            limits = data.get("limits", {})

            return {
                "status": "success",
                "error": None,
                "active": data.get("active", False),
                "state": data.get("state", "inactive"),
                "enforce_daily_limit": data.get("enforceDailyLimit", False),
                "limits": {
                    "daily": limits.get("daily", 0),
                    "monthly": limits.get("monthly", 0),
                },
                "daily": {
                    "used": daily.get("used", 0),
                    "remaining": daily.get("remaining", 0),
                    "percent_used": daily.get("percentUsed", 0.0),
                    # Convert epoch ms to seconds
                    "reset_at": daily.get("resetAt", 0) / 1000.0,
                },
                "monthly": {
                    "used": monthly.get("used", 0),
                    "remaining": monthly.get("remaining", 0),
                    "percent_used": monthly.get("percentUsed", 0.0),
                    "reset_at": monthly.get("resetAt", 0) / 1000.0,
                },
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
            lib_logger.warning(f"Failed to fetch NanoGPT subscription usage: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "active": False,
                "state": "unknown",
                "limits": {"daily": 0, "monthly": 0},
                "daily": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "monthly": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch NanoGPT subscription usage: {e}")
            return {
                "status": "error",
                "error": str(e),
                "active": False,
                "state": "unknown",
                "limits": {"daily": 0, "monthly": 0},
                "daily": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "monthly": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "fetched_at": time.time(),
            }

    def get_tier_from_state(self, state: str) -> str:
        """
        Map NanoGPT subscription state to tier name.

        Args:
            state: One of "active", "grace", "inactive"

        Returns:
            Tier name for priority mapping
        """
        state_to_tier = {
            "active": "subscription-active",
            "grace": "subscription-grace",
            "inactive": "no-subscription",
        }
        return state_to_tier.get(state, "no-subscription")

    def get_remaining_fraction(self, usage_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction from usage data.

        Uses daily limit by default, unless enforceDailyLimit is False
        (in which case only monthly matters).

        Args:
            usage_data: Response from fetch_subscription_usage()

        Returns:
            Remaining fraction (0.0 to 1.0)
        """
        limits = usage_data.get("limits", {})
        daily = usage_data.get("daily", {})

        daily_limit = limits.get("daily", 0)
        daily_remaining = daily.get("remaining", 0)

        if daily_limit <= 0:
            return 1.0  # No limit configured

        return min(1.0, max(0.0, daily_remaining / daily_limit))

    def get_reset_timestamp(self, usage_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the next reset timestamp from usage data.

        Args:
            usage_data: Response from fetch_subscription_usage()

        Returns:
            Unix timestamp when quota resets, or None
        """
        daily = usage_data.get("daily", {})
        reset_at = daily.get("reset_at", 0)
        return reset_at if reset_at > 0 else None

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_subscription_usage(
        self,
        api_key: str,
        credential_identifier: str,
    ) -> Dict[str, Any]:
        """
        Refresh and cache subscription usage for a credential.

        Args:
            api_key: NanoGPT API key
            credential_identifier: Identifier for caching

        Returns:
            Usage data from fetch_subscription_usage()
        """
        usage_data = await self.fetch_subscription_usage(api_key)

        if usage_data.get("status") == "success":
            self._subscription_cache[credential_identifier] = usage_data

            daily = usage_data.get("daily", {})
            limits = usage_data.get("limits", {})
            lib_logger.debug(
                f"NanoGPT subscription usage for {credential_identifier}: "
                f"daily={daily.get('remaining', 0)}/{limits.get('daily', 0)}, "
                f"state={usage_data.get('state')}"
            )

        return usage_data

    def get_cached_usage(self, credential_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get cached subscription usage for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Cached usage data or None
        """
        return self._subscription_cache.get(credential_identifier)

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
                        "daily": { ... },
                        "monthly": { ... },
                        "limits": { ... },
                    }
                },
                "summary": {
                    "total_credentials": int,
                    "active_subscriptions": int,
                },
                "timestamp": float,
            }
        """
        results = {}
        active_count = 0

        # Fetch quota for all credentials in parallel
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(identifier: str, api_key: str):
            async with semaphore:
                return identifier, await self.fetch_subscription_usage(api_key)

        tasks = [fetch_with_semaphore(ident, key) for ident, key in api_keys]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Quota fetch failed: {result}")
                continue

            identifier, usage_data = result

            if usage_data.get("active"):
                active_count += 1

            tier = self.get_tier_from_state(usage_data.get("state", "inactive"))

            results[identifier] = {
                "identifier": identifier,
                "tier": tier,
                "status": usage_data.get("status", "error"),
                "error": usage_data.get("error"),
                "active": usage_data.get("active", False),
                "state": usage_data.get("state"),
                "daily": usage_data.get("daily"),
                "monthly": usage_data.get("monthly"),
                "limits": usage_data.get("limits"),
                "remaining_fraction": self.get_remaining_fraction(usage_data),
                "fetched_at": usage_data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(api_keys),
                "active_subscriptions": active_count,
            },
            "timestamp": time.time(),
        }
