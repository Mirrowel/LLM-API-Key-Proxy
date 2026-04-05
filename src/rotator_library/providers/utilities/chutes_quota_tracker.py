# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Chutes Dollar-Based Quota Tracking Mixin

Provides quota tracking for the Chutes provider using a dollar-based credit
system.  Chutes subscription plans include a monthly PAYGO-equivalent
allowance (5× the subscription price) that is enforced across both a
monthly window and a 4-hour rolling window.

The ``/users/me/subscription_usage`` endpoint returns live dollar usage:

    {
        "subscription": true,
        "monthly_price": 10.0,
        "anchor_date": "2026-04-09T22:28:31",
        "effective_date": "2026-04-09T22:28:31",
        "monthly": {
            "usage": 0.02, "cap": 50.0, "remaining": 49.98,
            "reset_at": "2026-05-09T22:28:31+00:00"
        },
        "four_hour": {
            "usage": 0.01, "cap": 4.17, "remaining": 4.16,
            "reset_at": "2026-04-23T08:00:00+00:00"
        }
    }

The ``reset_at`` fields provide exact timestamps for when each window
resets, eliminating the need for manual configuration of reset dates.
The monthly reset is anchored to the subscription start date.

Cost per request is calculated from per-model pricing returned by the
Chutes ``/v1/models`` endpoint (USD per million tokens for input/output).

Required from provider:
    - self._pricing_cache: Dict[str, Dict[str, float]] = {}
    - self._balance_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# Chutes models endpoint for pricing discovery
CHUTES_MODELS_URL = "https://llm.chutes.ai/v1/models"

# Subscription usage endpoint (dollar-based, replaces legacy quota endpoint)
CHUTES_SUBSCRIPTION_USAGE_URL = "https://api.chutes.ai/users/me/subscription_usage"

# Legacy quota endpoint (fallback only)
CHUTES_LEGACY_QUOTA_URL = "https://api.chutes.ai/users/me/quota_usage/me"

# Scale factor: dollars → integer cents for UsageManager compatibility
CENTS_PER_DOLLAR = 100


class ChutesQuotaTracker:
    """
    Mixin class providing dollar-based quota tracking for the Chutes provider.

    This mixin adds:
    - Model pricing cache from the Chutes /v1/models API
    - Dollar-based cost calculation per request
    - Credit balance tracking via /users/me/subscription_usage
    - Both monthly and 4-hour rolling window enforcement

    Usage:
        class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._pricing_cache: Dict[str, Dict[str, float]] = {}
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300
    """

    # Type hints for attributes from provider
    _pricing_cache: Dict[str, Dict[str, float]]
    _balance_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # MODEL PRICING
    # =========================================================================

    async def fetch_model_pricing(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch per-model pricing from the Chutes /v1/models API.

        Args:
            api_key: Chutes API key
            client: Optional HTTP client for connection reuse

        Returns:
            Dict mapping model_id → {
                "input": float,   # USD per 1M input tokens
                "output": float,  # USD per 1M output tokens
                "input_cache_read": float,  # USD per 1M cached input tokens
            }
        """
        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            if client is not None:
                response = await client.get(
                    CHUTES_MODELS_URL, headers=headers, timeout=30
                )
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        CHUTES_MODELS_URL, headers=headers, timeout=30
                    )
            response.raise_for_status()
            data = response.json()

            pricing: Dict[str, Dict[str, float]] = {}
            for model in data.get("data", []):
                model_id = model.get("id", "")
                price_info = model.get("pricing") or model.get("price", {})
                if not model_id or not price_info:
                    continue

                # pricing field uses "prompt"/"completion" keys
                # price field uses nested "input"/"output" with "usd" sub-key
                if "prompt" in price_info:
                    # pricing format: {"prompt": 0.08, "completion": 0.24, ...}
                    input_cost = float(price_info.get("prompt", 0.0))
                    output_cost = float(price_info.get("completion", 0.0))
                    cache_read_cost = float(
                        price_info.get("input_cache_read", input_cost * 0.5)
                    )
                elif "input" in price_info:
                    # price format: {"input": {"usd": 0.08}, "output": {"usd": 0.24}}
                    input_data = price_info.get("input", {})
                    output_data = price_info.get("output", {})
                    cache_data = price_info.get("input_cache_read", {})
                    input_cost = float(
                        input_data.get("usd", 0.0)
                        if isinstance(input_data, dict)
                        else input_data
                    )
                    output_cost = float(
                        output_data.get("usd", 0.0)
                        if isinstance(output_data, dict)
                        else output_data
                    )
                    cache_read_cost = float(
                        cache_data.get("usd", input_cost * 0.5)
                        if isinstance(cache_data, dict)
                        else (cache_data if cache_data else input_cost * 0.5)
                    )
                else:
                    continue

                pricing[model_id] = {
                    "input": input_cost,
                    "output": output_cost,
                    "input_cache_read": cache_read_cost,
                }

            lib_logger.debug(f"Cached pricing for {len(pricing)} Chutes models")
            return pricing

        except Exception as e:
            lib_logger.warning(f"Failed to fetch Chutes model pricing: {e}")
            return {}

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        """
        Calculate the dollar cost of a request using cached model pricing.

        Args:
            model: Full model name (e.g. "chutes/Qwen/Qwen3-32B")
            prompt_tokens: Uncached prompt/input tokens
            completion_tokens: Total completion/output tokens (incl. thinking)
            cache_read_tokens: Tokens read from cache
            cache_creation_tokens: Tokens written to cache (unused by Chutes API currently but supported for parity)

        Returns:
            Cost in USD
        """
        # Strip provider prefix to get the Chutes model ID
        model_id = model.split("/", 1)[-1] if "/" in model else model

        pricing = self._pricing_cache.get(model_id)
        if not pricing:
            # Try without any prefix
            for cached_id, cached_pricing in self._pricing_cache.items():
                if cached_id.endswith(model_id) or model_id.endswith(cached_id):
                    pricing = cached_pricing
                    break

        if not pricing:
            lib_logger.debug(
                f"No cached pricing for Chutes model {model_id}, cost=0.0"
            )
            return 0.0

        # Prices are per 1M tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0.0)
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0.0)
        
        # Apply cached token pricing if available (default to 50% of input rate if not specified)
        cache_rate = pricing.get("input_cache_read", pricing.get("input", 0.0) * 0.5)
        cache_cost = (cache_read_tokens / 1_000_000) * cache_rate

        return input_cost + output_cost + cache_cost

    # =========================================================================
    # SUBSCRIPTION USAGE (NEW API)
    # =========================================================================

    async def fetch_subscription_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch dollar-based subscription usage from the Chutes API.

        Endpoint: GET /users/me/subscription_usage

        Returns data like:
            {
                "subscription": true,
                "monthly_price": 10.0,
                "monthly": {"usage": 0.02, "cap": 50.0, "remaining": 49.98},
                "four_hour": {"usage": 0.01, "cap": 4.17, "remaining": 4.16}
            }

        Args:
            api_key: Chutes API key
            client: Optional HTTP client for connection reuse

        Returns:
            Parsed subscription usage data, or error dict on failure
        """
        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            if client is not None:
                response = await client.get(
                    CHUTES_SUBSCRIPTION_USAGE_URL, headers=headers, timeout=30
                )
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        CHUTES_SUBSCRIPTION_USAGE_URL, headers=headers, timeout=30
                    )

            response.raise_for_status()
            data = response.json()

            # Validate expected structure
            if "monthly" not in data or "four_hour" not in data:
                lib_logger.warning(
                    f"Chutes subscription_usage response missing expected fields: "
                    f"{list(data.keys())}"
                )
                return {
                    "status": "error",
                    "error": "unexpected response format",
                    "raw": data,
                    "fetched_at": time.time(),
                }

            return {
                "status": "success",
                "subscription": data.get("subscription", False),
                "monthly_price": data.get("monthly_price", 0.0),
                "anchor_date": data.get("anchor_date"),
                "effective_date": data.get("effective_date"),
                "monthly": {
                    "usage": data["monthly"].get("usage", 0.0),
                    "cap": data["monthly"].get("cap", 0.0),
                    "remaining": data["monthly"].get("remaining", 0.0),
                    "reset_at": data["monthly"].get("reset_at"),
                },
                "four_hour": {
                    "usage": data["four_hour"].get("usage", 0.0),
                    "cap": data["four_hour"].get("cap", 0.0),
                    "remaining": data["four_hour"].get("remaining", 0.0),
                    "reset_at": data["four_hour"].get("reset_at"),
                },
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            lib_logger.warning(
                f"Failed to fetch Chutes subscription usage: HTTP {status}"
            )
            return {
                "status": "error",
                "error": f"HTTP {status}",
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Chutes subscription usage: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fetched_at": time.time(),
            }

    # =========================================================================
    # TIMESTAMP PARSING
    # =========================================================================

    @staticmethod
    def _parse_reset_at(reset_at_str: Optional[str]) -> Optional[float]:
        """
        Parse an ISO 8601 reset_at string from the Chutes API to a Unix timestamp.

        Handles formats like:
            "2026-05-09T22:28:31+00:00"
            "2026-04-23T08:00:00+00:00"

        Args:
            reset_at_str: ISO 8601 datetime string, or None

        Returns:
            Unix timestamp (float), or None if parsing fails
        """
        if not reset_at_str:
            return None

        try:
            dt = datetime.fromisoformat(reset_at_str)
            # Ensure timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError) as e:
            lib_logger.warning(
                f"Failed to parse Chutes reset_at timestamp '{reset_at_str}': {e}"
            )
            return None

    # =========================================================================
    # BALANCE TRACKING
    # =========================================================================

    def get_remaining_fraction(self, balance_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction from balance data.

        Uses the tighter of the two windows (monthly vs 4-hour).

        Args:
            balance_data: Cached balance data

        Returns:
            Remaining fraction (0.0 to 1.0)
        """
        monthly = balance_data.get("monthly", {})
        four_hour = balance_data.get("four_hour", {})

        monthly_cap = monthly.get("cap", 0)
        four_hour_cap = four_hour.get("cap", 0)

        monthly_frac = (
            monthly.get("remaining", 0) / monthly_cap if monthly_cap > 0 else 1.0
        )
        four_hour_frac = (
            four_hour.get("remaining", 0) / four_hour_cap if four_hour_cap > 0 else 1.0
        )

        # Return the tighter constraint
        return min(monthly_frac, four_hour_frac)

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_balance(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Refresh the pricing cache and fetch live balance from subscription API.

        Args:
            api_key: Chutes API key
            credential_identifier: Identifier for caching
            client: Optional HTTP client for connection reuse

        Returns:
            Balance data dict with monthly/four_hour usage from API
        """
        # Refresh model pricing cache
        pricing = await self.fetch_model_pricing(api_key, client)
        if pricing:
            self._pricing_cache.update(pricing)

        # Fetch live subscription usage
        sub_usage = await self.fetch_subscription_usage(api_key, client)

        if sub_usage.get("status") == "success":
            monthly = sub_usage.get("monthly", {})
            four_hour = sub_usage.get("four_hour", {})

            # Parse reset_at ISO timestamps to Unix timestamps
            monthly_reset_ts = self._parse_reset_at(monthly.get("reset_at"))
            four_hour_reset_ts = self._parse_reset_at(four_hour.get("reset_at"))

            balance_data = {
                "status": "success",
                "subscription": sub_usage.get("subscription", False),
                "monthly_price": sub_usage.get("monthly_price", 0.0),
                "anchor_date": sub_usage.get("anchor_date"),
                "effective_date": sub_usage.get("effective_date"),
                "monthly": monthly,
                "four_hour": four_hour,
                # Cents-based values for UsageManager compatibility
                # Use round() to avoid truncating sub-cent usage to 0
                "monthly_cap_cents": int(round(
                    monthly.get("cap", 0) * CENTS_PER_DOLLAR
                )),
                "monthly_used_cents": int(round(
                    monthly.get("usage", 0) * CENTS_PER_DOLLAR
                )),
                "four_hour_cap_cents": int(round(
                    four_hour.get("cap", 0) * CENTS_PER_DOLLAR
                )),
                "four_hour_used_cents": int(round(
                    four_hour.get("usage", 0) * CENTS_PER_DOLLAR
                )),
                # Unix timestamps for UsageManager quota_reset_ts
                "monthly_reset_ts": monthly_reset_ts,
                "four_hour_reset_ts": four_hour_reset_ts,
                # Raw dollar values for precise logging
                "monthly_usage_dollars": monthly.get("usage", 0.0),
                "four_hour_usage_dollars": four_hour.get("usage", 0.0),
                "pricing_models_cached": len(self._pricing_cache),
                "fetched_at": time.time(),
            }
        else:
            # API failed — return error but keep cached pricing
            balance_data = {
                "status": "error",
                "error": sub_usage.get("error", "unknown"),
                "pricing_models_cached": len(self._pricing_cache),
                "fetched_at": time.time(),
            }

        self._balance_cache[credential_identifier] = balance_data

        if balance_data.get("status") == "success":
            monthly = balance_data.get("monthly", {})
            four_hour = balance_data.get("four_hour", {})
            lib_logger.debug(
                f"Chutes balance refresh for {credential_identifier}: "
                f"monthly=${monthly.get('usage', 0):.4f}/${monthly.get('cap', 0):.2f}, "
                f"4h=${four_hour.get('usage', 0):.4f}/${four_hour.get('cap', 0):.2f}, "
                f"models_priced={len(self._pricing_cache)}"
            )

        return balance_data

    def get_cached_balance(
        self, credential_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached balance data for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Cached balance data or None
        """
        return self._balance_cache.get(credential_identifier)

    async def get_all_balance_info(
        self,
        api_keys: List[Tuple[str, str]],  # List of (identifier, api_key) tuples
    ) -> Dict[str, Any]:
        """
        Get balance info for all credentials.

        Args:
            api_keys: List of (identifier, api_key) tuples

        Returns:
            {
                "credentials": { identifier: { ... } },
                "summary": { ... },
                "timestamp": float,
            }
        """
        results: Dict[str, Any] = {}

        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(
            identifier: str, api_key: str, client: httpx.AsyncClient
        ) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                data = await self.refresh_balance(api_key, identifier, client)
                return identifier, data

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                fetch_with_semaphore(ident, key, client) for ident, key in api_keys
            ]
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Chutes balance fetch failed: {result}")
                continue

            identifier, data = result
            results[identifier] = {
                "identifier": identifier,
                "status": data.get("status", "error"),
                "monthly": data.get("monthly"),
                "four_hour": data.get("four_hour"),
                "pricing_models_cached": data.get("pricing_models_cached"),
                "fetched_at": data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(api_keys),
            },
            "timestamp": time.time(),
        }
