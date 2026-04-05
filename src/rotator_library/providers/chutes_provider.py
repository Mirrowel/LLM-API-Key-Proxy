# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Chutes Provider

Provider for Chutes (https://chutes.ai).
OpenAI-compatible API with dollar-based subscription quota tracking.

Features:
- Dynamic model discovery from /v1/models endpoint
- Per-model pricing cached from models API for accurate cost tracking
- Server-side dollar-based usage tracking via /users/me/subscription_usage
- Monthly and 4-hour rolling window enforcement

Quota system:
Chutes subscription plans include a PAYGO-equivalent allowance of 5×
the subscription price.  Limits are enforced across both a monthly window
and a 4-hour rolling window.

    $10/mo  →  $50   monthly cap  →  $4.17  per 4 h
    $15/mo  →  $75   monthly cap  →  $1.25  per 4 h
    $50/mo  →  $250  monthly cap  →  $4.17  per 4 h
    $100/mo →  $500  monthly cap  →  $8.33  per 4 h

The /users/me/subscription_usage endpoint returns live dollar usage for
both windows, eliminating the need for local cost estimation.

Environment variables:
    CHUTES_API_KEY_1=<api_key>
    CHUTES_QUOTA_REFRESH_INTERVAL=300  # optional, seconds
"""

import asyncio
import httpx
import os
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.chutes_quota_tracker import ChutesQuotaTracker

lib_logger = logging.getLogger("rotator_library")

# Concurrency limit for parallel balance fetches
BALANCE_FETCH_CONCURRENCY = 5


class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
    """
    Provider implementation for the chutes.ai API with dollar-based quota tracking.

    All models share the same credential-level credit balance pool.
    Cost is calculated from per-model pricing cached from the /v1/models API.
    Usage caps are tracked server-side and fetched via subscription_usage API.
    """

    @staticmethod
    def parse_quota_error(error: Exception, error_body: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parse Chutes-specific quota/rate-limit errors.

        Chutes returns two distinct error types:
        - 429 "Infrastructure is at maximum capacity, try again later"
          → transient rate limit, retry after ~30s
        - 402 "Subscription usage cap exceeded. Please add balance to continue."
          → permanent per-key quota exhaustion
        """
        body = error_body
        if not body:
            if hasattr(error, 'response') and hasattr(error.response, 'text'):
                body = error.response.text
            elif hasattr(error, 'body'):
                body = str(error.body) if not isinstance(error.body, str) else error.body
            else:
                body = str(error)

        body_lower = body.lower() if body else ""

        status_code = None
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code

        if status_code == 429 or "infrastructure" in body_lower or "maximum capacity" in body_lower:
            return {
                "retry_after": None,
                "reason": "infrastructure_capacity",
            }

        if status_code == 402 or "subscription usage cap" in body_lower or "add balance" in body_lower:
            return {
                "retry_after": None,
                "reason": "subscription_cap_exceeded",
            }

        return None


    # Cost is calculated via our own calculate_cost() method using cached
    # per-model pricing from the Chutes API.  The executor calls
    # plugin.calculate_cost() first, then falls back to LiteLLM (which
    # has no Chutes pricing) — so we must NOT set skip_cost_calculation
    # to True, or the executor would skip our calculator too.
    skip_cost_calculation = False

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    # Enable environment variable overrides (e.g., QUOTA_GROUPS_CHUTES_GLOBAL)
    provider_env_name = "chutes"

    # Two quota groups so the TUI shows both enforcement windows:
    #   4h-credits($)  — 4-hour rolling window (tighter, rate-limiter)
    #   monthly($)     — monthly cap (overall budget)
    model_quota_groups = {
        "4h-credits($)": ["_balance_4h"],
        "monthly($)": ["_balance_monthly"],
    }

    # 4-hour rolling window — the tighter of the two enforced windows.
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=14400,  # 4 hours
            mode="per_model",
            description="Chutes 4-hour credit window",
            field_name="4h",
        )
    }

    def __init__(self, *args, **kwargs):
        """Initialize ChutesProvider with dollar-based quota tracking."""
        super().__init__(*args, **kwargs)

        # Model pricing cache: model_id → {input, output, input_cache_read}
        self._pricing_cache: Dict[str, Dict[str, float]] = {}

        # Balance cache: credential_identifier → balance data dict
        self._balance_cache: Dict[str, Dict[str, Any]] = {}

        self._quota_refresh_interval: int = int(
            os.environ.get("CHUTES_QUOTA_REFRESH_INTERVAL", "60")
        )

    # =========================================================================
    # USAGE TRACKING CONFIGURATION
    # =========================================================================

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Chutes credentials.

        Uses per_model mode with a 4-hour window to match the tighter
        rolling window enforced by the API.
        """
        return {
            "mode": "per_model",
            "window_seconds": 14400,  # 4 hours
        }

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Chutes models share the same credential-level credit balance pool.
        The primary (tighter) group is 4h-credits($).

        Args:
            model: Model name (ignored — all models share one balance)

        Returns:
            Quota group name
        """
        return "4h-credits($)"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Return all models belonging to the given quota group.

        Args:
            group: Quota group identifier

        Returns:
            List of model names in the group
        """
        if group == "4h-credits($)":
            return ["_balance_4h"]
        if group == "monthly($)":
            return ["_balance_monthly"]
        return []

    def get_quota_groups(self) -> List[str]:
        """Return the list of quota groups for this provider."""
        return ["4h-credits($)", "monthly($)"]

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Chutes API.

        Also caches per-model pricing for cost calculation.

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
            data = response.json()

            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                if model_id:
                    models.append(f"chutes/{model_id}")

                    # Cache pricing while we're at it
                    price_info = model_data.get("pricing") or model_data.get(
                        "price", {}
                    )
                    if price_info:
                        if "prompt" in price_info:
                            self._pricing_cache[model_id] = {
                                "input": float(price_info.get("prompt", 0.0)),
                                "output": float(price_info.get("completion", 0.0)),
                                "input_cache_read": float(
                                    price_info.get(
                                        "input_cache_read",
                                        float(price_info.get("prompt", 0.0)) * 0.5,
                                    )
                                ),
                            }
                        elif "input" in price_info:
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
                            cache_cost = float(
                                cache_data.get("usd", input_cost * 0.5)
                                if isinstance(cache_data, dict)
                                else (cache_data if cache_data else input_cost * 0.5)
                            )
                            self._pricing_cache[model_id] = {
                                "input": input_cost,
                                "output": output_cost,
                                "input_cache_read": cache_cost,
                            }

            if self._pricing_cache:
                lib_logger.info(
                    f"Cached pricing for {len(self._pricing_cache)} Chutes models"
                )

            return models
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch chutes.ai models: {e}")
            return []

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure periodic credit balance refresh."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "chutes_balance_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh credit balance for all credentials from the subscription API.

        Fetches live dollar usage from /users/me/subscription_usage and pushes
        both the 4-hour window (as the primary tracked window) and monthly cap
        data to the UsageManager.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(BALANCE_FETCH_CONCURRENCY)

        async def refresh_single(api_key: str, client: httpx.AsyncClient) -> None:
            async with semaphore:
                try:
                    balance_data = await self.refresh_balance(
                        api_key,
                        credential_identifier=api_key,
                        client=client,
                    )

                    if balance_data.get("status") == "success":
                        # API is authoritative for the sliding window.
                        # Usage can go DOWN as old spending ages out,
                        # so we must use force=True.
                        # Between refreshes, UsageManager adds +1 per request
                        # (slight overcounting) but next refresh corrects it.

                        four_hour_cap_cents = balance_data.get(
                            "four_hour_cap_cents", 0
                        )
                        four_hour_used_cents = balance_data.get(
                            "four_hour_used_cents", 0
                        )
                        four_hour_reset_ts = balance_data.get(
                            "four_hour_reset_ts"
                        )

                        await usage_manager.update_quota_baseline(
                            api_key,
                            "chutes/_balance_4h",
                            quota_max_requests=four_hour_cap_cents,
                            quota_reset_ts=four_hour_reset_ts,
                            quota_used=four_hour_used_cents,
                            force=True,  # API is authoritative (sliding window)
                        )

                        monthly_cap_cents = balance_data.get(
                            "monthly_cap_cents", 0
                        )
                        monthly_used_cents = balance_data.get(
                            "monthly_used_cents", 0
                        )
                        monthly_reset_ts = balance_data.get(
                            "monthly_reset_ts"
                        )

                        await usage_manager.update_quota_baseline(
                            api_key,
                            "chutes/_balance_monthly",
                            quota_max_requests=monthly_cap_cents,
                            quota_reset_ts=monthly_reset_ts,
                            quota_used=monthly_used_cents,
                            quota_group="monthly($)",
                            force=True,  # API is authoritative
                        )

                        monthly = balance_data.get("monthly", {})
                        four_hour = balance_data.get("four_hour", {})
                        lib_logger.debug(
                            f"Updated Chutes balance baseline: "
                            f"4h=${four_hour.get('usage', 0):.4f}/"
                            f"${four_hour.get('cap', 0):.2f} "
                            f"(resets: {four_hour.get('reset_at', 'N/A')}), "
                            f"monthly=${monthly.get('usage', 0):.4f}/"
                            f"${monthly.get('cap', 0):.2f} "
                            f"(resets: {monthly.get('reset_at', 'N/A')}), "
                            f"models_priced={len(self._pricing_cache)}"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh Chutes balance: {e}"
                    )

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [refresh_single(api_key, client) for api_key in credentials]
            await asyncio.gather(*tasks, return_exceptions=True)
