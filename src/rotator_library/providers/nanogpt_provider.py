"""
NanoGPT Provider

Provider for NanoGPT API (https://nano-gpt.com).
OpenAI-compatible API with subscription-based usage tracking.

Features:
- Dynamic model discovery from /v1/models endpoint
- Environment variable model override (NANOGPT_MODELS)
- Subscription usage monitoring via /api/subscription/v1/usage
- Tier-based credential prioritization

Usage units:
NanoGPT tracks "usage units" (successful operations) rather than tokens.
All models share a daily/monthly usage pool at the credential level.
"""

import asyncio
import httpx
import os
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.nanogpt_quota_tracker import NanoGptQuotaTracker
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# NanoGPT API base URL
NANOGPT_API_BASE = "https://nano-gpt.com"

# Concurrency limit for parallel quota fetches
QUOTA_FETCH_CONCURRENCY = 5

# Fallback models if API discovery fails and no env override
NANOGPT_FALLBACK_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


class NanoGptProvider(NanoGptQuotaTracker, ProviderInterface):
    """
    Provider for NanoGPT API.

    Supports subscription-based usage tracking with daily/monthly limits.
    All models share the same usage pool at the credential level.
    """

    # Skip cost calculation - NanoGPT uses "usage units", not tokens
    skip_cost_calculation = True

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    provider_env_name = "nanogpt"

    # Tier priorities based on subscription state
    # Active subscriptions get highest priority
    tier_priorities = {
        "subscription-active": 1,  # Active subscription
        "subscription-grace": 2,   # Grace period (subscription lapsed but still has access)
        "no-subscription": 3,      # No active subscription (pay-as-you-go only)
    }
    default_tier_priority = 3

    # =========================================================================
    # USAGE TRACKING CONFIGURATION
    # =========================================================================

    # Daily quota resets at UTC midnight
    # NanoGPT tracks usage at credential level (all models share the pool)
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=24 * 60 * 60,  # 24 hours
            mode="credential",  # All models share daily quota
            description="Daily subscription quota (UTC midnight reset)",
            field_name="daily",
        ),
    }

    def __init__(self):
        self.model_definitions = ModelDefinitions()

        # Set the API base for litellm routing via existing _CUSTOM_API_BASE pattern
        # This allows the client's existing infrastructure to handle routing
        if not os.getenv("NANOGPT_CUSTOM_API_BASE"):
            os.environ["NANOGPT_CUSTOM_API_BASE"] = NANOGPT_API_BASE + "/v1"

        # Quota tracking cache
        self._subscription_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval = int(
            os.getenv("NANOGPT_QUOTA_REFRESH_INTERVAL", "300")
        )

        # Tier cache (credential -> tier name)
        self._tier_cache: Dict[str, str] = {}

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All NanoGPT models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "nanogpt_global"

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns NanoGPT models from:
        1. Environment variable (NANOGPT_MODELS) - priority
        2. Dynamic discovery from API
        3. Hardcoded fallback list

        Also refreshes subscription usage to determine tier.
        """
        models = []
        seen_ids = set()

        # Source 1: Environment variable models (via NANOGPT_MODELS)
        static_models = self.model_definitions.get_all_provider_models("nanogpt")
        if static_models:
            for model in static_models:
                model_id = model.split("/")[-1] if "/" in model else model
                models.append(model)
                seen_ids.add(model_id)
            lib_logger.debug(f"Loaded {len(static_models)} static models for nanogpt")

        # Source 2: Dynamic discovery from API
        try:
            response = await client.get(
                f"{NANOGPT_API_BASE}/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            dynamic_count = 0
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id and model_id not in seen_ids:
                    # Skip auto-model variants - these are internal routing models
                    if model_id.startswith("auto-model"):
                        continue
                    models.append(f"nanogpt/{model_id}")
                    seen_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} models for nanogpt from API"
                )

        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery failed for nanogpt: {e}")

            # Source 3: Fallback to hardcoded models if nothing discovered
            if not models:
                for model_id in NANOGPT_FALLBACK_MODELS:
                    if model_id not in seen_ids:
                        models.append(f"nanogpt/{model_id}")
                        seen_ids.add(model_id)
                lib_logger.debug(
                    f"Using {len(NANOGPT_FALLBACK_MODELS)} fallback models for nanogpt"
                )

        # Refresh subscription usage to get tier info
        await self._refresh_tier_from_api(api_key)

        return models

    # =========================================================================
    # TIER MANAGEMENT
    # =========================================================================

    async def _refresh_tier_from_api(self, api_key: str) -> Optional[str]:
        """
        Refresh subscription status and cache the tier.

        Args:
            api_key: NanoGPT API key

        Returns:
            Tier name or None if fetch failed
        """
        usage_data = await self.fetch_subscription_usage(api_key)

        if usage_data.get("status") == "success":
            state = usage_data.get("state", "inactive")
            tier = self.get_tier_from_state(state)
            self._tier_cache[api_key] = tier

            daily = usage_data.get("daily", {})
            limits = usage_data.get("limits", {})
            lib_logger.info(
                f"NanoGPT subscription: state={state}, "
                f"daily={daily.get('remaining', 0)}/{limits.get('daily', 0)}"
            )
            return tier

        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the tier name for a credential.

        Uses cached subscription state from API refresh.

        Args:
            credential: The API key

        Returns:
            Tier name or None if not yet discovered
        """
        return self._tier_cache.get(credential)

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic subscription usage refresh.

        Returns:
            Background job configuration
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "nanogpt_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh subscription usage for all credentials in parallel.

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
                    usage_data = await self.fetch_subscription_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        # Update tier cache
                        state = usage_data.get("state", "inactive")
                        tier = self.get_tier_from_state(state)
                        self._tier_cache[api_key] = tier

                        # Update subscription cache
                        self._subscription_cache[api_key] = usage_data

                        # Calculate remaining fraction for quota tracking
                        remaining = self.get_remaining_fraction(usage_data)
                        reset_ts = self.get_reset_timestamp(usage_data)

                        # Store baseline in usage manager
                        # Since NanoGPT uses credential-level quota, we use a special model key
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "nanogpt/_subscription",  # Virtual model for credential-level tracking
                            remaining,
                            max_requests=usage_data.get("limits", {}).get("daily", 0),
                            reset_timestamp=reset_ts,
                        )

                        lib_logger.debug(
                            f"Updated NanoGPT quota baseline: "
                            f"{usage_data.get('daily', {}).get('remaining', 0)}/"
                            f"{usage_data.get('limits', {}).get('daily', 0)} remaining"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh NanoGPT subscription usage: {e}"
                    )

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
