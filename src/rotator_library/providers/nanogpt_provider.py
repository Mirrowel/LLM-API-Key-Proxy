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

    # Quota groups for tracking daily and monthly limits
    # These are virtual models used to track subscription-level quota
    model_quota_groups = {
        "daily": ["_daily"],
        "monthly": ["_monthly"],
    }



    def __init__(self):
        self.model_definitions = ModelDefinitions()

        # Quota tracking cache
        self._subscription_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval = int(
            os.getenv("NANOGPT_QUOTA_REFRESH_INTERVAL", "300")
        )

        # Tier cache (credential -> tier name)
        self._tier_cache: Dict[str, str] = {}

        # Track discovered models for quota group sync
        self._discovered_models: set = set()

    # =========================================================================
    # USAGE TRACKING CONFIGURATION
    # =========================================================================

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for NanoGPT credentials.

        NanoGPT uses per_model mode to track usage at the model level,
        with daily and monthly quotas managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode
        """
        return {
            "mode": "per_model",
            "window_seconds": 86400,  # 24 hours (daily quota reset)
        }

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All NanoGPT models share the same subscription quota pool.
        - Real models + _daily: belong to "daily" group (enforces exhaustion)
        - _monthly: belongs to "monthly" group (separate limit tracking)

        Args:
            model: Model name

        Returns:
            Quota group name
        """
        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model

        # _monthly has its own group to prevent cross-syncing of limits
        if clean_model == "_monthly":
            return "monthly"

        # All other models (real models + _daily) share the daily quota
        return "daily"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get the canonical model(s) for quota group stats display.

        This returns only the virtual quota tracker models for display purposes.
        All real models share these quota pools via get_model_quota_group().

        Args:
            group: Quota group identifier

        Returns:
            List with the virtual quota tracker model
        """
        if group == "daily":
            # Only _daily for stats display - it has the baseline from API
            return ["_daily"]
        elif group == "monthly":
            return ["_monthly"]
        return []

    def get_quota_groups(self) -> List[str]:
        """
        Get the list of quota groups for this provider.

        Returns:
            List of quota group names
        """
        return ["daily", "monthly"]

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
                    # Track for quota group sync
                    self._discovered_models.add(model_id)

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
                # Track fallback models for quota group sync
                for model_id in NANOGPT_FALLBACK_MODELS:
                    self._discovered_models.add(model_id)

        # Also track static models for quota group sync
        for model in models:
            model_id = model.split("/")[-1] if "/" in model else model
            self._discovered_models.add(model_id)

        # Refresh subscription usage to get tier info (only if not already cached)
        if api_key not in self._tier_cache:
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

        Uses the mixin's refresh_subscription_usage method to avoid code duplication.

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
                    # Use mixin method for refresh (handles caching internally)
                    # Pass the shared client to respect concurrency control
                    usage_data = await self.refresh_subscription_usage(
                        api_key, credential_identifier=api_key, client=client
                    )

                    if usage_data.get("status") == "success":
                        # Update tier cache
                        state = usage_data.get("state", "inactive")
                        tier = self.get_tier_from_state(state)
                        self._tier_cache[api_key] = tier

                        # Extract quota data for daily and monthly limits
                        daily_data = usage_data.get("daily", {})
                        monthly_data = usage_data.get("monthly", {})
                        limits = usage_data.get("limits", {})

                        daily_limit = limits.get("daily", 0)
                        monthly_limit = limits.get("monthly", 0)
                        daily_remaining = daily_data.get("remaining", 0)
                        monthly_remaining = monthly_data.get("remaining", 0)

                        # Calculate remaining fractions
                        daily_fraction = daily_remaining / daily_limit if daily_limit > 0 else 1.0
                        monthly_fraction = monthly_remaining / monthly_limit if monthly_limit > 0 else 1.0

                        # Get reset timestamps
                        daily_reset_ts = daily_data.get("reset_at", 0)
                        monthly_reset_ts = monthly_data.get("reset_at", 0)

                        # Store daily quota baseline
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "nanogpt/_daily",
                            daily_fraction,
                            max_requests=daily_limit,
                            reset_timestamp=daily_reset_ts if daily_reset_ts > 0 else None,
                        )

                        # Store monthly quota baseline
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "nanogpt/_monthly",
                            monthly_fraction,
                            max_requests=monthly_limit,
                            reset_timestamp=monthly_reset_ts if monthly_reset_ts > 0 else None,
                        )

                        lib_logger.debug(
                            f"Updated NanoGPT quota baselines: "
                            f"daily={daily_remaining}/{daily_limit}, "
                            f"monthly={monthly_remaining}/{monthly_limit}"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh NanoGPT subscription usage: {e}"
                    )

        # Fetch all credentials in parallel using a shared client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
