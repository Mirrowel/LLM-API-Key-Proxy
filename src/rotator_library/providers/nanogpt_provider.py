# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""NanoGPT — OpenAI-compatible aggregator with two extra transport faces (G8 final).

The ``speaks`` envelope declares the three faces over one credential:
pay-as-you-go chat (the default), their OpenAI-compatible Responses
surface, and the subscription pool, which lives on its own base and is
addressed as ``nanogpt:subscription/model``. Everything each face shares
with its protocol — endpoints, bearer auth, the ``/models`` listing and
its ``data[].id`` shape — inherits; only the subscription base is a real
override.

The ``nanogpt`` adapter owns the genuinely custom wire handling (the
reasoning-field normalization and the usage fold); the ``max_tokens``
length spelling is a declared ``model_rules`` row (the same declaration
lands on Chutes). The quota tracker, quota groups, usage windows, and
background refresh job below are provider-owned state and stay untouched.

Model listing is the shared, protocol-aware interface implementation; a
failed listing is an honest empty.
"""

import httpx
import logging
import os
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface
from .utilities.nanogpt_quota_tracker import NanoGptQuotaTracker
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class NanoGPTProvider(NanoGptQuotaTracker, ProviderInterface):
    """NanoGPT — OpenAI-compatible aggregator with a subscription face."""

    # Deposit exhaustion is a 402 (rotate, never retry); subscription
    # exhaustion is a 429 until reset. The ``nanogpt`` wire adapter owns
    # the length-parameter mapping and the reasoning-field spelling.

    # -- transport (the envelope) ---------------------------------------
    # First entry is the default face. The subscription pool is a second
    # openai_chat face with its own base override (the historical profile
    # name ``subscription`` is preserved for addressing stability); the
    # Responses face inherits its route from the protocol defaults.
    speaks = (
        ("chat", "openai_chat", {}),
        "responses",
        (
            "subscription",
            "openai_chat",
            {"base": "https://nano-gpt.com/api/subscription/v1"},
        ),
    )
    native_streaming_supported = True
    default_api_base = "https://nano-gpt.com/api/v1"
    adapter_names = ("nanogpt",)
    model_rules = (
        {
            "match": "*",
            # NanoGPT spells the length parameter ``max_tokens``; the
            # shared rename is declared here instead of riding provider
            # code (Chutes carries the same row).
            "rename": {"max_completion_tokens": "max_tokens"},
        },
    )
    # NOTE(for-removal): dies with the cost phase.
    skip_cost_calculation = True
    provider_env_name = "nanogpt"

    model_quota_groups = {
        "daily": ["_daily"],
        "monthly": ["_monthly"],
    }

    def _resolve_quota_api_base(self) -> str:
        base = (self.get_provider_api_base() or "").rstrip("/")
        for suffix in ("/api/v1", "/v1"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def __init__(self):
        super().__init__(api_base=self._resolve_quota_api_base())
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

        # Track subscription-only models (subject to daily/monthly limits)
        self._subscription_models: set = set()

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

        NanoGPT has two quota types:
        - Daily: Soft limit (2000/day) - display only, does NOT block
        - Monthly: Hard limit (60000/month) - BLOCKS when exhausted

        Real models belong to "monthly" so they're only blocked by the
        hard limit. The "daily" group is just for display.

        Args:
            model: Model name

        Returns:
            Quota group name
        """
        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model

        # _daily is for soft limit display only
        if clean_model == "_daily":
            return "daily"

        # Real models + _monthly belong to monthly (hard limit)
        return "monthly"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models belonging to a quota group.

        This is used by UsageManager.update_quota_baseline to sync
        request_count, baseline, and cooldowns across all group members.

        Args:
            group: Quota group identifier

        Returns:
            List of model names in the group
        """
        if group == "daily":
            # Daily is soft limit - only virtual tracker for display
            return ["_daily"]
        elif group == "monthly":
            # Monthly is hard limit - include subscription models for sync
            models = ["_monthly"]
            models.extend(list(self._subscription_models))
            return models
        return []

    def get_quota_groups(self) -> List[str]:
        """
        Get the list of quota groups for this provider.

        Returns:
            List of quota group names
        """
        return ["daily", "monthly"]

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

                        # Extract quota data — the live subscription-usage
                        # API reports token-count spellings
                        # (dailyInputTokens/weeklyInputTokens under limits
                        # and top-level usage counts), while older shapes
                        # carried remaining/reset_at fractions. Both are
                        # accepted; whichever carries data wins.
                        limits = usage_data.get("limits", {}) or {}
                        daily_data = usage_data.get("daily", {}) or {}

                        daily_limit = limits.get("dailyInputTokens") or limits.get("daily") or 0
                        weekly_limit = limits.get("weeklyInputTokens") or limits.get("monthly") or 0
                        if usage_data.get("dailyInputTokens") is not None:
                            # New shape: counts + limits, no fractions.
                            daily_used = int(usage_data.get("dailyInputTokens") or 0)
                            daily_reset_ts = 0
                        else:
                            daily_remaining = daily_data.get("remaining", 0)
                            daily_fraction = (
                                daily_remaining / daily_limit if daily_limit > 0 else 1.0
                            )
                            daily_used = (
                                int((1.0 - daily_fraction) * daily_limit)
                                if daily_limit > 0
                                else 0
                            )
                            daily_reset_ts = daily_data.get("reset_at", 0)

                        await usage_manager.update_quota_baseline(
                            api_key,
                            "nanogpt/_daily",
                            quota_max_requests=daily_limit,
                            quota_reset_ts=daily_reset_ts
                            if daily_reset_ts > 0
                            else None,
                            quota_used=daily_used,
                        )

                        # Weekly baseline (the new API reports weekly token
                        # limits; the legacy spelling was monthly).
                        if usage_data.get("weeklyInputTokens") is not None:
                            weekly_used = int(usage_data.get("weeklyInputTokens") or 0)
                            weekly_reset_ts = 0
                        else:
                            monthly_data = usage_data.get("monthly", {}) or {}
                            monthly_remaining = monthly_data.get("remaining", 0)
                            monthly_fraction = (
                                monthly_remaining / weekly_limit if weekly_limit > 0 else 1.0
                            )
                            weekly_reset_ts = monthly_data.get("reset_at", 0)
                            weekly_used = (
                                int((1.0 - monthly_fraction) * weekly_limit)
                                if weekly_limit > 0
                                else 0
                            )
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "nanogpt/_monthly",
                            quota_max_requests=weekly_limit,
                            quota_reset_ts=weekly_reset_ts
                            if weekly_reset_ts > 0
                            else None,
                            quota_used=weekly_used,
                        )

                        lib_logger.debug(
                            f"Updated NanoGPT quota baselines: "
                            f"daily={daily_used}/{daily_limit}, "
                            f"weekly={weekly_used}/{weekly_limit}"
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
