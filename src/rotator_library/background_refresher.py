# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/background_refresher.py

import os
import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List

if TYPE_CHECKING:
    from .client import RotatingClient

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================
# These can be overridden via environment variables.

# OAuth token refresh interval in seconds
# Override: OAUTH_REFRESH_INTERVAL=<seconds>
DEFAULT_OAUTH_REFRESH_INTERVAL: int = 600  # 10 minutes

# Default interval for provider background jobs (quota refresh, etc.)
# Individual providers can override this in their get_background_job_config()
DEFAULT_BACKGROUND_JOB_INTERVAL: int = 300  # 5 minutes

# Whether to run background jobs immediately on start (before first interval)
DEFAULT_BACKGROUND_JOB_RUN_ON_START: bool = True


class BackgroundRefresher:
    """
    A background task manager that handles:
    1. Periodic OAuth token refresh for all providers
    2. Provider-specific background jobs (e.g., quota refresh) with independent timers

    Each provider can define its own background job via get_background_job_config()
    and run_background_job(). These run on their own schedules, independent of the
    OAuth refresh interval.
    """

    def __init__(self, client: "RotatingClient"):
        self._client = client
        self._task: Optional[asyncio.Task] = None
        self._provider_job_tasks: Dict[str, asyncio.Task] = {}  # provider -> task
        self._initialized = False
        try:
            interval_str = os.getenv(
                "OAUTH_REFRESH_INTERVAL", str(DEFAULT_OAUTH_REFRESH_INTERVAL)
            )
            self._interval = int(interval_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid OAUTH_REFRESH_INTERVAL '{interval_str}'. "
                f"Falling back to {DEFAULT_OAUTH_REFRESH_INTERVAL}s."
            )
            self._interval = DEFAULT_OAUTH_REFRESH_INTERVAL

    def start(self):
        """Starts the background refresh task."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            lib_logger.info(
                f"Background token refresher started. Check interval: {self._interval} seconds."
            )

    async def stop(self):
        """Stops all background tasks (main loop + provider jobs)."""
        # Cancel provider job tasks first
        for provider, task in self._provider_job_tasks.items():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                lib_logger.debug(f"Stopped background job for '{provider}'")

        self._provider_job_tasks.clear()

        # Cancel main task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            lib_logger.info("Background token refresher stopped.")

    async def _initialize_credentials(self):
        """
        Initialize all providers by loading credentials and persisted tier data.
        Called once before the main refresh loop starts.
        """
        if self._initialized:
            return

        api_summary = {}  # provider -> count
        oauth_summary = {}  # provider -> {"count": N, "tiers": {tier: count}}

        all_credentials = self._client.all_credentials
        oauth_providers = self._client.oauth_providers

        for provider, credentials in all_credentials.items():
            if not credentials:
                continue

            provider_plugin = self._client._get_provider_instance(provider)

            # Call initialize_credentials if provider supports it
            if provider_plugin and hasattr(provider_plugin, "initialize_credentials"):
                try:
                    await provider_plugin.initialize_credentials(credentials)
                except Exception as e:
                    lib_logger.error(
                        f"Error initializing credentials for provider '{provider}': {e}"
                    )

            # Build summary based on provider type
            if provider in oauth_providers:
                tier_breakdown = {}
                if provider_plugin and hasattr(
                    provider_plugin, "get_credential_tier_name"
                ):
                    for cred in credentials:
                        tier = provider_plugin.get_credential_tier_name(cred)
                        if tier:
                            tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1
                oauth_summary[provider] = {
                    "count": len(credentials),
                    "tiers": tier_breakdown,
                }
            else:
                api_summary[provider] = len(credentials)

        # Log 3-line summary
        total_providers = len(api_summary) + len(oauth_summary)
        total_credentials = sum(api_summary.values()) + sum(
            d["count"] for d in oauth_summary.values()
        )

        if total_providers > 0:
            lib_logger.info(
                f"Providers initialized: {total_providers} providers, {total_credentials} credentials"
            )

            # API providers line
            if api_summary:
                api_parts = [f"{p}:{c}" for p, c in sorted(api_summary.items())]
                lib_logger.info(f"  API: {', '.join(api_parts)}")

            # OAuth providers line with tier breakdown
            if oauth_summary:
                oauth_parts = []
                for provider, data in sorted(oauth_summary.items()):
                    if data["tiers"]:
                        tier_str = ", ".join(
                            f"{t}:{c}" for t, c in sorted(data["tiers"].items())
                        )
                        oauth_parts.append(f"{provider}:{data['count']} ({tier_str})")
                    else:
                        oauth_parts.append(f"{provider}:{data['count']}")
                lib_logger.info(f"  OAuth: {', '.join(oauth_parts)}")

        self._initialized = True

    def _start_provider_background_jobs(self):
        """
        Start independent background job tasks for providers that define them.

        Each provider with a get_background_job_config() that returns a config
        gets its own asyncio task running on its own schedule.
        """
        all_credentials = self._client.all_credentials

        for provider, credentials in all_credentials.items():
            if not credentials:
                lib_logger.debug(f"Skipping {provider} background job: no credentials")
                continue

            provider_plugin = self._client._get_provider_instance(provider)
            if not provider_plugin:
                lib_logger.debug(
                    f"Skipping {provider} background job: no provider instance"
                )
                continue

            # Check if provider has a background job
            if not hasattr(provider_plugin, "get_background_job_config"):
                lib_logger.debug(
                    f"Skipping {provider} background job: no get_background_job_config method"
                )
                continue

            config = provider_plugin.get_background_job_config()
            if not config:
                lib_logger.debug(f"Skipping {provider} background job: config is None")
                continue

            # Start the provider's background job task
            task = asyncio.create_task(
                self._run_provider_background_job(
                    provider, provider_plugin, credentials, config
                )
            )
            self._provider_job_tasks[provider] = task

            job_name = config.get("name", "background_job")
            interval = config.get("interval", DEFAULT_BACKGROUND_JOB_INTERVAL)
            lib_logger.info(f"Started {provider} {job_name} (interval: {interval}s)")

    async def _run_provider_background_job(
        self,
        provider_name: str,
        provider: Any,
        credentials: List[str],
        config: Dict[str, Any],
    ) -> None:
        """
        Independent loop for a single provider's background job.

        Args:
            provider_name: Name of the provider (for logging)
            provider: Provider plugin instance
            credentials: List of credential paths for this provider
            config: Background job configuration from get_background_job_config()
        """
        interval = config.get("interval", DEFAULT_BACKGROUND_JOB_INTERVAL)
        job_name = config.get("name", "background_job")
        run_on_start = config.get("run_on_start", DEFAULT_BACKGROUND_JOB_RUN_ON_START)

        # Run immediately on start if configured
        if run_on_start:
            try:
                await provider.run_background_job(
                    self._client.usage_manager, credentials
                )
                lib_logger.debug(f"{provider_name} {job_name}: initial run complete")
            except Exception as e:
                lib_logger.error(
                    f"Error in {provider_name} {job_name} (initial run): {e}"
                )

        # Main loop
        while True:
            try:
                await asyncio.sleep(interval)
                await provider.run_background_job(
                    self._client.usage_manager, credentials
                )
                lib_logger.debug(f"{provider_name} {job_name}: periodic run complete")
            except asyncio.CancelledError:
                lib_logger.debug(f"{provider_name} {job_name}: cancelled")
                break
            except Exception as e:
                lib_logger.error(f"Error in {provider_name} {job_name}: {e}")

    async def _run(self):
        """The main loop for OAuth token refresh."""
        # Initialize credentials (load persisted tiers) before starting
        await self._initialize_credentials()

        # Start provider-specific background jobs with their own timers
        self._start_provider_background_jobs()

        # Main OAuth refresh loop
        while True:
            try:
                oauth_configs = self._client.get_oauth_credentials()
                for provider, paths in oauth_configs.items():
                    provider_plugin = self._client._get_provider_instance(provider)
                    if provider_plugin and hasattr(
                        provider_plugin, "proactively_refresh"
                    ):
                        for path in paths:
                            try:
                                await provider_plugin.proactively_refresh(path)
                            except Exception as e:
                                lib_logger.error(
                                    f"Error during proactive refresh for '{path}': {e}"
                                )

                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Unexpected error in background refresher loop: {e}")
