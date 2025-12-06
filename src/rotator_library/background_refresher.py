# src/rotator_library/background_refresher.py

import os
import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client import RotatingClient

lib_logger = logging.getLogger("rotator_library")


class BackgroundRefresher:
    """
    A background task that periodically checks and refreshes OAuth tokens
    to ensure they remain valid.
    """

    def __init__(self, client: "RotatingClient"):
        try:
            interval_str = os.getenv("OAUTH_REFRESH_INTERVAL", "600")
            self._interval = int(interval_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid OAUTH_REFRESH_INTERVAL '{interval_str}'. Falling back to 600s."
            )
            self._interval = 600
        self._client = client
        self._task: Optional[asyncio.Task] = None
        self._initialized = False

    def start(self):
        """Starts the background refresh task."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            lib_logger.info(
                f"Background token refresher started. Check interval: {self._interval} seconds."
            )
            # [NEW] Log if custom interval is set

    async def stop(self):
        """Stops the background refresh task."""
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

    async def _run(self):
        """The main loop for the background task."""
        # Initialize credentials (load persisted tiers) before starting the refresh loop
        await self._initialize_credentials()

        while True:
            try:
                # lib_logger.info("Running proactive token refresh check...")

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
