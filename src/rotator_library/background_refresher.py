# src/rotator_library/background_refresher.py

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RotatingClient

lib_logger = logging.getLogger('rotator_library')

class BackgroundRefresher:
    """
    A background task that periodically checks and refreshes OAuth tokens
    to ensure they remain valid.
    """
    def __init__(self, client: 'RotatingClient', interval_seconds: int = 300):
        self._client = client
        self._interval = interval_seconds
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Starts the background refresh task."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            lib_logger.info(f"Background token refresher started. Check interval: {self._interval} seconds.")

    async def stop(self):
        """Stops the background refresh task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            lib_logger.info("Background token refresher stopped.")

    async def _run(self):
        """The main loop for the background task."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                lib_logger.info("Running proactive token refresh check...")

                oauth_configs = self._client.get_oauth_credentials()
                for provider, paths in oauth_configs.items():
                    provider_plugin = self._client._get_provider_instance(f"{provider}_oauth")
                    if provider_plugin and hasattr(provider_plugin, 'proactively_refresh'):
                        for path in paths:
                            try:
                                await provider_plugin.proactively_refresh(path)
                            except Exception as e:
                                lib_logger.error(f"Error during proactive refresh for '{path}': {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Unexpected error in background refresher loop: {e}")