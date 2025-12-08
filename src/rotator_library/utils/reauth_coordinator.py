# src/rotator_library/utils/reauth_coordinator.py

"""
Global Re-authentication Coordinator

Ensures only ONE interactive OAuth flow runs at a time across ALL providers.
This prevents port conflicts and user confusion when multiple credentials
need re-authentication simultaneously.

When a credential needs interactive re-auth (expired refresh token, revoked, etc.),
it queues a request here. The coordinator ensures only one re-auth happens at a time,
regardless of which provider the credential belongs to.
"""

import asyncio
import logging
import time
from typing import Callable, Optional, Dict, Any, Awaitable
from pathlib import Path

lib_logger = logging.getLogger("rotator_library")


class ReauthCoordinator:
    """
    Singleton coordinator for global re-authentication serialization.

    When a credential needs interactive re-auth (expired refresh token, revoked, etc.),
    it queues a request here. The coordinator ensures only one re-auth happens at a time.

    This is critical because:
    1. Different providers may use the same callback ports
    2. User can only complete one OAuth flow at a time
    3. Prevents race conditions in credential state management
    """

    _instance: Optional["ReauthCoordinator"] = None
    _initialized: bool = False  # Class-level declaration for Pylint

    def __new__(cls):
        # Singleton pattern - only one coordinator exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Global semaphore - only 1 re-auth at a time
        self._reauth_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

        # Tracking for observability
        self._pending_reauths: Dict[str, float] = {}  # credential -> queue_time
        self._current_reauth: Optional[str] = None
        self._current_provider: Optional[str] = None
        self._reauth_start_time: Optional[float] = None

        # Lock for tracking dict modifications
        self._tracking_lock: asyncio.Lock = asyncio.Lock()

        # Statistics
        self._total_reauths: int = 0
        self._successful_reauths: int = 0
        self._failed_reauths: int = 0
        self._timeout_reauths: int = 0

        self._initialized = True
        lib_logger.info("Global ReauthCoordinator initialized")

    def _get_display_name(self, credential_path: str) -> str:
        """Get a display-friendly name for a credential path."""
        if credential_path.startswith("env://"):
            return credential_path
        return Path(credential_path).name

    async def execute_reauth(
        self,
        credential_path: str,
        provider_name: str,
        reauth_func: Callable[[], Awaitable[Dict[str, Any]]],
        timeout: float = 300.0,  # 5 minutes default timeout
    ) -> Dict[str, Any]:
        """
        Execute a re-authentication function with global serialization.

        Only one re-auth can run at a time across all providers.
        Other requests wait in queue.

        Args:
            credential_path: Path/identifier of the credential needing re-auth
            provider_name: Name of the provider (for logging)
            reauth_func: Async function that performs the actual re-auth
            timeout: Maximum time to wait for re-auth to complete

        Returns:
            The result from reauth_func (new credentials dict)

        Raises:
            TimeoutError: If re-auth doesn't complete within timeout
            Exception: Any exception from reauth_func is re-raised
        """
        display_name = self._get_display_name(credential_path)

        # Track that this credential is waiting
        async with self._tracking_lock:
            self._pending_reauths[credential_path] = time.time()
            pending_count = len(self._pending_reauths)

            # Log queue status
            if self._current_reauth:
                current_display = self._get_display_name(self._current_reauth)
                lib_logger.info(
                    f"[ReauthCoordinator] Credential '{display_name}' ({provider_name}) queued for re-auth. "
                    f"Position in queue: {pending_count}. "
                    f"Currently processing: '{current_display}' ({self._current_provider})"
                )
            else:
                lib_logger.info(
                    f"[ReauthCoordinator] Credential '{display_name}' ({provider_name}) requesting re-auth."
                )

        try:
            # Acquire global semaphore - blocks until our turn
            async with self._reauth_semaphore:
                # Calculate how long we waited in queue
                async with self._tracking_lock:
                    queue_time = self._pending_reauths.pop(credential_path, time.time())
                    wait_duration = time.time() - queue_time
                    self._current_reauth = credential_path
                    self._current_provider = provider_name
                    self._reauth_start_time = time.time()
                    self._total_reauths += 1

                if wait_duration > 1.0:
                    lib_logger.info(
                        f"[ReauthCoordinator] Starting re-auth for '{display_name}' ({provider_name}) "
                        f"after waiting {wait_duration:.1f}s in queue"
                    )
                else:
                    lib_logger.info(
                        f"[ReauthCoordinator] Starting re-auth for '{display_name}' ({provider_name})"
                    )

                try:
                    # Execute the actual re-auth with timeout
                    result = await asyncio.wait_for(reauth_func(), timeout=timeout)

                    async with self._tracking_lock:
                        self._successful_reauths += 1
                        duration = time.time() - self._reauth_start_time

                    lib_logger.info(
                        f"[ReauthCoordinator] Re-auth SUCCESS for '{display_name}' ({provider_name}) "
                        f"in {duration:.1f}s"
                    )
                    return result

                except asyncio.TimeoutError:
                    async with self._tracking_lock:
                        self._failed_reauths += 1
                        self._timeout_reauths += 1
                    lib_logger.error(
                        f"[ReauthCoordinator] Re-auth TIMEOUT for '{display_name}' ({provider_name}) "
                        f"after {timeout}s. User did not complete OAuth flow in time."
                    )
                    raise TimeoutError(
                        f"Re-authentication timed out after {timeout}s. "
                        f"Please try again and complete the OAuth flow within the time limit."
                    )

                except Exception as e:
                    async with self._tracking_lock:
                        self._failed_reauths += 1
                    lib_logger.error(
                        f"[ReauthCoordinator] Re-auth FAILED for '{display_name}' ({provider_name}): {e}"
                    )
                    raise

                finally:
                    async with self._tracking_lock:
                        self._current_reauth = None
                        self._current_provider = None
                        self._reauth_start_time = None

                        # Log if there are still pending reauths
                        if self._pending_reauths:
                            lib_logger.info(
                                f"[ReauthCoordinator] {len(self._pending_reauths)} credential(s) "
                                f"still waiting for re-auth"
                            )

        finally:
            # Ensure we're removed from pending even if something goes wrong
            async with self._tracking_lock:
                self._pending_reauths.pop(credential_path, None)

    def is_reauth_in_progress(self) -> bool:
        """Check if a re-auth is currently in progress."""
        return self._current_reauth is not None

    def get_pending_count(self) -> int:
        """Get number of credentials waiting for re-auth."""
        return len(self._pending_reauths)

    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status for debugging/monitoring."""
        return {
            "current_reauth": self._current_reauth,
            "current_provider": self._current_provider,
            "reauth_in_progress": self._current_reauth is not None,
            "reauth_duration": (time.time() - self._reauth_start_time)
            if self._reauth_start_time
            else None,
            "pending_count": len(self._pending_reauths),
            "pending_credentials": list(self._pending_reauths.keys()),
            "stats": {
                "total": self._total_reauths,
                "successful": self._successful_reauths,
                "failed": self._failed_reauths,
                "timeouts": self._timeout_reauths,
            },
        }


# Global singleton instance
_coordinator: Optional[ReauthCoordinator] = None


def get_reauth_coordinator() -> ReauthCoordinator:
    """Get the global ReauthCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ReauthCoordinator()
    return _coordinator
