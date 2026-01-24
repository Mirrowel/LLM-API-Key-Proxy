# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Provider Hook Dispatcher for Usage Manager.

This module bridges provider plugins to the usage manager, allowing providers
to customize how requests are counted, cooled down, and tracked.

=============================================================================
OVERVIEW
=============================================================================

The HookDispatcher calls provider hooks at key points in the request lifecycle.
Currently, the main hook is `on_request_complete`, which is called after every
request (success or failure) and allows the provider to override:

    - Request count (how many requests to record)
    - Cooldown duration (custom cooldown to apply)
    - Exhaustion state (mark credential for fair cycle)

=============================================================================
IMPLEMENTING on_request_complete IN YOUR PROVIDER
=============================================================================

Add this method to your provider class:

    from rotator_library.core.types import RequestCompleteResult

    def on_request_complete(
        self,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        '''
        Called after each request completes.

        Args:
            credential: Credential accessor (file path or API key)
            model: Model that was called
            success: Whether the request succeeded
            response: Response object (if success=True)
            error: ClassifiedError object (if success=False)

        Returns:
            RequestCompleteResult to override behavior, or None for defaults.
        '''
        # Your logic here
        return None  # Use default behavior

=============================================================================
RequestCompleteResult FIELDS
=============================================================================

    count_override: Optional[int]
        How many requests to count for usage tracking.
        - 0 = Don't count this request (e.g., server errors)
        - N = Count as N requests (e.g., internal retries)
        - None = Use default (1)

    cooldown_override: Optional[float]
        Seconds to cool down this credential.
        - Applied in addition to any error-based cooldown.
        - Use for custom rate limiting logic.

    force_exhausted: bool
        Mark credential as exhausted for fair cycle.
        - True = Skip this credential until fair cycle resets.
        - Useful for quota errors without long cooldowns.

=============================================================================
USE CASE: COUNTING INTERNAL RETRIES
=============================================================================

If your provider performs internal retries (e.g., for transient errors, empty
responses, or malformed responses), each retry is an API call that should be
counted. Use the ContextVar pattern for thread-safe counting:

    from contextvars import ContextVar
    from rotator_library.core.types import RequestCompleteResult

    # Module-level: each async task gets its own isolated value
    _internal_attempt_count: ContextVar[int] = ContextVar(
        'my_provider_attempt_count', default=1
    )

    class MyProvider:

        async def _make_request_with_retry(self, ...):
            # Reset at start of request
            _internal_attempt_count.set(1)

            for attempt in range(max_attempts):
                try:
                    result = await self._call_api(...)
                    return result  # Success
                except RetryableError:
                    # Increment before retry
                    _internal_attempt_count.set(_internal_attempt_count.get() + 1)
                    continue

        def on_request_complete(self, credential, model, success, response, error):
            # Report actual API call count
            count = _internal_attempt_count.get()
            _internal_attempt_count.set(1)  # Reset for safety

            if count > 1:
                logging.debug(f"Request used {count} API calls (internal retries)")

            return RequestCompleteResult(count_override=count)

Why ContextVar?
    - Instance variables (self.count) are shared across concurrent requests
    - ContextVar gives each async task its own isolated value
    - Thread-safe without explicit locking

=============================================================================
USE CASE: CUSTOM ERROR HANDLING
=============================================================================

Override counting or cooldown based on error type:

    def on_request_complete(self, credential, model, success, response, error):
        if not success and error:
            # Don't count server errors against quota
            if error.error_type == "server_error":
                return RequestCompleteResult(count_override=0)

            # Force exhaustion on quota errors
            if error.error_type == "quota_exceeded":
                return RequestCompleteResult(
                    force_exhausted=True,
                    cooldown_override=3600.0,  # 1 hour
                )

            # Custom cooldown for rate limits
            if error.error_type == "rate_limit":
                retry_after = getattr(error, "retry_after", 60)
                return RequestCompleteResult(cooldown_override=retry_after)

        return None  # Default behavior

=============================================================================
"""

import asyncio
from typing import Any, Dict, Optional

from ...core.types import RequestCompleteResult


class HookDispatcher:
    """
    Dispatch optional provider hooks during request lifecycle.

    The HookDispatcher is instantiated by UsageManager with the provider plugins
    dict. It lazily instantiates provider instances and calls their hooks.

    Currently supported hooks:
        - on_request_complete: Called after each request completes

    Usage:
        dispatcher = HookDispatcher(provider_plugins)
        result = await dispatcher.dispatch_request_complete(
            provider="my_provider",
            credential="path/to/cred.json",
            model="my-model",
            success=True,
            response=response_obj,
            error=None,
        )
        if result and result.count_override is not None:
            request_count = result.count_override
    """

    def __init__(self, provider_plugins: Optional[Dict[str, Any]] = None):
        """
        Initialize the hook dispatcher.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes.
                              Classes are lazily instantiated on first hook call.
        """
        self._plugins = provider_plugins or {}

    def _get_instance(self, provider: str) -> Optional[Any]:
        """Get provider plugin instance (singleton via metaclass)."""
        plugin_class = self._plugins.get(provider)
        if not plugin_class:
            return None
        if isinstance(plugin_class, type):
            return plugin_class()  # Singleton - always returns same instance
        return plugin_class

    async def dispatch_request_complete(
        self,
        provider: str,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        """
        Dispatch the on_request_complete hook to a provider.

        Called by UsageManager after each request completes (success or failure).
        The provider can return a RequestCompleteResult to override default
        behavior for request counting, cooldowns, or exhaustion marking.

        Args:
            provider: Provider name (e.g., "antigravity", "openai")
            credential: Credential accessor (file path or API key)
            model: Model that was called (with provider prefix)
            success: Whether the request succeeded
            response: Response object if success=True, else None
            error: ClassifiedError if success=False, else None

        Returns:
            RequestCompleteResult from provider, or None if:
            - Provider not found in plugins
            - Provider doesn't implement on_request_complete
            - Provider returns None (use default behavior)
        """
        plugin = self._get_instance(provider)
        if not plugin or not hasattr(plugin, "on_request_complete"):
            return None

        result = plugin.on_request_complete(credential, model, success, response, error)
        if asyncio.iscoroutine(result):
            result = await result

        return result
