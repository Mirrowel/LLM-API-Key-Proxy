# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Usage API Facade for Reading and Updating Usage Data.

This module provides a clean, public API for programmatically interacting with
usage data. It's accessible via `usage_manager.api` and is intended for:

    - Admin endpoints (viewing/modifying credential state)
    - Background jobs (quota refresh, cleanup tasks)
    - Monitoring and alerting (checking remaining quota)
    - External tooling and integrations
    - Provider-specific logic that needs to inspect/modify state

=============================================================================
ACCESSING THE API
=============================================================================

The UsageAPI is available as a property on UsageManager:

    # From RotatingClient
    usage_manager = client.get_usage_manager("my_provider")
    api = usage_manager.api

    # Or if you have the manager directly
    api = usage_manager.api

=============================================================================
AVAILABLE METHODS
=============================================================================

Reading State
-------------

    # Get state for a specific credential
    state = api.get_state("path/to/credential.json")
    if state:
        print(f"Total requests: {state.totals.request_count}")
        print(f"Total successes: {state.totals.success_count}")
        print(f"Total failures: {state.totals.failure_count}")

    # Get all credential states
    all_states = api.get_all_states()
    for stable_id, state in all_states.items():
        print(f"{stable_id}: {state.totals.request_count} requests")

    # Check remaining quota in a window
    remaining = api.get_window_remaining(
        accessor="path/to/credential.json",
        window_name="5h",
        model="gpt-4o",  # Optional: specific model
        quota_group="gpt4",  # Optional: quota group
    )
    print(f"Remaining in 5h window: {remaining}")

Modifying State
---------------

    # Apply a manual cooldown
    await api.apply_cooldown(
        accessor="path/to/credential.json",
        duration=1800.0,  # 30 minutes
        reason="manual_override",
        model_or_group="gpt4",  # Optional: scope to model/group
    )

    # Clear a cooldown
    await api.clear_cooldown(
        accessor="path/to/credential.json",
        model_or_group="gpt4",  # Optional: scope
    )

    # Mark credential as exhausted for fair cycle
    await api.mark_exhausted(
        accessor="path/to/credential.json",
        model_or_group="gpt4",
        reason="quota_exceeded",
    )

=============================================================================
CREDENTIAL STATE STRUCTURE
=============================================================================

CredentialState contains:

    state.accessor           # File path or API key
    state.display_name       # Human-readable name (e.g., email)
    state.tier               # Tier name (e.g., "standard-tier")
    state.priority           # Priority level (1 = highest)
    state.active_requests    # Currently in-flight requests

    state.totals             # TotalStats - credential-level totals
    state.model_usage        # Dict[model, ModelStats]
    state.group_usage        # Dict[group, GroupStats]

    state.cooldowns          # Dict[key, CooldownState]
    state.fair_cycle         # Dict[key, FairCycleState]

ModelStats / GroupStats contain:

    stats.windows            # Dict[name, WindowStats] - time-based windows
    stats.totals             # TotalStats - all-time totals for this scope

TotalStats contains:

    totals.request_count
    totals.success_count
    totals.failure_count
    totals.prompt_tokens
    totals.completion_tokens
    totals.thinking_tokens
    totals.output_tokens
    totals.prompt_tokens_cache_read
    totals.prompt_tokens_cache_write
    totals.total_tokens
    totals.approx_cost
    totals.first_used_at
    totals.last_used_at

WindowStats contains:

    window.request_count
    window.success_count
    window.failure_count
    window.prompt_tokens
    window.completion_tokens
    window.thinking_tokens
    window.output_tokens
    window.prompt_tokens_cache_read
    window.prompt_tokens_cache_write
    window.total_tokens
    window.approx_cost
    window.started_at
    window.reset_at
    window.limit
    window.remaining         # Computed: limit - request_count (if limit set)

=============================================================================
EXAMPLE: BUILDING AN ADMIN ENDPOINT
=============================================================================

    from fastapi import APIRouter
    from rotator_library import RotatingClient

    router = APIRouter()

    @router.get("/admin/credentials/{provider}")
    async def list_credentials(provider: str):
        usage_manager = client.get_usage_manager(provider)
        if not usage_manager:
            return {"error": "Provider not found"}

        api = usage_manager.api
        result = []

        for stable_id, state in api.get_all_states().items():
            result.append({
                "id": stable_id,
                "accessor": state.accessor,
                "tier": state.tier,
                "priority": state.priority,
                "requests": state.totals.request_count,
                "successes": state.totals.success_count,
                "failures": state.totals.failure_count,
                "cooldowns": [
                    {"key": k, "remaining": v.remaining_seconds}
                    for k, v in state.cooldowns.items()
                    if v.is_active
                ],
            })

        return {"credentials": result}

    @router.post("/admin/credentials/{provider}/{accessor}/cooldown")
    async def apply_cooldown(provider: str, accessor: str, duration: float):
        usage_manager = client.get_usage_manager(provider)
        api = usage_manager.api
        await api.apply_cooldown(accessor, duration, reason="admin")
        return {"status": "cooldown applied"}

=============================================================================
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..types import CredentialState

if TYPE_CHECKING:
    from ..manager import UsageManager


class UsageAPI:
    """
    Public API facade for reading and updating usage data.

    Provides a clean interface for external code to interact with usage
    tracking without needing to understand the internal component structure.

    Access via: usage_manager.api

    Example:
        api = usage_manager.api
        state = api.get_state("path/to/credential.json")
        remaining = api.get_window_remaining("path/to/cred.json", "5h", "gpt-4o")
        await api.apply_cooldown("path/to/cred.json", 1800.0, "manual")
    """

    def __init__(self, manager: "UsageManager"):
        """
        Initialize the API facade.

        Args:
            manager: The UsageManager instance to wrap.
        """
        self._manager = manager

    def get_state(self, accessor: str) -> Optional[CredentialState]:
        """
        Get the credential state for a given accessor.

        Args:
            accessor: Credential file path or API key.

        Returns:
            CredentialState if found, None otherwise.

        Example:
            state = api.get_state("oauth_creds/my_cred.json")
            if state:
                print(f"Requests: {state.totals.request_count}")
        """
        stable_id = self._manager.registry.get_stable_id(
            accessor, self._manager.provider
        )
        return self._manager.states.get(stable_id)

    def get_all_states(self) -> Dict[str, CredentialState]:
        """
        Get all credential states.

        Returns:
            Dict mapping stable_id to CredentialState.

        Example:
            for stable_id, state in api.get_all_states().items():
                print(f"{stable_id}: {state.totals.request_count} requests")
        """
        return dict(self._manager.states)

    def get_window_remaining(
        self,
        accessor: str,
        window_name: str,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> Optional[int]:
        """
        Get remaining requests in a usage window.

        Args:
            accessor: Credential file path or API key.
            window_name: Window name (e.g., "5h", "daily").
            model: Optional model to check (uses model-specific window).
            quota_group: Optional quota group to check.

        Returns:
            Remaining requests (limit - used), or None if:
            - Credential not found
            - Window has no limit set

        Example:
            remaining = api.get_window_remaining("cred.json", "5h", model="gpt-4o")
            if remaining is not None and remaining < 10:
                print("Warning: low quota remaining")
        """
        state = self.get_state(accessor)
        if not state:
            return None
        return self._manager.limits.window_checker.get_remaining(
            state, window_name, model=model, quota_group=quota_group
        )

    async def apply_cooldown(
        self,
        accessor: str,
        duration: float,
        reason: str = "manual",
        model_or_group: Optional[str] = None,
    ) -> None:
        """
        Apply a cooldown to a credential.

        The credential will not be selected for requests until the cooldown
        expires or is cleared.

        Args:
            accessor: Credential file path or API key.
            duration: Cooldown duration in seconds.
            reason: Reason for cooldown (for logging/debugging).
            model_or_group: Optional scope (model name or quota group).
                           If None, applies to credential globally.

        Example:
            # Global cooldown
            await api.apply_cooldown("cred.json", 1800.0, "maintenance")

            # Model-specific cooldown
            await api.apply_cooldown("cred.json", 3600.0, "quota", "gpt-4o")
        """
        await self._manager.apply_cooldown(
            accessor=accessor,
            duration=duration,
            reason=reason,
            model_or_group=model_or_group,
        )

    async def clear_cooldown(
        self,
        accessor: str,
        model_or_group: Optional[str] = None,
    ) -> None:
        """
        Clear a cooldown from a credential.

        Args:
            accessor: Credential file path or API key.
            model_or_group: Optional scope to clear. If None, clears all.

        Example:
            # Clear specific cooldown
            await api.clear_cooldown("cred.json", "gpt-4o")

            # Clear all cooldowns
            await api.clear_cooldown("cred.json")
        """
        stable_id = self._manager.registry.get_stable_id(
            accessor, self._manager.provider
        )
        state = self._manager.states.get(stable_id)
        if state:
            await self._manager.tracking.clear_cooldown(
                state=state,
                model_or_group=model_or_group,
            )

    async def mark_exhausted(
        self,
        accessor: str,
        model_or_group: str,
        reason: str,
    ) -> None:
        """
        Mark a credential as exhausted for fair cycle.

        The credential will be skipped during selection until all other
        credentials in the same tier are also exhausted, at which point
        the fair cycle resets.

        Args:
            accessor: Credential file path or API key.
            model_or_group: Model name or quota group to mark exhausted.
            reason: Reason for exhaustion (for logging/debugging).

        Example:
            await api.mark_exhausted("cred.json", "gpt-4o", "quota_exceeded")
        """
        stable_id = self._manager.registry.get_stable_id(
            accessor, self._manager.provider
        )
        state = self._manager.states.get(stable_id)
        if state:
            await self._manager.tracking.mark_exhausted(
                state=state,
                model_or_group=model_or_group,
                reason=reason,
            )
