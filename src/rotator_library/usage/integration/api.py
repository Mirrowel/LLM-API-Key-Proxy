# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Usage API facade for reading and updating usage data.
"""

from typing import Any, Dict, Optional

from ..types import CredentialState


class UsageAPI:
    """Thin API wrapper around UsageManager internals."""

    def __init__(self, manager: "UsageManager"):
        self._manager = manager

    def get_state(self, accessor: str) -> Optional[CredentialState]:
        stable_id = self._manager.registry.get_stable_id(
            accessor, self._manager.provider
        )
        return self._manager.states.get(stable_id)

    def get_all_states(self) -> Dict[str, CredentialState]:
        return dict(self._manager.states)

    def get_window_remaining(
        self,
        accessor: str,
        window_name: str,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> Optional[int]:
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
