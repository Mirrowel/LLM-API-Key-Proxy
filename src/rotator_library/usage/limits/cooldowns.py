# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Cooldown checker.

Checks if a credential is currently in cooldown.
"""

import time
from typing import Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from .base import LimitChecker


class CooldownChecker(LimitChecker):
    """
    Checks cooldown status for credentials.

    Blocks credentials that are currently cooling down from
    rate limits, errors, or other causes.
    """

    @property
    def name(self) -> str:
        return "cooldowns"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if credential is in cooldown.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        now = time.time()
        group_key = quota_group or model

        # Check model/group-specific cooldowns
        keys_to_check = []
        if group_key:
            keys_to_check.append(group_key)
        if quota_group and quota_group != model:
            keys_to_check.append(model)

        for key in keys_to_check:
            cooldown = state.cooldowns.get(key)
            if cooldown and cooldown.until > now:
                return LimitCheckResult.blocked(
                    result=LimitResult.BLOCKED_COOLDOWN,
                    reason=f"Cooldown for '{key}': {cooldown.reason} (expires in {cooldown.remaining_seconds:.0f}s)",
                    blocked_until=cooldown.until,
                )

        # Check global cooldown
        global_cooldown = state.cooldowns.get("_global_")
        if global_cooldown and global_cooldown.until > now:
            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_COOLDOWN,
                reason=f"Global cooldown: {global_cooldown.reason} (expires in {global_cooldown.remaining_seconds:.0f}s)",
                blocked_until=global_cooldown.until,
            )

        return LimitCheckResult.ok()

    def reset(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Clear cooldown for a credential.

        Args:
            state: Credential state
            model: Optional model scope
            quota_group: Optional quota group scope
        """
        if quota_group:
            if quota_group in state.cooldowns:
                del state.cooldowns[quota_group]
        elif model:
            if model in state.cooldowns:
                del state.cooldowns[model]
        else:
            # Clear all cooldowns
            state.cooldowns.clear()

    def get_cooldown_end(
        self,
        state: CredentialState,
        model_or_group: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get when cooldown ends for a credential.

        Args:
            state: Credential state
            model_or_group: Optional scope to check

        Returns:
            Timestamp when cooldown ends, or None if not in cooldown
        """
        now = time.time()

        # Check specific scope
        if model_or_group:
            cooldown = state.cooldowns.get(model_or_group)
            if cooldown and cooldown.until > now:
                return cooldown.until

        # Check global
        global_cooldown = state.cooldowns.get("_global_")
        if global_cooldown and global_cooldown.until > now:
            return global_cooldown.until

        return None
