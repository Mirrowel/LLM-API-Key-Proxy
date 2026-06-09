# SPDX-License-Identifier: LGPL-3.0-only

"""
Monthly budget limit checker.

Enforces a monthly spending cap per credential. Tracks cumulative
approx_cost across all models/groups within a calendar month and
blocks the credential once the budget is exceeded.

Configuration via environment variables:
    MONTHLY_BUDGET_{PROVIDER}=200.0          (budget in dollars)
    MONTHLY_BUDGET_RESET_DAY_{PROVIDER}=1    (day of month to reset, default 1)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from .base import LimitChecker

lib_logger = logging.getLogger("rotator_library")


class MonthlyBudgetChecker(LimitChecker):
    """
    Checks monthly spending budget per credential.

    Aggregates approx_cost from all group and model windows that fall within
    the current billing period, then compares against the configured budget.
    The billing period resets on the configured day of the month (default: 1st).
    """

    def __init__(
        self,
        budgets: Dict[str, float],
        reset_day: int = 1,
    ):
        """
        Args:
            budgets: Maps provider name -> monthly budget in dollars.
                     Can also map "credential:<stable_id>" -> budget for
                     per-credential overrides.
            reset_day: Day of month when the budget resets (1-28).
        """
        self._budgets = budgets
        self._reset_day = max(1, min(28, reset_day))

    @property
    def name(self) -> str:
        return "monthly_budget"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        budget = self._get_budget(state)
        if budget is None:
            return LimitCheckResult.ok()

        current_spend = self._get_current_period_spend(state)
        if current_spend >= budget:
            reset_ts = self._next_reset_timestamp()
            remaining_hours = max(0.0, (reset_ts - time.time()) / 3600)
            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_CUSTOM_CAP,
                reason=(
                    f"Monthly budget exceeded: ${current_spend:.2f}/${budget:.2f} "
                    f"(resets in {remaining_hours:.1f}h)"
                ),
                blocked_until=reset_ts,
            )
        return LimitCheckResult.ok()

    def get_budget_status(self, state: CredentialState) -> Optional[Dict[str, Any]]:
        """Return current budget status for API/UI consumption."""
        budget = self._get_budget(state)
        if budget is None:
            return None
        current_spend = self._get_current_period_spend(state)
        period_start = self._current_period_start_timestamp()
        reset_ts = self._next_reset_timestamp()
        return {
            "budget": budget,
            "spent": round(current_spend, 4),
            "remaining": round(max(0.0, budget - current_spend), 4),
            "percent_used": round(min(100.0, current_spend / budget * 100), 1) if budget > 0 else 0,
            "period_start": period_start,
            "reset_at": reset_ts,
            "reset_day": self._reset_day,
        }

    def _get_budget(self, state: CredentialState) -> Optional[float]:
        per_cred_key = f"credential:{state.stable_id}"
        if per_cred_key in self._budgets:
            return self._budgets[per_cred_key]
        return self._budgets.get(state.provider)

    def _get_current_period_spend(self, state: CredentialState) -> float:
        """Sum approx_cost from all windows whose first_used_at falls in the current period."""
        period_start = self._current_period_start_timestamp()
        total_cost = 0.0

        for group_stats in state.group_usage.values():
            for window in group_stats.windows.values():
                if window.started_at is not None and window.started_at >= period_start:
                    total_cost += window.approx_cost

        seen_model_groups = set()
        for model_key in state.model_usage:
            for group_key in state.group_usage:
                if model_key in group_key or group_key in model_key:
                    seen_model_groups.add(model_key)

        for model_key, model_stats in state.model_usage.items():
            if model_key in seen_model_groups:
                continue
            for window in model_stats.windows.values():
                if window.started_at is not None and window.started_at >= period_start:
                    total_cost += window.approx_cost

        if total_cost == 0.0:
            total_cost = self._cost_from_totals_in_period(state, period_start)

        return total_cost

    def _cost_from_totals_in_period(
        self, state: CredentialState, period_start: float
    ) -> float:
        """Fallback: use credential totals if last_used_at is in the current period."""
        if state.totals.last_used_at and state.totals.last_used_at >= period_start:
            if state.totals.first_used_at and state.totals.first_used_at >= period_start:
                return state.totals.approx_cost
        return 0.0

    def _current_period_start_timestamp(self) -> float:
        now = datetime.now(timezone.utc)
        if now.day >= self._reset_day:
            period_start = now.replace(
                day=self._reset_day, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            if now.month == 1:
                period_start = now.replace(
                    year=now.year - 1, month=12, day=self._reset_day,
                    hour=0, minute=0, second=0, microsecond=0,
                )
            else:
                period_start = now.replace(
                    month=now.month - 1, day=self._reset_day,
                    hour=0, minute=0, second=0, microsecond=0,
                )
        return period_start.timestamp()

    def _next_reset_timestamp(self) -> float:
        now = datetime.now(timezone.utc)
        if now.day < self._reset_day:
            reset = now.replace(
                day=self._reset_day, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            if now.month == 12:
                reset = now.replace(
                    year=now.year + 1, month=1, day=self._reset_day,
                    hour=0, minute=0, second=0, microsecond=0,
                )
            else:
                reset = now.replace(
                    month=now.month + 1, day=self._reset_day,
                    hour=0, minute=0, second=0, microsecond=0,
                )
        return reset.timestamp()
