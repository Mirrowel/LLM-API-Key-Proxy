# SPDX-License-Identifier: LGPL-3.0-only

"""
Requests-per-day (RPD) limit checker.

Enforces daily request count limits on specific models, per-credential.
Designed for providers like Google AI Studio that impose free-tier RPD
quotas that reset at a fixed time (e.g., midnight Pacific).

The checker maintains its own lightweight daily counters that are independent
of the existing window system — this avoids coupling to the primary window
definitions which may be rolling/5h windows for quota groups.

Configuration:
    Providers set `rpd_limits` as a class attribute:
        rpd_limits = {
            "gemini-3-flash": 20,
            "gemma-4-26b": 1500,
        }

    Environment variable overrides:
        RPD_LIMIT_{PROVIDER}_{MODEL_UPPER}=500
        RPD_RESET_HOUR_{PROVIDER}=0  (hour in reset timezone, default 0)
        RPD_RESET_TZ_{PROVIDER}=US/Pacific  (default US/Pacific)
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from ..types import CredentialState, LimitCheckResult, LimitResult
from .base import LimitChecker

lib_logger = logging.getLogger("rotator_library")

DEFAULT_RESET_TZ = "America/Los_Angeles"
DEFAULT_RESET_HOUR = 0


class RPDLimitChecker(LimitChecker):
    """
    Checks per-model requests-per-day (RPD) limits.

    Tracks daily request counts per (credential, model) pair using the
    existing model_usage windows. Uses a dedicated "rpd" window that
    resets at midnight in the configured timezone (default: Pacific).
    """

    RPD_WINDOW_NAME = "rpd"

    def __init__(
        self,
        rpd_limits: Dict[str, int],
        aliases: Optional[Dict[str, str]] = None,
        reset_tz: str = DEFAULT_RESET_TZ,
        reset_hour: int = DEFAULT_RESET_HOUR,
    ):
        """
        Args:
            rpd_limits: Maps model name (without provider prefix) -> daily RPD limit.
            aliases: Maps alias model name -> canonical model name (both bare).
                     E.g. {"gemini-flash-latest": "gemini-3.5-flash"}
            reset_tz: Timezone name for the daily reset.
            reset_hour: Hour in reset_tz when counters reset (default 0 = midnight).
        """
        self._rpd_limits = {k.lower(): v for k, v in rpd_limits.items()}
        self._aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
        self._reset_tz = self._resolve_timezone(reset_tz)
        self._reset_hour = reset_hour

    @property
    def name(self) -> str:
        return "rpd_limit"

    def _resolve_alias(self, bare_model: str) -> str:
        """Resolve a model name through the alias map to its canonical name."""
        return self._aliases.get(bare_model.lower(), bare_model)

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        bare_model = self._resolve_alias(self._strip_provider_prefix(model))
        limit = self._get_limit(bare_model)
        if limit is None:
            return LimitCheckResult.ok()

        canonical_key = self._canonical_counter_key(model)
        count = self._get_today_count(state, canonical_key)
        if count >= limit:
            reset_ts = self._next_reset_timestamp()
            remaining_hours = max(0.0, (reset_ts - time.time()) / 3600)
            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_WINDOW,
                reason=(
                    f"RPD limit reached for {bare_model}: "
                    f"{count}/{limit} (resets in {remaining_hours:.1f}h)"
                ),
                blocked_until=reset_ts,
            )
        return LimitCheckResult.ok()

    def _canonical_counter_key(self, model: str) -> str:
        """Build the canonical counter key for a model (resolves aliases, keeps prefix)."""
        prefix, bare = ("", model)
        if "/" in model:
            prefix, bare = model.split("/", 1)
        resolved = self._resolve_alias(bare)
        return f"{prefix}/{resolved}" if prefix else resolved

    def record_request(self, state: CredentialState, model: str) -> None:
        """Increment today's RPD counter for a model on this credential."""
        bare_model = self._resolve_alias(self._strip_provider_prefix(model))
        if self._get_limit(bare_model) is None:
            return

        counter_key = self._canonical_counter_key(model)
        rpd_data = state.rpd_counters
        model_entry = rpd_data.get(counter_key)

        period_start = self._current_period_start_timestamp()

        if model_entry is None or model_entry.get("period_start", 0) < period_start:
            rpd_data[counter_key] = {
                "count": 1,
                "period_start": period_start,
                "reset_at": self._next_reset_timestamp(),
            }
        else:
            model_entry["count"] = model_entry.get("count", 0) + 1

    def get_rpd_status(
        self, state: CredentialState, model: str
    ) -> Optional[Dict[str, Any]]:
        """Return current RPD status for a model, or None if no limit applies."""
        bare_model = self._resolve_alias(self._strip_provider_prefix(model))
        limit = self._get_limit(bare_model)
        if limit is None:
            return None
        counter_key = self._canonical_counter_key(model)
        count = self._get_today_count(state, counter_key)
        return {
            "model": bare_model,
            "limit": limit,
            "used": count,
            "remaining": max(0, limit - count),
            "reset_at": self._next_reset_timestamp(),
        }

    def get_all_rpd_status(self, state: CredentialState) -> Dict[str, Dict[str, Any]]:
        """Return RPD status for all tracked models on this credential."""
        result: Dict[str, Dict[str, Any]] = {}
        period_start = self._current_period_start_timestamp()

        # Gather counts from stored counters, resolving to canonical names
        for model_key, entry in state.rpd_counters.items():
            bare = self._resolve_alias(self._strip_provider_prefix(model_key))
            limit = self._get_limit(bare)
            if limit is None:
                continue
            count = entry.get("count", 0) if entry.get("period_start", 0) >= period_start else 0
            if bare in result:
                result[bare]["used"] += count
                result[bare]["remaining"] = max(0, result[bare]["limit"] - result[bare]["used"])
            else:
                result[bare] = {
                    "limit": limit,
                    "used": count,
                    "remaining": max(0, limit - count),
                    "reset_at": self._next_reset_timestamp(),
                }

        # Fill in any configured limits that haven't been seen yet
        for bare_model, limit in self._rpd_limits.items():
            if bare_model not in result:
                result[bare_model] = {
                    "limit": limit,
                    "used": 0,
                    "remaining": limit,
                    "reset_at": self._next_reset_timestamp(),
                }
        return result

    def _get_limit(self, bare_model: str) -> Optional[int]:
        """Look up RPD limit by exact bare model name."""
        return self._rpd_limits.get(bare_model.lower())

    def _get_today_count(self, state: CredentialState, model: str) -> int:
        """Get today's request count for a model."""
        period_start = self._current_period_start_timestamp()
        entry = state.rpd_counters.get(model)
        if entry is None:
            return 0
        if entry.get("period_start", 0) < period_start:
            return 0
        return entry.get("count", 0)

    def _current_period_start_timestamp(self) -> float:
        now_local = datetime.now(self._reset_tz)
        if now_local.hour < self._reset_hour:
            period_start = (now_local - timedelta(days=1)).replace(
                hour=self._reset_hour, minute=0, second=0, microsecond=0
            )
        else:
            period_start = now_local.replace(
                hour=self._reset_hour, minute=0, second=0, microsecond=0
            )
        return period_start.timestamp()

    def _next_reset_timestamp(self) -> float:
        now_local = datetime.now(self._reset_tz)
        if now_local.hour < self._reset_hour:
            reset = now_local.replace(
                hour=self._reset_hour, minute=0, second=0, microsecond=0
            )
        else:
            reset = (now_local + timedelta(days=1)).replace(
                hour=self._reset_hour, minute=0, second=0, microsecond=0
            )
        return reset.timestamp()

    @staticmethod
    def _resolve_timezone(tz_name: str) -> Any:
        """Resolve a timezone name to a tzinfo, with robust fallbacks."""
        # Try canonical IANA names first, then common aliases
        candidates = [tz_name]
        alias_map = {
            "US/Pacific": "America/Los_Angeles",
            "US/Eastern": "America/New_York",
            "US/Central": "America/Chicago",
            "US/Mountain": "America/Denver",
        }
        if tz_name in alias_map:
            candidates.insert(0, alias_map[tz_name])
        elif tz_name == "America/Los_Angeles":
            candidates.append("US/Pacific")

        for name in candidates:
            try:
                return ZoneInfo(name)
            except Exception:
                continue

        # Last resort: use a fixed UTC-7 offset (Pacific standard-ish)
        lib_logger.warning(
            f"Could not resolve timezone '{tz_name}', using fixed UTC-7 offset"
        )
        return timezone(timedelta(hours=-7))

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        if "/" in model:
            return model.split("/", 1)[1]
        return model
