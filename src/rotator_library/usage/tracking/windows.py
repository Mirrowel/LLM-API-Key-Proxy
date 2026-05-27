# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Window management for usage tracking.

Handles time-based usage windows with various reset modes.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dt_time
from typing import Any, Dict, List, Optional, Tuple

from ..types import WindowStats, ResetMode
from ..config import WindowDefinition

lib_logger = logging.getLogger("rotator_library")


class WindowManager:
    """
    Manages usage tracking windows for credentials.

    Handles:
    - Rolling windows (e.g., last 5 hours)
    - Fixed daily windows (reset at specific UTC time)
    - Calendar windows (weekly, monthly)
    - API-authoritative windows (provider determines reset)
    """

    def __init__(
        self,
        window_definitions: List[WindowDefinition],
        daily_reset_time_utc: str = "03:00",
    ):
        """
        Initialize window manager.

        Args:
            window_definitions: List of window configurations
            daily_reset_time_utc: Time for daily reset in HH:MM format
        """
        self.definitions = {w.name: w for w in window_definitions}
        self.daily_reset_time_utc = self._parse_time(daily_reset_time_utc)

    def get_active_window(
        self,
        windows: Dict[str, WindowStats],
        window_name: str,
    ) -> Optional[WindowStats]:
        """
        Get an active (non-expired) window by name.

        Args:
            windows: Current windows dict for a credential
            window_name: Name of window to get

        Returns:
            WindowStats if active, None if expired or doesn't exist
        """
        window = windows.get(window_name)
        if window is None:
            return None

        definition = self.definitions.get(window_name)
        if definition is None:
            return window  # Unknown window, return as-is

        # Check if window needs reset
        if self._should_reset(window, definition):
            return None

        return window

    def get_or_create_window(
        self,
        windows: Dict[str, WindowStats],
        window_name: str,
        limit: Optional[int] = None,
    ) -> WindowStats:
        """
        Get an active window or create a new one.

        Args:
            windows: Current windows dict for a credential
            window_name: Name of window to get/create
            limit: Optional request limit for the window

        Returns:
            Active WindowStats (may be newly created)
        """
        window = self.get_active_window(windows, window_name)
        if window is not None:
            return window

        # Preserve fields from expired window (if exists)
        old_max = None
        old_max_at = None
        old_limit = None
        old_window = windows.get(window_name)
        if old_window is not None:
            # Preserve limit across window resets (until new API baseline arrives)
            old_limit = old_window.limit

            # Take max of the old window's recorded max and its final request count
            old_recorded_max = old_window.max_recorded_requests or 0
            if old_window.request_count > old_recorded_max:
                old_max = old_window.request_count
                old_max_at = old_window.last_used_at or time.time()
            elif old_recorded_max > 0:
                old_max = old_recorded_max
                old_max_at = old_window.max_recorded_at

        # Create new window
        # Note: started_at and reset_at are left as None until first actual usage
        # This prevents bogus reset times from being displayed for unused windows
        new_window = WindowStats(
            name=window_name,
            started_at=None,
            reset_at=None,
            limit=limit or old_limit,  # Use passed limit, fall back to preserved limit
            max_recorded_requests=old_max,  # Carry forward historical max
            max_recorded_at=old_max_at,
        )

        windows[window_name] = new_window
        return new_window

    def get_primary_window(
        self,
        windows: Dict[str, WindowStats],
    ) -> Optional[WindowStats]:
        """
        Get the primary window used for rotation decisions.

        Args:
            windows: Current windows dict for a credential

        Returns:
            Primary WindowStats or None
        """
        for name, definition in self.definitions.items():
            if definition.is_primary:
                return self.get_active_window(windows, name)
        return None

    def get_primary_definition(self) -> Optional[WindowDefinition]:
        """Get the primary window definition."""
        for definition in self.definitions.values():
            if definition.is_primary:
                return definition
        return None

    def get_window_remaining(
        self,
        windows: Dict[str, WindowStats],
        window_name: str,
    ) -> Optional[int]:
        """
        Get remaining requests in a window.

        Args:
            windows: Current windows dict for a credential
            window_name: Name of window to check

        Returns:
            Remaining requests, or None if unlimited/unknown
        """
        window = self.get_active_window(windows, window_name)
        if window is None:
            return None
        return window.remaining

    def update_limit(
        self,
        windows: Dict[str, WindowStats],
        window_name: str,
        new_limit: int,
    ) -> None:
        """
        Update the limit for a window (e.g., from API response).

        Args:
            windows: Current windows dict for a credential
            window_name: Name of window to update
            new_limit: New request limit
        """
        window = windows.get(window_name)
        if window is not None:
            window.limit = new_limit

    def update_reset_time(
        self,
        windows: Dict[str, WindowStats],
        window_name: str,
        reset_timestamp: float,
    ) -> None:
        """
        Update the reset time for a window (e.g., from API response).

        Args:
            windows: Current windows dict for a credential
            window_name: Name of window to update
            reset_timestamp: New reset timestamp
        """
        window = windows.get(window_name)
        if window is not None:
            window.reset_at = reset_timestamp

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _should_reset(self, window: WindowStats, definition: WindowDefinition) -> bool:
        """
        Check if a window should be reset based on its definition.
        """
        now = time.time()

        # If window has an explicit reset time, use it
        if window.reset_at is not None:
            return now >= window.reset_at

        # If window has no start time, it hasn't been used yet - no need to reset
        if window.started_at is None:
            return False

        # Check based on reset mode
        if definition.reset_mode == ResetMode.ROLLING:
            if definition.duration_seconds is None:
                return False  # Infinite window
            return now >= window.started_at + definition.duration_seconds

        elif definition.reset_mode == ResetMode.FIXED_DAILY:
            return self._past_daily_reset(window.started_at, now)

        elif definition.reset_mode == ResetMode.CALENDAR_WEEKLY:
            return self._past_weekly_reset(window.started_at, now)

        elif definition.reset_mode == ResetMode.CALENDAR_MONTHLY:
            return self._past_monthly_reset(window.started_at, now)

        elif definition.reset_mode == ResetMode.API_AUTHORITATIVE:
            # Only reset if explicit reset_at is set and passed
            return False

        return False

    def _calculate_reset_time(
        self,
        definition: WindowDefinition,
        start_time: float,
    ) -> Optional[float]:
        """
        Calculate when a window should reset based on its definition.
        """
        if definition.reset_mode == ResetMode.ROLLING:
            if definition.duration_seconds is None:
                return None  # Infinite window
            return start_time + definition.duration_seconds

        elif definition.reset_mode == ResetMode.FIXED_DAILY:
            return self._next_daily_reset(start_time)

        elif definition.reset_mode == ResetMode.CALENDAR_WEEKLY:
            return self._next_weekly_reset(start_time)

        elif definition.reset_mode == ResetMode.CALENDAR_MONTHLY:
            return self._next_monthly_reset(start_time)

        elif definition.reset_mode == ResetMode.API_AUTHORITATIVE:
            return None  # Will be set by API response

        return None

    def _parse_time(self, time_str: str) -> dt_time:
        """Parse HH:MM time string."""
        try:
            parts = time_str.split(":")
            return dt_time(hour=int(parts[0]), minute=int(parts[1]))
        except (ValueError, IndexError):
            return dt_time(hour=3, minute=0)  # Default 03:00

    def _past_daily_reset(self, started_at: float, now: float) -> bool:
        """Check if we've passed the daily reset time since window started."""
        start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)

        # Get reset time for the day after start
        reset_dt = start_dt.replace(
            hour=self.daily_reset_time_utc.hour,
            minute=self.daily_reset_time_utc.minute,
            second=0,
            microsecond=0,
        )
        if reset_dt <= start_dt:
            # Reset time already passed today, use tomorrow
            from datetime import timedelta

            reset_dt += timedelta(days=1)

        return now_dt >= reset_dt

    def _next_daily_reset(self, from_time: float) -> float:
        """Calculate next daily reset timestamp."""
        from datetime import timedelta

        from_dt = datetime.fromtimestamp(from_time, tz=timezone.utc)
        reset_dt = from_dt.replace(
            hour=self.daily_reset_time_utc.hour,
            minute=self.daily_reset_time_utc.minute,
            second=0,
            microsecond=0,
        )
        if reset_dt <= from_dt:
            reset_dt += timedelta(days=1)

        return reset_dt.timestamp()

    def _past_weekly_reset(self, started_at: float, now: float) -> bool:
        """Check if we've passed the weekly reset (Sunday 03:00 UTC)."""
        start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)

        # Get start of next week (Sunday 03:00 UTC)
        days_until_sunday = (6 - start_dt.weekday()) % 7
        if days_until_sunday == 0 and start_dt.hour >= 3:
            days_until_sunday = 7

        from datetime import timedelta

        reset_dt = start_dt.replace(
            hour=3, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_until_sunday)

        return now_dt >= reset_dt

    def _next_weekly_reset(self, from_time: float) -> float:
        """Calculate next weekly reset timestamp."""
        from datetime import timedelta

        from_dt = datetime.fromtimestamp(from_time, tz=timezone.utc)
        days_until_sunday = (6 - from_dt.weekday()) % 7
        if days_until_sunday == 0 and from_dt.hour >= 3:
            days_until_sunday = 7

        reset_dt = from_dt.replace(
            hour=3, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_until_sunday)

        return reset_dt.timestamp()

    def _past_monthly_reset(self, started_at: float, now: float) -> bool:
        """Check if we've passed the monthly reset (1st 03:00 UTC)."""
        start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)

        # Get 1st of next month
        if start_dt.month == 12:
            reset_dt = start_dt.replace(
                year=start_dt.year + 1,
                month=1,
                day=1,
                hour=3,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            reset_dt = start_dt.replace(
                month=start_dt.month + 1,
                day=1,
                hour=3,
                minute=0,
                second=0,
                microsecond=0,
            )

        return now_dt >= reset_dt

    def _next_monthly_reset(self, from_time: float) -> float:
        """Calculate next monthly reset timestamp."""
        from_dt = datetime.fromtimestamp(from_time, tz=timezone.utc)

        if from_dt.month == 12:
            reset_dt = from_dt.replace(
                year=from_dt.year + 1,
                month=1,
                day=1,
                hour=3,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            reset_dt = from_dt.replace(
                month=from_dt.month + 1,
                day=1,
                hour=3,
                minute=0,
                second=0,
                microsecond=0,
            )

        return reset_dt.timestamp()
