# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Retry, cooldown, and target-failover policy helpers.

This module centralizes decisions that sit above the existing error classifier.
It deliberately delegates parsing and classification to `error_handler.py` so the
proxy keeps its current retry-after parser and credential-rotation semantics.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from .error_handler import ClassifiedError, classify_error, should_retry_same_key, should_rotate_on_error
from .routing import FallbackPolicy
from .routing.policy import normalize_route_error_type
from .routing.types import FallbackGroup

DEFAULT_PROVIDER_COOLDOWN_DEFAULT_SECONDS = 30


@dataclass(frozen=True)
class ProviderCooldownDecision:
    """Decision describing whether a provider-level cooldown should start."""

    should_start: bool
    duration: int = 0
    reason: str = "not_applicable"
    scope: str = "provider"
    model: Optional[str] = None
    backoff_level: int = 0


@dataclass(frozen=True)
class FailureHistoryEntry:
    """One sanitized provider/model failure event kept in memory only."""

    timestamp: float
    provider: str
    model: Optional[str]
    error_type: str
    scope: str
    duration: int
    reason: str


def classify_route_error(error: BaseException, provider: Optional[str] = None) -> str:
    """Map an exception into the vocabulary consumed by fallback policy."""

    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    explicit = getattr(error, "error_type", None)
    if explicit:
        return normalize_route_error_type(str(explicit))
    return normalize_route_error_type(classify_error(error, provider).error_type)


def should_retry_same_credential(classified_error: ClassifiedError, small_cooldown_threshold: int) -> bool:
    """Return whether the current credential should be retried before rotation."""

    return should_retry_same_key(classified_error, small_cooldown_threshold)


def should_rotate_credential(classified_error: ClassifiedError) -> bool:
    """Return whether a classified failure should rotate to another credential."""

    return should_rotate_on_error(classified_error)


def decide_provider_cooldown(
    classified_error: ClassifiedError,
    *,
    small_cooldown_threshold: int,
    provider_cooldown_min_seconds: int,
    default_duration: int = DEFAULT_PROVIDER_COOLDOWN_DEFAULT_SECONDS,
    cooldown_on_quota: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    original_error: Any = None,
    failure_history: "FailureHistory | None" = None,
) -> ProviderCooldownDecision:
    """Return whether a provider-wide cooldown should be activated.

    Small retry-after values are intentionally left to same-credential retry to
    preserve cache/session locality. Larger retry-after values can indicate a
    provider-wide or IP-level throttle and are therefore safe to coordinate via
    the provider cooldown manager.
    """

    error_type = classified_error.error_type
    if error_type == "quota_exceeded" and not cooldown_on_quota:
        return ProviderCooldownDecision(False, reason="quota_cooldown_disabled")
    if error_type not in {"rate_limit", "server_error", "api_connection", "quota_exceeded"}:
        return ProviderCooldownDecision(False, reason="non_provider_cooldown_error")
    scope = "model" if model and is_model_capacity_error(original_error or classified_error.original_exception) else "provider"

    retry_after = classified_error.retry_after
    if retry_after is not None:
        if retry_after <= 0:
            return ProviderCooldownDecision(False, reason="non_positive_retry_after")
        if retry_after < small_cooldown_threshold:
            return ProviderCooldownDecision(False, reason="small_retry_after")
        if retry_after < provider_cooldown_min_seconds:
            return ProviderCooldownDecision(False, reason="below_provider_cooldown_minimum")
        return ProviderCooldownDecision(True, duration=int(retry_after), reason="retry_after", scope=scope, model=model if scope == "model" else None)

    if error_type in {"server_error", "api_connection"} and default_duration >= provider_cooldown_min_seconds:
        backoff_level = 0
        duration = int(default_duration)
        if failure_history is not None:
            backoff = failure_history.backoff_for(provider=provider, error_type=error_type, scope=scope, model=model if scope == "model" else None, default_duration=duration)
            duration = backoff.duration
            backoff_level = backoff.level
        return ProviderCooldownDecision(True, duration=duration, reason="model_capacity_cooldown" if scope == "model" else "default_transient_cooldown", scope=scope, model=model if scope == "model" else None, backoff_level=backoff_level)
    return ProviderCooldownDecision(False, reason="missing_retry_after")


@dataclass(frozen=True)
class BackoffDecision:
    """Bounded backoff duration derived from recent transient failures."""

    duration: int
    level: int = 0


class FailureHistory:
    """Bounded in-memory provider/model failure history.

    This is intentionally process-local. It provides enough recent context for
    conservative cooldown backoff and future observability without introducing a
    persistence layer or changing credential usage accounting.
    """

    def __init__(self, *, max_entries: int | None = None, clock: Any = None) -> None:
        settings = _retry_settings()
        self.max_entries = max(1, max_entries if max_entries is not None else settings.failure_history_max_entries)
        self._entries: deque[FailureHistoryEntry] = deque(maxlen=self.max_entries)
        self._clock = clock or time.time

    def record(self, *, provider: str, model: Optional[str], error_type: str, scope: str, duration: int, reason: str) -> None:
        """Record one sanitized cooldown/failure event."""

        self._entries.append(
            FailureHistoryEntry(
                timestamp=float(self._clock()),
                provider=provider,
                model=model,
                error_type=error_type,
                scope=scope,
                duration=duration,
                reason=reason,
            )
        )

    def snapshot(self) -> tuple[FailureHistoryEntry, ...]:
        """Return recent entries for tests and future read-only reporting."""

        return tuple(self._entries)

    def backoff_for(self, *, provider: Optional[str], error_type: str, scope: str, model: Optional[str], default_duration: int) -> BackoffDecision:
        """Return bounded backoff for repeated transient failures."""

        settings = _retry_settings()
        window = settings.provider_backoff_window_seconds
        threshold = max(1, settings.provider_backoff_threshold)
        base = max(1, settings.provider_backoff_base_seconds or default_duration)
        max_seconds = max(base, settings.provider_backoff_max_seconds)
        now = float(self._clock())
        recent = [
            entry
            for entry in self._entries
            if now - entry.timestamp <= window
            and entry.provider == provider
            and entry.error_type == error_type
            and entry.scope == scope
            and (scope != "model" or entry.model == model)
        ]
        if len(recent) + 1 < threshold:
            return BackoffDecision(default_duration, level=0)
        level = len(recent) + 1 - threshold + 1
        return BackoffDecision(min(max_seconds, base * (2 ** (level - 1))), level=level)


def is_model_capacity_error(error: Any) -> bool:
    """Return whether an error indicates model/deployment capacity exhaustion."""

    if error is None:
        return False
    if isinstance(error, dict):
        text = str(error).lower()
    else:
        parts = [str(error)]
        response = getattr(error, "response", None)
        if response is not None:
            parts.append(str(getattr(response, "text", "")))
        body = getattr(error, "body", None)
        if body is not None:
            parts.append(str(body))
        text = " ".join(parts).lower()
    return "model_capacity_exhausted" in text or "model capacity" in text or "capacity exhausted" in text


def provider_cooldown_env() -> tuple[int, int, bool]:
    """Read provider-cooldown env controls with conservative defaults."""

    settings = _retry_settings()
    return settings.provider_cooldown_min_seconds, settings.provider_cooldown_default_seconds, settings.provider_cooldown_on_quota


def is_target_failover_eligible(
    error_type: str,
    *,
    group: FallbackGroup | None = None,
    stream: bool = False,
    emitted_output: bool = False,
) -> bool:
    """Return whether a target failure may advance to the next target."""

    return FallbackPolicy().should_fallback(error_type, group=group, stream=stream, emitted_output=emitted_output)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _retry_settings() -> Any:
    """Load retry settings lazily to avoid config import cycles at startup."""

    from .config.experimental import get_retry_runtime_settings

    return get_retry_runtime_settings()
