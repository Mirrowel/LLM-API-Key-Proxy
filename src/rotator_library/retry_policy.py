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
from dataclasses import dataclass
from typing import Optional

from .error_handler import ClassifiedError, classify_error, should_retry_same_key, should_rotate_on_error
from .routing import FallbackPolicy
from .routing.types import FallbackGroup

DEFAULT_PROVIDER_COOLDOWN_DEFAULT_SECONDS = 30


@dataclass(frozen=True)
class ProviderCooldownDecision:
    """Decision describing whether a provider-level cooldown should start."""

    should_start: bool
    duration: int = 0
    reason: str = "not_applicable"


def classify_route_error(error: BaseException, provider: Optional[str] = None) -> str:
    """Map an exception into the vocabulary consumed by fallback policy."""

    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    explicit = getattr(error, "error_type", None)
    if explicit:
        return str(explicit).lower()
    return classify_error(error, provider).error_type


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

    retry_after = classified_error.retry_after
    if retry_after is not None:
        if retry_after <= 0:
            return ProviderCooldownDecision(False, reason="non_positive_retry_after")
        if retry_after < small_cooldown_threshold:
            return ProviderCooldownDecision(False, reason="small_retry_after")
        if retry_after < provider_cooldown_min_seconds:
            return ProviderCooldownDecision(False, reason="below_provider_cooldown_minimum")
        return ProviderCooldownDecision(True, duration=int(retry_after), reason="retry_after")

    if error_type in {"server_error", "api_connection"} and default_duration >= provider_cooldown_min_seconds:
        return ProviderCooldownDecision(True, duration=int(default_duration), reason="default_transient_cooldown")
    return ProviderCooldownDecision(False, reason="missing_retry_after")


def provider_cooldown_env() -> tuple[int, int, bool]:
    """Read provider-cooldown env controls with conservative defaults."""

    min_seconds = _env_int("PROVIDER_COOLDOWN_MIN_SECONDS", 10)
    default_seconds = _env_int("PROVIDER_COOLDOWN_DEFAULT_SECONDS", DEFAULT_PROVIDER_COOLDOWN_DEFAULT_SECONDS)
    cooldown_on_quota = os.environ.get("PROVIDER_COOLDOWN_ON_QUOTA", "").strip().lower() in {"1", "true", "yes", "on"}
    return max(0, min_seconds), max(0, default_seconds), cooldown_on_quota


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
