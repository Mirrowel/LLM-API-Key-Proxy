# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Side-effect-free decisions for streaming failures."""

from __future__ import annotations

from dataclasses import dataclass

from ..error_handler import ClassifiedError, classify_error, should_retry_same_key, should_rotate_on_error
from ..retry_policy import ProviderCooldownDecision, decide_provider_cooldown
from .policy import can_retry_stream_after_error, is_visible_stream_output


@dataclass(frozen=True)
class StreamingErrorDecision:
    """Decision returned by streaming error policy before executor side effects."""

    classified: ClassifiedError
    action: str
    start_provider_cooldown: bool = False
    provider_cooldown_duration: int = 0
    provider_cooldown_scope: str = "provider"
    provider_cooldown_model: str | None = None
    reason: str = ""


def decide_streaming_error_action(
    error: Exception,
    *,
    provider: str,
    last_streamed_chunk: str | None,
    attempt: int,
    max_retries: int,
    small_cooldown_threshold: int,
    provider_cooldown_min_seconds: int,
    provider_cooldown_default_seconds: int,
    cooldown_on_quota: bool = False,
    allow_reasoning_only_retry: bool = False,
    model: str | None = None,
) -> StreamingErrorDecision:
    """Classify a stream failure without sleeping or mutating state.

    The executor remains responsible for logging, credential state, sleeping,
    provider cooldown mutation, and fallback. This helper only makes the same
    decision consistently across streaming exception branches.
    """

    classified = classify_error(error, provider)
    cooldown = _cooldown_decision(
        classified,
        small_cooldown_threshold=small_cooldown_threshold,
        provider_cooldown_min_seconds=provider_cooldown_min_seconds,
        provider_cooldown_default_seconds=provider_cooldown_default_seconds,
        cooldown_on_quota=cooldown_on_quota,
        last_streamed_chunk=last_streamed_chunk,
        provider=provider,
        model=model,
        original_error=error,
    )
    if not should_rotate_on_error(classified):
        return _decision(classified, "fail", cooldown, "non_rotatable")
    if not can_retry_stream_after_error(last_streamed_chunk, allow_reasoning_only_retry):
        return _decision(classified, "fallback_blocked_after_output", cooldown, "visible_output")
    if should_retry_same_key(classified, small_cooldown_threshold) and attempt < max_retries - 1:
        return _decision(classified, "retry_same", cooldown, "retry_same_credential")
    return _decision(classified, "rotate", cooldown, "rotate_credential")


def _cooldown_decision(
    classified: ClassifiedError,
    *,
    small_cooldown_threshold: int,
    provider_cooldown_min_seconds: int,
    provider_cooldown_default_seconds: int,
    cooldown_on_quota: bool,
    last_streamed_chunk: str | None,
    provider: str,
    model: str | None,
    original_error: Exception,
) -> ProviderCooldownDecision:
    if is_visible_stream_output(last_streamed_chunk):
        return ProviderCooldownDecision(False, reason="visible_output")
    return decide_provider_cooldown(
        classified,
        small_cooldown_threshold=small_cooldown_threshold,
        provider_cooldown_min_seconds=provider_cooldown_min_seconds,
        default_duration=provider_cooldown_default_seconds,
        cooldown_on_quota=cooldown_on_quota,
        provider=provider,
        model=model,
        original_error=original_error,
    )


def _decision(
    classified: ClassifiedError,
    action: str,
    cooldown: ProviderCooldownDecision,
    reason: str,
) -> StreamingErrorDecision:
    return StreamingErrorDecision(
        classified=classified,
        action=action,
        start_provider_cooldown=cooldown.should_start,
        provider_cooldown_duration=cooldown.duration,
        provider_cooldown_scope=cooldown.scope,
        provider_cooldown_model=cooldown.model,
        reason=reason if not cooldown.should_start else f"{reason};cooldown:{cooldown.reason}",
    )
