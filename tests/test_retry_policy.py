from __future__ import annotations

import asyncio

from rotator_library.error_handler import ClassifiedError, PreRequestCallbackError, classify_error
from rotator_library.retry_policy import (
    classify_route_error,
    decide_provider_cooldown,
    is_target_failover_eligible,
    should_retry_same_credential,
    should_rotate_credential,
)


class ExplicitRouteError(Exception):
    error_type = "unsupported_operation"


def _classified(error_type: str, **kwargs) -> ClassifiedError:
    return ClassifiedError(error_type, original_exception=Exception(error_type), **kwargs)


def test_classifier_output_maps_to_fallback_policy_categories() -> None:
    retryable = ["rate_limit", "quota_exceeded", "server_error", "api_connection", "unsupported_operation"]
    stopped = [
        "authentication",
        "forbidden",
        "invalid_request",
        "context_window_exceeded",
        "credential_reauth_needed",
        "pre_request_callback_error",
        "cancelled",
    ]

    assert all(is_target_failover_eligible(error_type) for error_type in retryable)
    assert all(not is_target_failover_eligible(error_type) for error_type in stopped)
    assert not is_target_failover_eligible("unknown")


def test_classify_route_error_preserves_explicit_and_cancelled_types() -> None:
    assert classify_route_error(ExplicitRouteError()) == "unsupported_operation"
    assert classify_route_error(asyncio.CancelledError()) == "cancelled"
    assert classify_route_error(PreRequestCallbackError("boom")) == "pre_request_callback_error"


def test_retry_and_rotation_helpers_delegate_to_existing_semantics() -> None:
    small_rate_limit = _classified("rate_limit", retry_after=3)
    invalid_request = _classified("invalid_request")

    assert should_retry_same_credential(small_rate_limit, small_cooldown_threshold=10) is True
    assert should_rotate_credential(small_rate_limit) is True
    assert should_rotate_credential(invalid_request) is False


def test_provider_cooldown_uses_large_retry_after_not_small_retry_after() -> None:
    small = decide_provider_cooldown(
        _classified("rate_limit", retry_after=3),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
    )
    large = decide_provider_cooldown(
        _classified("rate_limit", retry_after=60),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
    )

    assert small.should_start is False
    assert small.reason == "small_retry_after"
    assert large.should_start is True
    assert large.duration == 60


def test_provider_cooldown_is_conservative_for_quota_by_default() -> None:
    disabled = decide_provider_cooldown(
        _classified("quota_exceeded", retry_after=3600),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
    )
    enabled = decide_provider_cooldown(
        _classified("quota_exceeded", retry_after=3600),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        cooldown_on_quota=True,
    )

    assert disabled.should_start is False
    assert disabled.reason == "quota_cooldown_disabled"
    assert enabled.should_start is True


def test_shared_classifier_handles_structured_dict_status_codes() -> None:
    assert classify_error({"error": {"status": 401}}).error_type == "authentication"
    assert classify_error({"error": {"details": {"status_code": 403}}}).error_type == "forbidden"
    assert classify_error({"error": {"code": 429}}).error_type == "rate_limit"
