from __future__ import annotations

import asyncio

from rotator_library.error_handler import ClassifiedError, PreRequestCallbackError, classify_error
from rotator_library.retry_policy import (
    FailureHistory,
    classify_route_error,
    decide_provider_cooldown,
    is_model_capacity_error,
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
    assert large.scope == "provider"


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


def test_model_capacity_error_uses_model_scoped_cooldown() -> None:
    error = Exception("503 MODEL_CAPACITY_EXHAUSTED")
    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=30,
        model="gpt-test",
        original_error=error,
    )

    assert is_model_capacity_error(error) is True
    assert decision.should_start is True
    assert decision.scope == "model"
    assert decision.model == "gpt-test"
    assert decision.reason == "model_capacity_cooldown"


def test_failure_history_escalates_repeated_transient_backoff(monkeypatch) -> None:
    now = 1000.0
    history = FailureHistory(clock=lambda: now)
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "3")
    monkeypatch.setenv("PROVIDER_BACKOFF_BASE_SECONDS", "10")
    monkeypatch.setenv("PROVIDER_BACKOFF_MAX_SECONDS", "40")
    history.record(provider="openai", model=None, error_type="server_error", scope="provider", duration=10, reason="test")
    history.record(provider="openai", model=None, error_type="server_error", scope="provider", duration=10, reason="test")

    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="openai",
        failure_history=history,
    )

    assert decision.duration == 10
    assert decision.backoff_level == 1
    history.record(provider="openai", model=None, error_type="server_error", scope="provider", duration=10, reason="test")
    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="openai",
        failure_history=history,
    )
    assert decision.duration == 20
    assert decision.backoff_level == 2


def test_single_generic_transient_without_retry_after_does_not_cooldown(monkeypatch) -> None:
    history = FailureHistory(clock=lambda: 1000.0)
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "3")

    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="openai",
        failure_history=history,
    )

    assert decision.should_start is False
    assert decision.reason == "transient_backoff_threshold_not_met"


def test_failure_history_clear_resets_repeated_transient_backoff(monkeypatch) -> None:
    history = FailureHistory(clock=lambda: 1000.0)
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "2")
    history.record(provider="openai", model=None, error_type="server_error", scope="provider", duration=0, reason="first")

    history.clear(provider="openai", model="gpt-test")
    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="openai",
        failure_history=history,
    )

    assert history.snapshot() == ()
    assert decision.should_start is False


def test_failure_history_backoff_is_provider_scoped(monkeypatch) -> None:
    history = FailureHistory(clock=lambda: 1000.0)
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "2")
    monkeypatch.setenv("PROVIDER_BACKOFF_BASE_SECONDS", "10")
    history.record(provider="provider-a", model=None, error_type="server_error", scope="provider", duration=10, reason="test")

    provider_b = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="provider-b",
        failure_history=history,
    )
    provider_a = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        provider="provider-a",
        failure_history=history,
    )

    assert provider_b.backoff_level == 0
    assert provider_a.backoff_level == 1


def test_failure_history_backoff_requires_matching_provider(monkeypatch) -> None:
    history = FailureHistory(clock=lambda: 1000.0)
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "2")
    history.record(provider="provider-a", model=None, error_type="server_error", scope="provider", duration=10, reason="test")

    decision = decide_provider_cooldown(
        _classified("server_error"),
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        default_duration=10,
        failure_history=history,
    )

    assert decision.backoff_level == 0


def test_shared_classifier_handles_structured_dict_status_codes() -> None:
    assert classify_error({"error": {"status": 401}}).error_type == "authentication"
    assert classify_error({"error": {"details": {"status_code": 403}}}).error_type == "forbidden"
    assert classify_error({"error": {"code": 429}}).error_type == "rate_limit"


def test_shared_classifier_handles_structured_dict_type_and_code_text() -> None:
    assert classify_error({"error": {"type": "authentication"}}).error_type == "authentication"
    assert classify_error({"error": {"code": "permission_denied"}}).error_type == "forbidden"
    assert classify_error({"error": {"code": "context_length_exceeded"}}).error_type == "context_window_exceeded"
    assert classify_error({"error": {"status_code": 400, "code": "context_length_exceeded"}}).error_type == "context_window_exceeded"
    assert classify_error({"error": {"type": "rate_limit"}}).error_type == "rate_limit"


def test_shared_classifier_preserves_explicit_error_type_attributes() -> None:
    class ConfigurationFailure(Exception):
        error_type = "configuration_error"

    assert classify_error(ConfigurationFailure("raw secret message")).error_type == "configuration_error"
