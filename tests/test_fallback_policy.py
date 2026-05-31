from __future__ import annotations

from rotator_library.routing import FallbackPolicy, parse_route_target
from rotator_library.routing.policy import normalize_route_error_type
from rotator_library.routing.types import FallbackGroup


def test_policy_falls_back_on_retryable_categories() -> None:
    policy = FallbackPolicy()

    assert policy.should_fallback("rate_limit") is True
    assert policy.should_fallback("quota_exceeded") is True
    assert policy.should_fallback("server_error") is True
    assert policy.should_fallback("api_connection") is True


def test_policy_stops_on_permanent_categories() -> None:
    policy = FallbackPolicy()

    assert policy.should_fallback("authentication") is False
    assert policy.should_fallback("forbidden") is False
    assert policy.should_fallback("invalid_request") is False
    assert policy.should_fallback("context_window_exceeded") is False
    assert policy.should_fallback("credential_reauth_needed") is False
    assert policy.should_fallback("pre_request_callback_error") is False
    assert policy.should_fallback("cancelled") is False


def test_policy_blocks_stream_fallback_after_visible_output() -> None:
    assert FallbackPolicy().should_fallback("rate_limit", stream=True, emitted_output=True) is False


def test_policy_allows_stream_fallback_before_visible_output() -> None:
    assert FallbackPolicy().should_fallback("rate_limit", stream=True, emitted_output=False) is True


def test_policy_respects_safe_group_overrides() -> None:
    group = FallbackGroup(
        name="auth_safe",
        targets=(parse_route_target("a/model"), parse_route_target("b/model")),
        failover_on=frozenset({"network"}),
        stop_on=frozenset({"validation"}),
    )

    assert FallbackPolicy().should_fallback("api_connection", group=group) is True
    assert FallbackPolicy().should_fallback("validation", group=group) is False


def test_policy_hard_stops_cannot_be_overridden_by_group_failover() -> None:
    group = FallbackGroup(
        name="unsafe",
        targets=(parse_route_target("a/model"), parse_route_target("b/model")),
        failover_on=frozenset({"auth", "configuration"}),
        stop_on=frozenset(),
    )

    assert FallbackPolicy().should_fallback("authentication", group=group) is False
    assert FallbackPolicy().should_fallback("configuration_error", group=group) is False


def test_policy_normalizes_user_facing_aliases() -> None:
    assert normalize_route_error_type("auth") == "authentication"
    assert normalize_route_error_type("permission-denied") == "forbidden"
    assert normalize_route_error_type("bad request") == "invalid_request"
    assert normalize_route_error_type("context_length_exceeded") == "context_window_exceeded"
    assert FallbackPolicy().should_fallback("network") is True
    assert FallbackPolicy().should_fallback("validation") is False


def test_policy_normalizes_common_structured_provider_aliases() -> None:
    assert normalize_route_error_type("invalid_api_key") == "authentication"
    assert normalize_route_error_type("unauthorized") == "authentication"
    assert normalize_route_error_type("invalid_argument") == "invalid_request"
    assert normalize_route_error_type("max_tokens_exceeded") == "context_window_exceeded"
    assert normalize_route_error_type("rate_limited") == "rate_limit"
    assert normalize_route_error_type("too_many_requests") == "rate_limit"
    assert normalize_route_error_type("resource_exhausted") == "quota_exceeded"
    assert normalize_route_error_type("unavailable") == "server_error"
    assert normalize_route_error_type("deadline_exceeded") == "api_connection"
