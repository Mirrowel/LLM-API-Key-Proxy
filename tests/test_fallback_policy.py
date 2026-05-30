from __future__ import annotations

from rotator_library.routing import FallbackPolicy, parse_route_target
from rotator_library.routing.types import FallbackGroup


def test_policy_falls_back_on_retryable_categories() -> None:
    policy = FallbackPolicy()

    assert policy.should_fallback("rate_limit") is True
    assert policy.should_fallback("server_error") is True
    assert policy.should_fallback("api_connection") is True


def test_policy_stops_on_permanent_categories() -> None:
    policy = FallbackPolicy()

    assert policy.should_fallback("auth") is False
    assert policy.should_fallback("validation") is False
    assert policy.should_fallback("pre_request_callback") is False


def test_policy_blocks_stream_fallback_after_visible_output() -> None:
    assert FallbackPolicy().should_fallback("rate_limit", stream=True, emitted_output=True) is False


def test_policy_allows_stream_fallback_before_visible_output() -> None:
    assert FallbackPolicy().should_fallback("rate_limit", stream=True, emitted_output=False) is True


def test_policy_respects_group_overrides() -> None:
    group = FallbackGroup(
        name="auth_safe",
        targets=(parse_route_target("a/model"), parse_route_target("b/model")),
        failover_on=frozenset({"auth"}),
        stop_on=frozenset({"validation"}),
    )

    assert FallbackPolicy().should_fallback("auth", group=group) is True
    assert FallbackPolicy().should_fallback("validation", group=group) is False
