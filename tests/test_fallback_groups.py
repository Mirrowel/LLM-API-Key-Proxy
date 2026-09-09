from __future__ import annotations

import pytest

from rotator_library.routing import FallbackResolver, FallbackGroup, RouteTarget, RoutingConfig
from rotator_library.routing.config import RoutingConfigError, load_routing_config_from_env, parse_route_target
from rotator_library.routing.executor import FallbackAttemptRunner


def test_parse_route_target_supports_execution_suffix() -> None:
    target = parse_route_target("openai/gpt-5@litellm_fallback")

    assert target.provider == "openai"
    assert target.model == "gpt-5"
    assert target.execution == "litellm_fallback"


def test_env_fallback_group_config_loads_current_routing_types() -> None:
    config = load_routing_config_from_env(
        {
            "FALLBACK_GROUPS": "main",
            "FALLBACK_GROUP_MAIN": "openai/gpt-5@litellm_fallback,anthropic/claude@native",
            "MODEL_ROUTE_GPT5": "group:main",
        },
        config=RoutingConfig(),
    )

    group = config.fallback_groups["main"]
    assert [target.prefixed_model for target in group.targets] == ["openai/gpt-5", "anthropic/claude"]
    assert group.targets[0].execution == "litellm_fallback"
    assert group.targets[1].execution == "native"
    assert config.model_routes["gpt5"] == "group:main"


def test_resolver_promotes_requested_provider_model_inside_group() -> None:
    group = FallbackGroup(
        name="main",
        targets=(
            RouteTarget("openai", "gpt-5"),
            RouteTarget("anthropic", "claude"),
            RouteTarget("google", "gemini"),
        ),
    )
    decision = FallbackResolver(RoutingConfig(fallback_groups={"main": group})).resolve("anthropic/claude")

    assert decision.reason == "provider_model_group_promoted"
    assert [target.prefixed_model for target in decision.targets] == ["anthropic/claude", "openai/gpt-5", "google/gemini"]


def test_resolver_promotes_requested_model_for_group_route_alias() -> None:
    group = FallbackGroup(
        name="main",
        targets=(RouteTarget("openai", "gpt-5"), RouteTarget("anthropic", "claude")),
    )
    config = RoutingConfig(fallback_groups={"main": group}, model_routes={"anthropic/claude": "group:main"})

    decision = FallbackResolver(config).resolve("anthropic/claude")

    assert decision.reason == "model_route_group_promoted"
    assert [target.prefixed_model for target in decision.targets] == ["anthropic/claude", "openai/gpt-5"]


def test_resolver_rejects_missing_group_route() -> None:
    with pytest.raises(RoutingConfigError):
        FallbackResolver(RoutingConfig(model_routes={"alias": "group:missing"})).resolve("alias")


@pytest.mark.asyncio
async def test_fallback_runner_uses_decision_group_policy() -> None:
    group = FallbackGroup(name="main", targets=(RouteTarget("a", "one"), RouteTarget("b", "two")), failover_on=frozenset({"rate_limit"}))
    decision = FallbackResolver(RoutingConfig(fallback_groups={"main": group}, model_routes={"alias": "group:main"})).resolve("alias")
    attempts: list[str] = []

    class RateLimitError(RuntimeError):
        error_type = "rate_limit"

    async def attempt(target, index):
        attempts.append(target.prefixed_model)
        if target.provider == "a":
            raise RateLimitError("rate limit")
        return "ok"

    result = await FallbackAttemptRunner().run(decision, attempt)

    assert result == "ok"
    assert attempts == ["a/one", "b/two"]
