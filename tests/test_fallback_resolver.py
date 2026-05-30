from __future__ import annotations

import pytest

from rotator_library.routing import FallbackResolver, RoutingConfigError, load_routing_config_from_env


def test_resolver_maps_alias_to_fallback_group_in_order() -> None:
    config = load_routing_config_from_env(
        {
            "FALLBACK_GROUPS": "code_chain",
            "FALLBACK_GROUP_CODE_CHAIN": "codex/gpt-5.1-codex,openai/gpt-5.1",
            "MODEL_ROUTE_CODEX": "group:code_chain",
        }
    )

    decision = FallbackResolver(config).resolve("codex")

    assert decision.group_name == "code_chain"
    assert [target.prefixed_model for target in decision.targets] == ["codex/gpt-5.1-codex", "openai/gpt-5.1"]
    assert decision.reason == "model_route_group"


def test_resolver_keeps_provider_prefixed_model_as_direct_target() -> None:
    decision = FallbackResolver(load_routing_config_from_env({})).resolve("openai/gpt-5.1")

    assert decision.group_name is None
    assert decision.targets[0].provider == "openai"
    assert decision.targets[0].model == "gpt-5.1"
    assert decision.reason == "direct_provider_model"


def test_resolver_maps_alias_to_single_target() -> None:
    config = load_routing_config_from_env({"MODEL_ROUTE_FAST": "openai/gpt-5.1"})

    decision = FallbackResolver(config).resolve("fast")

    assert decision.targets[0].prefixed_model == "openai/gpt-5.1"
    assert decision.reason == "model_route_target"


def test_resolver_rejects_unprefixed_model_without_route() -> None:
    with pytest.raises(RoutingConfigError):
        FallbackResolver(load_routing_config_from_env({})).resolve("gpt-5.1")
