from __future__ import annotations

import pytest

from rotator_library.config.experimental import load_config_from_mapping
from rotator_library.routing import RoutingConfigError, load_routing_config_from_env


def test_json_routing_config_loads_fallback_group_and_route() -> None:
    experimental = load_config_from_mapping(
        {
            "routing": {
                "fallback_groups": {
                    "code_chain": {
                        "targets": ["codex/gpt-5.1@native", "openai/gpt-5.1@litellm_fallback"],
                        "failover_on": ["rate_limit"],
                        "stop_on": ["authentication"],
                    }
                },
                "model_routes": {"code": "group:code_chain"},
            }
        }
    )

    config = load_routing_config_from_env({}, config=experimental)

    assert config.fallback_groups["code_chain"].targets[0].execution == "native"
    assert config.fallback_groups["code_chain"].failover_on == frozenset({"rate_limit"})
    assert config.model_routes["code"] == "group:code_chain"


def test_env_group_overrides_json_group_targets() -> None:
    experimental = load_config_from_mapping(
        {"routing": {"fallback_groups": {"chain": {"targets": ["codex/gpt"]}}, "model_routes": {"code": "group:chain"}}}
    )

    config = load_routing_config_from_env(
        {"FALLBACK_GROUPS": "chain", "FALLBACK_GROUP_CHAIN": "openai/gpt-5.1", "MODEL_ROUTE_CODE": "group:chain"},
        config=experimental,
    )

    assert config.fallback_groups["chain"].targets[0].provider == "openai"


def test_env_route_can_reference_json_group() -> None:
    experimental = load_config_from_mapping({"routing": {"fallback_groups": {"chain": {"targets": ["codex/gpt"]}}}})

    config = load_routing_config_from_env({"MODEL_ROUTE_CODE": "group:chain"}, config=experimental)

    assert config.model_routes["code"] == "group:chain"


def test_json_route_rejects_unknown_group() -> None:
    experimental = load_config_from_mapping({"routing": {"model_routes": {"code": "group:missing"}}})

    with pytest.raises(RoutingConfigError):
        load_routing_config_from_env({}, config=experimental)
