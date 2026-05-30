from __future__ import annotations

import pytest

from rotator_library.routing import RoutingConfigError, load_routing_config_from_env, parse_route_target


def test_parse_route_target_supports_execution_suffix() -> None:
    target = parse_route_target("codex/gpt-5.1-codex@native")

    assert target.provider == "codex"
    assert target.model == "gpt-5.1-codex"
    assert target.execution == "native"
    assert target.prefixed_model == "codex/gpt-5.1-codex"


def test_parse_route_target_rejects_missing_provider() -> None:
    with pytest.raises(RoutingConfigError):
        parse_route_target("gpt-5.1")


def test_load_routing_config_from_env_parses_group_and_model_route() -> None:
    config = load_routing_config_from_env(
        {
            "FALLBACK_GROUPS": "sonnet_chain",
            "FALLBACK_GROUP_SONNET_CHAIN": "claude_code/claude-sonnet-4-5,copilot/claude-sonnet-4-5@litellm_fallback",
            "MODEL_ROUTE_CLAUDE_SONNET": "group:sonnet_chain",
        }
    )

    assert tuple(config.fallback_groups) == ("sonnet_chain",)
    assert config.fallback_groups["sonnet_chain"].targets[1].execution == "litellm_fallback"
    assert config.model_routes["claude_sonnet"] == "group:sonnet_chain"


def test_load_routing_config_rejects_empty_group() -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_config_from_env({"FALLBACK_GROUPS": "empty", "FALLBACK_GROUP_EMPTY": ""})


def test_load_routing_config_rejects_duplicate_group_names() -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_config_from_env({"FALLBACK_GROUPS": "a,a", "FALLBACK_GROUP_A": "openai/gpt"})


def test_load_routing_config_rejects_unknown_group_route() -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_config_from_env({"MODEL_ROUTE_CODEX": "group:missing"})
