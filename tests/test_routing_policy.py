import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.routing_policy import RoutingPolicy, RoutingPolicyError


def make_policy(overrides=None, provider_models=None, providers=None):
    return RoutingPolicy(
        model_overrides=overrides
        or {
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
                "reason": "Only available on Ollama Cloud",
            }
        },
        available_providers=providers or {"ollama", "chutes"},
        provider_models=provider_models
        or {
            "ollama": {"nemotron-3-super", "qwen3.5"},
            "chutes": {"qwen3.5"},
        },
    )


def test_single_override_rewrites_weighted_router_model():
    decision = make_policy().resolve("weighted-router/nemotron-3-super")

    assert decision.selected_provider == "ollama"
    assert decision.rewritten_model == "ollama/nemotron-3-super"
    assert decision.selection_source == "model_override"
    assert decision.override_applied is True


def test_non_weighted_router_model_passes_through():
    decision = make_policy().resolve("ollama/nemotron-3-super")

    assert decision.rewritten_model == "ollama/nemotron-3-super"
    assert decision.override_applied is False
    assert decision.selection_source == "passthrough"


def test_missing_override_for_weighted_router_model_fails_closed():
    with pytest.raises(RoutingPolicyError, match="No routing override configured"):
        make_policy().resolve("weighted-router/qwen3.5")


def test_unknown_provider_fails_validation():
    with pytest.raises(RoutingPolicyError, match="unknown provider 'go'"):
        make_policy(
            overrides={
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "go",
                    "allowed_providers": ["go"],
                    "fallback_providers": [],
                }
            }
        )


@pytest.mark.parametrize(
    "override, expected_error",
    [
        (
            {"nemotron-3-super": {"primary": "ollama", "allowed_providers": ["ollama"]}},
            "strategy 'single'",
        ),
        (
            {
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "ollama",
                    "allowed_providers": ["ollama", "chutes"],
                }
            },
            "restrict 'allowed_providers'",
        ),
        (
            {
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "ollama",
                    "allowed_providers": ["ollama"],
                    "fallback_providers": ["chutes"],
                }
            },
            "cannot define 'fallback_providers'",
        ),
    ],
)
def test_invalid_single_override_shapes_fail_validation(override, expected_error):
    with pytest.raises(RoutingPolicyError, match=expected_error):
        make_policy(overrides=override)


def test_provider_model_mismatch_fails_validation_when_models_are_known():
    with pytest.raises(RoutingPolicyError, match="does not expose model 'nemotron-3-super'"):
        make_policy(
            overrides={
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "chutes",
                    "allowed_providers": ["chutes"],
                    "fallback_providers": [],
                }
            }
        )
