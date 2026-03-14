import sys
import asyncio
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.client import RotatingClient
from rotator_library.routing_policy import RoutingPolicy, RoutingPolicyError


def test_client_helper_rewrites_weighted_router_model():
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = RoutingPolicy(
        model_overrides={
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
            }
        },
        available_providers={"ollama"},
        provider_models={"ollama": {"nemotron-3-super"}},
    )

    model, decision = client._apply_routing_policy("weighted-router/nemotron-3-super")

    assert model == "ollama/nemotron-3-super"
    assert decision is not None
    assert decision.override_applied is True


def test_client_helper_passthrough_without_routing_policy():
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = None

    model, decision = client._apply_routing_policy("ollama/nemotron-3-super")

    assert model == "ollama/nemotron-3-super"
    assert decision is None


def test_load_model_routing_overrides_from_env(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    monkeypatch.setenv(
        "MODEL_ROUTING_OVERRIDES",
        '{"nemotron-3-super":{"strategy":"single","primary":"ollama","allowed_providers":["ollama"],"fallback_providers":[]}}',
    )

    overrides = client._load_model_routing_overrides_from_env()

    assert overrides["nemotron-3-super"]["primary"] == "ollama"


def test_invalid_model_routing_overrides_env_fails_closed(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    monkeypatch.setenv("MODEL_ROUTING_OVERRIDES", "{invalid")

    with pytest.raises(RoutingPolicyError, match="Invalid JSON"):
        client._load_model_routing_overrides_from_env()


def test_acompletion_rewrites_model_before_dispatch(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = RoutingPolicy(
        model_overrides={
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
            }
        },
        available_providers={"ollama"},
        provider_models={"ollama": {"nemotron-3-super"}},
    )

    captured = {}

    async def fake_execute_with_retry(api_call, request=None, pre_request_callback=None, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(client, "_execute_with_retry", fake_execute_with_retry)
    monkeypatch.setattr(client, "_log_route_decision", lambda decision: None)

    result = asyncio.run(client.acompletion(model="weighted-router/nemotron-3-super", stream=False))

    assert result == {"ok": True}
    assert captured["model"] == "ollama/nemotron-3-super"
