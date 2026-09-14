import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.providers.nvidia_provider import NvidiaProvider


def test_mistral_medium_adds_reasoning_effort_via_extra_body():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/mistralai/mistral-medium-3.5-128b", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-medium-3.5-128b")
    assert payload["extra_body"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload["extra_body"]


def test_mistral_small_adds_reasoning_effort_via_extra_body():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/mistralai/mistral-small-4-latest", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-small-4-latest")
    assert payload["extra_body"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in payload


def test_mistral_pops_top_level_reasoning_effort():
    provider = NvidiaProvider()
    payload = {
        "model": "nvidia_nim/mistralai/mistral-medium-3.5-128b",
        "reasoning_effort": "medium",
        "messages": [],
    }
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-medium-3.5-128b")
    assert "reasoning_effort" not in payload
    assert payload["extra_body"]["reasoning_effort"] == "high"


def test_mistral_disabled_reasoning_effort_pops_key():
    provider = NvidiaProvider()
    payload = {
        "model": "nvidia_nim/mistralai/mistral-medium-3.5-128b",
        "reasoning_effort": "none",
        "messages": [],
    }
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-medium-3.5-128b")
    assert "reasoning_effort" not in payload
    assert "extra_body" not in payload


def test_mistral_disabled_reasoning_effort_off():
    provider = NvidiaProvider()
    payload = {
        "model": "nvidia_nim/mistralai/mistral-small-4",
        "reasoning_effort": "off",
        "messages": [],
    }
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-small-4")
    assert "reasoning_effort" not in payload
    assert "extra_body" not in payload


def test_mistral_disabled_reasoning_effort_disable():
    provider = NvidiaProvider()
    payload = {
        "model": "nvidia_nim/mistralai/mistral-medium-3.5-128b",
        "reasoning_effort": "disable",
        "messages": [],
    }
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-medium-3.5-128b")
    assert "reasoning_effort" not in payload
    assert "extra_body" not in payload


def test_non_mistral_non_deepseek_model_untouched():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/meta/llama-3.1-405b", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/meta/llama-3.1-405b")
    assert payload == {"model": "nvidia_nim/meta/llama-3.1-405b", "messages": []}


def test_deepseek_v3_still_works():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/deepseek-ai/deepseek-v3.1", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/deepseek-ai/deepseek-v3.1")
    kwargs = payload["extra_body"]["chat_template_kwargs"]
    assert kwargs["thinking"] is True
    assert "reasoning_effort" not in kwargs


def test_deepseek_v4_still_works():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/deepseek-ai/deepseek-v4-pro", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/deepseek-ai/deepseek-v4-pro")
    kwargs = payload["extra_body"]["chat_template_kwargs"]
    assert kwargs["thinking"] is True
    assert kwargs["reasoning_effort"] == "max"


def test_mistral_no_thinking_param_in_payload():
    provider = NvidiaProvider()
    payload = {"model": "nvidia_nim/mistralai/mistral-medium-3.5-128b", "messages": []}
    provider.handle_thinking_parameter(payload, "nvidia_nim/mistralai/mistral-medium-3.5-128b")
    assert "thinking" not in payload
    assert "chat_template_kwargs" not in payload.get("extra_body", {})
