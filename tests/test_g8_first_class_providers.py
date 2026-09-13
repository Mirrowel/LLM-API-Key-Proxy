# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Wave-1 first-class provider pins: openrouter, groq, cohere."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_openrouter_declares_three_faces():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["openrouter"]
    profiles = plugin.transport_profiles
    assert set(profiles) == {"chat", "responses", "anthropic"}
    assert profiles["chat"]["protocol"] == "openai_chat"
    assert profiles["responses"]["protocol"] == "responses"
    assert profiles["anthropic"]["protocol"] == "anthropic_messages"
    assert plugin.default_profile == "chat"
    assert plugin.default_api_base == "https://openrouter.ai/api/v1"


def test_openrouter_face_addressing_routes():
    from rotator_library.routing.profiles import resolve_profile

    from rotator_library.providers import PROVIDER_PLUGINS

    profiles = PROVIDER_PLUGINS["openrouter"].transport_profiles
    from rotator_library.routing.profiles import parse_model_reference

    ref = parse_model_reference("openrouter:responses/some-model")
    assert ref.profile == "responses"
    assert profiles[ref.profile]["protocol"] == "responses"
    ref = parse_model_reference("openrouter:anthropic/vendor/model")
    assert ref.profile == "anthropic"
    assert profiles[ref.profile]["protocol"] == "anthropic_messages"


def test_openrouter_models_parse_and_prefix():
    from rotator_library.providers.openrouter_provider import OpenRouterProvider

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "moonshotai/kimi-k2.5:free"}, {"no_id": True}, "junk"]}

    class _Client:
        async def get(self, url, headers=None):
            return _FakeResponse()

    models = asyncio.run(OpenRouterProvider().get_models("k", _Client()))
    assert models == ["openrouter/moonshotai/kimi-k2.5:free"]


def test_groq_adapter_request_hygiene():
    from rotator_library.adapters.groq import GroqAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = GroqAdapter()
    context = AdapterContext(provider="groq", model="llama-3.3-70b-versatile")
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "frequency_penalty": 0.5,
        "logit_bias": {"1": 2},
        "n": 3,
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    result = asyncio.run(adapter.transform_request(payload, context))
    assert result["temperature"] == 1e-8
    assert "frequency_penalty" not in result and "logit_bias" not in result
    assert result["n"] == 1
    assert result["reasoning_format"] == "parsed"
    plain = asyncio.run(
        adapter.transform_request({"model": "m", "messages": [], "temperature": 0.7}, context)
    )
    assert "reasoning_format" not in plain


def test_groq_adapter_reasoning_rename():
    from rotator_library.adapters.groq import GroqAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = GroqAdapter()
    context = AdapterContext(provider="groq", model="m")
    response = {"choices": [{"message": {"role": "assistant", "reasoning": "thinking..."}}]}
    fixed = asyncio.run(adapter.transform_response(response, context))
    assert fixed["choices"][0]["message"]["reasoning_content"] == "thinking..."
    chunk = {"choices": [{"delta": {"reasoning": "step"}}]}
    fixed = asyncio.run(adapter.transform_stream_event(chunk, context))
    assert fixed["choices"][0]["delta"]["reasoning_content"] == "step"


def test_groq_declaration():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["groq"]
    assert plugin.default_api_base == "https://api.groq.com/openai/v1"
    assert plugin.adapter_names == ("groq",)


def test_cohere_compat_models_parse_both_shapes():
    from rotator_library.providers.cohere_provider import CohereProvider

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, payload):
            self._payload = payload

        async def get(self, url, headers=None):
            return _FakeResponse(self._payload)

    provider = CohereProvider()
    compat = asyncio.run(provider.get_models("k", _Client({"data": [{"id": "command-a-03-2025"}]})))
    assert compat == ["cohere/command-a-03-2025"]
    native = asyncio.run(
        provider.get_models(
            "k",
            _Client({"models": [{"name": "command-r7b-12-2024", "endpoints": ["chat"]}, {"name": "embed-x", "endpoints": ["embed"]}]}),
        )
    )
    assert native == ["cohere/command-r7b-12-2024"]


def test_cohere_declaration():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["cohere"]
    assert plugin.default_api_base == "https://api.cohere.ai/compatibility/v1"
    assert plugin.protocol_name == "openai_chat"
    assert plugin.adapter_names == ("cohere",)


def test_cohere_effort_narrowing():
    from rotator_library.adapters.cohere import CohereAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = CohereAdapter()
    context = AdapterContext(provider="cohere", model="command-a-03-2025")
    narrowed = asyncio.run(
        adapter.transform_request(
            {"model": "m", "messages": [], "reasoning_effort": "medium"}, context
        )
    )
    assert narrowed["reasoning_effort"] == "high"
    kept = asyncio.run(
        adapter.transform_request(
            {"model": "m", "messages": [], "reasoning_effort": "none"}, context
        )
    )
    assert kept["reasoning_effort"] == "none"
    untouched = asyncio.run(adapter.transform_request({"model": "m", "messages": []}, context))
    assert "reasoning_effort" not in untouched


def test_groq_include_reasoning_popped_and_xgroq_usage_lifted():
    from rotator_library.adapters.groq import GroqAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = GroqAdapter()
    context = AdapterContext(provider="groq", model="m")
    payload = {
        "model": "m",
        "messages": [],
        "include_reasoning": False,
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    result = asyncio.run(adapter.transform_request(payload, context))
    assert "include_reasoning" not in result and result["reasoning_format"] == "parsed"
    terminal = {"choices": [], "x_groq": {"id": "x", "usage": {"prompt_tokens": 3}}}
    lifted = asyncio.run(adapter.transform_stream_event(terminal, context))
    assert lifted["usage"] == {"prompt_tokens": 3}


def test_cohere_effort_aliases_and_null_drop():
    from rotator_library.adapters.cohere import CohereAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = CohereAdapter()
    context = AdapterContext(provider="cohere", model="m")
    off = asyncio.run(adapter.transform_request({"reasoning_effort": "off"}, context))
    assert off["reasoning_effort"] == "none"
    dropped = asyncio.run(adapter.transform_request({"reasoning_effort": None}, context))
    assert "reasoning_effort" not in dropped


def test_groq_explicit_raw_with_tools_is_forced_parsed():
    from rotator_library.adapters.groq import GroqAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = GroqAdapter()
    context = AdapterContext(provider="groq", model="m")
    payload = {
        "model": "m",
        "messages": [],
        "reasoning_format": "raw",
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    result = asyncio.run(adapter.transform_request(payload, context))
    assert result["reasoning_format"] == "parsed"
