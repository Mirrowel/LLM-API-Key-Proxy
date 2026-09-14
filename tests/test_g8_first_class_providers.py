# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Wave-1 first-class provider pins: openrouter, groq, cohere.

Re-expressed on the G8 envelope: ``speaks`` replaces ``transport_profiles``
/ ``protocol_name``, groq's parameter hygiene lives in ``model_rules``
(with the adapter keeping only the conditional wire surgery), cohere is a
pure declaration (its effort narrowing is the capability cascade's
``effort_accept`` now), and listing is the shared, protocol-aware
interface implementation with an honest empty on failure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_provider_bases(monkeypatch):
    """Endpoint assertions use the class-declared bases; the operator's
    ``.env`` (loaded by other suites) must not leak into them."""

    for name in ("OPENAI", "GROQ", "COHERE", "CHUTES", "NANOGPT", "OPENROUTER"):
        monkeypatch.delenv(f"{name}_API_BASE", raising=False)


# --- openrouter: speaks resolution --------------------------------------------


def test_openrouter_declares_three_faces():
    from rotator_library.providers import PROVIDER_PLUGINS

    provider_class = PROVIDER_PLUGINS["openrouter"]
    plugin = provider_class()
    profiles, default = plugin.get_declared_profiles()
    assert set(profiles) == {"chat", "responses", "anthropic"}
    assert profiles["chat"]["protocol"] == "openai_chat"
    assert profiles["responses"]["protocol"] == "responses"
    assert profiles["anthropic"]["protocol"] == "anthropic_messages"
    assert default == "chat"
    assert plugin.default_api_base == "https://openrouter.ai/api/v1"
    # The legacy wiring is gone: speaks is the one transport declaration.
    assert plugin.transport_profiles is None
    assert plugin.protocol_name is None
    assert plugin.default_profile is None


def test_openrouter_face_addressing_routes():
    from rotator_library.routing.profiles import resolve_profile

    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["openrouter"]()
    profiles, default = plugin.get_declared_profiles()
    from rotator_library.routing.profiles import parse_model_reference

    ref = parse_model_reference("openrouter:responses/some-model")
    assert ref.profile == "responses"
    assert profiles[ref.profile]["protocol"] == "responses"
    ref = parse_model_reference("openrouter:anthropic/vendor/model")
    assert ref.profile == "anthropic"
    assert profiles[ref.profile]["protocol"] == "anthropic_messages"
    # The resolved face's endpoints are the real OpenRouter routes: the
    # anthropic face sits on /messages of the shared base (not the
    # protocol default's /v1/messages), and authenticates with Bearer.
    assert plugin.get_native_endpoint("m", "messages", profile="anthropic") == (
        "https://openrouter.ai/api/v1/messages"
    )
    assert plugin.get_native_headers("KEY", profile="anthropic") == {
        "Authorization": "Bearer KEY"
    }
    assert default == "chat"


def test_openrouter_models_parse_and_prefix():
    """Listing is the shared implementation (openai_chat face): the
    descriptor's /models route, bearer auth, ``data[].id`` shape."""

    from rotator_library.providers.openrouter_provider import OpenRouterProvider

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "moonshotai/kimi-k2.5:free"}, {"no_id": True}, "junk"]}

    class _Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, headers=None, **kwargs):
            self.calls.append((url, headers))
            return _FakeResponse()

    client = _Client()
    models = asyncio.run(OpenRouterProvider().get_models("k", client))
    assert models == ["openrouter/moonshotai/kimi-k2.5:free"]
    assert client.calls[0][0] == "https://openrouter.ai/api/v1/models"
    assert client.calls[0][1] == {"Authorization": "Bearer k"}


def test_openrouter_failed_listing_is_an_honest_empty():
    from rotator_library.providers.openrouter_provider import OpenRouterProvider

    class _Client:
        async def get(self, url, headers=None, **kwargs):
            raise RuntimeError("network down")

    assert asyncio.run(OpenRouterProvider().get_models("k", _Client())) == []
    # The hand-rolled listing is deleted; the class inherits the shared one.
    assert "get_models" not in vars(OpenRouterProvider)


# --- groq: declaration + adapter split ------------------------------------------


def test_groq_declaration():
    from rotator_library.providers import PROVIDER_PLUGINS

    provider_class = PROVIDER_PLUGINS["groq"]
    plugin = provider_class()
    assert plugin.default_api_base == "https://api.groq.com/openai/v1"
    assert plugin.adapter_names == ("groq",)
    assert plugin.speaks == ("openai_chat",)
    assert plugin.transport_profiles is None
    assert plugin.protocol_name is None
    assert plugin.get_adapter_names("m") == ("param_rules", "groq")


def test_groq_param_hygiene_is_declared_in_model_rules():
    """The strip list, temperature floor, and ``n`` pin are rows on the
    capability cascade — no provider code carries them anymore."""

    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["groq"]()
    (row,) = plugin.model_rules
    assert row["match"] == "*"
    assert set(row["strip"]) == {
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    }
    assert row["clamp"] == {"temperature": [1e-8, 2.0], "n": [1, 1]}
    # The resolved tables ride the always-on param engine's config key.
    rules = plugin.get_adapter_config("llama-3.3-70b-versatile")["param_rules"]
    assert "temperature" in rules["clamp"] and "n" in rules["clamp"]
    assert "frequency_penalty" in rules["strip"]


def test_groq_chain_enforces_declared_hygiene():
    """The full chain (always-on param engine + groq adapter) strips the
    unsupported knobs, floats temperature off zero, pins n, and forces
    parsed reasoning with tools."""

    from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["groq"]()
    model = "llama-3.3-70b-versatile"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "frequency_penalty": 0.5,
        "logit_bias": {"1": 2},
        "n": 3,
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    context = AdapterContext(
        provider="groq",
        model=model,
        protocol="openai_chat",
        adapter_config=plugin.get_adapter_config(model),
    )
    adapters = [get_adapter(name) for name in plugin.get_adapter_names(model)]
    result = asyncio.run(
        run_adapter_chain(adapters, payload, context, stage="request")
    )
    assert result["temperature"] == 1e-8
    assert "frequency_penalty" not in result and "logit_bias" not in result
    assert result["n"] == 1
    assert result["reasoning_format"] == "parsed"
    plain_context = AdapterContext(provider="groq", model=model, protocol="openai_chat")
    plain = asyncio.run(
        run_adapter_chain(
            adapters,
            {"model": "m", "messages": [], "temperature": 0.7, "n": 1},
            plain_context,
            stage="request",
        )
    )
    assert "reasoning_format" not in plain
    assert plain["temperature"] == 0.7


def test_groq_adapter_keeps_only_conditional_wire_surgery():
    """What a flat row cannot express stays in the adapter: the parsed
    selection on tools/JSON and the include_reasoning mutual exclusion."""

    from rotator_library.adapters.groq import GroqAdapter
    from rotator_library.adapters.base import AdapterContext

    adapter = GroqAdapter()
    context = AdapterContext(provider="groq", model="m")
    json_mode = asyncio.run(
        adapter.transform_request(
            {"model": "m", "messages": [], "response_format": {"type": "json_object"}},
            context,
        )
    )
    assert json_mode["reasoning_format"] == "parsed"
    untouched = asyncio.run(
        adapter.transform_request({"model": "m", "messages": [], "temperature": 0.5}, context)
    )
    assert "reasoning_format" not in untouched
    # The strip list is NOT the adapter's job anymore: a raw payload that
    # reaches it directly keeps the knobs (the param engine runs first in
    # the chain).
    raw = asyncio.run(
        adapter.transform_request(
            {"model": "m", "messages": [], "frequency_penalty": 0.5, "temperature": 0}, context
        )
    )
    assert raw["frequency_penalty"] == 0.5 and raw["temperature"] == 0


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


# --- cohere: shared listing + declaration-only shaping --------------------------


def test_cohere_compat_models_via_shared_listing():
    """The compat face's listing is openai-shaped: the shared descriptor
    route, bearer auth, ``data[].id``; the old native-shape fallback was
    the compat base's dead branch and is gone."""

    from rotator_library.providers.cohere_provider import CohereProvider

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "command-a-03-2025"}, {"no_id": True}]}

    class _Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, headers=None, **kwargs):
            self.calls.append((url, headers))
            return _FakeResponse()

    client = _Client()
    models = asyncio.run(CohereProvider().get_models("k", client))
    assert models == ["cohere/command-a-03-2025"]
    assert client.calls[0][0] == "https://api.cohere.ai/compatibility/v1/models"
    assert client.calls[0][1] == {"Authorization": "Bearer k"}


def test_cohere_failed_listing_is_an_honest_empty():
    from rotator_library.providers.cohere_provider import CohereProvider

    class _Client:
        async def get(self, url, headers=None, **kwargs):
            raise RuntimeError("network down")

    assert asyncio.run(CohereProvider().get_models("k", _Client())) == []


def test_cohere_declaration():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["cohere"]()
    assert plugin.default_api_base == "https://api.cohere.ai/compatibility/v1"
    assert plugin.speaks == ("openai_chat",)
    assert plugin.get_protocol_name("m") == "openai_chat"
    # The effort adapter is dead; the always-on param engine is the chain.
    assert plugin.adapter_names == ()
    assert plugin.get_adapter_names("m") == ("param_rules",)
    assert plugin.default_profile is None


def test_cohere_capability_row_declares_effort_and_tool_choice():
    """One wildcard row carries the whole compat-face vocabulary: the
    {off, high} effort acceptance and the required→any tool_choice fold."""

    from rotator_library.protocols.effort import resolve_accepted_effort
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["cohere"]()
    (row,) = plugin.model_rules
    assert row["match"] == "*"
    assert row["effort_accept"] == ["off", "high"]
    assert row["map"] == {"tool_choice": {"required": "any"}}
    accepted, source = resolve_accepted_effort(
        plugin, "command-a-03-2025", protocol_family="openai_chat"
    )
    assert accepted == ("off", "high")
    assert source.startswith("model_rules:")


def test_cohere_effort_words_fold_through_the_ladder():
    """The ladder owns the narrowing the adapter used to do: every ON word
    lands on the nearest accepted rung (high), off keeps the wire's own
    ``none`` spelling, and unknown words drop with a recorded note."""

    from rotator_library.native_provider.effort_emission import normalize_wire_effort
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["cohere"]()

    medium = {"reasoning_effort": "medium"}
    normalize_wire_effort(
        medium, provider_plugin=plugin, model="command-a-03-2025", protocol_name="openai_chat"
    )
    assert medium["reasoning_effort"] == "high"

    off = {"reasoning_effort": "none"}
    normalize_wire_effort(
        off, provider_plugin=plugin, model="command-a-03-2025", protocol_name="openai_chat"
    )
    assert off["reasoning_effort"] == "none"

    unknown = {"reasoning_effort": "banana"}
    normalize_wire_effort(
        unknown, provider_plugin=plugin, model="command-a-03-2025", protocol_name="openai_chat"
    )
    assert "reasoning_effort" not in unknown
