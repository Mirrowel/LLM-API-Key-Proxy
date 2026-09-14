# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 envelope pins for the migrated providers (G8 final).

One pin set over the whole migration: every provider declares its faces
through ``speaks`` (no ``transport_profiles`` / ``protocol_name`` /
``default_profile`` survives), the always-on param engine heads every
adapter chain, shared listing resolves the descriptor's route with the
provider prefix and returns an honest empty on failure, and no hardcoded
fallback listing remains. Provider-specific extras (groq's declared
parameter cascade, nanogpt's subscription base, gemini's native listing
hint, chutes' genuinely custom pseudo-model filter) are pinned here too.
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

MIGRATED = ("openai", "groq", "cohere", "chutes", "nanogpt", "openrouter", "gemini", "ollama")


@pytest.fixture(autouse=True)
def _isolate_provider_bases(monkeypatch):
    """Endpoint assertions use the class-declared bases; the operator's
    ``.env`` (loaded by other suites) must not leak into them."""

    for name in MIGRATED:
        monkeypatch.delenv(f"{name.upper()}_API_BASE", raising=False)

#: Default face name + profile table per provider (all migrated to speaks).
FACES = {
    "openai": ("responses", {"responses": "responses", "chat": "openai_chat"}),
    "groq": ("openai_chat", {"openai_chat": "openai_chat"}),
    "cohere": ("openai_chat", {"openai_chat": "openai_chat"}),
    "chutes": ("openai_chat", {"openai_chat": "openai_chat"}),
    "nanogpt": (
        "chat",
        {
            "chat": "openai_chat",
            "responses": "responses",
            "subscription": "openai_chat",
        },
    ),
    "openrouter": (
        "chat",
        {
            "chat": "openai_chat",
            "responses": "responses",
            "anthropic": "anthropic_messages",
        },
    ),
    "gemini": ("native", {"native": "gemini", "openai": "openai_chat"}),
    "ollama": ("ollama", {"ollama": "ollama"}),
}

#: Shared-listing expectations: route, auth headers, response body, ids.
LISTING = {
    "openai": (
        "https://api.openai.com/v1/models",
        {"Authorization": "Bearer k"},
        {"data": [{"id": "gpt-x"}, {"no_id": True}]},
        ["openai/gpt-x"],
    ),
    "groq": (
        "https://api.groq.com/openai/v1/models",
        {"Authorization": "Bearer k"},
        {"data": [{"id": "llama-3.3-70b-versatile"}]},
        ["groq/llama-3.3-70b-versatile"],
    ),
    "cohere": (
        "https://api.cohere.ai/compatibility/v1/models",
        {"Authorization": "Bearer k"},
        {"data": [{"id": "command-a-03-2025"}]},
        ["cohere/command-a-03-2025"],
    ),
    "nanogpt": (
        "https://nano-gpt.com/api/v1/models",
        {"Authorization": "Bearer k"},
        {"data": [{"id": "gpt-5.2"}]},
        ["nanogpt/gpt-5.2"],
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1/models",
        {"Authorization": "Bearer k"},
        {"data": [{"id": "moonshotai/kimi-k2.5:free"}]},
        ["openrouter/moonshotai/kimi-k2.5:free"],
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/models",
        {"x-goog-api-key": "k"},
        {"models": [{"name": "models/gemini-2.5-pro"}, {"no_name": True}]},
        ["gemini/gemini-2.5-pro"],
    ),
    "ollama": (
        "http://localhost:11434/api/tags",
        # A supplied real credential rides the listing too (authenticated
        # Ollama behind a proxy), plus the provider's JSON content header;
        # the no-auth sentinel yields only the content header.
        {"Authorization": "Bearer k", "Content-Type": "application/json"},
        {"models": [{"name": "llama3:latest"}]},
        ["ollama/llama3:latest"],
    ),
}


def _plugin(name: str):
    from rotator_library.providers import PROVIDER_PLUGINS

    return PROVIDER_PLUGINS[name]()


# --- speaks resolution ----------------------------------------------------------


@pytest.mark.parametrize("provider", MIGRATED)
def test_speaks_is_the_transport_declaration(provider: str) -> None:
    plugin = _plugin(provider)
    expected_default, expected_profiles = FACES[provider]
    profiles, default = plugin.get_declared_profiles()
    assert default == expected_default
    assert {name: entry["protocol"] for name, entry in profiles.items()} == expected_profiles
    # The legacy declaration surface is gone.
    assert plugin.transport_profiles is None
    assert plugin.protocol_name is None
    assert plugin.default_profile is None


@pytest.mark.parametrize("provider", MIGRATED)
def test_param_engine_heads_the_adapter_chain(provider: str) -> None:
    plugin = _plugin(provider)
    adapter_names = plugin.get_adapter_names("test-model")
    assert adapter_names[0] == "param_rules"


@pytest.mark.parametrize(
    ("provider", "declared"),
    [
        ("openai", ()),
        ("groq", ("groq",)),
        ("cohere", ("cohere",)),
        ("chutes", ("chutes",)),
        ("nanogpt", ("nanogpt",)),
        ("openrouter", ()),
        ("gemini", ()),
        ("ollama", ()),
    ],
)
def test_adapter_names_only_where_wire_logic_is_custom(
    provider: str, declared: tuple[str, ...]
) -> None:
    plugin = _plugin(provider)
    assert plugin.adapter_names == declared


# --- listing: shared implementation, descriptor route, honest empty ------------


@pytest.mark.parametrize("provider", LISTING)
def test_listing_rides_the_descriptor_route(provider: str) -> None:
    plugin = _plugin(provider)
    url, headers, body, expected = LISTING[provider]
    calls: list[dict] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return body

    class _Client:
        async def get(self, request_url, headers=None, **kwargs):
            calls.append({"url": request_url, "headers": dict(headers or {})})
            return _Response()

    models = asyncio.run(plugin.get_models("k", _Client()))
    assert models == expected
    assert calls[0]["url"] == url
    assert calls[0]["headers"] == headers


@pytest.mark.parametrize("provider", LISTING)
def test_failed_listing_is_an_honest_empty(provider: str) -> None:
    plugin = _plugin(provider)

    class _Client:
        async def get(self, request_url, headers=None, **kwargs):
            raise RuntimeError("network down")

    assert asyncio.run(plugin.get_models("k", _Client())) == []


@pytest.mark.parametrize(
    "provider", [name for name in MIGRATED if name != "chutes"]
)
def test_shared_listing_providers_declare_no_custom_get_models(provider: str) -> None:
    """Chutes is the one genuinely custom lister (routing pseudo-models);
    every other migrated provider inherits the shared implementation — the
    hand-rolled fetches (and their fallback lists) are gone."""

    from rotator_library.providers import PROVIDER_PLUGINS

    assert "get_models" not in vars(PROVIDER_PLUGINS[provider])


# --- provider-specific envelope extras -----------------------------------------


def test_groq_declared_cascade_enforced_through_the_chain() -> None:
    from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain

    plugin = _plugin("groq")
    model = "llama-3.3-70b-versatile"
    context = AdapterContext(
        provider="groq",
        model=model,
        protocol="openai_chat",
        adapter_config=plugin.get_adapter_config(model),
    )
    adapters = [get_adapter(name) for name in plugin.get_adapter_names(model)]
    result = asyncio.run(
        run_adapter_chain(
            adapters,
            {
                "model": model,
                "messages": [],
                "temperature": 0,
                "presence_penalty": 0.4,
                "top_logprobs": 3,
                "n": 2,
            },
            context,
            stage="request",
        )
    )
    assert result["temperature"] == 1e-8
    assert "presence_penalty" not in result and "top_logprobs" not in result
    assert result["n"] == 1


def test_nanogpt_subscription_face_uses_its_own_base() -> None:
    plugin = _plugin("nanogpt")
    assert plugin.get_native_endpoint("m", "chat", profile="subscription") == (
        "https://nano-gpt.com/api/subscription/v1/chat/completions"
    )
    assert plugin.get_native_endpoint("m", "chat") == (
        "https://nano-gpt.com/api/v1/chat/completions"
    )


def test_openrouter_anthropic_face_routes_and_auth() -> None:
    plugin = _plugin("openrouter")
    assert plugin.get_native_endpoint("m", "messages", profile="anthropic") == (
        "https://openrouter.ai/api/v1/messages"
    )
    # OpenRouter authenticates every face with Bearer (not the Anthropic
    # protocol's conventional x-api-key).
    assert plugin.get_native_headers("KEY", profile="anthropic") == {
        "Authorization": "Bearer KEY"
    }


def test_gemini_native_listing_hint_and_endpoints() -> None:
    plugin = _plugin("gemini")
    assert plugin.listing_profile == "native"
    assert plugin.get_native_endpoint("gemini-2.5-pro", "generate") == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    )
    assert plugin.get_native_headers("KEY") == {"x-goog-api-key": "KEY"}
    # The compat face is addressable and Bearer-authed.
    assert plugin.get_native_endpoint("gemini-2.5-pro", "chat", profile="openai") == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert plugin.get_native_headers("KEY", profile="openai") == {
        "Authorization": "Bearer KEY"
    }


def test_chutes_custom_listing_filters_routing_pseudo_models() -> None:
    """The one kept hand-rolled lister: the gateway advertises ``default``
    and comma-separated fallback chains that are not callable ids."""

    from rotator_library.providers.chutes_provider import ChutesProvider

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "deepseek-ai/DeepSeek-V3.2-TEE"},
                    {"id": "default"},
                    {"id": "a,b:latency"},
                ]
            }

    class _Client:
        async def get(self, url, headers=None, **kwargs):
            return _FakeResponse()

    assert asyncio.run(ChutesProvider().get_models("k", _Client())) == [
        "chutes/deepseek-ai/DeepSeek-V3.2-TEE"
    ]
