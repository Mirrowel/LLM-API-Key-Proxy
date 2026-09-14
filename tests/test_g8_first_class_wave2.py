# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Wave-2 first-class provider pins: chutes, nanogpt, openai."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_chutes_declaration_and_listing():
    from rotator_library.providers import PROVIDER_PLUGINS

    provider_class = PROVIDER_PLUGINS["chutes"]
    plugin = provider_class()
    assert plugin.default_api_base == "https://llm.chutes.ai/v1"
    assert plugin.adapter_names == ("chutes",)
    # The envelope declares the chat face; discovery rides the SHARED
    # listing cascade with the declarable routing pseudo-model exclusions.
    assert plugin.speaks == ("openai_chat",)
    assert plugin.get_protocol_name("m") == "openai_chat"
    assert plugin.transport_profiles is None
    assert plugin.listing_filters == ("default*", "*,*")
    assert "get_models" not in vars(provider_class)

    (row,) = plugin.model_rules
    assert row["match"] == "*"
    assert row["rename"] == {"max_completion_tokens": "max_tokens"}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "deepseek-ai/DeepSeek-V3.2-TEE"},
                    {"id": "default"},
                    {"id": "a,b:latency"},
                    {"no_id": True},
                ]
            }

    class _Client:
        async def get(self, url, headers=None, **kwargs):
            return _FakeResponse()

    models = asyncio.run(plugin.get_models("k", _Client()))
    assert models == ["chutes/deepseek-ai/DeepSeek-V3.2-TEE"]


def test_nanogpt_declaration_and_profiles():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["nanogpt"]()
    profiles, default = plugin.get_declared_profiles()
    assert set(profiles) == {"chat", "responses", "subscription"}
    assert profiles["subscription"]["protocol"] == "openai_chat"
    assert profiles["responses"]["protocol"] == "responses"
    assert default == "chat"
    assert plugin.adapter_names == ("nanogpt",)
    # The shared length rename is declared on the capability cascade.
    (row,) = plugin.model_rules
    assert row["match"] == "*"
    assert row["rename"] == {"max_completion_tokens": "max_tokens"}
    # The subscription pool's own base is a real override on the face —
    # not the inert legacy ``base_url`` key the profile machinery ignored.
    assert plugin.get_native_endpoint("m", "chat", profile="subscription") == (
        "https://nano-gpt.com/api/subscription/v1/chat/completions"
    )
    assert plugin.get_native_endpoint("m", "chat") == (
        "https://nano-gpt.com/api/v1/chat/completions"
    )
    assert plugin.transport_profiles is None
    assert plugin.default_profile is None


def test_openai_declaration_responses_first():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["openai"]()
    profiles, default = plugin.get_declared_profiles()
    # Responses is the default face (first speaks entry) and the token
    # counter's diverging route is the ONE real override on it.
    assert default == "responses"
    assert profiles["chat"]["protocol"] == "openai_chat"
    assert plugin.get_native_endpoint("m", "count_tokens") == (
        "https://api.openai.com/v1/responses/input_tokens"
    )
    # The chat face is addressed explicitly, exactly as before.
    assert plugin.get_native_endpoint("m", "chat", profile="chat") == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert plugin.get_native_endpoint("m", "chat") == "https://api.openai.com/v1/responses"


def test_chutes_nanogpt_adapters_and_declared_length_rename():
    from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
    from rotator_library.adapters.chutes import ChutesAdapter
    from rotator_library.adapters.nanogpt import NanoGPTAdapter
    from rotator_library.providers import PROVIDER_PLUGINS

    context = AdapterContext(provider="chutes", model="m")
    payload = {
        "model": "m",
        "messages": [],
        "max_completion_tokens": 512,
        "frequency_penalty": 0.3,
        "logprobs": True,
        "best_of": 2,
        "n": 3,
    }
    # The Chutes adapter owns the data-center sampling hygiene only: the
    # shared length rename is declared, so a raw payload keeps its spelling...
    raw = asyncio.run(ChutesAdapter().transform_request(payload, context))
    assert raw["max_completion_tokens"] == 512 and "max_tokens" not in raw
    # Penalties stay: the live sampling whitelist advertises them per model.
    assert "frequency_penalty" in raw
    assert "logprobs" not in raw and "best_of" not in raw
    assert raw["n"] == 1

    # ...while the full declared chain performs the rename on BOTH providers.
    for provider in ("chutes", "nanogpt"):
        plugin = PROVIDER_PLUGINS[provider]()
        chain_context = AdapterContext(
            provider=provider,
            model="m",
            protocol="openai_chat",
            adapter_config=plugin.get_adapter_config("m"),
        )
        adapters = [get_adapter(name) for name in plugin.get_adapter_names("m")]
        mapped = asyncio.run(
            run_adapter_chain(
                adapters,
                {"model": "m", "messages": [], "max_completion_tokens": 9},
                chain_context,
                stage="request",
            )
        )
        assert mapped["max_tokens"] == 9 and "max_completion_tokens" not in mapped

    response = {"choices": [{"message": {"role": "assistant", "reasoning": "hmm"}}]}
    fixed = asyncio.run(ChutesAdapter().transform_response(response, context))
    assert fixed["choices"][0]["message"]["reasoning_content"] == "hmm"
    chunk = {"choices": [{"delta": {"reasoning": "vllm"}}]}
    nanogpt_context = AdapterContext(provider="nanogpt", model="m")
    fixed = asyncio.run(NanoGPTAdapter().transform_stream_event(chunk, nanogpt_context))
    assert fixed["choices"][0]["delta"]["reasoning_content"] == "vllm"

    # NanoGPT additionally folds a top-level reasoning_tokens count into the
    # standard completion_tokens_details shape.
    usage_payload = {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 20, "reasoning_tokens": 7},
    }
    folded = asyncio.run(NanoGPTAdapter().transform_response(usage_payload, nanogpt_context))
    assert folded["usage"]["completion_tokens_details"]["reasoning_tokens"] == 7
    assert "reasoning_tokens" not in folded["usage"]


def test_402_balance_bodies_classify_as_quota():
    from rotator_library.core.errors import structured_api_response_error

    chutes = structured_api_response_error(
        {"error": {"message": "Quota exceeded and account balance is $0.0", "status_code": 402}},
    )
    assert chutes is not None and chutes.error_type == "quota_exceeded"
    nanogpt = structured_api_response_error(
        {"error": "Insufficient balance", "status_code": 402},
    )
    assert nanogpt is not None and nanogpt.error_type == "quota_exceeded"
