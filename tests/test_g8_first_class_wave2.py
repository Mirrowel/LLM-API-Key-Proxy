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
    from rotator_library.providers.chutes_provider import ChutesProvider

    plugin = PROVIDER_PLUGINS["chutes"]
    assert plugin.default_api_base == "https://llm.chutes.ai/v1"
    assert plugin.adapter_names == ("chutes",)

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
        async def get(self, url, headers=None):
            return _FakeResponse()

    models = asyncio.run(ChutesProvider().get_models("k", _Client()))
    assert models == ["chutes/deepseek-ai/DeepSeek-V3.2-TEE"]


def test_nanogpt_declaration_and_profiles():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["nanogpt"]
    profiles = plugin.transport_profiles
    assert set(profiles) == {"chat", "responses", "subscription"}
    assert profiles["subscription"]["base_url"] == "https://nano-gpt.com/api/subscription/v1"
    assert profiles["subscription"]["protocol"] == "openai_chat"
    assert plugin.default_profile == "chat"
    assert plugin.adapter_names == ("nanogpt",)


def test_openai_declaration_responses_first():
    from rotator_library.providers import PROVIDER_PLUGINS

    plugin = PROVIDER_PLUGINS["openai"]
    assert plugin.default_profile == "responses"
    profiles = plugin.transport_profiles
    assert profiles["responses"]["endpoint_paths"]["count_tokens"] == "/responses/input_tokens"
    assert profiles["chat"]["protocol"] == "openai_chat"


def test_chutes_nanogpt_adapters():
    from rotator_library.adapters.chutes_nanogpt import ChutesAdapter, NanoGPTAdapter
    from rotator_library.adapters.base import AdapterContext

    chutes = ChutesAdapter()
    nanogpt = NanoGPTAdapter()
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
    fixed = asyncio.run(chutes.transform_request(payload, context))
    assert fixed["max_tokens"] == 512 and "max_completion_tokens" not in fixed
    # Penalties stay: the live sampling whitelist advertises them per model.
    assert "frequency_penalty" in fixed
    assert "logprobs" not in fixed and "best_of" not in fixed
    assert fixed["n"] == 1
    mapped = asyncio.run(nanogpt.transform_request({"max_completion_tokens": 9}, context))
    assert mapped["max_tokens"] == 9

    response = {"choices": [{"message": {"role": "assistant", "reasoning": "hmm"}}]}
    fixed = asyncio.run(chutes.transform_response(response, context))
    assert fixed["choices"][0]["message"]["reasoning_content"] == "hmm"
    chunk = {"choices": [{"delta": {"reasoning": "vllm"}}]}
    fixed = asyncio.run(nanogpt.transform_stream_event(chunk, context))
    assert fixed["choices"][0]["delta"]["reasoning_content"] == "vllm"


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
