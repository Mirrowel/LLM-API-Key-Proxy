from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rotator_library.providers.provider_interface import SingletonABCMeta
from rotator_library.providers.vertex_provider import VertexProvider


@pytest.fixture
def vertex_provider(monkeypatch):
    SingletonABCMeta._instances.pop(VertexProvider, None)
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_LOCATION", "global")

    cache = MagicMock()
    cache.retrieve.return_value = None

    with patch(
        "rotator_library.providers.vertex_provider.ProviderCache",
        return_value=cache,
    ):
        provider = VertexProvider()

    yield provider, cache
    SingletonABCMeta._instances.pop(VertexProvider, None)


class TestVertexProviderThinkingPolicy:
    @pytest.mark.asyncio
    async def test_plain_request_does_not_auto_enable_thinking(self, vertex_provider):
        provider, _ = vertex_provider
        provider._non_stream_completion = AsyncMock(return_value="ok")

        result = await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-3.5-flash",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert result == "ok"
        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["model"] == "google/gemini-3.5-flash"
        assert "extra_body" not in payload

    @pytest.mark.asyncio
    async def test_explicit_thinking_config_is_forwarded(self, vertex_provider):
        provider, _ = vertex_provider
        provider._non_stream_completion = AsyncMock(return_value="ok")

        await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-3.5-flash",
            messages=[{"role": "user", "content": "Solve this"}],
            thinking={"include_thoughts": True, "budget_tokens": 128},
        )

        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True,
            "thinking_budget": 128,
        }
        assert payload["extra_body"]["google"]["thought_tag_marker"] == "thought"
        assert "reasoning_effort" not in payload

    @pytest.mark.asyncio
    async def test_reasoning_effort_passes_through_without_thinking_config(self, vertex_provider):
        provider, _ = vertex_provider
        provider._non_stream_completion = AsyncMock(return_value="ok")

        await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-3.5-flash",
            messages=[{"role": "user", "content": "Summarize this."}],
            reasoning_effort="medium",
        )

        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["reasoning_effort"] == "medium"
        assert "extra_body" not in payload

    @pytest.mark.asyncio
    async def test_explicit_thinking_wins_over_reasoning_effort(self, vertex_provider):
        provider, _ = vertex_provider
        provider._non_stream_completion = AsyncMock(return_value="ok")

        await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-3.5-flash",
            messages=[{"role": "user", "content": "Solve this."}],
            thinking={"include_thoughts": True},
            reasoning_effort="high",
        )

        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True,
        }
        assert "reasoning_effort" not in payload

    @pytest.mark.asyncio
    async def test_reasoning_history_enables_thinking_continuation(self, vertex_provider):
        provider, _ = vertex_provider
        provider._non_stream_completion = AsyncMock(return_value="ok")

        await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-3.5-flash",
            messages=[
                {
                    "role": "assistant",
                    "content": "I'll check.",
                    "reasoning_content": "Need to inspect the tool result first.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                {"role": "user", "content": "Continue."},
            ],
        )

        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True,
        }
        assert payload["messages"][0]["tool_calls"][0]["extra_content"]["google"][
            "thought_signature"
        ] == "skip_thought_signature_validator"

    @pytest.mark.asyncio
    async def test_cached_thought_signature_is_rehydrated_dynamically(self, vertex_provider):
        provider, cache = vertex_provider
        cache.retrieve.return_value = "cached-signature"
        provider._non_stream_completion = AsyncMock(return_value="ok")

        await provider.acompletion(
            MagicMock(),
            credential_identifier="test-project:test-key",
            model="vertex/gemini-2.5-flash",
            messages=[
                {
                    "role": "assistant",
                    "content": "Calling tool",
                    "tool_calls": [
                        {
                            "id": "call_cached",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                }
            ],
        )

        payload = provider._non_stream_completion.await_args.args[3]
        assert payload["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True,
        }
        assert payload["messages"][0]["tool_calls"][0]["extra_content"]["google"][
            "thought_signature"
        ] == "cached-signature"
