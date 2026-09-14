# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 reasoning-effort system pins.

Three layers are locked here:

- the ladder math (``protocols.effort``): nearest accepted rung, searching
  upward first, ties rounding up; only explicit OFF words produce OFF;
  unknown words drop with a note;
- the resolution chain (base -> database seam -> provider code -> model
  rows -> config), later layers winning;
- the emission wiring (``native_provider.effort_emission``): the canonical
  word is normalized ONCE before the protocol builders, a provider's
  declared toggle shapes the chat wire (OFF drops the effort word and
  emits the disabled thinking object; ON rides next to enabled), and
  providers without a declaration keep their existing emission.
"""

from __future__ import annotations

import asyncio

import pytest

from rotator_library.native_provider import (
    NativeHTTPTransport,
    NativeProviderContext,
    NativeProviderExecutor,
)
from rotator_library.native_provider.effort_emission import (
    apply_reasoning_emission,
    normalize_request_effort,
    normalize_wire_effort,
)
from rotator_library.protocols.effort import (
    CHAT_BASE_ACCEPTED,
    normalize_effort,
    register_effort_database_resolver,
    resolve_accepted_effort,
    resolve_effort_toggle,
)
from rotator_library.protocols.types import UnifiedRequest


class _DeclaredPlugin:
    provider_env_name = "declared"
    reasoning_effort_accept = ("off", "low", "medium", "high", "max")
    reasoning_effort_toggle = True
    model_rules = (
        {"match": "old-*", "effort_accept": ["off", "low", "high", "max"]},
    )


class _PlainPlugin:
    provider_env_name = "plain"
    model_rules = ()


# --- ladder pins ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "accepted", "expected", "transformed"),
    [
        ("medium", ("low", "high"), "high", True),  # tie rounds up
        ("xhigh", ("off", "low", "high", "max"), "high", True),  # nearest down
        ("minimal", ("off", "low", "high", "max"), "low", True),
        ("ultra", ("off", "low", "medium", "high", "max"), "max", True),
        ("medium", ("off", "low", "medium", "high", "max"), "medium", False),  # native
        ("MAX", ("off", "low", "high", "max"), "max", False),  # spelling folds
    ],
)
def test_ladder_nearest_rung_and_tie_up(word, accepted, expected, transformed) -> None:
    normalized, note = normalize_effort(word, accepted)
    assert normalized == expected
    # Every real transformation is disclosed; vocabulary spelling is not.
    assert (note is not None) is transformed


@pytest.mark.parametrize("word", ["none", "off", "disable", "disabled", "NONE"])
def test_ladder_off_group_collapses(word) -> None:
    assert normalize_effort(word, ("off", "low", "high")) == ("off", None)


def test_ladder_off_requested_but_not_accepted_drops() -> None:
    normalized, note = normalize_effort("none", ("low", "high"))
    assert normalized is None
    assert "does not accept an off control" in note


def test_ladder_drop_when_no_on_rung_is_accepted() -> None:
    normalized, note = normalize_effort("high", ("off",))
    assert normalized is None
    assert "no on-reasoning rung" in note


def test_ladder_unknown_word_drops_with_note() -> None:
    normalized, note = normalize_effort("turbo", ("off", "high"))
    assert normalized is None
    assert "not a known reasoning-effort word" in note


# --- resolution chain (later layer wins) ---------------------------------------


def test_chain_starts_at_the_protocol_base() -> None:
    accepted, source = resolve_accepted_effort(None, "m", protocol_family="openai_chat")
    assert accepted == CHAT_BASE_ACCEPTED
    assert source == "protocol_base"


def test_chain_database_seam_overrides_base_and_provider_overrides_database() -> None:
    class DatabasePlugin(_PlainPlugin):
        provider_env_name = "database"

    try:
        register_effort_database_resolver(lambda provider, model: {"low", "high"} if provider == "database" else None)
        accepted, source = resolve_accepted_effort(DatabasePlugin(), "m")
        assert accepted == ("high", "low")
        assert source == "model_database"

        class ProviderOverride(DatabasePlugin):
            reasoning_effort_accept = ("off", "max")

        accepted, source = resolve_accepted_effort(ProviderOverride(), "m")
        assert accepted == ("off", "max")
        assert source == "provider_code"
    finally:
        register_effort_database_resolver(None)


def test_chain_model_rows_override_provider_then_config_wins() -> None:
    accepted, source = resolve_accepted_effort(_DeclaredPlugin(), "old-snapshot")
    assert accepted == ("off", "low", "high", "max")
    assert source == "model_rules:old-*"

    accepted, source = resolve_accepted_effort(
        _DeclaredPlugin(),
        "old-snapshot",
        runtime_config={"reasoning_effort_accept": ("off", "low")},
    )
    assert accepted == ("off", "low")
    assert source == "config"


def test_toggle_resolution_layers() -> None:
    assert resolve_effort_toggle(_DeclaredPlugin(), "current") is True
    assert resolve_effort_toggle(_DeclaredPlugin(), "old-snapshot") is True
    assert resolve_effort_toggle(_PlainPlugin(), "m") is False
    assert resolve_effort_toggle(_PlainPlugin(), "m", runtime_config={"reasoning_effort_toggle": True}) is True

    class RowToggle(_PlainPlugin):
        model_rules = (
            {"match": "*", "toggle": True},
            {"match": "off-*", "toggle": False},
        )

    assert resolve_effort_toggle(RowToggle(), "m") is True
    assert resolve_effort_toggle(RowToggle(), "off-me") is False


# --- canonical normalization (before any protocol build) -----------------------


def test_canonical_normalization_folds_the_word_once() -> None:
    request = UnifiedRequest(generation_params={"reasoning": {"effort": "medium"}})
    normalize_request_effort(
        request,
        provider_plugin=_DeclaredPlugin(),
        model="old-snapshot",
        protocol_name="openai_chat",
    )
    assert request.generation_params["reasoning"]["effort"] == "high"
    assert [warning.code for warning in request.warnings] == ["reasoning_effort_normalized"]

    native = UnifiedRequest(generation_params={"reasoning": {"effort": "medium"}})
    normalize_request_effort(
        native,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="openai_chat",
    )
    assert native.generation_params["reasoning"]["effort"] == "medium"
    assert native.warnings == []


def test_canonical_normalization_drops_unaccepted_off_and_unknown_words() -> None:
    class OnOnlyPlugin(_PlainPlugin):
        provider_env_name = "ononly"
        model_rules = ({"match": "*", "effort_accept": ["low", "high"]},)

    request = UnifiedRequest(generation_params={"reasoning": {"effort": "none"}})
    normalize_request_effort(
        request,
        provider_plugin=OnOnlyPlugin(),
        model="m",
        protocol_name="responses",
    )
    assert "reasoning" not in request.generation_params
    assert [warning.code for warning in request.warnings] == ["reasoning_effort_normalized"]

    unknown = UnifiedRequest(generation_params={"reasoning": {"effort": "turbo"}})
    normalize_request_effort(
        unknown,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="openai_chat",
    )
    assert "reasoning" not in unknown.generation_params
    assert [warning.code for warning in unknown.warnings] == ["reasoning_effort_normalized"]


def test_no_declaration_keeps_the_protocol_vocabulary_untouched() -> None:
    request = UnifiedRequest(generation_params={"reasoning": {"effort": "xhigh"}})
    normalize_request_effort(
        request,
        provider_plugin=_PlainPlugin(),
        model="m",
        protocol_name="openai_chat",
    )
    assert request.generation_params["reasoning"]["effort"] == "xhigh"
    assert request.warnings == []


def test_wire_normalization_folds_flat_key_and_records_the_note() -> None:
    request = UnifiedRequest()
    payload = {"reasoning_effort": "medium"}
    normalize_wire_effort(
        payload,
        unified_request=request,
        provider_plugin=_DeclaredPlugin(),
        model="old-snapshot",
        protocol_name="openai_chat",
    )
    assert payload["reasoning_effort"] == "high"
    assert [warning.code for warning in request.warnings] == ["reasoning_effort_normalized"]

    # already-legal words pass through with no second note
    normalize_wire_effort(
        payload,
        unified_request=request,
        provider_plugin=_DeclaredPlugin(),
        model="old-snapshot",
        protocol_name="openai_chat",
    )
    assert payload["reasoning_effort"] == "high"
    assert len(request.warnings) == 1


def test_wire_normalization_keeps_the_wire_off_spelling() -> None:
    payload = {"reasoning_effort": "none"}
    normalize_wire_effort(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="openai_chat",
    )
    assert payload["reasoning_effort"] == "none"


# --- toggle emission (chat wire) -----------------------------------------------


def test_toggle_off_emits_disabled_and_drops_the_effort_word() -> None:
    payload = {"reasoning_effort": "none"}
    assert apply_reasoning_emission(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="openai_chat",
    ) is True
    assert payload == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in payload


def test_toggle_on_rides_next_to_the_folded_word() -> None:
    payload = {"reasoning_effort": "medium"}
    normalize_wire_effort(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="old-snapshot",
        protocol_name="openai_chat",
    )
    assert apply_reasoning_emission(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="old-snapshot",
        protocol_name="openai_chat",
    ) is True
    assert payload == {"reasoning_effort": "high", "thinking": {"type": "enabled"}}


def test_toggle_never_touches_other_wires() -> None:
    payload = {"reasoning": {"effort": "none"}}
    assert apply_reasoning_emission(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="responses",
    ) is False
    assert payload == {"reasoning": {"effort": "none"}}
    assert apply_reasoning_emission(
        payload,
        provider_plugin=_DeclaredPlugin(),
        model="current",
        protocol_name="anthropic_messages",
    ) is False


def test_non_toggle_provider_keeps_plain_effort_emission() -> None:
    class NonTogglePlugin(_PlainPlugin):
        provider_env_name = "nontoggle"
        model_rules = ({"match": "*", "effort_accept": ["off", "high"]},)

    payload = {"reasoning_effort": "medium"}
    normalize_wire_effort(
        payload,
        provider_plugin=NonTogglePlugin(),
        model="m",
        protocol_name="openai_chat",
    )
    assert apply_reasoning_emission(
        payload,
        provider_plugin=NonTogglePlugin(),
        model="m",
        protocol_name="openai_chat",
    ) is False
    assert payload == {"reasoning_effort": "high"}


# --- executor wiring (the provider build is the seam) --------------------------


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append(json)
        return _FakeHTTPResponse(self.response)


_CHAT_RESPONSE = {
    "id": "chat_1",
    "object": "chat.completion",
    "model": "deepseek-v4-pro",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}


def _deepseek_plugin():
    from rotator_library.providers import PROVIDER_PLUGINS

    return PROVIDER_PLUGINS["deepseek"]()


def _context(plugin, model: str, *, raw_client_request: dict | None = None) -> NativeProviderContext:
    return NativeProviderContext(
        provider="deepseek",
        model=model,
        protocol_name="openai_chat",
        endpoint="https://api.deepseek.com/chat/completions",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        provider_plugin=plugin,
        raw_client_request=raw_client_request,
    )


def test_executor_rebuild_normalizes_and_emits_the_toggle() -> None:
    plugin = _deepseek_plugin()
    client = _FakeHTTPClient(_CHAT_RESPONSE)

    async def _drive():
        await NativeProviderExecutor().execute(
            {
                "model": "deepseek-v4-pro-0813",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "medium",
            },
            _context(plugin, "deepseek-v4-pro-0813"),
            NativeHTTPTransport(client),
        )

    asyncio.run(_drive())
    sent = client.calls[0]
    assert sent["reasoning_effort"] == "high"
    assert sent["thinking"] == {"type": "enabled"}


class _FakeStreamingClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[dict] = []

    async def stream_json_lines(self, endpoint, *, headers, json):
        self.calls.append(json)
        for chunk in self.chunks:
            yield chunk


def test_cache_family_idle_default_is_three_days() -> None:
    """G8 riding fix: no per-rule TTLs anymore — the storage engine's cache
    family owns retention, now 3 days of inactivity."""

    from rotator_library.storage.engine import get_engine

    assert get_engine("cache")._idle_prune == 3 * 86400.0


def test_executor_cross_protocol_normalizes_the_canonical_word() -> None:
    plugin = _deepseek_plugin()
    client = _FakeHTTPClient(_CHAT_RESPONSE)
    context = NativeProviderContext(
        provider="deepseek",
        model="deepseek-v4-pro-0813",
        protocol_name="openai_chat",
        endpoint="https://api.deepseek.com/chat/completions",
        input_protocol_name="responses",
        client_protocol_name="responses",
        provider_plugin=plugin,
    )

    async def _drive():
        await NativeProviderExecutor().execute(
            {
                "model": "deepseek-v4-pro-0813",
                "input": "hi",
                "reasoning": {"effort": "medium"},
            },
            context,
            NativeHTTPTransport(client),
        )

    asyncio.run(_drive())
    sent = client.calls[0]
    # The canonical word was folded BEFORE the chat build (the formatter
    # emitted the provider-legal rung), then the toggle rode along.
    assert sent["reasoning_effort"] == "high"
    assert sent["thinking"] == {"type": "enabled"}


def test_executor_stream_path_normalizes_and_emits_the_toggle() -> None:
    plugin = _deepseek_plugin()
    client = _FakeStreamingClient(
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
            "[DONE]",
        ]
    )

    async def _drive():
        async for _ in NativeProviderExecutor().stream(
            {
                "model": "deepseek-v4-flash-0813",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "medium",
            },
            _context(plugin, "deepseek-v4-flash-0813"),
            NativeHTTPTransport(client),
        ):
            pass

    asyncio.run(_drive())
    sent = client.calls[0]
    assert sent["reasoning_effort"] == "high"
    assert sent["thinking"] == {"type": "enabled"}


def test_executor_raw_chat_fast_path_normalizes_the_flat_key() -> None:
    plugin = _deepseek_plugin()
    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
    }
    client = _FakeHTTPClient(_CHAT_RESPONSE)

    async def _drive():
        await NativeProviderExecutor().execute(
            dict(payload),
            _context(plugin, "deepseek-v4-pro", raw_client_request=dict(payload)),
            NativeHTTPTransport(client),
        )

    asyncio.run(_drive())
    sent = client.calls[0]
    assert "reasoning_effort" not in sent
    assert sent["thinking"] == {"type": "disabled"}
