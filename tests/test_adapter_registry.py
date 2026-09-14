from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.adapters import (
    AdapterContext,
    PayloadAdapter,
    get_adapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
    resolve_adapter_name,
    run_adapter_chain,
)
from rotator_library.transaction_logger import TransactionLogger
from tests.txn_helpers import changes




def _trace_entries(logger):
    return changes(logger)



def test_adapter_registry_auto_discovers_builtins_and_aliases() -> None:
    adapters = list_adapters()

    assert "noop" in adapters
    assert "model_override" in adapters
    assert "suppress_developer_role" in adapters
    assert "antigravity_envelope" in adapters
    # The retired escape-hatch-thin adapters are gone (declarable behavior).
    assert "field_rename" not in adapters
    assert "reasoning_content" not in adapters
    assert resolve_adapter_name("passthrough") == "noop"
    assert get_adapter_class("override_model").name == "model_override"
    assert get_adapter("none") is get_adapter("noop")
    with pytest.raises(KeyError):
        resolve_adapter_name("field_copy")


def test_adapter_registry_rejects_duplicate_names_and_alias_collisions() -> None:
    class DuplicateNoop(PayloadAdapter):
        name = "noop"

    class AliasCollision(PayloadAdapter):
        name = "custom_alias_collision"
        aliases = ("noop",)

    with pytest.raises(ValueError):
        register_adapter(DuplicateNoop)
    with pytest.raises(ValueError):
        register_adapter(AliasCollision)


@pytest.mark.asyncio
async def test_noop_adapter_preserves_payload_without_mutating_original() -> None:
    payload = {"model": "original", "messages": []}

    result = await run_adapter_chain([get_adapter("noop")], payload, AdapterContext(), stage="request")

    assert result == payload
    assert result is not payload


@pytest.mark.asyncio
async def test_model_override_adapter_changes_model_from_config() -> None:
    payload = {"model": "public-name", "messages": []}
    context = AdapterContext(adapter_config={"model_override": {"model": "native-name"}})

    result = await run_adapter_chain([get_adapter("model_override")], payload, context, stage="request")

    assert result["model"] == "native-name"
    assert payload["model"] == "public-name"


@pytest.mark.asyncio
async def test_suppress_developer_role_converts_or_drops_messages() -> None:
    payload = {"messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]}

    system_result = await run_adapter_chain([get_adapter("suppress_developer_role")], payload, AdapterContext(), stage="request")
    drop_result = await run_adapter_chain(
        [get_adapter("suppress_developer_role")],
        payload,
        AdapterContext(adapter_config={"suppress_developer_role": {"mode": "drop"}}),
        stage="request",
    )

    assert system_result["messages"][0]["role"] == "system"
    assert [message["role"] for message in drop_result["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_antigravity_envelope_wraps_user_request_key() -> None:
    adapter = get_adapter("antigravity_envelope")

    result = await adapter.transform_request(
        {"model": "gemini-3-flash", "request": {"user_supplied": True}, "contents": [{"parts": [{"text": "hi"}]}]},
        AdapterContext(adapter_config={"antigravity_envelope": {"request_type": "CHAT_COMPLETION", "user_agent": "test-agent"}}),
    )

    assert result["requestType"] == "CHAT_COMPLETION"
    assert result["requestId"]
    assert result["request"]["request"] == {"user_supplied": True}
    assert result["request"]["contents"][0]["parts"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_antigravity_envelope_is_idempotent_for_controlled_envelope() -> None:
    adapter = get_adapter("antigravity_envelope")
    payload = {"model": "gemini-3-flash", "request": {"contents": []}, "requestType": "CHAT_COMPLETION", "requestId": "id"}

    result = await adapter.transform_request(payload, AdapterContext())

    assert result == payload


@pytest.mark.asyncio
async def test_adapter_chain_order_is_preserved() -> None:
    payload = {"model": "public", "messages": [{"role": "developer", "content": "rules"}]}
    context = AdapterContext(adapter_config={"model_override": {"model": "native"}})

    result = await run_adapter_chain(
        [get_adapter("model_override"), get_adapter("suppress_developer_role")],
        payload,
        context,
        stage="request",
    )

    assert result["model"] == "native"
    assert result["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_adapter_chain_traces_final_summary(tmp_path) -> None:
    logger = TransactionLogger("native", "native/test", parent_dir=tmp_path)
    payload = {"model": "public", "messages": []}
    context = AdapterContext(
        adapter_config={"model_override": {"model": "native"}},
        transaction_logger=logger,
        protocol="openai_chat",
    )

    result = await run_adapter_chain([get_adapter("model_override")], payload, context, stage="request")

    assert result["model"] == "native"
    entries = _trace_entries(logger)
    pass_names = [entry["pass_name"] for entry in entries]
    assert pass_names == ["before_adapter_chain", "after_adapter", "after_adapter_chain"]
    assert entries[1]["data"]["model"] == "native"
    assert entries[-1]["data"]["model"] == "native"
    assert entries[-1]["stage"] == "adapter"
    assert entries[-1]["detail"] == "after_adapter_chain/request/adapter"
