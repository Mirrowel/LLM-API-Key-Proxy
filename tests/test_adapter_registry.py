from __future__ import annotations

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


def test_adapter_registry_auto_discovers_builtins_and_aliases() -> None:
    adapters = list_adapters()

    assert "noop" in adapters
    assert "model_override" in adapters
    assert resolve_adapter_name("passthrough") == "noop"
    assert get_adapter_class("override_model").name == "model_override"
    assert get_adapter("none") is get_adapter("noop")


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
async def test_reasoning_content_adapter_copies_common_reasoning_field() -> None:
    payload = {"choices": [{"message": {"role": "assistant", "reasoning": "hidden"}}]}

    result = await run_adapter_chain([get_adapter("reasoning_content")], payload, AdapterContext(), stage="response")

    assert result["choices"][0]["message"]["reasoning_content"] == "hidden"


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
