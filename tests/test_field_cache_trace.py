from __future__ import annotations

import json

import pytest

from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
from rotator_library.field_cache import FieldCacheContext, FieldCacheEngine, FieldCacheInjection, FieldCacheRule
from rotator_library.transaction_logger import TransactionLogger


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_adapter_chain_emits_before_after_trace_entries(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    context = AdapterContext(
        provider="openai",
        model="gpt-test",
        protocol="openai_chat",
        credential_id="cred_1",
        transport="http",
        transaction_logger=logger,
        adapter_config={"model_override": {"model": "native"}},
    )

    result = await run_adapter_chain([get_adapter("model_override")], {"model": "public"}, context, stage="request")

    entries = _trace_entries(logger.log_dir)
    assert result["model"] == "native"
    assert [entry["pass_name"] for entry in entries] == ["before_adapter_chain", "after_adapter", "after_adapter_chain"]
    assert entries[1]["metadata"]["adapter"] == "model_override"
    assert entries[1]["metadata"]["changed"] is True
    assert entries[1]["credential_id"] == "cred_1"


@pytest.mark.asyncio
async def test_field_cache_extract_and_inject_emit_before_after_trace_entries(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="reasoning_content",
        source="response",
        path="choices.*.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="messages[-1].reasoning_content"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="openai", model="gpt-test", session_id="session_1", classifier="global")

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "hidden"}}]}, context, transaction_logger=logger)
    updated, _ = await engine.inject("request", {"messages": [{"role": "user"}]}, context, transaction_logger=logger)

    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert updated["messages"][-1]["reasoning_content"] == "hidden"
    assert pass_names == [
        "before_field_cache_extraction",
        "after_field_cache_extraction",
        "before_field_cache_injection",
        "after_field_cache_injection",
    ]
    assert entries[1]["metadata"]["rule_name"] == "reasoning_content"
    assert entries[1]["metadata"]["matched"] == 1
    assert entries[3]["metadata"]["hit"] is True
    assert entries[3]["metadata"]["changed"] is True


@pytest.mark.asyncio
async def test_field_cache_errors_emit_transform_log_error(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="bad_injection",
        source="response",
        path="choices.*.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="messages.*.reasoning_content"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="openai", model="gpt-test", session_id="session_1", classifier="global")

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "hidden"}}]}, context, transaction_logger=logger)
    with pytest.raises(Exception):
        await engine.inject("request", {"messages": [{"role": "user"}]}, context, transaction_logger=logger)

    entries = _trace_entries(logger.log_dir)
    error_entry = next(entry for entry in entries if entry["pass_name"] == "transform_log_error")
    assert error_entry["data"]["failed_pass_name"] == "field_cache_inject"
    assert error_entry["metadata"]["rule_name"] == "bad_injection"
