from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
from rotator_library.field_cache import FieldCacheContext, FieldCacheEngine, FieldCacheInjection, FieldCacheRule
from rotator_library.transaction_logger import TransactionLogger
from tests.txn_helpers import changes, error_records


def _trace_entries(logger):
    return changes(logger)



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

    entries = _trace_entries(logger)
    assert result["model"] == "native"
    assert [entry["pass_name"] for entry in entries] == ["before_adapter_chain", "after_adapter", "after_adapter_chain"]
    assert entries[1]["data"]["model"] == "native"
    assert entries[1]["detail"] == "after_adapter/request/adapter"


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
    context = FieldCacheContext(provider="openai", model="gpt-test", credential_id="credential_1", session_id="session_1", classifier="global")

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "hidden"}}]}, context, transaction_logger=logger)
    updated, _ = await engine.inject("request", {"messages": [{"role": "user"}, {"role": "assistant"}]}, context, transaction_logger=logger)

    entries = _trace_entries(logger)
    pass_names = [entry["pass_name"] for entry in entries]
    assert updated["messages"][-1]["reasoning_content"] == "hidden"
    assert pass_names == [
        "field_cache_extraction_start",
        "before_field_cache_extraction",
        "after_field_cache_extraction",
        "field_cache_extraction_complete",
        "field_cache_injection_start",
        "before_field_cache_injection",
        "after_field_cache_injection",
        "field_cache_injection_complete",
    ]
    # Extraction/injection passes carry the mutated payload in the change log.
    assert entries[2]["data"] is not None
    assert entries[6]["data"] is not None


@pytest.mark.asyncio
async def test_stream_sourced_rule_injection_trace_uses_request_direction(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="provider_session_id",
        source="stream_event",
        path="metadata.provider_session_id",
        inject=FieldCacheInjection(target="request", path="metadata.provider_session_id"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="openai", model="gpt-test", credential_id="credential_1", session_id="session_1", classifier="global")

    await engine.extract("stream_event", {"metadata": {"provider_session_id": "sid_1"}}, context, transaction_logger=logger)
    await engine.inject("request", {"metadata": {}}, context, transaction_logger=logger)

    entries = _trace_entries(logger)
    injection_entries = [entry for entry in entries if "injection" in entry["pass_name"]]
    assert injection_entries
    assert {entry["detail"].split("/")[1] for entry in injection_entries} == {"request"}


@pytest.mark.asyncio
async def test_field_cache_errors_emit_transform_log_error(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="bad_injection",
        source="response",
        path="choices.*.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.*.cached"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="openai", model="gpt-test", credential_id="credential_1", session_id="session_1", classifier="global")

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "hidden"}}]}, context, transaction_logger=logger)
    # G2 containment: the wildcard injection error is traced and the rule is
    # skipped; the request is not failed.
    updated, operations = await engine.inject("request", {"messages": [{"role": "user"}]}, context, transaction_logger=logger)
    assert operations[0].skipped is True
    assert operations[0].reason == "rule_error:FieldCachePathError"

    records = error_records(logger)
    error_entry = next(entry for entry in records if entry["failed_pass_name"] == "field_cache_inject")
    assert error_entry["stage"] == "adapter"
    assert updated == {"messages": [{"role": "user"}]}
