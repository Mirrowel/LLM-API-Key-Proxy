from __future__ import annotations

import json
from types import SimpleNamespace

from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.transaction_logger import TransactionLogger


def _executor() -> RequestExecutor:
    return RequestExecutor({}, None, None, None, {}, None)


def test_executor_accounts_for_non_streaming_usage_and_cost_trace(tmp_path, monkeypatch) -> None:
    executor = _executor()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=30,
            prompt_tokens_details={"cached_tokens": 40},
            completion_tokens_details={"reasoning_tokens": 10},
        )
    )
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    context = RequestContext(
        provider="openai",
        model="gpt-test",
        kwargs={},
        streaming=False,
        credentials=[],
        deadline=0,
        transaction_logger=logger,
    )
    monkeypatch.setattr(
        "rotator_library.usage.costs.litellm.get_model_info",
        lambda model: {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )

    usage, cost = executor._account_for_response_usage("openai", "gpt-test", response, context)

    assert usage.input_tokens == 60
    assert usage.cache_read_tokens == 40
    assert usage.completion_tokens == 20
    assert usage.reasoning_tokens == 10
    assert cost.total_cost > 0
    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["pass_name"] == "usage_accounting_summary"
    assert entries[-1]["data"]["usage"]["total_tokens"] == usage.total_tokens
