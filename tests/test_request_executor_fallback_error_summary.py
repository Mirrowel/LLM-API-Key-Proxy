from __future__ import annotations

from types import MethodType

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.routing import parse_route_target


def _context() -> RequestContext:
    return RequestContext(
        model="code",
        provider="requested",
        kwargs={"model": "code", "messages": []},
        streaming=False,
        credentials=["cred-a"],
        deadline=9999999999.0,
        routing_targets=(parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1")),
        routing_group_name="code_chain",
    )


@pytest.mark.asyncio
async def test_fallback_summary_includes_all_structured_target_failures_without_credentials() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_execute(self, context):
        attempts.append(context.provider)
        return {
            "error": {
                "type": "proxy_all_credentials_exhausted",
                "message": f"{context.provider} failed for cred-secret-value",
                "details": {"normal_error_summary": "1 rate_limit"},
            }
        }

    executor._execute_non_streaming = MethodType(fake_execute, executor)

    response = await executor._execute_non_streaming_with_fallback(_context())

    fallback_targets = response["error"]["details"]["fallback_targets"]
    assert attempts == ["codex", "openai"]
    assert [failure["provider"] for failure in fallback_targets] == ["codex", "openai"]
    assert [failure["error_type"] for failure in fallback_targets] == ["rate_limit", "rate_limit"]
    assert "credential" not in str(fallback_targets).lower()
    assert "cred-secret-value" not in str(fallback_targets)
