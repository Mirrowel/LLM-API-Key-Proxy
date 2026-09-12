"""G2 proxy-tools reference demo: strip/inject/intercept seams + end-to-end."""

from __future__ import annotations

import pytest

from rotator_library.hooks.demo.proxy_tools import (
    STATE_HISTORY,
    STATE_INJECTED,
    STATE_INTERCEPTED,
    STATE_REENTER,
    STATE_STRIPPED,
    ToolCallInterceptorHook,
    ToolInjectorHook,
    ToolStripperHook,
    proxy_tools_registry_snapshot,
    reset_proxy_tools_registry,
)
from rotator_library.hooks.runner import PipelineRun, run_slot
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport

PROXY_TOOL = {
    "name": "proxy_time",
    "description": "Demo proxy-owned tool.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
}

WEATHER_TOOL = {"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}}
STOCKS_TOOL = {"type": "function", "function": {"name": "stocks", "description": "Get quotes", "parameters": {"type": "object"}}}


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_proxy_tools_registry()
    yield
    reset_proxy_tools_registry()


def _context(**overrides) -> NativeProviderContext:
    base = dict(
        provider="openai_test",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://provider.example/v1/chat/completions",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        operation="chat",
        headers={"Authorization": "Bearer test"},
        session_id="sess-1",
        scope_key="scope-1",
    )
    base.update(overrides)
    return NativeProviderContext(**base)


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, response_body: dict):
        self._response_body = response_body
        self.calls: list[dict] = []

    async def post(self, endpoint, headers=None, json=None, **kwargs):
        self.calls.append({"endpoint": endpoint, "headers": dict(headers or {}), "json": json})
        return _FakeResponse(self._response_body)


def _chat_response() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _tool_call_response(name: str = "proxy_time") -> dict:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{\"timezone\":\"UTC\"}"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _sent_wire_tool_names(client: _FakeClient) -> set[str]:
    tools = client.calls[0]["json"].get("tools") or []
    names = set()
    for entry in tools:
        if isinstance(entry, dict) and isinstance(entry.get("function"), dict):
            names.add(entry["function"].get("name"))
        elif isinstance(entry, dict):
            names.add(entry.get("name"))
    return {name for name in names if name}


# -- strip -------------------------------------------------------------------


async def test_strip_at_canonical_removes_tool_from_wire():
    stripper = ToolStripperHook(("weather",), stages=("parsed_canonical",))
    context = _context(hook_class_declarations=(stripper,))
    client = _FakeClient(_chat_response())
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
               "tools": [WEATHER_TOOL, STOCKS_TOOL]}
    await NativeProviderExecutor().execute(request, context, NativeHTTPTransport(client))

    assert _sent_wire_tool_names(client) == {"stocks"}
    # canonical edit is visible: rebuild basis recorded, never a silent strip
    overlays = context.request_transport_overlays or []
    assert any(o.get("kind") == "canonical_rebuild" for o in overlays)
    # per-request state bag + per-session registry records
    state = context.pipeline_run.context.state
    assert state[STATE_STRIPPED] == ["weather"]
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["stripped"] == [{"stage": "parsed_canonical", "names": ["weather"]}]


async def test_strip_at_wire_only_keeps_raw_basis():
    stripper = ToolStripperHook(("weather",), stages=("provider_built",))
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
               "tools": [WEATHER_TOOL, STOCKS_TOOL]}
    context = _context(raw_client_request=request, hook_class_declarations=(stripper,))
    client = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(request, context, NativeHTTPTransport(client))

    assert _sent_wire_tool_names(client) == {"stocks"}
    overlays = context.request_transport_overlays or []
    assert not any(o.get("kind") == "canonical_rebuild" for o in overlays)
    assert any(o.get("kind") == "hook_edit" and o.get("stage") == "provider_built" for o in overlays)
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["stripped"] == [{"stage": "provider_built", "names": ["weather"]}]


async def test_strip_wire_tolerant_to_dialect_shapes():
    stripper = ToolStripperHook(("a",))

    async def strip(payload):
        outcome = await run_slot(PipelineRun(class_hooks=[stripper]), "provider_built", payload)
        return outcome.payload, outcome.modified

    # openai nested shape
    payload, modified = await strip({"tools": [WEATHER_TOOL, {"type": "function", "function": {"name": "a", "parameters": {}}}]})
    assert modified and [e["function"]["name"] for e in payload["tools"]] == ["weather"]
    # anthropic flat shape
    payload, modified = await strip({"tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}]})
    assert modified and [e["name"] for e in payload["tools"]] == ["b"]
    # gemini container shape
    payload, modified = await strip({"tools": [{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}]})
    assert modified and payload["tools"] == [{"functionDeclarations": [{"name": "b"}]}]
    # no match -> identity preserved (no modification flagged)
    payload, modified = await strip({"tools": [STOCKS_TOOL]})
    assert not modified and payload["tools"] == [STOCKS_TOOL]


# -- inject ------------------------------------------------------------------


async def test_inject_at_both_levels_idempotent():
    injector = ToolInjectorHook([PROXY_TOOL])
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
    context = _context(hook_class_declarations=(injector,))
    client = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(request, context, NativeHTTPTransport(client))

    wire_tools = client.calls[0]["json"]["tools"]
    names = [entry["function"]["name"] for entry in wire_tools]
    assert names == ["proxy_time"]  # canonical inject + wire no-op = exactly once

    # direct double-invocation is idempotent at both levels
    canonical_run = PipelineRun(class_hooks=[injector])
    outcome_first = await run_slot(canonical_run, "parsed_canonical", _parse(request))
    outcome_second = await run_slot(PipelineRun(class_hooks=[injector]), "parsed_canonical", outcome_first.payload)
    assert outcome_second.modified is False  # second pass finds the name present

    state = context.pipeline_run.context.state
    assert state[STATE_INJECTED] == ["proxy_time"]
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["proxy_owned"]["proxy_time"]["description"] == PROXY_TOOL["description"]


def _parse(request):
    from rotator_library.protocols import get_protocol

    protocol = get_protocol("openai_chat")
    return protocol.parse_request(dict(request), None)


async def test_inject_at_wire_only_appends_matching_dialect():
    injector = ToolInjectorHook([PROXY_TOOL], stages=("provider_built",))
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "tools": [STOCKS_TOOL]}
    context = _context(raw_client_request=request, hook_class_declarations=(injector,))
    client = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(request, context, NativeHTTPTransport(client))

    wire_tools = client.calls[0]["json"]["tools"]
    assert [e["function"]["name"] for e in wire_tools] == ["stocks", "proxy_time"]
    injected = wire_tools[1]
    assert injected["type"] == "function"
    assert injected["function"]["description"] == PROXY_TOOL["description"]
    assert injected["function"]["parameters"] == PROXY_TOOL["parameters"]
    # unknown-tool requests keep working: injector with no canonical stage
    # never touched the parsed request
    assert proxy_tools_registry_snapshot("scope-1", "sess-1")["proxy_owned"]["proxy_time"]


# -- intercept ---------------------------------------------------------------


async def test_intercept_with_custom_executor_records_and_flags_reenter():
    async def executor(tool_name: str, arguments) -> str:
        return "42 o'clock"

    injector = ToolInjectorHook([PROXY_TOOL])
    interceptor = ToolCallInterceptorHook(executor=executor)
    context = _context(hook_class_declarations=(injector, interceptor))
    client = _FakeClient(_tool_call_response())
    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "what time is it"}]},
        context, NativeHTTPTransport(client),
    )

    # no RESPOND: the provider response flows through (tool call visible)
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "proxy_time"
    state = context.pipeline_run.context.state
    assert state[STATE_INTERCEPTED][0]["result"] == "42 o'clock"
    assert state[STATE_REENTER] is True
    history = state[STATE_HISTORY]
    assert history[0]["role"] == "assistant"
    assert history[0]["tool_calls"][0]["function"]["name"] == "proxy_time"
    assert history[1] == {"role": "tool", "tool_call_id": "call_1", "name": "proxy_time", "content": "42 o'clock"}
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["interceptions"][0]["result"] == "42 o'clock"
    assert snapshot["history"][1]["content"] == "42 o'clock"


async def test_intercept_without_executor_responds_with_synthetic_answer():
    injector = ToolInjectorHook([PROXY_TOOL])
    interceptor = ToolCallInterceptorHook()  # stub executor + RESPOND
    context = _context(hook_class_declarations=(injector, interceptor))
    client = _FakeClient(_tool_call_response())
    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "what time is it"}]},
        context, NativeHTTPTransport(client),
    )

    message = result["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "proxy tool proxy_time executed: proxy-tool-result:proxy_time"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["object"] == "chat.completion"
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    state = context.pipeline_run.context.state
    assert state[STATE_REENTER] is True
    assert state[STATE_INTERCEPTED][0]["result"] == "proxy-tool-result:proxy_time"
    assert proxy_tools_registry_snapshot("scope-1", "sess-1")["interceptions"][0]["name"] == "proxy_time"


async def test_intercept_ignores_non_proxy_tool_calls():
    interceptor = ToolCallInterceptorHook()
    context = _context(hook_class_declarations=(interceptor,))
    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        context, NativeHTTPTransport(_FakeClient(_tool_call_response(name="client_tool"))),
    )
    # client-owned tool call passes through untouched
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "client_tool"
    assert STATE_REENTER not in context.pipeline_run.context.state


# -- session registry scoping ------------------------------------------------


async def test_registry_marks_proxy_owned_per_session():
    injector = ToolInjectorHook([PROXY_TOOL])
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
    await NativeProviderExecutor().execute(
        request, _context(session_id="s1", hook_class_declarations=(injector,)),
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    await NativeProviderExecutor().execute(
        request, _context(session_id="s2", hook_class_declarations=(injector,)),
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    await NativeProviderExecutor().execute(
        request, _context(session_id="s2", scope_key="scope-1", hook_class_declarations=()),
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    assert "proxy_time" in proxy_tools_registry_snapshot("scope-1", "s1")["proxy_owned"]
    assert "proxy_time" in proxy_tools_registry_snapshot("scope-1", "s2")["proxy_owned"]
    assert proxy_tools_registry_snapshot("scope-2", "s1")["proxy_owned"] == {}


# -- combined ----------------------------------------------------------------


async def test_strip_and_inject_combined():
    stripper = ToolStripperHook(("weather",))
    injector = ToolInjectorHook([PROXY_TOOL])
    context = _context(hook_class_declarations=(stripper, injector))
    client = _FakeClient(_chat_response())
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
               "tools": [WEATHER_TOOL, STOCKS_TOOL]}
    await NativeProviderExecutor().execute(request, context, NativeHTTPTransport(client))

    assert _sent_wire_tool_names(client) == {"stocks", "proxy_time"}
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["stripped"][0]["names"] == ["weather"]
    assert "proxy_time" in snapshot["proxy_owned"]


async def test_end_to_end_injected_tool_call_intercepted_synthetic_answer():
    """Full loop: inject -> model calls the proxy tool -> intercept -> answer."""

    async def executor(tool_name: str, arguments) -> str:
        return f"stubbed:{tool_name}"

    injector = ToolInjectorHook([PROXY_TOOL])
    interceptor = ToolCallInterceptorHook(executor=executor)
    context = _context(hook_class_declarations=(injector, interceptor))
    client = _FakeClient(_tool_call_response())
    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "what time is it"}]},
        context, NativeHTTPTransport(client),
    )

    # the wire request actually carried the injected proxy tool
    assert "proxy_time" in _sent_wire_tool_names(client)
    # interception ran through the caller-supplied executor
    state = context.pipeline_run.context.state
    assert state[STATE_INTERCEPTED][0]["result"] == "stubbed:proxy_time"
    assert state[STATE_REENTER] is True
    # custom executor => provider response is the client's answer
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "proxy_time"
    # registry carries the full session story
    snapshot = proxy_tools_registry_snapshot("scope-1", "sess-1")
    assert snapshot["proxy_owned"] and snapshot["interceptions"] and snapshot["history"]
