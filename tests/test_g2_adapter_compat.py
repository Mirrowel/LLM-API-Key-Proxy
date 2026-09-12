"""G2 adapters-as-hooks compatibility bridge: mapping, mirroring, parity."""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from rotator_library.adapters.base import AdapterContext, PayloadAdapter
from rotator_library.adapters.builtin import ModelOverrideAdapter, NoOpAdapter
from rotator_library.adapters.registry import register_adapter
from rotator_library.hooks import adapters_compatible_hook
from rotator_library.hooks.adapter_compat import AdapterHookBridge
from rotator_library.hooks.registry import get_hook, list_hooks, register_hook
from rotator_library.hooks.runner import PipelineRun, run_slot
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport


class _CaptureAdapter(PayloadAdapter):
    """Records every AdapterContext it sees; mutates dict payloads."""

    name = "g2_capture_adapter"
    supported_stages = ("request", "response", "stream_event")

    def __init__(self):
        self.contexts: list[AdapterContext] = []
        self.stages: list[str] = []

    async def transform(self, stage: str, payload, context: AdapterContext):
        self.contexts.append(context)
        self.stages.append(stage)
        if isinstance(payload, dict):
            updated = dict(payload)
            updated["captured"] = True
            return updated
        return payload


class _AppendContentAdapter(PayloadAdapter):
    """Appends a marker to the assistant content (response parity vehicle)."""

    name = "g2_append_content"
    supported_stages = ("response",)

    async def transform_response(self, payload, context: AdapterContext):
        if not isinstance(payload, dict) or not payload.get("choices"):
            return payload
        updated = dict(payload)
        updated["choices"] = [dict(choice) for choice in updated["choices"]]
        message = dict(updated["choices"][0].get("message") or {})
        if isinstance(message.get("content"), str):
            message["content"] = message["content"] + " [adapted]"
        updated["choices"][0]["message"] = message
        return updated


class _UpperStreamAdapter(PayloadAdapter):
    """Uppercases text deltas on stream events (parity-test vehicle)."""

    name = "g2_upper_stream"
    supported_stages = ("stream_event",)

    async def transform_stream_event(self, payload, context: AdapterContext):
        delta = getattr(payload, "delta", None)
        if delta is None:
            return payload
        blocks = [
            _replace(block, text=(block.text or "").upper()) if getattr(block, "type", "") == "text" else block
            for block in (delta.content or [])
        ]
        return _replace(payload, delta=_replace(delta, content=blocks))


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


def _chat_response(**message_overrides) -> dict:
    message = {"role": "assistant", "content": "hi"}
    message.update(message_overrides)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


# -- unit pins ---------------------------------------------------------------


def test_stage_mapping_from_supported_stages():
    request_only = adapters_compatible_hook(ModelOverrideAdapter())
    assert request_only.name == "adapter:model_override"
    assert request_only.stages == ("mutated",)
    assert request_only.aliases == ("adapter:override_model",)
    assert request_only.critical is True  # matches run_adapter_chain fail-fast

    all_stages = adapters_compatible_hook(NoOpAdapter())
    assert all_stages.stages == ("mutated", "response_received", "stream_event")


def test_registered_adapters_are_mirrored_as_hooks():
    for name in ("noop", "model_override", "field_rename", "reasoning_content"):
        assert f"adapter:{name}" in list_hooks()
    bridge = get_hook("adapter:model_override")
    assert isinstance(bridge, AdapterHookBridge)
    # alias mirror resolves to the same declaration
    assert get_hook("adapter:passthrough") is get_hook("adapter:noop")


async def test_adapter_context_built_from_hook_context_fields():
    capture = _CaptureAdapter()
    bridge = AdapterHookBridge(
        capture,
        protocol="openai_chat",
        adapter_config={"g2_capture_adapter": {"x": 1}},
    )
    run = PipelineRun(
        provider="prov", model="mdl", credential_id="cred1", session_id="sess1",
        scope_key="scope1", classifier="cls", operation="chat",
        class_hooks=[bridge],
    )
    outcome = await run_slot(run, "mutated", {"k": 1})
    assert outcome.payload == {"k": 1, "captured": True}
    assert capture.stages == ["request"]
    ctx = capture.contexts[0]
    assert isinstance(ctx, AdapterContext)
    assert ctx.provider == "prov"
    assert ctx.model == "mdl"
    assert ctx.credential_id == "cred1"
    assert ctx.session_id == "sess1"
    assert ctx.scope_key == "scope1"
    assert ctx.classifier == "cls"
    assert ctx.protocol == "openai_chat"
    assert ctx.metadata["operation"] == "chat"
    assert ctx.config_for("g2_capture_adapter") == {"x": 1}


async def test_adapter_config_state_bag_fallback():
    capture = _CaptureAdapter()
    bridge = AdapterHookBridge(capture)  # no static config declared
    run = PipelineRun(class_hooks=[bridge])
    run.context.state["adapter_config"] = {"g2_capture_adapter": {"from_state": True}}
    await run_slot(run, "response_received", {})
    assert capture.contexts[-1].config_for("g2_capture_adapter") == {"from_state": True}


async def test_stream_event_stage_routes_to_transform_stream_event():
    capture = _CaptureAdapter()
    bridge = AdapterHookBridge(capture)
    run = PipelineRun(class_hooks=[bridge])
    await run_slot(run, "stream_event", {"evt": 1}, direction="stream", event_index=0)
    assert capture.stages == ["stream_event"]


def test_collision_rules_mirror_adapter_registry():
    class _CollideAdapter(PayloadAdapter):
        name = "g2_collide"
        supported_stages = ("request",)

    # squat the hook name first: the adapter registry does not know it yet
    register_hook(adapters_compatible_hook(_CollideAdapter()))
    with pytest.raises(ValueError):
        register_adapter(_CollideAdapter)
    # replace clears the collision, same as the adapter registry contract
    register_adapter(_CollideAdapter, replace=True)
    assert "adapter:g2_collide" in list_hooks()


# -- parity pins: hook declaration vs adapter_names --------------------------


async def test_request_parity_hook_declaration_vs_adapter_names():
    """THE migration pin: same adapter, same wire output, both surfaces."""
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}
    config = {"model_override": {"model": "provider/native-model-x"}}

    via_adapters = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(
        request,
        _context(adapter_names=("model_override",), adapter_config=config),
        NativeHTTPTransport(via_adapters),
    )

    via_hooks = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(
        request,
        _context(
            adapter_names=(),
            adapter_config=config,
            hook_class_declarations=(adapters_compatible_hook("model_override", adapter_config={"model_override": {"model": "provider/native-model-x"}}),
        ),
        ),
        NativeHTTPTransport(via_hooks),
    )

    assert via_adapters.calls[0]["json"]["model"] == "provider/native-model-x"
    assert via_hooks.calls[0]["json"] == via_adapters.calls[0]["json"]


async def test_response_parity_hook_declaration_vs_adapter_names():
    register_adapter(_AppendContentAdapter, replace=True)
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}
    raw_response = _chat_response(content="hi")

    via_adapters = _FakeClient(raw_response)
    result_a = await NativeProviderExecutor().execute(
        request, _context(adapter_names=("g2_append_content",)), NativeHTTPTransport(via_adapters),
    )

    via_hooks = _FakeClient(raw_response)
    result_b = await NativeProviderExecutor().execute(
        request,
        _context(hook_class_declarations=(adapters_compatible_hook("g2_append_content"),)),
        NativeHTTPTransport(via_hooks),
    )
    assert result_a["choices"][0]["message"]["content"] == "hi [adapted]"
    assert result_b == result_a


async def test_stream_parity_hook_declaration_vs_adapter_names():
    register_adapter(_UpperStreamAdapter, replace=True)

    def chunks() -> list[dict]:
        return [
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}, "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
             "choices": [{"index": 0, "delta": {"content": "y"}, "finish_reason": "stop"}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
             "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        ]

    class _FakeStreamClient:
        def __init__(self, payload: list[dict]):
            self._chunks = payload

        def stream(self, method, endpoint, headers=None, json=None, **kwargs):
            import json as _json
            lines = ["data: " + _json.dumps(chunk) for chunk in self._chunks]
            lines.append("data: [DONE]")
            text = "\n\n".join(lines) + "\n\n"

            class _Resp:
                status_code = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                def aiter_lines(self):
                    async def _gen():
                        for line in text.splitlines():
                            yield line
                    return _gen()

            return _Resp()

    async def texts_for(context) -> list[str]:
        collected: list[str] = []
        async for item in NativeProviderExecutor().stream(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            context,
            NativeHTTPTransport(_FakeStreamClient(chunks())),
        ):
            for event in getattr(item, "events", [item]):
                if event.type == "message_delta" and event.delta is not None:
                    collected.append("".join(b.text or "" for b in (event.delta.content or []) if getattr(b, "type", "") == "text"))
        return collected

    via_adapters = await texts_for(_context(adapter_names=("g2_upper_stream",)))
    via_hooks = await texts_for(
        _context(hook_class_declarations=(adapters_compatible_hook("g2_upper_stream"),))
    )
    assert via_adapters == ["HE", "Y"]
    assert via_hooks == via_adapters
