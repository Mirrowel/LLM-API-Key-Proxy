# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI Responses protocol adapter.

Responses is important enough to model natively rather than forcing it through a
chat-completions shape. This adapter focuses on loss-conscious parsing and
formatting; storage, routes, and WebSocket transport are later phases.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .canonical import (
    record_instruction_merge,
    disclose_response_drops,
    STOP_REASON_CONTENT_FILTER,
    format_reasoning_controls,
    normalize_reasoning_controls,
    attach_conversion_summary,
    add_conversion_warning,
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
    canonical_tool_choice,
    coalesce_assistant_message,
    conversation_messages,
    format_stop_reason,
    format_structured_output,
    format_tool_choice,
    instruction_blocks,
    is_same_protocol,
    message_reasoning,
    message_tool_calls,
    message_tool_results,
    ordered_message_blocks,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_text,
    tool_result_text,
)
from .operation import OPERATION_GENERATE, OPERATION_RESPONSES
from .validation import validate_generative_request, validate_generative_response
from .types import (
    Annotation,
    BuiltinToolCall,
    ContentBlock,
    ConversionWarning,
    CostDetails,
    MediaSource,
    OutputItem,
    ProtocolContext,
    ReasoningBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedStreamEvent,
    Usage,
    first_text,
    text_blocks,
)

_GENERATION_PARAMS = {
    "background",
    "context_management",
    "conversation",
    "include",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "moderation",
    "parallel_tool_calls",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream_options",
    "temperature",
    "text",
    "tool_choice",
    "top_p",
    "top_logprobs",
    "truncation",
    "user",
}

_REQUEST_CORE_FIELDS = {
    "model",
    "input",
    "metadata",
    "modalities",
    "previous_response_id",
    "stream",
    "tools",
    *_GENERATION_PARAMS,
}


class ResponsesProtocol(ProtocolAdapter):
    """Adapter for OpenAI Responses request, response, and event stream shapes.

    The protocol keeps output items in addition to parsed messages because later
    response storage and continuation features need item-level fidelity.
    """

    name: ClassVar[str] = "responses"
    aliases: ClassVar[tuple[str, ...]] = ("openai_responses", "response_api")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_RESPONSES,)
    future_transports: ClassVar[tuple[str, ...]] = ("websocket",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        source_generation = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request and k != "instructions"}
        generation_params = _parse_responses_generation_params(source_generation)
        if "tool_choice" in generation_params:
            generation_params["tool_choice"] = canonical_tool_choice(generation_params["tool_choice"], self.name)
        return UnifiedRequest(
            operation=OPERATION_RESPONSES,
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=resolve_tool_result_names(self._parse_input(request.get("input"))),
            system=text_blocks(request.get("instructions")) if request.get("instructions") is not None else [],
            tools=[self._parse_tool(tool) for tool in request.get("tools") or []],
            stream=bool(request.get("stream", False)),
            modalities=[str(value).lower() for value in request.get("modalities") or []],
            generation_params=generation_params,
            response_format=deepcopy(generation_params.get("structured_output")),
            previous_response_id=request.get("previous_response_id"),
            metadata=deepcopy(request.get("metadata") or {}),
            source_protocol=self.name,
            extensions={self.name: {"generation_params": source_generation}},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "input": self._format_input(
                # D4 identity: same-protocol keeps role:system input messages
                # in the input array; cross-protocol promotes them into the
                # single instructions field below.
                list(unified_request.messages) if preserve_source else conversation_messages(unified_request),
                preserve_source=preserve_source,
                warnings=unified_request.warnings,
            ),
        }
        if preserve_source:
            # D4 identity: the instructions field and role:system input
            # messages are BOTH legal on this wire and stay exactly where
            # the client put them — no merge, no promotion, no warning.
            instructions_text = "\n\n".join(block.text or "" for block in unified_request.system if block.text)
            if instructions_text:
                payload["instructions"] = instructions_text
        else:
            instruction_parts = instruction_blocks(unified_request)
            instructions = "\n\n".join(block.text or "" for block in instruction_parts if block.text)
            if instructions:
                payload["instructions"] = instructions
            dropped_non_text = [block for block in instruction_parts if block.text is None]
            if dropped_non_text:
                add_conversion_warning(
                    unified_request,
                    code="instruction_block_dropped",
                    message=f"{len(dropped_non_text)} non-text instruction block(s) have no Responses instructions representation",
                    field="system",
                    target_protocol=self.name,
                )
        if not preserve_source:
            # The merge record describes the CROSS-PROTOCOL promotion; the
            # same-protocol path keeps both homes untouched (nothing merged).
            record_instruction_merge(unified_request, self.name)
        if unified_request.previous_response_id:
            payload["previous_response_id"] = unified_request.previous_response_id
        if unified_request.tools:
            payload["tools"] = [self._format_tool(tool, preserve_source=preserve_source) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        # NOTE: Responses has NO modalities field — the canonical concept
        # never leaks into upstream payloads; a foreign or rebuilt request
        # carrying it drops with a recorded warning (the raw fast path keeps
        # the client's own payload verbatim).
        if unified_request.modalities:
            add_conversion_warning(
                unified_request,
                code="unsupported_optional_control",
                message="modalities has no Responses representation; dropped",
                field="modalities",
                target_protocol=self.name,
            )
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(self._format_generation_params(unified_request, preserve_source=preserve_source))
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        output = deepcopy(response.get("output") or [])
        messages: list[UnifiedMessage] = []
        items: list[OutputItem] = []
        for index, item in enumerate(output):
            if isinstance(item, dict):
                parsed = self._parse_output_item(item)
                if parsed:
                    parsed.extra["_output_index"] = index
                    for block in parsed.content:
                        # Output-item blocks carry the item position as their
                        # content-block identity.
                        if block.index is None:
                            block.index = index
                    messages.append(parsed)
                    items.append(_output_item_from_message(parsed, item))
        stop_reason = canonical_stop_reason(response.get("status"))
        if stop_reason == "stop" and any(message_tool_calls(message) for message in messages):
            stop_reason = "tool_use"
        if stop_reason == "incomplete":
            # incomplete_details.reason is authoritative: content_filter
            # incompletions are safety stops, not token-budget stops.
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            if reason == "content_filter":
                stop_reason = "content_filter"
        return UnifiedResponse(
            operation=OPERATION_RESPONSES,
            logical_operation=OPERATION_GENERATE,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            items=items,
            output=output,
            stop_reason=stop_reason,
            usage=self.extract_usage(response, context),
            modalities=_responses_output_modalities(messages),
            metadata={"object": response.get("object"), "created_at": response.get("created_at"), "native_status": response.get("status"), "incomplete_details": deepcopy(response.get("incomplete_details"))},
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created_at", "model", "output", "usage", "status"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        disclose_response_drops(unified_response, self.name)
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        assistants = [m for m in unified_response.messages if m.role in {"assistant", "model"}]
        candidate_backed = len(assistants) > 1 and all(m.index is not None for m in assistants)
        if not preserve_source:
            dropped_media = [
                block.type
                for message in assistants
                for block in message.content
                if block.type in {"audio", "video", "image"}
            ]
            if dropped_media:
                _warn_responses_once(
                    unified_response,
                    code="media_dropped",
                    message="media output has no Responses output-part representation; dropped",
                    field=f"content[{dropped_media[0]}]",
                )
        if preserve_source and unified_response.output:
            output = deepcopy(unified_response.output)
            for fallback_index, message in enumerate(unified_response.messages):
                output_index = message.extra.get("_output_index", fallback_index)
                if isinstance(output_index, int) and 0 <= output_index < len(output):
                    output[output_index] = self._format_output_message(message, output_index)
                else:
                    output.append(self._format_output_message(message, fallback_index))
        elif candidate_backed:
            # D9 first-wins: a Responses object is one logical answer; extra
            # candidates degrade to the first with a recorded summary, never
            # concatenation.
            _warn_responses_once(
                unified_response,
                code="candidates_first_wins",
                message=f"{len(assistants) - 1} additional candidate(s) dropped: single-response object (first candidate wins)",
            )
            output = self._format_canonical_output(assistants[0], unified_response)
        else:
            output = self._format_canonical_output(coalesce_assistant_message(unified_response.messages), unified_response)
        native_status = unified_response.metadata.get("native_status")
        non_terminal = native_status in {"in_progress", "queued", "cancelled"}
        payload = {
            "id": unified_response.id,
            "object": unified_response.metadata.get("object", "response"),
            "created_at": unified_response.metadata.get("created_at"),
            "model": unified_response.model,
            # First-wins keeps the FIRST candidate's stop status (D9), never
            # the response-level reason that parse derived from the LAST one.
            # Non-terminal native statuses (queued/in_progress/cancelled)
            # replay verbatim — they are not incompletions.
            "status": native_status if (preserve_source and non_terminal) else format_stop_reason(
                (assistants[0].stop_reason or unified_response.stop_reason)
                if candidate_backed and assistants
                else unified_response.stop_reason,
                self.name,
            ),
            "output": output,
            "usage": _format_responses_usage(unified_response.usage),
        }
        if unified_response.metadata.get("incomplete_details"):
            payload["incomplete_details"] = deepcopy(unified_response.metadata["incomplete_details"])
        elif payload["status"] == "incomplete" and not preserve_source:
            # Incomplete reasons are evidence-based: max_tokens maps exactly;
            # content_filter only when the canonical stop says so; anything
            # else defaults to max_output_tokens (the SDK's default reason —
            # never a speculative content_filter claim).
            payload["incomplete_details"] = {
                "reason": "content_filter" if unified_response.stop_reason == STOP_REASON_CONTENT_FILTER else "max_output_tokens"
            }
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
        return attach_conversion_summary({k: v for k, v in payload.items() if v is not None}, unified_response)

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="done", raw=deepcopy(raw_event))
        data = _as_dict(event)
        event_type = str(data.get("type") or data.get("event") or "chunk")
        if event_type in {"error", "response.error"} or data.get("error") is not None:
            return UnifiedStreamEvent(type="error", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, error=deepcopy(data.get("error", data)), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            response_payload = data.get("response") if isinstance(data.get("response"), dict) else {}
            response = self.parse_response(response_payload, context)
            error = None
            if event_type == "response.failed":
                error = deepcopy(response_payload.get("error") or data.get("error") or {"type": "upstream_error", "message": "Provider response failed"})
            return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, message=response.messages[0] if response.messages else None, usage=response.usage, error=error, stop_reason=response.stop_reason, raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "response.output_text.delta":
            message = UnifiedMessage(role="assistant", content=text_blocks(data.get("delta") or ""))
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, output_index=data.get("output_index"), content_index=data.get("content_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            # Reasoning deltas (summary AND full-text families) stream into
            # canonical reasoning blocks — never silently dropped.
            reasoning = ReasoningBlock(type="reasoning", text=data.get("delta") or "")
            message = UnifiedMessage(role="assistant", content=[ContentBlock(type="reasoning", reasoning=reasoning)])
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, output_index=data.get("output_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "response.refusal.delta":
            message = UnifiedMessage(role="assistant", content=[ContentBlock(type="refusal", refusal=data.get("delta") or "")])
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, output_index=data.get("output_index"), content_index=data.get("content_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "response.output_text.annotation.added":
            annotation_payload = data.get("annotation")
            if isinstance(annotation_payload, dict):
                annotation = Annotation(type=str(annotation_payload.get("type") or "citation"), url=annotation_payload.get("url"), title=annotation_payload.get("title"), raw=deepcopy(annotation_payload))
                message = UnifiedMessage(role="assistant", content=[ContentBlock(type="citations_delta", annotations=[annotation])])
                return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, output_index=data.get("output_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type in {"response.web_search_call.in_progress", "response.web_search_call.searching", "response.web_search_call.completed"}:
            item = data.get("item") if isinstance(data.get("item"), dict) else {"type": "web_search_call", "id": data.get("item_id"), "status": "in_progress"}
            builtin = BuiltinToolCall(kind="web_search", call_id=item.get("id"), status=str(item.get("status") or "in_progress"), raw=deepcopy(item))
            message = UnifiedMessage(role="assistant", content=[ContentBlock(type="builtin_tool", builtin_tool=builtin)])
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, output_index=data.get("output_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "response.function_call_arguments.delta":
            call = ToolCall(id=data.get("call_id") or data.get("item_id"), arguments=data.get("delta") or "", index=data.get("output_index"))
            message = UnifiedMessage(role="assistant", content=[ContentBlock(type="tool_call", tool_call=call)], tool_calls=[call])
            return UnifiedStreamEvent(type="tool_call_delta", operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, delta=message, tool_call=call, item_id=data.get("item_id"), output_index=data.get("output_index"), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type in {"response.output_item.added", "response.output_item.done"} and isinstance(data.get("item"), dict):
            message = self._parse_output_item(data["item"])
            if message and message.tool_calls:
                for call in message.tool_calls:
                    call.index = data.get("output_index")
                    if call.arguments in ({}, ""):
                        call.arguments = None
            # Item events carry their output position — following deltas join
            # the same block identity instead of minting a second item.
            item_index = data.get("output_index") if isinstance(data.get("output_index"), int) else None
            return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, message=message, output_index=item_index, raw=deepcopy(raw_event), extra={"payload": data})
        return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type=event_type, raw=deepcopy(raw_event), extra={"payload": data})

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else payload
        if not isinstance(usage, dict) or not any(key.endswith("tokens") for key in usage):
            return None
        input_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
        output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
        cost = None
        cost_details = usage.get("cost_details")
        if isinstance(cost_details, dict):
            provider_cost = cost_details.get("total_cost") or cost_details.get("request_cost_usd") or cost_details.get("cost") or cost_details.get("estimated_cost")
            cost = CostDetails(
                provider_reported_cost=float(provider_cost) if provider_cost is not None else None,
                currency=str(cost_details.get("currency") or "USD"),
                source="usage.cost_details",
                metadata={k: deepcopy(v) for k, v in cost_details.items() if k not in {"total_cost", "cost", "currency"}},
            )
        return Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cache_read_tokens=int(input_details.get("cached_tokens") or 0),
            # Official spelling is cache_write_tokens; the creation_tokens
            # variant is retained as a lenient fallback.
            cache_write_tokens=int(
                input_details.get("cache_write_tokens")
                or input_details.get("cache_creation_tokens")
                or usage.get("cache_creation_tokens")
                or 0
            ),
            reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
            audio_tokens=int(input_details.get("audio_tokens") or 0),
            output_audio_tokens=int(output_details.get("audio_tokens") or 0),
            cost=cost,
            raw=deepcopy(usage),
        )

    def _parse_input(self, input_value: Any) -> list[UnifiedMessage]:
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [UnifiedMessage(role="user", content=text_blocks(input_value), raw=input_value)]
        if not isinstance(input_value, list):
            return [UnifiedMessage(role="user", content=[ContentBlock(type="unknown", raw=deepcopy(input_value))], raw=deepcopy(input_value))]
        messages = []
        for item in input_value:
            if isinstance(item, dict):
                messages.append(self._parse_input_item(item))
            else:
                messages.append(UnifiedMessage(role="user", content=text_blocks(str(item)), raw=deepcopy(item)))
        return messages

    def _parse_input_item(self, item: dict[str, Any]) -> UnifiedMessage:
        item_type = item.get("type")
        if item_type in {"message", None}:
            return UnifiedMessage(
                role=str(item.get("role") or "user"),
                content=self._parse_content(item.get("content")),
                raw=deepcopy(item),
                extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "role", "content"}},
            )
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            # Custom tool outputs share the function-output contract: the
            # neutral home is the same ToolResult (text content), so a
            # Responses conversation that used a custom tool replays to any
            # provider instead of hard-rejecting cross-protocol.
            result_content = canonical_tool_arguments(item.get("output"))
            return UnifiedMessage(
                role="tool",
                content=[ContentBlock(type="tool_result", tool_result=ToolResult(tool_call_id=item.get("call_id"), content=result_content), raw=deepcopy(item))],
                tool_call_id=item.get("call_id"),
                raw=deepcopy(item),
            )
        if item_type in {"function_call", "custom_tool_call"}:
            call = ToolCall(
                id=item.get("call_id") or item.get("id"),
                name=item.get("name"),
                arguments=canonical_tool_arguments(item.get("arguments") or item.get("input")),
                type="function" if item_type == "function_call" else str(item_type),
                raw=deepcopy(item),
            )
            return UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="tool_call", tool_call=call, raw=deepcopy(item))],
                tool_calls=[call],
                raw=deepcopy(item),
            )
        if item_type == "reasoning":
            reasoning = ReasoningBlock(
                type="reasoning",
                text=_reasoning_text(item),
                encrypted_content=item.get("encrypted_content") if isinstance(item.get("encrypted_content"), str) else None,
                # SDK requires id+summary on replay: the item id/status ride
                # extra for canonical rebuilds; content[] rides raw.
                extra={
                    "reasoning_item_id": item.get("id") if isinstance(item.get("id"), str) else None,
                    **{k: deepcopy(v) for k, v in item.items() if k not in {"type", "summary", "encrypted_content", "id"}},
                },
                raw=deepcopy(item),
            )
            return UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="reasoning", reasoning=reasoning, raw=deepcopy(item))],
                reasoning=[reasoning],
                raw=deepcopy(item),
            )
        return UnifiedMessage(role=str(item.get("role") or "user"), content=[ContentBlock(type=str(item_type or "unknown"), raw=deepcopy(item))], raw=deepcopy(item))

    def _format_input(self, messages: Iterable[UnifiedMessage], *, preserve_source: bool, warnings: list | None = None) -> list[dict[str, Any]]:
        """Format canonical turns into ordered Responses input items."""

        items: list[dict[str, Any]] = []
        for message in messages:
            visible: list[ContentBlock] = []

            def flush_visible() -> None:
                if not visible:
                    return
                residual_message = deepcopy(message)
                residual_message.content = list(visible)
                residual_message.tool_calls = []
                residual_message.reasoning = []
                items.append(self._format_input_message(residual_message, preserve_source=preserve_source, warnings=warnings))
                visible.clear()

            for block in ordered_message_blocks(message):
                if block.reasoning:
                    flush_visible()
                    if block.reasoning.text or block.reasoning.encrypted_content:
                        reasoning_item: dict[str, Any] = {"type": "reasoning"}
                        # SDK ResponseReasoningItemParam requires id +
                        # summary; id/status/content[] ride extra on
                        # same-protocol replay (verbatim when present).
                        if isinstance(block.reasoning.extra.get("reasoning_item_id"), str):
                            reasoning_item["id"] = block.reasoning.extra["reasoning_item_id"]
                        elif preserve_source and isinstance(message.raw, dict) and isinstance(message.raw.get("id"), str):
                            reasoning_item["id"] = message.raw["id"]
                        if block.reasoning.text:
                            reasoning_item["summary"] = [{"type": "summary_text", "text": block.reasoning.text}]
                        if block.reasoning.encrypted_content:
                            # Bound opaque state (D8): survives same-protocol
                            # rebuilds byte-for-byte; never emitted by foreign
                            # formatters.
                            reasoning_item["encrypted_content"] = block.reasoning.encrypted_content
                        if preserve_source and isinstance(block.raw, dict):
                            # Full-shape replay: content[] (full reasoning
                            # text items) + status ride the raw item.
                            for raw_key in ("content", "status"):
                                if raw_key in block.raw and raw_key not in reasoning_item:
                                    reasoning_item[raw_key] = deepcopy(block.raw[raw_key])
                        items.append(reasoning_item)
                elif block.builtin_tool is not None and isinstance(block.builtin_tool.raw, dict):
                    # Hosted-tool input items (web_search_call, mcp_list_tools,
                    # computer_call, ...) are top-level items with their own
                    # shapes — verbatim replay, never a message envelope.
                    flush_visible()
                    items.append(deepcopy(block.builtin_tool.raw))
                elif block.type not in {"text", "image", "audio", "file", "document", "refusal", "tool_call", "tool_result", "reasoning", "builtin_tool"} and isinstance(block.raw, dict) and block.raw.get("type"):
                    # Native Responses input item of an unrecognized type
                    # (item_reference, apply_patch_call, mcp_approval_request,
                    # ...): verbatim passthrough, never wrapped in
                    # {role, content}.
                    flush_visible()
                    items.append(deepcopy(block.raw))
                elif block.tool_call:
                    flush_visible()
                    items.append(self._format_function_call(block.tool_call, preserve_source=preserve_source))
                elif block.tool_result:
                    flush_visible()
                    items.append(self._format_function_result(block.tool_result, preserve_source=preserve_source))
                else:
                    visible.append(block)
            flush_visible()
        return items

    def _format_input_message(self, message: UnifiedMessage, *, preserve_source: bool = True, warnings: list | None = None) -> dict[str, Any]:
        if preserve_source and isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            if payload.get("type") == "function_call_output":
                payload["call_id"] = message.tool_call_id or payload.get("call_id")
                result = message.content[0].tool_result if message.content and message.content[0].tool_result else None
                if result:
                    payload["output"] = deepcopy(result.content)
                return payload
            if payload.get("type") not in (None, "message"):
                # Non-message raw items (item_reference, hosted-tool calls,
                # approvals, ...) replay verbatim — role/content are not
                # members of their shapes.
                return payload
            payload["role"] = message.role
            payload["content"] = self._format_content(message.content, role=message.role, preserve_source=preserve_source, warnings=warnings)
            return payload
        role = "assistant" if message.role in {"assistant", "model"} else "user"
        return {"type": "message", "role": role, "content": self._format_content(message.content, role=role, preserve_source=preserve_source, warnings=warnings)}

    def _parse_output_item(self, item: dict[str, Any]) -> UnifiedMessage | None:
        item_type = item.get("type")
        if item_type == "message":
            return UnifiedMessage(
                role=str(item.get("role") or "assistant"),
                content=self._parse_content(item.get("content")),
                raw=deepcopy(item),
                extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "role", "content"}},
            )
        if item_type == "reasoning":
            reasoning = ReasoningBlock(
                type="reasoning",
                text=_reasoning_text(item),
                encrypted_content=item.get("encrypted_content") if isinstance(item.get("encrypted_content"), str) else None,
                extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "summary", "encrypted_content"}},
            )
            reasoning.raw = deepcopy(item)
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="reasoning", reasoning=reasoning, raw=deepcopy(item))], reasoning=[reasoning], raw=deepcopy(item))
        if item_type in {"function_call", "custom_tool_call"}:
            call = ToolCall(id=item.get("call_id") or item.get("id"), name=item.get("name"), arguments=canonical_tool_arguments(item.get("arguments") or item.get("input")), type="function" if item_type == "function_call" else str(item_type), raw=deepcopy(item))
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="tool_call", tool_call=call, raw=deepcopy(item))], tool_calls=[call], raw=deepcopy(item))
        if item_type in _BUILTIN_TOOL_ITEM_TYPES:
            # Provider-executed tool records (web search, file search, code
            # interpreter, ...) are canonical capabilities (W2): never dropped,
            # never turned into empty successes.
            builtin = BuiltinToolCall(
                kind=str(item_type).removesuffix("_call") or str(item_type),
                call_id=item.get("call_id") or item.get("id"),
                status=item.get("status"),
                output=deepcopy(item.get("results") if "results" in item else item.get("output") if "output" in item else item.get("result")),
                raw=deepcopy(item),
            )
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="builtin_tool", builtin_tool=builtin, raw=deepcopy(item))], raw=deepcopy(item))
        if item_type and item_type not in {"message", "reasoning", "function_call", "custom_tool_call"}:
            # Unknown provider-executed / server-side item kinds (MCP list/
            # approval, shell calls, future tools) become catch-all builtin
            # records: retained with raw for same-protocol fidelity, guarded
            # cross-protocol — never silently dropped.
            builtin = BuiltinToolCall(
                kind=str(item_type),
                call_id=item.get("call_id") or item.get("id"),
                status=item.get("status"),
                output=deepcopy({k: v for k, v in item.items() if k not in {"type", "id", "call_id", "status"}}),
                raw=deepcopy(item),
                extra={"synthesized_kind": True},
            )
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="builtin_tool", builtin_tool=builtin, raw=deepcopy(item))], raw=deepcopy(item))
        return None

    def _format_output_message(self, message: UnifiedMessage, index: int) -> dict[str, Any]:
        if isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            item_type = payload.get("type")
            if item_type == "message":
                payload["role"] = message.role
                payload["content"] = self._format_content(message.content, role="assistant", output=True, preserve_source=True)
                return payload
            if item_type == "reasoning" and message.reasoning:
                payload["summary"] = [{"type": "summary_text", "text": message.reasoning[0].text or ""}]
                return payload
            if item_type in {"function_call", "custom_tool_call"} and message.tool_calls:
                call = message.tool_calls[0]
                payload["call_id"] = call.id
                payload["name"] = call.name
                if item_type == "custom_tool_call" and "input" in payload:
                    # Native input member present: NEVER stamp arguments
                    # alongside it (hybrid shapes 400 upstream).
                    return payload
                payload["arguments"] = tool_arguments_text(call.arguments)
                return payload
            if item_type in _BUILTIN_TOOL_ITEM_TYPES:
                # Provider-executed tool items round-trip their native shape.
                return payload
            if item_type and item_type != "message":
                # Unknown/extension items round-trip their native shape; only
                # genuine message items rebuild through canonical formatting.
                return payload
        return {"id": f"msg_{index}", "type": "message", "role": message.role, "content": self._format_content(message.content, role=message.role, output=True, preserve_source=False)}

    def _format_canonical_output(self, message: UnifiedMessage, unified_response: UnifiedResponse | None = None) -> list[dict[str, Any]]:
        """Build ordered Responses output items from one canonical assistant turn."""

        output: list[dict[str, Any]] = []
        visible: list[ContentBlock] = []
        item_index = 0

        def flush_visible() -> None:
            nonlocal item_index
            if not visible:
                return
            output.append(
                {
                    "id": f"msg_{item_index}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": self._format_content(visible, role="assistant", output=True, preserve_source=False),
                }
            )
            visible.clear()
            item_index += 1

        for block in ordered_message_blocks(message):
            if block.reasoning:
                flush_visible()
                if block.reasoning.text or block.reasoning.encrypted_content:
                    reasoning_output: dict[str, Any] = {
                        "id": f"rs_{item_index}",
                        "type": "reasoning",
                        "status": "completed",
                    }
                    if block.reasoning.text:
                        reasoning_output["summary"] = [{"type": "summary_text", "text": block.reasoning.text}]
                    if block.reasoning.encrypted_content:
                        reasoning_output["encrypted_content"] = block.reasoning.encrypted_content
                    output.append(reasoning_output)
                    item_index += 1
            elif block.tool_call:
                flush_visible()
                item = self._format_function_call(block.tool_call, preserve_source=False)
                item.setdefault("id", f"fc_{item_index}")
                item["status"] = "completed"
                output.append(item)
                item_index += 1
            elif block.tool_result:
                flush_visible()
                output.append(self._format_function_result(block.tool_result, preserve_source=False))
                item_index += 1
            elif block.type == "refusal" and block.refusal is not None:
                flush_visible()
                if block.annotations and unified_response is not None:
                    # Refusal parts carry no annotations slot (W2 carry-in):
                    # the loss is recorded, never silent.
                    _warn_responses_once(
                        unified_response,
                        code="annotations_dropped",
                        message="annotations on a refusal part have no Responses refusal representation; dropped",
                        field="content[refusal]",
                    )
                output.append(
                    {
                        "id": f"msg_{item_index}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": block.refusal}],
                    }
                )
                item_index += 1
            elif block.type == "builtin_tool" and block.builtin_tool is not None:
                flush_visible()
                builtin = block.builtin_tool
                raw = builtin.raw if isinstance(builtin.raw, dict) else None
                if raw is not None and raw.get("type") in _BUILTIN_TOOL_ITEM_TYPES:
                    # Responses-native item shape: round-trip verbatim.
                    item = deepcopy(raw)
                else:
                    # Foreign raw shapes (e.g. Anthropic server_tool_use blocks)
                    # never leak onto the Responses wire: synthesize the
                    # equivalent native call item.
                    item_type = builtin.kind if builtin.extra.get("synthesized_kind") else f"{builtin.kind}_call"
                    item = {
                        "id": builtin.call_id or f"bc_{item_index}",
                        "call_id": builtin.call_id or f"bc_{item_index}",
                        "type": item_type,
                        "status": builtin.status or "completed",
                    }
                    if builtin.output is not None:
                        item["results" if builtin.kind == "file_search" else "output"] = deepcopy(builtin.output)
                output.append(item)
                item_index += 1
            else:
                visible.append(block)
        flush_visible()
        return output

    def _parse_content(self, content: Any) -> list[ContentBlock]:
        if content is None:
            return []
        if isinstance(content, str):
            return text_blocks(content)
        if not isinstance(content, list):
            return [ContentBlock(type="unknown", raw=deepcopy(content))]
        blocks = []
        for block_index, block in enumerate(content):
            parsed_blocks = self._parse_content_block(block, block_index)
            blocks.extend(parsed_blocks)
        return blocks

    def _parse_content_block(self, block: Any, block_index: int = 0) -> list[ContentBlock]:
        if isinstance(block, str):
            return [ContentBlock(type="input_text", text=block, raw=block, index=block_index)]
        if not isinstance(block, dict):
            return [ContentBlock(type="unknown", raw=deepcopy(block), index=block_index)]
        block_type = str(block.get("type") or "text")
        if block_type in {"input_text", "output_text", "text"}:
            return [ContentBlock(
                type="text",
                text=block.get("text", ""),
                annotations=_parse_responses_annotations(block.get("annotations")),
                index=block_index,
                raw=deepcopy(block),
                extra={"source_type": block_type, **_without(block, {"type", "text", "annotations"})},
            )]
        if block_type == "refusal":
            return [ContentBlock(type="refusal", refusal=str(block.get("refusal") or ""), index=block_index, raw=deepcopy(block), extra=_without(block, {"type", "refusal"}))]
        if block_type in {"input_image", "image_url"}:
            source = _parse_responses_media_source(block)
            return [ContentBlock(type="image", source=source, index=block_index, raw=deepcopy(block), extra={"source_type": block_type, **_without(block, {"type", "image_url", "source"})})]
        if block_type in {"input_file", "file"}:
            source = _parse_responses_media_source(block)
            return [ContentBlock(type="file", source=source, index=block_index, raw=deepcopy(block), extra={"source_type": block_type, **_without(block, {"type", "file_id", "file_data", "file_url"})})]
        return [ContentBlock(type=block_type, index=block_index, raw=deepcopy(block), extra=_without(block, {"type"}))]

    def _format_content(self, blocks: Iterable[ContentBlock], *, role: str = "user", output: bool = False, preserve_source: bool = True, warnings: list | None = None) -> list[dict[str, Any]]:
        formatted = []
        for block in blocks:
            if block.type == "text":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                payload["type"] = "output_text" if output or role in {"assistant", "model"} else "input_text"
                payload["text"] = block.text or ""
                if block.annotations:
                    payload["annotations"] = _format_responses_annotations(block.annotations)
                if preserve_source:
                    payload.update({k: deepcopy(v) for k, v in block.extra.items() if k not in {"source_type", "annotations"}})
                formatted.append(payload)
            elif block.type == "refusal" and block.refusal is not None:
                if output:
                    formatted.append({"type": "refusal", "refusal": block.refusal})
                else:
                    # Request-side history (any role, including assistant
                    # turns): Responses input has no refusal part; degrade to
                    # input_text per D7 (validator admits refusal).
                    if warnings is not None and not preserve_source:
                        _warn_responses_list_once(
                            warnings,
                            code="incompatible_content_downgrade",
                            message="refusal has no Responses input part; degraded to input_text (the refusal semantics carry as text)",
                            field="content",
                        )
                    formatted.append({"type": "input_text", "text": block.refusal})
            elif block.type == "image":
                if output:
                    # Output parts are output_text/refusal only: assistant
                    # images drop with a recorded media_dropped summary —
                    # never fabricated as input_image output parts.
                    continue
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "input_image"}
                payload["type"] = "input_image"
                payload.update(_format_responses_image_source(block.source))
                if preserve_source:
                    payload.update({k: deepcopy(v) for k, v in block.extra.items() if k != "source_type"})
                formatted.append(payload)
            elif block.type in {"file", "document"}:
                payload = {"type": "input_file"}
                payload.update(_format_responses_file_source(block.source))
                if preserve_source:
                    # Cache hints (prompt_cache_breakpoint etc.) ride the
                    # part like the text/image branches — never dropped.
                    payload.update({k: deepcopy(v) for k, v in block.extra.items() if k != "source_type"})
                formatted.append(payload)
            elif preserve_source and isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
        return formatted

    def _parse_tool(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        parameters = payload.get("parameters") or payload.get("input_schema") or {}
        tool_type = str(payload.get("type") or "function")
        if tool_type not in {"function", "custom"}:
            # Hosted tools (web_search, file_search, code_interpreter,
            # image_generation, computer_use_preview, mcp, local_shell, ...):
            # identity is the type; no name/parameters exist to mint.
            return ToolDefinition(
                name=tool_type,
                description=payload.get("description"),
                input_schema={},
                type=tool_type,
                extra={"raw": deepcopy(tool), "hosted": True, **{k: deepcopy(v) for k, v in payload.items() if k not in {"type", "name", "description", "parameters", "input_schema"}}},
            )
        return ToolDefinition(
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
            input_schema=deepcopy(parameters),
            type=tool_type,
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"type", "name", "description", "parameters", "input_schema"}},
        )

    def _format_tool(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        if tool.type not in {"function", "custom"}:
            # Hosted tools: {type, ...config} only — name/parameters are not
            # members of their shapes (strict-param 400 upstream).
            if preserve_source and isinstance(tool.extra.get("raw"), dict):
                return deepcopy(tool.extra["raw"])
            payload: dict[str, Any] = {"type": tool.type}
            payload.update(deepcopy({k: v for k, v in tool.extra.items() if k not in {"raw", "hosted"}}))
            return payload
        payload = {"type": "function" if tool.type == "function" else tool.type, "name": tool.name, "parameters": deepcopy(tool.input_schema)}
        if tool.description is not None:
            payload["description"] = tool.description
        if preserve_source:
            payload.update(deepcopy(tool.extra))
        return payload

    def _format_function_call(self, call: ToolCall, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        if call.type in {"custom", "custom_tool_call"}:
            # Custom tool calls keep their native spelling (input, not
            # arguments) — coercing to function_call would change how the
            # provider pairs the call with its custom_tool_output.
            if preserve_source and isinstance(call.raw, dict) and ("input" in call.raw or "arguments" in call.raw):
                # Verbatim replay: the raw item carries its own member
                # spelling — never inject the OTHER member alongside (hybrid
                # input+arguments shapes 400 upstream).
                payload.update(
                    {
                        "type": "custom_tool_call",
                        "call_id": call.id or payload.get("call_id") or "",
                        "name": call.name or payload.get("name") or "",
                    }
                )
                return payload
            payload.update(
                {
                    "type": "custom_tool_call",
                    "call_id": call.id or "",
                    "name": call.name or "",
                    "input": tool_arguments_text(call.arguments),
                }
            )
            return payload
        payload.update(
            {
                "type": "function_call",
                "call_id": call.id or "",
                "name": call.name or "",
                "arguments": tool_arguments_text(call.arguments),
            }
        )
        return payload

    def _format_function_result(self, result: ToolResult, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(result.raw) if preserve_source and isinstance(result.raw, dict) else {}
        result_content = {"error": result.content} if result.is_error else result.content
        if isinstance(result_content, list):
            # SDK allows function_call_output.output to be a string OR an
            # item list (images/files): lists stay lists, never stringified.
            output_value = deepcopy(result_content)
        else:
            output_value = tool_result_text(result_content)
        payload.update({"type": "function_call_output", "call_id": result.tool_call_id or "", "output": output_value})
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> dict[str, Any]:
        params = deepcopy(request.generation_params)
        original = request.extensions.get(self.name, {}).get("generation_params") if preserve_source else None
        payload = deepcopy(original) if isinstance(original, dict) else {}
        if "max_output_tokens" in params:
            payload["max_output_tokens"] = params.pop("max_output_tokens")
        if "stop_sequences" in params:
            # Responses currently has no universal stop field. Keep it only when
            # an explicitly compatible provider extension supplied one.
            params.pop("stop_sequences")
            add_conversion_warning(
                request,
                code="unsupported_optional_control",
                message="responses has no portable stop-sequence request field",
                field="stop_sequences",
                target_protocol=self.name,
            )
        reasoning = params.pop("reasoning", None)
        if not preserve_source:
            # Cross-protocol mapping only; same-protocol passthrough keeps
            # the preserved original verbatim.
            payload.update(format_reasoning_controls(reasoning, self.name, request))
        structured = params.pop("structured_output", None)
        verbosity = params.pop("text_verbosity", None)
        if isinstance(structured, dict):
            formatted_format = format_structured_output(structured, self.name)
            text_config: dict[str, Any] = payload["text"] if isinstance(payload.get("text"), dict) else {}
            if formatted_format is not None:
                text_config["format"] = formatted_format
            elif not preserve_source:
                add_conversion_warning(
                    request,
                    code="unsupported_optional_control",
                    message=f"structured output type {structured.get('type')!r} has no Responses representation; dropped",
                    field="text.format",
                    target_protocol=self.name,
                )
            if verbosity is not None:
                text_config["verbosity"] = verbosity
            if text_config:
                payload["text"] = text_config
        elif verbosity is not None:
            payload.setdefault("text", {})["verbosity"] = verbosity
        if "tool_choice" in params:
            choice = params.get("tool_choice")
            if isinstance(choice, dict) and choice.get("allowed_names"):
                add_conversion_warning(
                    request,
                    code="unsupported_optional_control",
                    message="tool-choice allowlist has no Responses representation; narrowed to the mode",
                    field="tool_choice",
                    target_protocol=self.name,
                )
            payload["tool_choice"] = format_tool_choice(params.pop("tool_choice"), self.name)
        supported = {
            "background",
            # NOTE: context_management/moderation are doc-unverified create
            # params with no canonical producer — deliberately NOT in the
            # supported set (they would emit unverified fields upstream).
            "conversation",
            "include",
            "max_tool_calls",
            "parallel_tool_calls",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "store",
            "stream_options",
            "temperature",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        }
        payload.update(
            retain_supported_generation_params(
                request,
                params,
                supported=supported,
                target_protocol=self.name,
            )
        )
        return payload


def _responses_output_modalities(messages: list[UnifiedMessage]) -> list[str]:
    """Declare output modalities the output actually carries (W2)."""

    blocks = [block for message in messages for block in message.content]
    modalities: list[str] = []
    if any(block.type == "text" and block.text for block in blocks):
        modalities.append("text")
    if any(block.type == "builtin_tool" and block.builtin_tool is not None and block.builtin_tool.kind == "image_generation" for block in blocks):
        modalities.append("image")
    return modalities


def _warn_responses_list_once(warnings: list | None, *, code: str, message: str, field: str | None = None) -> None:
    """Append a deduplicated ConversionWarning to a plain list sink."""

    if warnings is None:
        return
    for warning in warnings:
        if warning.code == code and warning.message == message and warning.field == field:
            return
    warnings.append(ConversionWarning(code=code, message=message, field=field, source_protocol=None, target_protocol="responses"))


def _warn_responses_once(unified_response: UnifiedResponse, *, code: str, message: str, field: str | None = None) -> None:
    """Append a deduplicated ConversionWarning (formatting may run twice)."""

    for warning in unified_response.warnings:
        if warning.code == code and warning.message == message and warning.field == field:
            return
    unified_response.warnings.append(
        ConversionWarning(code=code, message=message, field=field, source_protocol=unified_response.source_protocol, target_protocol="responses")
    )


_BUILTIN_TOOL_ITEM_TYPES = {
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "image_generation_call",
    "mcp_call",
    "local_shell_call",
}


def _parse_responses_annotations(payload: Any) -> list[Annotation]:
    """Normalize Responses output-text annotations to canonical form."""

    if not isinstance(payload, list):
        return []
    annotations: list[Annotation] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        citation = None
        for nested_key in ("url_citation", "citation"):
            if isinstance(entry.get(nested_key), dict):
                citation = entry[nested_key]
                break
        annotations.append(
            Annotation(
                type=str(entry.get("type") or "url_citation"),
                url=(citation or {}).get("url") or entry.get("url"),
                title=(citation or {}).get("title") or entry.get("title"),
                citation=(citation or {}).get("snippet") or (citation or {}).get("quote"),
                start_index=(citation or {}).get("start_index") if isinstance((citation or {}).get("start_index"), int) else None,
                end_index=(citation or {}).get("end_index") if isinstance((citation or {}).get("end_index"), int) else None,
                raw=deepcopy(entry),
            )
        )
    return annotations


def _format_responses_annotations(annotations: list[Annotation]) -> list[dict[str, Any]]:
    """Format canonical annotations back into Responses annotation objects."""

    formatted: list[dict[str, Any]] = []
    for annotation in annotations:
        if isinstance(annotation.raw, dict) and annotation.raw.get("type"):
            formatted.append(deepcopy(annotation.raw))
            continue
        formatted.append(
            {
                "type": annotation.type or "url_citation",
                "url_citation": {
                    "url": annotation.url,
                    "title": annotation.title,
                    "snippet": annotation.citation,
                    "start_index": annotation.start_index,
                    "end_index": annotation.end_index,
                },
            }
        )
    return [{k: v for k, v in entry.items() if v is not None} for entry in formatted]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _decode_sse_data(raw_event: Any) -> Any:
    from .streaming import decode_sse_data

    return decode_sse_data(raw_event)


def _reasoning_text(item: dict[str, Any]) -> str | None:
    summary = item.get("summary")
    if isinstance(summary, list):
        parts = []
        for part in summary:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts) if parts else None
    return str(summary) if summary else None


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}


def _format_responses_usage(usage: Usage | None) -> dict[str, Any] | None:
    """Format normalized usage using OpenAI Responses public field names."""

    if usage is None:
        return None
    payload: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens or (usage.input_tokens + usage.output_tokens),
    }
    input_details: dict[str, Any] = {}
    if usage.cache_read_tokens:
        input_details["cached_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        # Official detail spelling (ResponseUsage.InputTokensDetails).
        input_details["cache_write_tokens"] = usage.cache_write_tokens
    if usage.audio_tokens:
        input_details["audio_tokens"] = usage.audio_tokens
    if input_details:
        payload["input_tokens_details"] = input_details
    output_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        output_details["reasoning_tokens"] = usage.reasoning_tokens
    if usage.output_audio_tokens:
        output_details["audio_tokens"] = usage.output_audio_tokens
    if output_details:
        payload["output_tokens_details"] = output_details
    if usage.cost:
        cost_details: dict[str, Any] = dict(usage.cost.metadata)
        if usage.cost.provider_reported_cost is not None:
            cost_details["total_cost"] = usage.cost.provider_reported_cost
        elif usage.cost.estimated_cost is not None:
            cost_details["estimated_cost"] = usage.cost.estimated_cost
        cost_details["currency"] = usage.cost.currency
        if usage.cost.source:
            cost_details["source"] = usage.cost.source
        payload["cost_details"] = cost_details
    return payload


def _parse_responses_generation_params(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize Responses controls into canonical names."""

    params = deepcopy(source)
    text = params.pop("text", None)
    if isinstance(text, dict):
        if isinstance(text.get("format"), dict):
            params["structured_output"] = canonical_structured_output(text["format"], "responses")
        if text.get("verbosity") is not None:
            # text.verbosity is a first-class output control (low|medium|high)
            # — carried explicitly, never dropped silently.
            params["text_verbosity"] = text["verbosity"]
    reasoning = params.get("reasoning")
    if isinstance(reasoning, dict):
        # Source-native dict preserved verbatim for same-protocol rebuild;
        # normalized (summary -> include_thoughts) for cross-protocol mapping.
        params["reasoning"] = normalize_reasoning_controls(deepcopy(reasoning))
    return params


def _parse_responses_media_source(block: dict[str, Any]) -> MediaSource:
    """Normalize Responses image and file content fields."""

    value = block.get("image_url") or block.get("file_url") or block.get("source")
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, data = value.split(",", 1)
            return MediaSource(kind="base64", media_type=prefix[5:].split(";", 1)[0], data=data, raw=deepcopy(block))
        return MediaSource(kind="url", url=value, detail=block.get("detail"), raw=deepcopy(block))
    return MediaSource(
        kind="file" if block.get("file_id") else "base64" if block.get("file_data") else "url",
        media_type=block.get("mime_type") or block.get("media_type"),
        url=block.get("file_url"),
        data=block.get("file_data"),
        file_id=block.get("file_id"),
        detail=block.get("detail"),
        raw=deepcopy(block),
    )


def _coerce_media_source(value: Any) -> MediaSource:
    """Coerce legacy media dictionaries into canonical form."""

    if isinstance(value, MediaSource):
        return value
    if isinstance(value, str):
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="file" if payload.get("file_id") else "base64" if payload.get("data") or payload.get("file_data") else "url",
        media_type=payload.get("mime_type") or payload.get("media_type"),
        url=payload.get("url") or payload.get("file_url"),
        data=payload.get("data") or payload.get("file_data"),
        file_id=payload.get("file_id"),
        detail=payload.get("detail"),
        raw=deepcopy(value),
    )


def _format_responses_image_source(value: Any) -> dict[str, Any]:
    """Format a canonical image source for Responses input content."""

    source = _coerce_media_source(value)
    if source.url:
        image_url = source.url
    elif source.data:
        image_url = f"data:{source.media_type or 'application/octet-stream'};base64,{source.data}"
    else:
        image_url = source.file_id or ""
    payload: dict[str, Any] = {"image_url": image_url}
    if source.detail:
        payload["detail"] = source.detail
    return payload


def _format_responses_file_source(value: Any) -> dict[str, Any]:
    """Format a canonical file source for Responses input content."""

    source = _coerce_media_source(value)
    if source.file_id:
        return {"file_id": source.file_id}
    if source.data:
        return {"file_data": source.data}
    if source.url:
        return {"file_url": source.url}
    return {"file_data": ""}


def _output_item_from_message(message: UnifiedMessage, raw: dict[str, Any]) -> OutputItem:
    """Create an ordered canonical output item alongside compatibility messages."""

    item_type = str(raw.get("type") or "message")
    if item_type == "reasoning":
        reasoning = message_reasoning(message)
        return OutputItem(type="reasoning", id=raw.get("id"), reasoning=reasoning[0] if reasoning else None, status=raw.get("status"), raw=deepcopy(raw))
    if item_type in {"function_call", "custom_tool_call"}:
        calls = message_tool_calls(message)
        return OutputItem(type="tool_call", id=raw.get("id"), tool_call=calls[0] if calls else None, status=raw.get("status"), raw=deepcopy(raw))
    return OutputItem(type="message", id=raw.get("id"), role=message.role, content=deepcopy(message.content), status=raw.get("status"), raw=deepcopy(raw))
