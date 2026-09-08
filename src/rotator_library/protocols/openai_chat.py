# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI Chat Completions protocol adapter.

The adapter models the common OpenAI-compatible chat shape used by many current
providers. It is a reusable base, not a final authority: providers can subclass
or override pieces when they need non-standard fields, stricter ordering, or
different stream semantics.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from typing import Any, ClassVar, Iterable, Optional

from .base import ProtocolAdapter
from .canonical import (
    add_conversion_warning,
    disclose_response_drops,
    attach_conversion_summary,
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
    coalesce_assistant_message,
    canonical_tool_choice,
    format_reasoning_controls,
    format_stop_reason,
    format_structured_output,
    format_tool_choice,
    conversation_messages,
    instruction_layout,
    instruction_messages,
    normalize_reasoning_controls,
    is_same_protocol,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_text,
)
from .operation import OPERATION_CHAT, OPERATION_GENERATE
from .validation import validate_generative_request, validate_generative_response
from .types import (
    Annotation,
    ContentBlock,
    ConversionWarning,
    CostDetails,
    MediaSource,
    ProtocolContext,
    ProtocolError,
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
    serialize_value,
    text_blocks,
)

_GENERATION_PARAMS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning_effort",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
    "web_search_options",
}

_REQUEST_CORE_FIELDS = {
    "model",
    "messages",
    "modalities",
    "audio",
    "tools",
    "stream",
    "response_format",
    "metadata",
    *_GENERATION_PARAMS,
}


class OpenAIChatProtocol(ProtocolAdapter):
    """Adapter for OpenAI Chat Completions request, response, and stream chunks.

    Unknown OpenAI-compatible extension fields are preserved in ``extra`` so a
    custom provider can still use them through later adapter or field-cache
    phases. Lossy conversions are avoided unless the source shape itself uses a
    compact representation, such as string message content.
    """

    name: ClassVar[str] = "openai_chat"
    aliases: ClassVar[tuple[str, ...]] = (
        "openai",
        "chat",
        "chat_completions",
        "openai_chat_completions",
    )
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_CHAT,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        warnings_list: list[ConversionWarning] = []
        messages = resolve_tool_result_names([self._parse_message(message) for message in request.get("messages") or []])
        tools = [self._parse_tool_definition(tool) for tool in request.get("tools") or []]
        source_generation_params = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request}
        generation_params = _parse_openai_generation_params(source_generation_params)
        structured_output = canonical_structured_output(request.get("response_format"), self.name)
        if structured_output:
            generation_params["structured_output"] = structured_output
        if "tool_choice" in generation_params:
            generation_params["tool_choice"] = canonical_tool_choice(generation_params["tool_choice"], self.name)
        if request.get("audio") is not None:
            generation_params["audio_output"] = deepcopy(request["audio"])
        extra = {k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS}
        modalities = [str(value).lower() for value in request.get("modalities") or []]
        if modalities:
            # Chat output modalities are text|audio: unknown values (e.g.
            # "image") never sail through to providers that must reject them.
            legal = [value for value in modalities if value in {"text", "audio"}]
            dropped = [value for value in modalities if value not in {"text", "audio"}]
            if dropped:
                warnings_list.append(
                    ConversionWarning(
                        code="unsupported_optional_control",
                        message=f"modalities {dropped} are not legal Chat output modalities; dropped",
                        field="modalities",
                        source_protocol=self.name,
                        target_protocol=self.name,
                    )
                )
            modalities = legal

        return UnifiedRequest(
            operation=OPERATION_CHAT,
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=messages,
            tools=tools,
            stream=bool(request.get("stream", False)),
            modalities=modalities,
            generation_params=generation_params,
            response_format=structured_output,
            metadata=deepcopy(request.get("metadata") or {}),
            source_protocol=self.name,
            extensions={self.name: {"generation_params": source_generation_params, "response_format": deepcopy(request.get("response_format"))}},
            warnings=warnings_list,
            raw=deepcopy(raw_request),
            extra=extra,
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        # Chat can express system/developer messages anywhere: canonical
        # message order is preserved verbatim (D7 level 1 — interleaved
        # instructions never hoisted). A separate canonical `system` field
        # (non-chat sources) is promoted to a leading system message —
        # including when explicit system messages also exist, so neither
        # instruction source is dropped.
        interleaved, _ = instruction_layout(unified_request)
        has_inline_instructions = any(m.role in {"system", "developer"} for m in unified_request.messages)
        if unified_request.system and interleaved:
            # The canonical field leads as its own turn; interleaved inline
            # instructions keep their conversation positions verbatim.
            wire_messages = [
                UnifiedMessage(role="system", content=deepcopy(unified_request.system)),
                *unified_request.messages,
            ]
        elif unified_request.system or not (interleaved or has_inline_instructions):
            wire_messages = [*instruction_messages(unified_request), *conversation_messages(unified_request)]
        else:
            wire_messages = list(unified_request.messages)
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "messages": self._format_request_messages(
                wire_messages,
                preserve_source=preserve_source,
                warnings=unified_request.warnings,
            ),
        }
        if unified_request.tools:
            payload["tools"] = [self._format_tool_definition(tool, preserve_source=preserve_source) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.modalities:
            payload["modalities"] = deepcopy(unified_request.modalities)
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(self._format_generation_params(unified_request, preserve_source=preserve_source))
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        error_payload = response.get("error")
        if error_payload is not None:
            # Provider errors are structured failures, never empty successes
            # (W6/D1 contract; mirrors the stream parser's error branch).
            raise ProtocolError(
                "openai_chat provider returned an error payload",
                protocol="openai_chat",
                pass_name="parse_response",
                payload={"error": deepcopy(error_payload)},
            )
        messages: list[UnifiedMessage] = []
        stop_reason = None
        for choice_position, choice in enumerate(response.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            message_payload = choice.get("message") or {}
            if message_payload:
                message = self._parse_message(message_payload)
                # Candidate identity (W2/D9): every alternative keeps its own
                # choice index and finish status; n>1 survives cross-protocol.
                try:
                    message.index = int(choice.get("index", choice_position))
                except (TypeError, ValueError):
                    message.index = choice_position
                message.stop_reason = canonical_stop_reason(choice.get("finish_reason"))
                if choice.get("logprobs") is not None:
                    # Choice-level logprobs ride the message for same-protocol
                    # replay (cross-protocol drops as untranslatable evidence).
                    message.extra["logprobs"] = deepcopy(choice["logprobs"])
                message_annotations = _parse_openai_annotations(message_payload.get("annotations"))
                if message_annotations:
                    if message.content:
                        message.content[0].annotations.extend(message_annotations)
                    else:
                        # Annotation-bearing refusal-only/empty messages keep
                        # their citations on a synthesized empty text block.
                        message.content.append(
                            ContentBlock(type="text", text=message_payload.get("content") if isinstance(message_payload.get("content"), str) else None, annotations=message_annotations)
                        )
                messages.append(message)
            if choice.get("finish_reason") is not None:
                stop_reason = choice.get("finish_reason")

        return UnifiedResponse(
            operation=OPERATION_CHAT,
            logical_operation=OPERATION_GENERATE,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            stop_reason=canonical_stop_reason(stop_reason),
            usage=self.extract_usage(response, context),
            modalities=_openai_output_modalities(messages),
            metadata={
                "object": response.get("object"),
                "created": response.get("created"),
                "system_fingerprint": response.get("system_fingerprint"),
                "native_stop_reason": stop_reason,
            },
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:

        disclose_response_drops(unified_response, self.name)
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        # Chat natively supports alternatives: candidate multiplicity is
        # preserved cross-protocol too (D9 direct mapping), never coalesced.
        messages = unified_response.messages if preserve_source else _cross_protocol_assistant_messages(unified_response)
        if not preserve_source:
            for message in messages:
                has_builtin = any(block.type == "builtin_tool" for block in message.content)
                has_representable = any(
                    block.type in {"text", "reasoning", "tool_call", "tool_result", "image", "audio", "file", "refusal"}
                    for block in message.content
                ) or bool(message.tool_calls or message.reasoning)
                if has_builtin:
                    _warn_chat_once(
                        unified_response,
                        code="builtin_tool_dropped",
                        message="provider-executed tool records have no Chat Completions representation; dropped",
                    )
                if has_builtin and not has_representable:
                    # D7: server-side tool records have no Chat representation;
                    # a record-only response is rejected, never an empty success.
                    raise ProtocolError(
                        "OpenAI Chat Completions cannot represent a provider-executed tool record as a successful message",
                        protocol=self.name,
                        pass_name="format_response",
                        payload={"dropped_blocks": ["builtin_tool"]},
                    )
                unsynthesizable_audio = any(
                    block.type == "audio" and not getattr(block.source, "data", None)
                    for block in message.content
                )
                if unsynthesizable_audio and message.role == "assistant":
                    _warn_chat_once(
                        unified_response,
                        code="media_dropped",
                        message="audio without inline data cannot synthesize the Chat message-level audio field; dropped",
                        field="content[audio]",
                    )
                has_video = any(block.type == "video" for block in message.content)
                if has_video:
                    _warn_chat_once(
                        unified_response,
                        code="media_dropped",
                        message="video output has no Chat Completions representation; dropped",
                        field="content[video]",
                    )
                has_image = any(block.type == "image" for block in message.content)
                if has_image and message.role == "assistant":
                    _warn_chat_once(
                        unified_response,
                        code="media_dropped",
                        message="assistant image output has no standard Chat Completions message representation; dropped",
                        field="content[image]",
                    )
        choices = []
        for position, message in enumerate(messages):
            per_choice_reason = message.stop_reason or unified_response.stop_reason
            formatted_reason = format_stop_reason(per_choice_reason, self.name)
            native_reason = unified_response.metadata.get("native_stop_reason")
            if formatted_reason is None:
                # Non-stream finish_reason is a required enum — never null.
                # Unknown/absent foreign reasons degrade to the honest
                # fallback with a recorded summary (D7 level 5).
                if per_choice_reason is not None or native_reason is not None:
                    _warn_chat_once(
                        unified_response,
                        code="stop_reason_approximated",
                        message=f"native stop reason '{per_choice_reason or native_reason}' has no Chat enum value; emitted 'stop'",
                        field="finish_reason",
                    )
                formatted_reason = "stop"
            choice_entry: dict[str, Any] = {
                "index": message.index if message.index is not None else position,
                "message": _format_response_message(
                    self._format_message(message, preserve_source=preserve_source, direction="response", warnings=unified_response.warnings),
                    message,
                ),
                "finish_reason": formatted_reason,
            }
            choice_logprobs = (message.extra or {}).get("logprobs")
            if choice_logprobs is not None:
                choice_entry["logprobs"] = deepcopy(choice_logprobs)
            choices.append(choice_entry)
        payload = {
            "id": unified_response.id or f"chatcmpl-{uuid.uuid4().hex}",
            "object": unified_response.metadata.get("object", "chat.completion"),
            "created": unified_response.metadata.get("created") or int(time.time()),
            "model": unified_response.model,
            "choices": choices,
            "usage": _format_openai_usage(unified_response.usage),
        }
        # Determinism markers survive the canonical round-trip (their
        # documented purpose is client-side seed/fingerprint checks).
        fingerprint = unified_response.metadata.get("system_fingerprint")
        if fingerprint is not None:
            payload["system_fingerprint"] = fingerprint
        service_tier = unified_response.metadata.get("service_tier") or unified_response.extra.get("service_tier")
        if service_tier is not None:
            payload["service_tier"] = service_tier
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
        return attach_conversion_summary({k: v for k, v in payload.items() if v is not None}, unified_response)

    def parse_stream_events(self, raw_event: Any, context: ProtocolContext | None = None) -> list[UnifiedStreamEvent]:
        """One canonical event per choice — ``n>1`` frames carry several
        choices in one SSE chunk and every candidate must survive."""

        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return [UnifiedStreamEvent(type="done", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="done", raw=deepcopy(raw_event))]
        data = _as_dict(event)
        if data.get("error") is not None:
            return [UnifiedStreamEvent(type="error", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="error", error=deepcopy(data["error"]), raw=deepcopy(raw_event), extra={"payload": data})]

        events: list[UnifiedStreamEvent] = []
        saw_choice = False
        for choice_position, choice in enumerate(data.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            saw_choice = True
            try:
                choice_index = int(choice.get("index", choice_position))
            except (TypeError, ValueError):
                choice_index = choice_position
            delta = choice.get("delta") or {}
            delta_message = None
            finish_reason = None
            if delta:
                delta_message = self._parse_message({"role": delta.get("role", "assistant"), **delta})
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            if delta_message is None and finish_reason is None:
                continue
            events.append(self._stream_event(data, delta_message, finish_reason, raw_event, choice_index, logprobs=choice.get("logprobs")))

        usage = self.extract_usage(data, context)
        if usage is not None and not saw_choice:
            # Terminal usage-only chunk (choices == []): its own event.
            events.append(self._stream_event(data, None, None, raw_event, 0, usage=usage))
        elif usage is not None and events:
            events[0].usage = usage
        if not events:
            events.append(self._stream_event(data, None, None, raw_event, 0, usage=usage))
        return events

    def _stream_event(
        self,
        data: dict[str, Any],
        delta_message: Optional[UnifiedMessage],
        finish_reason: Optional[str],
        raw_event: Any,
        choice_index: int,
        usage: Optional[Usage] = None,
        logprobs: Any = None,
    ) -> UnifiedStreamEvent:
        extra = {
            "id": data.get("id"),
            "model": data.get("model"),
            "finish_reason": canonical_stop_reason(finish_reason),
            "payload": data,
        }
        if logprobs is not None:
            # Chunk-level logprobs ride alongside (same-protocol replay;
            # cross-protocol drops them as untranslatable evidence).
            extra["logprobs"] = deepcopy(logprobs)
        return UnifiedStreamEvent(
            type="message_delta" if delta_message else "chunk",
            operation=OPERATION_CHAT,
            logical_operation=OPERATION_GENERATE,
            source_protocol=self.name,
            native_type="chat.completion.chunk",
            delta=delta_message,
            usage=usage,
            stop_reason=canonical_stop_reason(finish_reason),
            output_index=choice_index,
            raw=deepcopy(raw_event),
            extra=extra,
        )

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        events = self.parse_stream_events(raw_event, context)
        return events[0]

    def format_stream_event(self, unified_event: UnifiedStreamEvent, context: ProtocolContext | None = None) -> Any:
        from .streaming import format_canonical_stream_event

        return format_canonical_stream_event(unified_event, self.name, context)

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        if not isinstance(completion_details, dict):
            completion_details = {}
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
            input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cache_read_tokens=int(prompt_details.get("cached_tokens") or usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(
                prompt_details.get("cache_write_tokens")
                or prompt_details.get("cache_creation_tokens")
                or usage.get("cache_creation_tokens")
                or 0
            ),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0),
            audio_tokens=int(
                prompt_details.get("audio_tokens")
                or usage.get("audio_tokens")
                or 0
            ),
            output_audio_tokens=int(
                completion_details.get("audio_tokens")
                or usage.get("output_audio_tokens")
                or 0
            ),
            # Prediction tokens live in completion_tokens_details (spec
            # placement); prompt-side reads stay as a lenient fallback.
            accepted_prediction_tokens=int(
                completion_details.get("accepted_prediction_tokens")
                or prompt_details.get("accepted_prediction_tokens")
                or usage.get("accepted_prediction_tokens")
                or 0
            ),
            rejected_prediction_tokens=int(
                completion_details.get("rejected_prediction_tokens")
                or prompt_details.get("rejected_prediction_tokens")
                or usage.get("rejected_prediction_tokens")
                or 0
            ),
            cost=cost,
            raw=deepcopy(usage),
        )

    def _parse_message(self, message: dict[str, Any]) -> UnifiedMessage:
        payload = dict(message or {})
        reasoning = _extract_reasoning(payload)
        # A missing role mirrors the wire verbatim (assistant default); the
        # upstream API enforces role presence — the adapter does not invent
        # stricter semantics than the wire contract.
        role = str(payload.get("role") or "assistant")
        content = self._parse_content(payload.get("content"))
        if role == "tool":
            result_content = canonical_tool_arguments(self._concat_text_content(payload.get("content")))
            content = [
                ContentBlock(
                    type="tool_result",
                    tool_result=ToolResult(
                        tool_call_id=payload.get("tool_call_id"),
                        name=payload.get("name"),
                        content=result_content,
                        raw=deepcopy(message),
                    ),
                    raw=deepcopy(message),
                )
            ]
        elif role == "function":
            # Legacy function-role results unify into the canonical tool
            # result (the assistant function_call counterpart already does);
            # `extra["legacy_function_role"]` marks the spelling so the chat
            # formatter can replay it verbatim same-protocol.
            result_content = canonical_tool_arguments(self._concat_text_content(payload.get("content")))
            content = [
                ContentBlock(
                    type="tool_result",
                    tool_result=ToolResult(
                        tool_call_id=payload.get("tool_call_id"),
                        name=payload.get("name"),
                        content=result_content,
                        raw=deepcopy(message),
                    ),
                    raw=deepcopy(message),
                    extra={"legacy_function_role": True},
                )
            ]
        refusal = payload.get("refusal")
        if isinstance(refusal, str) and refusal and not any(block.type == "refusal" for block in content):
            content.append(ContentBlock(type="refusal", refusal=refusal, raw=refusal))
        audio = payload.get("audio")
        if isinstance(audio, dict) and audio:
            parsed_audio = _openai_media_source(audio, kind="audio")
            # Response audio objects carry a compact format label (mp3) —
            # normalize to the MIME type so canonical media_type stays
            # MIME-typed for every downstream target (M2).
            if parsed_audio.media_type and "/" not in parsed_audio.media_type:
                parsed_audio.media_type = f"audio/{parsed_audio.media_type}"
            content.append(ContentBlock(type="audio", source=parsed_audio, raw=deepcopy(audio)))
        return UnifiedMessage(
            role=role,
            content=content,
            name=payload.get("name"),
            tool_call_id=payload.get("tool_call_id"),
            tool_calls=self._parse_message_tool_calls(payload),
            reasoning=reasoning,
            raw=deepcopy(message),
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning", "reasoning_content"}},
        )

    @staticmethod
    def _concat_text_content(content: Any) -> Any:
        """Tool-message content is string or an array of text parts; arrays
        concatenate their text (never JSON-stringify the parts array)."""

        if isinstance(content, list):
            texts = [part.get("text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
            if texts and len(texts) == sum(1 for part in content if isinstance(part, dict)):
                return "".join(texts)
        return content

    def _format_request_messages(self, messages: Iterable[UnifiedMessage], *, preserve_source: bool, warnings: Optional[list[ConversionWarning]] = None) -> list[dict[str, Any]]:
        """Format messages, expanding protocols that embed tool results in user turns."""

        formatted: list[dict[str, Any]] = []
        for message in messages:
            result_blocks = [block for block in message.content if block.tool_result]
            if result_blocks and not (message.role == "tool" and len(result_blocks) == 1):
                residual = [block for block in message.content if not block.tool_result]
                if residual:
                    residual_message = deepcopy(message)
                    residual_message.content = residual
                    residual_message.tool_call_id = None
                    formatted.append(self._format_message(residual_message, preserve_source=preserve_source, warnings=warnings))
                for block in result_blocks:
                    result = block.tool_result
                    if result is None:
                        continue
                    if block.extra.get("legacy_function_role"):
                        # Legacy spelling replays verbatim (role=function).
                        formatted.append(
                            {
                                "role": "function",
                                "name": result.name or message.name,
                                "content": _tool_result_text(result.content),
                            }
                        )
                        continue
                    formatted.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": _tool_result_text(result.content),
                        }
                    )
                continue
            formatted.append(self._format_message(message, preserve_source=preserve_source, warnings=warnings))
        return formatted

    def _format_message(self, message: UnifiedMessage, *, preserve_source: bool = True, direction: str = "request", warnings: Optional[list[ConversionWarning]] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.role == "assistant" and not preserve_source:
            # Standard assistant messages carry no image parts: drop with the
            # recorded media_dropped summary (never emit against the summary).
            message = UnifiedMessage(
                role=message.role,
                content=[block for block in message.content if block.type != "image"],
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=message.tool_calls,
                reasoning=message.reasoning,
                index=message.index,
                stop_reason=message.stop_reason,
                extra=message.extra,
                raw=message.raw,
            )
        result_blocks = [block.tool_result for block in message.content if block.tool_result]
        if message.role == "tool" and result_blocks:
            result = result_blocks[0]
            payload["tool_call_id"] = result.tool_call_id or message.tool_call_id
            if result.name:
                payload["name"] = result.name
            if result.is_error and not preserve_source and warnings is not None:
                # Chat has no error flag on tool results: the degrade to an
                # {"error": ...} content object is disclosed, never silent
                # (the spelling is deliberately NOT parsed back — legitimate
                # payloads may carry an "error" field without being errors).
                _warn_once(
                        warnings,
                        code="tool_result_is_error_downgraded",
                        message="tool result error flag has no Chat field; degraded to an {'error': ...} content object",
                        field="content",
                    )
            content = _tool_result_text({"error": result.content} if result.is_error else result.content)
        else:
            content = self._format_content(message.content, preserve_source=preserve_source, warnings=warnings)
        if isinstance(content, list) and not content:
            if any(block.type == "refusal" and block.refusal for block in message.content):
                # Refusal-only assistant history: content is null on the wire,
                # the refusal field carries the meaning.
                payload["content"] = None
            elif any(block.type == "audio" for block in message.content):
                # Audio-only messages carry the payload at message level
                # (`audio` field); content stays null, no duplicated part.
                payload["content"] = None
            elif _message_tool_calls(message) or any(block.reasoning is not None for block in message.content):
                # Tool-call-only / reasoning-only assistant turns: content is
                # OPTIONAL when tool_calls is present, and OpenAI itself sends
                # null — an empty parts array fails strict client validation.
                payload["content"] = None
            else:
                payload["content"] = content
        elif content is not None:
            payload["content"] = content
        extra = deepcopy(message.extra) if preserve_source else {}
        if not preserve_source:
            audio_blocks = [block for block in message.content if block.type == "audio"]
            if audio_blocks and message.role == "assistant" and "audio" not in extra:
                # Response-direction only: synthesize the Chat RESPONSE audio
                # object {id, data, transcript} (docs shape — no format key,
                # which belongs to the request-side audio parameter).
                # Deterministic content digest id (W12 reconstruction
                # parity); unsynthesizable audio records media_dropped via
                # the response handle (never silent).
                synthesized: dict[str, Any] | None = None
                for block in audio_blocks:
                    source = block.source
                    data = getattr(source, "data", None)
                    if data:
                        digest = hashlib.sha256(str(data).encode("utf-8", "replace")).hexdigest()[:8]
                        synthesized = {
                            "id": (getattr(source, "file_id", None) or f"audio_{digest}"),
                            "data": data,
                            "transcript": getattr(source, "transcript", None) or "",
                        }
                        break
                if synthesized is not None:
                    extra["audio"] = synthesized
        legacy_function_call = extra.get("function_call")
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            legacy_only = all(call.extra.get("legacy_function_call") for call in tool_calls)
            if legacy_only:
                # Pure legacy history: replay the function_call spelling.
                call = tool_calls[0]
                extra["function_call"] = {"name": call.name or "", "arguments": tool_arguments_text(call.arguments)}
            else:
                payload["tool_calls"] = [self._format_tool_call(call, preserve_source=preserve_source, warnings=warnings) for call in tool_calls]
        if message.reasoning and (
            preserve_source
            or direction == "response"
            or (direction == "request" and message.role == "assistant")
        ):
            # reasoning_content is a provider-extension field (DeepSeek-style),
            # not an OpenAI spec field. Replay verbatim on same-protocol
            # upstream payloads; cross-protocol, only ASSISTANT history turns
            # may carry it (reasoning replay is legitimate history; synthesizing
            # it onto user/tool turns is fabrication strict providers reject).
            # Responses to chat CLIENTS always render reasoning text through it
            # (the convention every chat consumer reads).
            text = "".join(block.text or "" for block in message.reasoning if block.text)
            if text:
                payload["reasoning_content"] = text
        refusal_text = "".join(
            block.refusal or "" for block in message.content if block.type == "refusal" and block.refusal
        )
        if refusal_text and direction == "response":
            # Responses carry refusal at message level (the spec's primary
            # home for assistant refusals); request-side history keeps the
            # content-part form only — never both (no double emission).
            payload["refusal"] = refusal_text
        annotations = [
            annotation
            for block in message.content
            for annotation in block.annotations
        ]
        if annotations:
            payload["annotations"] = _format_openai_annotations(annotations)
        # Computed fields (annotations/refusal) have typed homes; stale
        # parse-time copies in extra must never clobber them.
        payload.update({k: v for k, v in extra.items() if k not in {"annotations", "refusal"}})
        return payload

    def _parse_message_tool_calls(self, payload: dict[str, Any]) -> list[ToolCall]:
        """Return modern and legacy OpenAI function calls as unified tools."""

        modern_calls = payload.get("tool_calls") or []
        if modern_calls:
            return [self._parse_tool_call(call) for call in modern_calls]
        legacy_call = payload.get("function_call")
        if isinstance(legacy_call, dict):
            return [
                ToolCall(
                    id=None,
                    name=legacy_call.get("name"),
                    arguments=legacy_call.get("arguments"),
                    type="function",
                    raw=deepcopy(legacy_call),
                    extra={"legacy_function_call": True},
                )
            ]
        return []

    def _parse_content(self, content: Any) -> list[ContentBlock]:
        if content is None:
            return []
        if isinstance(content, str):
            return text_blocks(content)
        if not isinstance(content, list):
            return [ContentBlock(type="unknown", raw=deepcopy(content))]
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append(ContentBlock(type="text", text=block, raw=block))
                continue
            if not isinstance(block, dict):
                blocks.append(ContentBlock(type="unknown", raw=deepcopy(block)))
                continue
            block_type = block.get("type", "text")
            if block_type in {"text", "output_text"}:
                blocks.append(ContentBlock(
                    type="text",
                    text=block.get("text", ""),
                    annotations=_parse_openai_annotations(block.get("annotations")),
                    raw=deepcopy(block),
                    extra=_without(block, {"type", "text", "annotations"}),
                ))
            elif block_type in {"image_url", "input_image"}:
                raw_source = deepcopy(block.get("image_url") or block.get("source"))
                source = _openai_media_source(raw_source, kind="image")
                blocks.append(ContentBlock(type="image", source=source, raw=deepcopy(block), extra=_without(block, {"type", "image_url", "source"})))
            elif block_type in {"input_audio", "audio"}:
                raw_source = deepcopy(block.get("input_audio") or block.get("audio") or block.get("source"))
                source = _openai_media_source(raw_source, kind="audio")
                blocks.append(ContentBlock(type="audio", source=source, raw=deepcopy(block), extra=_without(block, {"type", "input_audio", "audio", "source"})))
            elif block_type in {"file", "input_file"}:
                # Documented Chat shape nests identity/data under "file":
                # {"type":"file","file":{"file_id"|"file_data","filename"}}.
                # Flat spellings (Responses input_file) are accepted leniently.
                nested = block.get("file") if isinstance(block.get("file"), dict) else None
                source_payload = nested if nested is not None else _without(block, {"type"})
                source = _openai_media_source(source_payload, kind="file")
                blocks.append(ContentBlock(type="file", source=source, raw=deepcopy(block), extra=_without(block, {"type", "file", "file_id", "file_data", "filename"})))
            elif block_type == "refusal":
                blocks.append(ContentBlock(
                    type="refusal",
                    refusal=block.get("refusal"),
                    raw=deepcopy(block),
                    extra=_without(block, {"type", "refusal"}),
                ))
            else:
                blocks.append(ContentBlock(type=str(block_type), raw=deepcopy(block), extra=_without(block, {"type"})))
        return blocks

    def _format_content(self, blocks: Iterable[ContentBlock], *, preserve_source: bool = True, warnings: Optional[list[ConversionWarning]] = None) -> Any:
        block_list = list(blocks)
        if not block_list:
            return None
        warnings_before = len(warnings) if warnings is not None else 0
        if all(block.type == "text" and (not preserve_source or not isinstance(block.raw, dict)) and (not preserve_source or not block.extra) for block in block_list):
            return first_text(block_list) or ""
        formatted = []
        for block in block_list:
            if block.type == "text":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "text"}
                payload["type"] = "text"
                payload["text"] = block.text or ""
                if preserve_source:
                    payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type == "image":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "image_url"}
                payload["type"] = "image_url"
                payload["image_url"] = _format_openai_image_source(block.source)
                if preserve_source:
                    payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type == "audio":
                if preserve_source:
                    # Typed request input_audio parts round-trip their native
                    # shape. Response audio (untyped raw) round-trips at message
                    # level via the `audio` field — never as a content part.
                    if isinstance(block.raw, dict) and block.raw.get("type"):
                        formatted.append(deepcopy(block.raw))
                    continue
                source = _media_source(block.source)
                audio_format = _audio_format(source.media_type)
                if audio_format is None:
                    # Unmappable MIME type: drop with a recorded warning
                    # instead of emitting an illegal format label.
                    if warnings is not None:
                        _warn_once(
                        warnings,
                        code="media_dropped",
                        message=f"audio part with MIME type {source.media_type!r} has no legal Chat audio format label; dropped",
                        field="content[audio]",
                    )
                    continue
                if str(source.media_type or "").strip().lower() == "audio/ogg" and warnings is not None:
                    # Container-to-codec guess: opus dominates ogg deliveries,
                    # but the container may hold Vorbis — disclose the label.
                    _warn_once(
                        warnings,
                        code="media_approximated",
                        message="audio/ogg container labeled 'opus' (dominant codec in ogg deliveries; Vorbis content would decode incorrectly)",
                        field="content[audio]",
                    )
                payload = {"type": "input_audio", "input_audio": {"data": source.data or "", "format": audio_format}}
                formatted.append(payload)
            elif block.type in {"file", "document"}:
                source = _media_source(block.source)
                file_obj: dict[str, Any] = {}
                if source.file_id:
                    file_obj["file_id"] = source.file_id
                elif source.data:
                    file_obj["file_data"] = source.data
                if source.filename:
                    file_obj["filename"] = source.filename
                if not file_obj or ("file_id" not in file_obj and "file_data" not in file_obj):
                    # No documented Chat home for URL-only files: record the
                    # drop honestly rather than invent a non-spec key.
                    if warnings is not None:
                        _warn_once(warnings, code="media_dropped", message=f"file without file_id/file_data cannot be represented as a Chat file part (url={source.url})", field="file")
                    continue
                formatted.append({"type": "file", "file": file_obj})
            elif block.type == "refusal":
                payload = {"type": "refusal", "refusal": block.refusal or ""}
                formatted.append(payload)
            elif preserve_source and isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
            elif block.type not in {"text", "image", "audio", "file", "document", "refusal", "tool_call", "tool_result", "reasoning", "builtin_tool"}:
                # Unknown block type that will NOT be raw-replayed: record
                # the drop (response paths have no fail-fast validation) —
                # never silent, raw-carrying or not.
                if warnings is not None:
                    _warn_once(
                        warnings,
                        code="unsupported_optional_control",
                        message=f"content block type '{block.type}' has no Chat representation; dropped",
                        field="content",
                    )
                continue
        if not formatted and block_list and warnings is not None and len(warnings) > warnings_before:
            # Every part was DROPPED with a recorded warning (e.g. URL-only
            # files): an empty parts array is illegal wire — degrade to an
            # empty string (legal for every role). Audio/refusal-only
            # messages that intentionally produce no parts (message-level
            # fields carry them) keep their [] -> None handling.
            return ""
        return formatted

    def _parse_tool_definition(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        if str(payload.get("type") or "") == "custom":
            # Custom tools: {type:"custom", custom:{name, input}} — never
            # force-wrapped into a function shape.
            custom = payload.get("custom") if isinstance(payload.get("custom"), dict) else {}
            return ToolDefinition(
                name=str(custom.get("name") or ""),
                description=custom.get("description"),
                input_schema=deepcopy(custom.get("input") or custom.get("parameters") or {}),
                type="custom",
                extra={"raw": deepcopy(tool), **_without(payload, {"type", "custom"})},
            )
        function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
        return ToolDefinition(
            name=str(function.get("name") or ""),
            description=function.get("description"),
            input_schema=deepcopy(function.get("parameters") or function.get("input_schema") or {}),
            type=str(payload.get("type") or "function"),
            extra={
                "raw": deepcopy(tool),
                # Strictness changes enforcement semantics — carried explicitly
                # so cross-protocol targets can preserve or warn, never drop
                # silently.
                **({"strict": deepcopy(function["strict"])} if "strict" in function else {}),
                **_without(payload, {"type", "function"}),
            },
        )

    def _format_tool_definition(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        if tool.type == "custom":
            payload = deepcopy(tool.extra.get("raw")) if preserve_source and isinstance(tool.extra.get("raw"), dict) else {}
            payload["type"] = "custom"
            custom = payload.get("custom") if isinstance(payload.get("custom"), dict) else {}
            custom["name"] = tool.name
            if tool.input_schema:
                custom.setdefault("input", deepcopy(tool.input_schema))
            payload["custom"] = custom
            return payload
        raw = tool.extra.get("raw")
        payload = deepcopy(raw) if preserve_source and isinstance(raw, dict) else {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.input_schema),
            },
        }
        payload["type"] = "function"
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        function.update({"name": tool.name, "description": tool.description, "parameters": deepcopy(tool.input_schema)})
        if "strict" in tool.extra:
            # Strictness changes enforcement semantics — carried on every
            # rebuild (Chat function tools honor it), never dropped silently.
            function["strict"] = deepcopy(tool.extra["strict"])
        payload["function"] = {k: v for k, v in function.items() if v is not None}
        return payload

    def _parse_tool_call(self, call: dict[str, Any]) -> ToolCall:
        payload = dict(call or {})
        if str(payload.get("type") or "") == "custom":
            # Custom tool calls: {id, type:"custom", custom:{name, input}}.
            custom = payload.get("custom") if isinstance(payload.get("custom"), dict) else {}
            return ToolCall(
                id=payload.get("id"),
                name=custom.get("name") or payload.get("name"),
                arguments=canonical_tool_arguments(custom.get("input")),
                type="custom",
                index=payload.get("index"),
                raw=deepcopy(call),
                extra={**_without(custom, {"name", "input"}), **_without(payload, {"id", "custom", "type", "index", "name"})},
            )
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        arguments: Any = canonical_tool_arguments(function.get("arguments"))
        return ToolCall(
            id=payload.get("id"),
            name=function.get("name") or payload.get("name"),
            arguments=arguments,
            type=str(payload.get("type") or "function"),
            index=payload.get("index"),
            raw=deepcopy(call),
            extra={**_without(function, {"name", "arguments"}), **_without(payload, {"id", "function", "type", "index", "name"})},
        )

    def _format_tool_call(self, call: ToolCall, *, preserve_source: bool = True, warnings: Optional[list[ConversionWarning]] = None) -> dict[str, Any]:
        if call.type in {"custom", "custom_tool_call"}:
            # Responses custom_tool_call maps onto Chat's native custom
            # envelope — the narrowing is disclosed, never silent (the
            # provider pairs custom calls with custom outputs differently).
            if call.type == "custom_tool_call" and warnings is not None:
                _warn_once(
                    warnings,
                    code="custom_tool_narrowed",
                    message="custom tool call maps onto Chat's custom envelope (native custom_tool_call spelling is Responses-only)",
                    field="tool_calls",
                )
            payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
            payload["type"] = "custom"
            if call.id:
                payload["id"] = call.id
            if call.index is not None:
                payload["index"] = call.index
            custom = deepcopy(payload.get("custom")) if isinstance(payload.get("custom"), dict) else {}
            custom["name"] = call.name or ""
            custom["input"] = tool_arguments_text(call.arguments)
            payload["custom"] = custom
            return payload
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        payload["type"] = "function"
        if call.id:
            payload["id"] = call.id
        if call.index is not None:
            payload["index"] = call.index
        function = deepcopy(payload.get("function")) if isinstance(payload.get("function"), dict) else {}
        function["name"] = call.name or ""
        function["arguments"] = tool_arguments_text(call.arguments)
        payload["function"] = function
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> dict[str, Any]:
        """Format canonical controls into OpenAI Chat field names."""

        params = deepcopy(request.generation_params)
        if preserve_source:
            original = request.extensions.get(self.name, {}).get("generation_params")
            payload = deepcopy(original) if isinstance(original, dict) else {}
            # store replays verbatim via extensions on this path — popping
            # it from params avoids the spurious unsupported-control warning
            # (same pattern as max_output_tokens / stop / reasoning).
            params.pop("store", None)
            # response_format rides in its own extensions slot (verbatim
            # original, unknown/custom types included).
            original_format = request.extensions.get(self.name, {}).get("response_format")
            if original_format is not None:
                payload["response_format"] = deepcopy(original_format)
        else:
            payload = {}
        if "max_output_tokens" in params:
            value = params.pop("max_output_tokens")
            # Same-protocol: the client's original spelling (max_tokens or
            # max_completion_tokens) is already restored verbatim — never
            # emit both keys (o-series providers reject the deprecated one).
            if not (preserve_source and ("max_tokens" in payload or "max_completion_tokens" in payload)):
                payload["max_completion_tokens"] = value
        if "stop_sequences" in params:
            value = params.pop("stop_sequences")
            # Same-protocol: original stop spelling (string or array) wins.
            if not (preserve_source and "stop" in payload):
                payload["stop"] = value
        reasoning = params.pop("reasoning", None)
        if not preserve_source:
            # Cross-protocol: map canonical controls onto Chat spellings.
            # Same-protocol passthrough keeps the preserved original verbatim
            # (no normalization, no warnings — "none" stays "none").
            payload.update(format_reasoning_controls(reasoning, self.name, request))
        candidate_count = params.pop("candidate_count", None)
        if candidate_count is not None:
            # D9 direct mapping: canonical multiplicity -> Chat `n`.
            payload["n"] = candidate_count
        if "structured_output" in params:
            structured = params.pop("structured_output")
            if preserve_source:
                # Same-protocol: the original response_format is restored
                # verbatim from extensions (unknown/custom types included) —
                # never force-wrap into a canonical json_schema shape.
                pass
            else:
                formatted_output = format_structured_output(structured, self.name)
                if formatted_output is not None:
                    payload["response_format"] = formatted_output
                elif structured.get("type") == "text":
                    # Chat's default IS text — dropping the explicit text
                    # format is lossless, not a drop worth warning about.
                    pass
                else:
                    add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message=f"structured output type {structured.get('type')!r} has no Chat representation; dropped",
                        field="response_format",
                        target_protocol=self.name,
                    )
        if "tool_choice" in params:
            choice = params.pop("tool_choice")
            if preserve_source and "tool_choice" in payload:
                # Same-protocol: the client's original spelling (including
                # the allowed_tools variant) is already restored verbatim —
                # never clobber it with the canonical re-format.
                pass
            else:
                if isinstance(choice, dict) and choice.get("namespaced") is not None:
                    add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message=f"namespaced tool_choice ({choice['namespaced'].get('type')}) has no Chat representation; narrowed to the mode",
                        field="tool_choice",
                        target_protocol=self.name,
                    )
                payload["tool_choice"] = format_tool_choice(choice, self.name)
                if isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True:
                    # Exact native sibling (D7 level 1): the parallelism
                    # constraint survives as parallel_tool_calls:false.
                    # The conflict guard reads the CANONICAL param — the
                    # payload's explicit value merges later (retain pass).
                    if params.get("parallel_tool_calls") is True:
                        add_conversion_warning(
                            request,
                            code="generation_control_conflict",
                            message="client-explicit parallel_tool_calls=true overrides tool_choice.disable_parallel_tool_use; parallelism stays enabled",
                            field="parallel_tool_calls",
                            target_protocol=self.name,
                        )
                    else:
                        payload["parallel_tool_calls"] = False
                if (
                    isinstance(choice, dict)
                    and choice.get("allowed_names")
                    and choice.get("allowed_tools") is None
                    and isinstance(payload["tool_choice"], str)
                ):
                    add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message="tool-choice allowlist has no Chat mode-only representation; constraint narrowed to the mode",
                        field="tool_choice",
                        target_protocol=self.name,
                    )
        if "audio_output" in params:
            payload["audio"] = deepcopy(params.pop("audio_output"))
            # The audio parameter is defined in terms of modalities:["audio"]
            # — absent or audio-less modalities get the pairing synthesized
            # with a recorded warning (never a silently reject-able build).
            if "audio" not in (request.modalities or []):
                add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message="audio output requested without modalities including audio; added",
                        field="modalities",
                        target_protocol=self.name,
                    )
                base_modalities = [m for m in (request.modalities or []) if m in {"text", "audio"}] or ["text"]
                payload["modalities"] = [*base_modalities, "audio"]
        supported = {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "seed",
            "service_tier",
            # NOTE: "store" is deliberately absent — provider persistence is
            # OpenAI-bound state; cross-protocol targets drop it WITH a
            # recorded warning, same-protocol replays it via extensions.
            "stream_options",
            "temperature",
            "top_logprobs",
            "top_p",
            "user",
            "verbosity",
            "web_search_options",
        }
        payload.update(
            retain_supported_generation_params(
                request,
                params,
                supported=supported,
                target_protocol=self.name,
            )
        )
        if not request.stream and "stream_options" in payload:
            # stream_options is only legal alongside stream:true; a stored
            # value without streaming is client error, never forwarded.
            payload.pop("stream_options")
            add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message="stream_options dropped: only legal when stream is true",
                        field="stream_options",
                        target_protocol=self.name,
                    )
        if "top_logprobs" in payload and not payload.get("logprobs"):
            # The pair is required by the API: logprobs gates top_logprobs.
            payload["logprobs"] = True
            add_conversion_warning(
                request,
                code="generation_control_synthesized",
                message="logprobs=true synthesized (top_logprobs requires its logprobs gate)",
                field="logprobs",
                target_protocol=self.name,
            )
        if payload.get("logprobs") is True and "top_logprobs" not in payload:
            payload["top_logprobs"] = 0
            add_conversion_warning(
                request,
                code="generation_control_synthesized",
                message="top_logprobs=0 synthesized (logprobs=true requires its pair)",
                field="top_logprobs",
                target_protocol=self.name,
            )
        return payload


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


def _extract_reasoning(payload: dict[str, Any]) -> list[ReasoningBlock]:
    blocks = []
    for field_name in ("reasoning_content", "reasoning"):
        value = payload.get(field_name)
        if value:
            blocks.append(ReasoningBlock(type=field_name, text=str(value), extra={"source_field": field_name}))
    return blocks


def _format_openai_usage(usage: Usage | None) -> dict[str, Any] | None:
    """Format normalized usage using OpenAI Chat's public field names."""

    if usage is None:
        return None
    payload: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens or (usage.input_tokens + usage.output_tokens),
    }
    prompt_details: dict[str, Any] = {}
    if usage.cache_read_tokens:
        prompt_details["cached_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        prompt_details["cache_creation_tokens"] = usage.cache_write_tokens
    if usage.audio_tokens:
        prompt_details["audio_tokens"] = usage.audio_tokens
    if prompt_details:
        payload["prompt_tokens_details"] = prompt_details
    completion_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        completion_details["reasoning_tokens"] = usage.reasoning_tokens
    if usage.output_audio_tokens:
        completion_details["audio_tokens"] = usage.output_audio_tokens
    if usage.accepted_prediction_tokens:
        completion_details["accepted_prediction_tokens"] = usage.accepted_prediction_tokens
    if usage.rejected_prediction_tokens:
        completion_details["rejected_prediction_tokens"] = usage.rejected_prediction_tokens
    if completion_details:
        payload["completion_tokens_details"] = completion_details
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


def _format_response_message(payload: dict[str, Any], message: UnifiedMessage) -> dict[str, Any]:
    """Return Chat Completions response-message shape.

    Request messages may legitimately preserve content-part arrays. Assistant
    response messages from non-chat native protocols often arrive as text parts;
    Chat Completions clients expect the final message content to be a string in
    that common case. Refusal-only messages use content=null per the wire shape.
    """

    if payload.get("content") is not None and message.content:
        if all(block.type in {"text", "input_text", "output_text", "refusal", "audio"} and not block.extra for block in message.content):
            payload = dict(payload)
            payload["content"] = _first_response_text(message.content) or ""
    if payload.get("content") is not None and not payload["content"]:
        if any(block.type == "refusal" for block in message.content) and not any(
            block.type in {"text", "input_text", "output_text"} and block.text for block in message.content
        ):
            payload = dict(payload)
            payload["content"] = None
        elif any(block.type == "audio" for block in message.content):
            # Audio-only responses carry the payload at message level.
            payload = dict(payload)
            payload["content"] = None
    return payload


def _first_response_text(blocks: Iterable[ContentBlock]) -> Optional[str]:
    parts = [block.text for block in blocks if block.type in {"text", "input_text", "output_text"} and block.text]
    return "".join(parts) if parts else first_text(blocks)


def _parse_openai_generation_params(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI Chat controls into protocol-independent field names."""

    params = deepcopy(source)
    max_tokens = params.pop("max_completion_tokens", params.pop("max_tokens", None))
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    stop = params.pop("stop", None)
    if stop is not None:
        params["stop_sequences"] = [stop] if isinstance(stop, str) else deepcopy(stop)
    reasoning_effort = params.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        params["reasoning"] = {"effort": reasoning_effort}
    if params.get("reasoning") is not None:
        params["reasoning"] = normalize_reasoning_controls(params["reasoning"])
    # D9 multiplicity: Chat `n` and Gemini `candidateCount` share one
    # canonical control.
    n = params.pop("n", None)
    if n is not None and "candidate_count" not in params:
        params["candidate_count"] = n
    return params


def _format_openai_annotations(annotations: list[Annotation]) -> list[dict[str, Any]]:
    """Format canonical annotations back into Chat annotation objects."""

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
                    "start_index": annotation.start_index,
                    "end_index": annotation.end_index,
                },
            }
        )
    return [{k: v for k, v in entry.items() if v is not None} for entry in formatted]


def _warn_once(
    warnings: Optional[list],
    *,
    code: str,
    message: str,
    field: Optional[str] = None,
) -> None:
    """Append a deduplicated ConversionWarning to a plain list sink.

    Retry/rotation rebuild the same request more than once — direct appends
    clone identical warnings on every pass (the rendered summary dedupes,
    the internal list must too).
    """

    if warnings is None:
        return
    for warning in warnings:
        if warning.code == code and warning.message == message and warning.field == field:
            return
    warnings.append(ConversionWarning(code=code, message=message, field=field, source_protocol=None, target_protocol="openai_chat"))


def _warn_chat_once(unified_response: UnifiedResponse, *, code: str, message: str, field: Optional[str] = None) -> None:
    """Append a deduplicated ConversionWarning (formatting may run twice)."""

    for warning in unified_response.warnings:
        if warning.code == code and warning.message == message and warning.field == field:
            return
    unified_response.warnings.append(
        ConversionWarning(code=code, message=message, field=field, source_protocol=unified_response.source_protocol, target_protocol="openai_chat")
    )


def _cross_protocol_assistant_messages(unified_response: UnifiedResponse) -> list[UnifiedMessage]:
    """Cross-protocol message view: preserve candidates, else coalesce.

    When the neutral response carries candidate identity (W2/D9 — n>1 from
    chat or candidateCount from Gemini), every alternative is emitted as its
    own choice.     Identity-less multi-message responses (e.g. mixed roles)
    keep the legacy coalesced single-answer shape.
    """

    assistants = [m for m in unified_response.messages if m.role in {"assistant", "model"}]
    if len(assistants) > 1 and all(m.index is not None for m in assistants):
        return assistants
    return [coalesce_assistant_message(unified_response.messages)]


def _parse_openai_annotations(payload: Any) -> list[Annotation]:
    """Normalize OpenAI-style url_citation annotations to canonical form."""

    if not isinstance(payload, list):
        return []
    annotations: list[Annotation] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        citation = entry.get("url_citation") if isinstance(entry.get("url_citation"), dict) else entry
        annotations.append(
            Annotation(
                type=str(entry.get("type") or "url_citation"),
                url=citation.get("url"),
                title=citation.get("title"),
                citation=citation.get("citation") or citation.get("quote"),
                start_index=citation.get("start_index") if isinstance(citation.get("start_index"), int) else None,
                end_index=citation.get("end_index") if isinstance(citation.get("end_index"), int) else None,
                raw=deepcopy(entry),
            )
        )
    return annotations


def _openai_output_modalities(messages: list[UnifiedMessage]) -> list[str]:
    """Declare output modalities the response actually carries (W2)."""

    modalities: list[str] = []
    if any(block.type == "text" and block.text for message in messages for block in message.content):
        modalities.append("text")
    if any(block.type == "audio" for message in messages for block in message.content):
        modalities.append("audio")
    return modalities


def _openai_media_source(value: Any, *, kind: str) -> MediaSource:
    """Parse OpenAI media fields into a canonical media source."""

    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, data = value.split(",", 1)
            return MediaSource(kind="base64", media_type=prefix[5:].split(";", 1)[0], data=data, raw=value)
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    url = payload.get("url") or payload.get("file_url")
    data = payload.get("data") or payload.get("file_data")
    media_type = payload.get("media_type") or payload.get("mime_type") or payload.get("format")
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        prefix, encoded = url.split(",", 1)
        media_type = media_type or prefix[5:].split(";", 1)[0]
        data = encoded
        url = None
    file_id = payload.get("file_id")
    if file_id is None and kind == "audio" and isinstance(payload.get("id"), str):
        # Response audio objects carry their multi-turn chain handle as `id`
        # — it maps onto the canonical file identity so cross-protocol
        # rebuilds keep the chain instead of minting a digest.
        file_id = payload["id"]
    source_kind = "file" if file_id else "base64" if data else "url"
    return MediaSource(
        kind=source_kind,
        media_type=media_type,
        url=url,
        data=data,
        file_id=file_id,
        filename=payload.get("filename") if isinstance(payload.get("filename"), str) else None,
        detail=payload.get("detail"),
        transcript=payload.get("transcript") if isinstance(payload.get("transcript"), str) else None,
        raw=deepcopy(value),
        extra=_without(payload, {"url", "file_url", "data", "file_data", "file_id", "filename", "transcript", "media_type", "mime_type", "format", "detail"}),
    )


def _media_source(value: Any) -> MediaSource:
    """Coerce legacy dictionary media sources into the canonical type."""

    if isinstance(value, MediaSource):
        return value
    return _openai_media_source(value, kind="media")


def _format_openai_image_source(value: Any) -> dict[str, Any]:
    """Format a canonical image source for Chat Completions."""

    source = _media_source(value)
    if source.url:
        url = source.url
    elif source.data:
        media_type = source.media_type or "application/octet-stream"
        url = f"data:{media_type};base64,{source.data}"
    elif source.file_id:
        url = source.file_id
    else:
        url = ""
    payload: dict[str, Any] = {"url": url}
    if source.detail:
        payload["detail"] = source.detail
    return payload


# Legal Chat audio format labels (docs: wav|aac|mp3|flac|opus|pcm16) and
# the MIME types that map onto them.
_AUDIO_FORMAT_LABELS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/aac": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/ogg": "opus",
    "audio/opus": "opus",
    "audio/pcm": "pcm16",
    "audio/l16": "pcm16",
    "wav": "wav",
    "aac": "aac",
    "mp3": "mp3",
    "flac": "flac",
    "opus": "opus",
    "pcm16": "pcm16",
}


def _audio_format(media_type: Optional[str]) -> Optional[str]:
    """Return OpenAI's compact audio format label for a MIME type or label.

    Returns None when no legal label exists (callers drop with a recorded
    warning instead of emitting an illegal format value)."""

    if not media_type:
        return "wav"
    return _AUDIO_FORMAT_LABELS.get(str(media_type).strip().lower())


def _message_tool_calls(message: UnifiedMessage) -> list[ToolCall]:
    """Return de-duplicated calls from both canonical message representations."""

    calls = list(message.tool_calls)
    seen = {(call.id, call.name, tool_arguments_text(call.arguments)) for call in calls}
    for block in message.content:
        if not block.tool_call:
            continue
        key = (block.tool_call.id, block.tool_call.name, tool_arguments_text(block.tool_call.arguments))
        if key not in seen:
            calls.append(block.tool_call)
            seen.add(key)
    return calls


def _tool_result_text(value: Any) -> str:
    """Serialize a canonical tool result for Chat's tool message content."""

    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}
