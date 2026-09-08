# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Gemini generateContent protocol adapter.

The adapter preserves Gemini-native content parts, thought signatures, safety
settings, tools, and generation configuration so later native providers can use
the same base without forcing an OpenAI-compatible intermediate shape.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .canonical import (
    disclose_response_drops,
    format_reasoning_controls,
    attach_conversion_summary,
    add_conversion_warning,
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
    coalesce_assistant_message,
    conversation_messages,
    format_stop_reason,
    format_structured_output,
    format_tool_choice,
    STOP_REASON_CONTENT_FILTER,
    instruction_blocks,
    record_instruction_merge,
    is_same_protocol,
    message_reasoning,
    message_tool_calls,
    message_tool_results,
    may_emit_opaque_provider_state,
    normalize_tool_result_messages,
    ordered_message_blocks,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_object,
    tool_result_object,
)
from .operation import OPERATION_CHAT, OPERATION_COUNT_TOKENS, OPERATION_GENERATE, OPERATION_UNKNOWN, normalize_operation
from .validation import validate_generative_request, validate_generative_response
from .types import (
    Annotation,
    ContentBlock,
    ConversionWarning,
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
    serialize_value,
)

_REQUEST_CORE_FIELDS = {
    "model",
    "contents",
    "systemInstruction",
    "system_instruction",
    "tools",
    "generationConfig",
    "generation_config",
    "safetySettings",
    "safety_settings",
    "toolConfig",
    "tool_config",
    "stream",
}


def _warn_gemini_once(unified_response: UnifiedResponse, *, code: str, message: str, field: str | None = None) -> None:
    """Append a deduplicated ConversionWarning (formatting may run twice)."""

    for warning in unified_response.warnings:
        if warning.code == code and warning.message == message and warning.field == field:
            return
    unified_response.warnings.append(
        ConversionWarning(code=code, message=message, field=field, source_protocol=unified_response.source_protocol, target_protocol="gemini")
    )


class GeminiProtocol(ProtocolAdapter):
    """Adapter for Gemini ``generateContent`` and stream event shapes.

    Gemini parts are richer than simple chat messages. Unknown part fields remain
    in ``extra`` and raw payloads are preserved so provider-specific subclasses
    can refine behavior without losing data.
    """

    name: ClassVar[str] = "gemini"
    aliases: ClassVar[tuple[str, ...]] = ("google_gemini", "generate_content")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_CHAT, OPERATION_COUNT_TOKENS, "generate", "stream_generate")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        generation_config = deepcopy(request.get("generationConfig") or request.get("generation_config") or {})
        safety_settings = deepcopy(request.get("safetySettings") or request.get("safety_settings") or [])
        tool_config = deepcopy(request.get("toolConfig") or request.get("tool_config") or {})
        generation_params = _parse_gemini_generation_params(generation_config, tool_config)
        if safety_settings:
            generation_params["safety_settings"] = safety_settings
        return UnifiedRequest(
            operation=_operation_from_context(context, OPERATION_CHAT),
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=resolve_tool_result_names(
                normalize_tool_result_messages([self._parse_content(content, default_role="user") for content in request.get("contents") or []])
            ),
            system=self._parse_system(request.get("systemInstruction") or request.get("system_instruction")),
            tools=self._parse_tools(request.get("tools") or []),
            stream=bool(request.get("stream", False)),
            modalities=[str(value).lower() for value in generation_config.get("responseModalities") or []],
            generation_params=generation_params,
            response_format=deepcopy(generation_params.get("structured_output")),
            source_protocol=self.name,
            extensions={self.name: {"generationConfig": generation_config, "safetySettings": safety_settings, "toolConfig": tool_config}},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        emit_opaque_state = may_emit_opaque_provider_state(context, preserve_source=preserve_source)
        payload: dict[str, Any] = {
            "contents": [
                self._format_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state, warnings=unified_request.warnings)
                for message in resolve_tool_result_names(deepcopy(conversation_messages(unified_request)))
            ],
        }
        # NOTE: GenerateContentRequest has NO model field (the model rides
        # the URL path) and NO stream field (streaming is the
        # :streamGenerateContent?alt=sse endpoint) — neither may leak into
        # the upstream body (unknown-field 400).
        instructions = instruction_blocks(unified_request)
        if instructions:
            payload["systemInstruction"] = {
                "parts": self._format_parts(
                    instructions,
                    preserve_source=preserve_source,
                    emit_opaque_state=emit_opaque_state,
                )
            }
        # D7 level 5: one systemInstruction field mandates the ordered merge.
        record_instruction_merge(unified_request, self.name)
        generation_config, safety_settings, tool_config = self._format_generation_params(unified_request, preserve_source=preserve_source)
        if generation_config:
            payload["generationConfig"] = deepcopy(generation_config)
        if safety_settings:
            payload["safetySettings"] = deepcopy(safety_settings)
        if tool_config:
            payload["toolConfig"] = deepcopy(tool_config)
        if unified_request.tools:
            payload["tools"] = self._format_tools(unified_request.tools, preserve_source=preserve_source)
        if not preserve_source:
            # Provider-bound envelope fields dropping at foreign targets is
            # RECORDED (cachedContent/labels/serviceTier/store ride extra;
            # same-protocol replay restores them via source_extensions).
            for bound_field in ("cachedContent", "labels", "serviceTier", "store"):
                if unified_request.extra.get(bound_field) is not None:
                    add_conversion_warning(
                        unified_request,
                        code="unsupported_optional_control",
                        message=f"{bound_field} is Gemini-provider-bound state; dropped cross-protocol",
                        target_protocol=self.name,
                        field=bound_field,
                    )
            system_blocks = unified_request.system if isinstance(unified_request.system, list) else None
            if system_blocks and any(
                getattr(block, "type", "text") not in {"text"} or getattr(block, "source", None) is not None
                for block in system_blocks
            ):
                add_conversion_warning(
                    unified_request,
                    code="unsupported_optional_control",
                    message="systemInstruction supports text parts only; non-text instruction blocks dropped",
                    target_protocol=self.name,
                    field="systemInstruction",
                )
        if not payload.get("contents") and not instructions:
            # Contents are required by the API: surface the malformed shape
            # loudly (recorded) without hard-failing minimal/synthetic flows.
            add_conversion_warning(
                unified_request,
                code="unsupported_optional_control",
                message="Gemini generateContent requires non-empty contents; request carries none",
                target_protocol=self.name,
                field="contents",
            )
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
        # Defensive: neither key is ever legal in the body (belt for raw
        # fast-path replays and extension leakage).
        payload.pop("model", None)
        payload.pop("stream", None)
        if unified_request.operation == OPERATION_COUNT_TOKENS:
            # CountTokensRequest = {contents, generateContentRequest}: the
            # generate-only members (tools, toolConfig, safetySettings,
            # generationConfig, systemInstruction) ride the nested envelope
            # — never the count body top level.
            nested_keys = ("tools", "toolConfig", "safetySettings", "generationConfig", "systemInstruction")
            nested = {key: payload.pop(key) for key in nested_keys if key in payload}
            return {
                "contents": payload.get("contents", []),
                "generateContentRequest": nested,
            }
        return payload

    def _attach_grounding_annotations(self, message: UnifiedMessage, candidate: dict[str, Any]) -> None:
        """Lift Gemini grounding metadata onto text parts as annotations (W2)."""

        grounding = candidate.get("groundingMetadata")
        if not isinstance(grounding, dict):
            return
        chunks = grounding.get("groundingChunks")
        web_query = grounding.get("webSearchQueries")
        derived = []
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict) and isinstance(chunk.get("web"), dict):
                    web = chunk["web"]
                    derived.append(
                        Annotation(
                            type="url_citation",
                            url=web.get("uri") or web.get("url"),
                            title=web.get("title"),
                            raw=deepcopy(chunk),
                        )
                    )
        if not derived and isinstance(grounding.get("searchEntryPoint"), dict):
            # At minimum preserve the search entry point as evidence metadata.
            message.extra["grounding_search_entry_point"] = deepcopy(grounding["searchEntryPoint"])
        if web_query is not None:
            message.extra["grounding_web_search_queries"] = deepcopy(web_query)
        if derived:
            for block in message.content:
                if block.type == "text" and block.text:
                    block.annotations.extend(derived)
                    break
            else:
                message.extra["grounding_annotations"] = [serialize_value(a) for a in derived]

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        messages: list[UnifiedMessage] = []
        stop_reason = None
        for candidate_position, candidate in enumerate(response.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
            message = self._parse_content(content)
            message.extra["candidate"] = _without(candidate, {"content"})
            # Candidate identity (W2/D9): each alternative keeps its index and
            # finish status so multiplicity survives cross-protocol conversion.
            try:
                message.index = int(candidate.get("index", candidate_position))
            except (TypeError, ValueError):
                message.index = candidate_position
            message.stop_reason = canonical_stop_reason(candidate.get("finishReason"))
            self._attach_grounding_annotations(message, candidate)
            messages.append(message)
            if candidate.get("finishReason") is not None:
                stop_reason = candidate.get("finishReason")
        modality_blocks = [block for message in messages for block in message.content]
        modalities: list[str] = []
        if any(block.type == "text" and block.text for block in modality_blocks):
            modalities.append("text")
        if any(block.type == "audio" for block in modality_blocks):
            modalities.append("audio")
        if any(block.type == "image" for block in modality_blocks):
            modalities.append("image")
        canonical_reason = canonical_stop_reason(stop_reason)
        if canonical_reason == "stop" and any(message_tool_calls(message) for message in messages):
            canonical_reason = "tool_use"
        if not messages:
            # Blocked prompt (promptFeedback.blockReason, no candidates): a
            # structured refusal outcome, never an empty success — the block
            # reason maps to the content-filter stop so every destination
            # observes the failure.
            block_reason = (response.get("promptFeedback") or {}).get("blockReason") if isinstance(response.get("promptFeedback"), dict) else None
            if block_reason:
                canonical_reason = STOP_REASON_CONTENT_FILTER
                metadata_block_reason = block_reason
            else:
                metadata_block_reason = None
        else:
            metadata_block_reason = None
        return UnifiedResponse(
            operation=_response_operation(response, context),
            logical_operation=OPERATION_GENERATE,
            id=response.get("responseId") or response.get("id"),
            model=response.get("modelVersion") or getattr(context, "model", None),
            messages=messages,
            stop_reason=canonical_reason,
            usage=self.extract_usage(response, context),
            modalities=modalities,
            metadata={"promptFeedback": deepcopy(response.get("promptFeedback")), "modelVersion": response.get("modelVersion"), "native_stop_reason": stop_reason, "block_reason": metadata_block_reason},
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"responseId", "id", "modelVersion", "candidates", "usageMetadata", "promptFeedback"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        disclose_response_drops(unified_response, self.name)
        if unified_response.operation == OPERATION_COUNT_TOKENS:
            usage = unified_response.usage
            payload = deepcopy(unified_response.extra)
            # Normalized usage wins over raw preserved fields so later adapters
            # can correct counts without stale provider keys shadowing them.
            payload["totalTokens"] = usage.total_tokens if usage else 0
            return payload
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        emit_opaque_state = may_emit_opaque_provider_state(context, preserve_source=preserve_source)
        # Gemini natively supports candidates: multiplicity is preserved
        # cross-protocol too (D9 direct mapping), never coalesced.
        if preserve_source:
            messages = unified_response.messages
        else:
            assistants = [m for m in unified_response.messages if m.role in {"assistant", "model"}]
            if len(assistants) > 1 and all(m.index is not None for m in assistants):
                messages = assistants
            else:
                messages = [coalesce_assistant_message(unified_response.messages)]
            has_annotations = any(block.annotations for m in messages for block in m.content)
            if has_annotations:
                _warn_gemini_once(
                    unified_response,
                    code="annotations_dropped",
                    message="citations/annotations have no Gemini representation; dropped",
                )
            for m in messages:
                has_builtin = any(block.type == "builtin_tool" for block in m.content)
                has_representable = any(
                    block.type in {"text", "reasoning", "tool_call", "tool_result", "image", "audio", "video", "file", "document", "refusal"}
                    for block in m.content
                ) or bool(m.tool_calls or m.reasoning)
                if has_builtin:
                    _warn_gemini_once(
                        unified_response,
                        code="builtin_tool_dropped",
                        message="provider-executed tool records have no Gemini representation; dropped",
                    )
                if has_builtin and not has_representable:
                    # D7: a record-only response the target cannot express is
                    # rejected, never an empty success.
                    raise ProtocolError(
                        "Gemini generateContent cannot represent a provider-executed tool record as a successful candidate",
                        protocol=self.name,
                        pass_name="format_response",
                        payload={"dropped_blocks": ["builtin_tool"]},
                    )
        candidates = []
        kept_messages = [
            message
            for message in messages
            if message.content or message.reasoning or message.tool_calls or message.stop_reason or unified_response.stop_reason
        ]
        for index, message in enumerate(kept_messages):
            candidate: dict[str, Any] = {"index": message.index if message.index is not None else index}
            if message.content or message.reasoning or message.tool_calls:
                candidate["content"] = self._format_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state, warnings=unified_response.warnings)
            per_candidate_reason = message.stop_reason or unified_response.stop_reason
            if per_candidate_reason:
                candidate["finishReason"] = format_stop_reason(per_candidate_reason, self.name)
            if preserve_source:
                candidate.update(deepcopy(message.extra.get("candidate") or {}))
            candidates.append(candidate)
        if not candidates and unified_response.stop_reason:
            # Blocked/filtered outcomes with no content: the candidate
            # carries the finish reason (never an empty 200 envelope).
            candidates.append({"index": 0, "finishReason": format_stop_reason(unified_response.stop_reason, self.name)})
        payload = {
            "responseId": unified_response.id,
            "modelVersion": unified_response.model,
            "candidates": candidates,
            "usageMetadata": self._format_usage(unified_response.usage),
            "promptFeedback": deepcopy(unified_response.metadata.get("promptFeedback")),
        }
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
        return attach_conversion_summary({k: v for k, v in payload.items() if v is not None}, unified_response)

    def parse_stream_events(self, raw_event: Any, context: ProtocolContext | None = None) -> list[UnifiedStreamEvent]:
        """One canonical event per candidate — candidateCount>1 streams carry
        several candidates in one chunk and every one must survive (the
        usage rides the first event)."""

        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return [UnifiedStreamEvent(type="done", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="done", raw=deepcopy(raw_event))]
        data = _as_dict(event)
        if data.get("error") is not None:
            return [UnifiedStreamEvent(type="error", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="error", error=deepcopy(data["error"]), raw=deepcopy(raw_event), extra={"payload": data})]
        response = self.parse_response(data, context)
        if not response.messages:
            return [
                UnifiedStreamEvent(
                    type="chunk",
                    operation=response.operation,
                    logical_operation=OPERATION_GENERATE,
                    source_protocol=self.name,
                    native_type="gemini.chunk",
                    usage=response.usage,
                    stop_reason=response.stop_reason,
                    raw=deepcopy(raw_event),
                    extra={"payload": data, "finish_reason": response.stop_reason},
                )
            ]
        events: list[UnifiedStreamEvent] = []
        for position, message in enumerate(response.messages):
            events.append(
                UnifiedStreamEvent(
                    type="message_delta",
                    operation=response.operation,
                    logical_operation=OPERATION_GENERATE,
                    source_protocol=self.name,
                    native_type="gemini.chunk",
                    delta=message,
                    usage=response.usage if position == 0 else None,
                    # Per-candidate finish only: response.stop_reason derives
                    # from the LAST finished candidate — leaking it onto
                    # unfinished siblings prematurely closes them in every
                    # client target.
                    stop_reason=message.stop_reason,
                    output_index=message.index if message.index is not None else position,
                    raw=deepcopy(raw_event),
                    extra={"payload": data, "finish_reason": message.stop_reason},
                )
            )
        return events

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        return self.parse_stream_events(raw_event, context)[0]

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else payload
        if not isinstance(usage, dict) or (not any(key.endswith("TokenCount") for key in usage) and "totalTokens" not in usage):
            return None
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)
        tool_prompt_tokens = int(usage.get("toolUsePromptTokenCount") or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("totalTokenCount") or usage.get("totalTokens") or input_tokens + output_tokens + reasoning_tokens),
            cache_read_tokens=int(usage.get("cachedContentTokenCount") or 0),
            reasoning_tokens=reasoning_tokens,
            # Tool-prompt accounting rides extra (Gemini-only bucket; the
            # detail lists stay in raw for same-protocol replay).
            extra={"tool_use_prompt_tokens": tool_prompt_tokens} if tool_prompt_tokens else {},
            raw=deepcopy(usage),
        )

    def _parse_system(self, system: Any) -> list[ContentBlock]:
        if system is None:
            return []
        if isinstance(system, str):
            return [ContentBlock(type="text", text=system, raw=system)]
        if isinstance(system, dict):
            return self._parse_parts(system.get("parts") or [])
        return []

    def _parse_content(self, content: dict[str, Any], *, default_role: str = "model") -> UnifiedMessage:
        payload = dict(content or {})
        role = str(payload.get("role") or default_role)
        # Gemini uses "model" where chat protocols usually say "assistant".
        normalized_role = "assistant" if role == "model" else role
        message = UnifiedMessage(
            role=normalized_role,
            content=self._parse_parts(payload.get("parts") or []),
            raw=deepcopy(content),
            extra={"gemini_role": role, **_without(payload, {"role", "parts"})},
        )
        for block in message.content:
            if block.tool_call:
                message.tool_calls.append(block.tool_call)
            if block.reasoning:
                message.reasoning.append(block.reasoning)
        return message

    def _format_content(
        self,
        message: UnifiedMessage,
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
        warnings: list | None = None,
    ) -> dict[str, Any]:
        role = message.extra.get("gemini_role") if preserve_source else None
        role = role or ("model" if message.role in {"assistant", "model"} else "user")
        if role not in {"user", "model"}:
            # Content.role is a user|model enum: unknown roles (foreign
            # sources or malformed input) degrade to user — the only legal
            # alternative — never leak to the wire.
            role = "user"
        parts = self._format_parts(
            ordered_message_blocks(message),
            preserve_source=preserve_source,
            emit_opaque_state=emit_opaque_state,
            warnings=warnings,
        )
        payload = {"role": role, "parts": parts}
        if preserve_source:
            payload.update(
                {
                    k: deepcopy(v)
                    for k, v in message.extra.items()
                    if k != "gemini_role" and not k.startswith("grounding_") and k != "candidate"
                }
            )
        return payload

    def _parse_parts(self, parts: Iterable[Any]) -> list[ContentBlock]:
        blocks = []
        for position, part in enumerate(parts):
            block = self._parse_part(part)
            if block.index is None:
                block.index = position
            blocks.append(block)
        return blocks

    def _parse_part(self, part: Any) -> ContentBlock:
        if isinstance(part, str):
            return ContentBlock(type="text", text=part, raw=part)
        if not isinstance(part, dict):
            return ContentBlock(type="unknown", raw=deepcopy(part))
        if "text" in part:
            reasoning = None
            signature = part.get("thoughtSignature") or part.get("thought_signature")
            if part.get("thought"):
                reasoning = ReasoningBlock(type="reasoning", text=part.get("text"), signature=signature, raw=deepcopy(part), extra=_without(part, {"text", "thought", "thoughtSignature", "thought_signature"}))
                return ContentBlock(type="reasoning", text=part.get("text", ""), reasoning=reasoning, raw=deepcopy(part), extra=_without(part, {"text"}))
            if signature:
                # Plain answer text carrying a thought signature stays TEXT
                # (signatures may attach to any part; only `thought:true`
                # marks thinking) — the signature rides extra for verbatim
                # same-protocol replay.
                return ContentBlock(type="text", text=part.get("text", ""), raw=deepcopy(part), extra=_without(part, {"text"}))
            return ContentBlock(type="text", text=part.get("text", ""), raw=deepcopy(part), extra=_without(part, {"text"}))
        if "inlineData" in part or "inline_data" in part:
            source = part.get("inlineData") or part.get("inline_data")
            media = _parse_gemini_media_source(source, inline=True)
            return ContentBlock(type=_media_block_type(media.media_type), source=media, raw=deepcopy(part), extra=_without(part, {"inlineData", "inline_data"}))
        if "fileData" in part or "file_data" in part:
            source = part.get("fileData") or part.get("file_data")
            media = _parse_gemini_media_source(source, inline=False)
            return ContentBlock(type=_media_block_type(media.media_type), source=media, raw=deepcopy(part), extra=_without(part, {"fileData", "file_data"}))
        if "functionCall" in part or "function_call" in part:
            call = part.get("functionCall") or part.get("function_call") or {}
            # raw is the OUTER part: same-protocol replay keeps part-level
            # metadata (thoughtSignature!) instead of corrupting the union.
            return ContentBlock(
                type="tool_call",
                tool_call=ToolCall(
                    id=call.get("id"),
                    name=call.get("name"),
                    arguments=canonical_tool_arguments(call.get("args")),
                    type="function",
                    signature=part.get("thoughtSignature") or part.get("thought_signature"),
                    raw=deepcopy(part),
                    extra=_without(part, {"functionCall", "function_call", "thoughtSignature", "thought_signature"}),
                ),
                raw=deepcopy(part),
                extra=_without(part, {"functionCall", "function_call"}),
            )
        if "functionResponse" in part or "function_response" in part:
            response = part.get("functionResponse") or part.get("function_response") or {}
            result_content = canonical_tool_arguments(response.get("response"))
            # Track whether the wire actually carried an id: Gemini 2.x
            # responses have none (identity is the name), and fabricating
            # one from the name on rebuild is a wire violation.
            had_id = response.get("id") is not None
            part_signature = part.get("thoughtSignature") or part.get("thought_signature")
            return ContentBlock(
                type="tool_result",
                tool_result=ToolResult(
                    tool_call_id=response.get("id") or response.get("name"),
                    name=response.get("name"),
                    content=result_content,
                    raw=deepcopy(part),
                    extra={
                        **_without(part, {"functionResponse", "function_response"}),
                        "had_function_response_id": had_id,
                        **({"thought_signature": part_signature} if part_signature else {}),
                    },
                ),
                raw=deepcopy(part),
                extra=_without(part, {"functionResponse", "function_response"}),
            )
        if "executableCode" in part or "codeExecutionResult" in part or "toolCall" in part or "toolResponse" in part:
            # Official union members without dedicated canonical types round-
            # trip as builtin records (verbatim raw replay; foreign targets
            # reject — no fabricated text).
            from .types import BuiltinToolCall

            kind = next(k for k in ("executableCode", "codeExecutionResult", "toolCall", "toolResponse") if k in part)
            builtin = BuiltinToolCall(kind=kind, call_id=None, status="completed", raw=deepcopy(part))
            return ContentBlock(type="builtin_tool", builtin_tool=builtin, raw=deepcopy(part))
        return ContentBlock(type="unknown", raw=deepcopy(part), extra=deepcopy(part))

    def _format_parts(
        self,
        blocks: Iterable[ContentBlock],
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
        warnings: list | None = None,
    ) -> list[dict[str, Any]]:
        parts = []
        for block in blocks:
            if block.tool_call:
                parts.append(self._format_tool_call(block.tool_call, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state))
            elif block.tool_result:
                parts.append(self._format_tool_result(block.tool_result, preserve_source=preserve_source))
            elif block.type in {"image", "audio", "video", "file", "document"}:
                part = _format_gemini_media(block, preserve_source=preserve_source, warnings=warnings)
                if part is not None:
                    parts.append(part)
            elif block.reasoning:
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                payload["text"] = block.reasoning.text or ""
                payload["thought"] = True
                if emit_opaque_state and block.reasoning.signature:
                    payload["thoughtSignature"] = block.reasoning.signature
                elif not emit_opaque_state:
                    payload.pop("thoughtSignature", None)
                parts.append(payload)
            elif block.type == "refusal" and block.refusal is not None:
                # Gemini has no refusal part; the text survives as a plain
                # text part (stop_reason carries the refusal semantics).
                parts.append({"text": block.refusal})
            elif block.type == "builtin_tool":
                # Native union members (executableCode, server toolCall, ...)
                # replay their raw part verbatim; foreign-source builtin
                # records have no gemini shape (handled by format_response
                # guards — never fabricated into empty text here).
                if isinstance(block.raw, dict) and any(k in block.raw for k in ("executableCode", "codeExecutionResult", "toolCall", "toolResponse")):
                    parts.append(deepcopy(block.raw))
                continue
            else:
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                if isinstance(block.raw, dict) and not any(k in block.raw for k in ("text", "inlineData", "inline_data", "fileData", "file_data", "functionCall", "function_call", "functionResponse", "function_response")):
                    # Unknown union part (no text member): verbatim replay —
                    # never inject a fabricated "text" key (illegal union).
                    if preserve_source:
                        payload.update(deepcopy(block.extra))
                    parts.append(payload)
                    continue
                payload["text"] = block.text or ""
                if preserve_source:
                    payload.update(deepcopy(block.extra))
                parts.append(payload)
        return parts

    def _parse_tools(self, tools: Iterable[dict[str, Any]]) -> list[ToolDefinition]:
        parsed: list[ToolDefinition] = []
        for container_index, tool in enumerate(tools):
            payload = dict(tool or {})
            # Hosted Gemini tools (googleSearch, codeExecution, urlContext,
            # fileSearch, googleSearchRetrieval, ...): identity is the
            # single-key envelope; no declarations.
            hosted_key = next(
                (
                    k
                    for k in (
                        "googleSearch",
                        "google_search",
                        "codeExecution",
                        "code_execution",
                        "urlContext",
                        "url_context",
                        "googleMaps",
                        "fileSearch",
                        "file_search",
                        "googleSearchRetrieval",
                        "google_search_retrieval",
                    )
                    if k in payload
                ),
                None,
            )
            if hosted_key and not payload.get("functionDeclarations"):
                parsed.append(
                    ToolDefinition(
                        name=hosted_key,
                        type="server",
                        input_schema={},
                        extra={"raw": deepcopy(tool), "gemini_hosted_tool": hosted_key},
                    )
                )
                continue
            declarations = payload.get("functionDeclarations") or payload.get("function_declarations") or []
            if declarations:
                for index, declaration in enumerate(declarations):
                    if not isinstance(declaration, dict):
                        continue
                    parsed.append(
                        ToolDefinition(
                            name=str(declaration.get("name") or ""),
                            description=declaration.get("description"),
                            # parametersJsonSchema is the Gemini-3 spelling of
                            # the same object schema; either is accepted.
                            input_schema=deepcopy(
                                declaration.get("parameters")
                                or declaration.get("parametersJsonSchema")
                                or declaration.get("parameters_json_schema")
                                or {}
                            ),
                            type="function",
                            extra={"raw_container": deepcopy(tool), "container_index": container_index, "declaration_index": index},
                        )
                    )
                continue
            parsed.append(
                ToolDefinition(
                    name=str(payload.get("name") or payload.get("type") or "gemini_tool"),
                    description=payload.get("description"),
                    input_schema=deepcopy(payload.get("parameters") or {}),
                    type=str(payload.get("type") or next(iter(payload.keys()), "tool")),
                    extra={"raw": deepcopy(tool)},
                )
            )
        return parsed

    def _format_tools(self, tools: Iterable[ToolDefinition], *, preserve_source: bool = True) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        ungrouped: list[dict[str, Any]] = []
        for tool in tools:
            hosted_kind = tool.extra.get("gemini_hosted_tool")
            if hosted_kind:
                raw = tool.extra.get("raw")
                if preserve_source and isinstance(raw, dict):
                    ungrouped.append(deepcopy(raw))
                else:
                    ungrouped.append({hosted_kind: {}})
                continue
            server_type = str(tool.extra.get("server_tool_type") or "")
            # Cross-protocol hosted tools map onto their Gemini native
            # envelopes — NEVER fabricated as functionDeclarations (the
            # model would call a function the client never declared and
            # the hosted tool never executes). Matching normalizes case and
            # underscores (validation admits both spellings); retrieval is
            # ordered BEFORE search so the prefix never folds it.
            normalized_server = server_type.lower().replace("_", "")
            hosted_map = [
                ("websearch", "googleSearch"),
                ("googlesearchretrieval", "googleSearchRetrieval"),
                ("googlesearch", "googleSearch"),
                ("codeexecution", "codeExecution"),
                ("urlcontext", "urlContext"),
                ("googlemaps", "googleMaps"),
                ("filesearch", "fileSearch"),
            ]
            mapped_envelope = None
            for prefix, native in hosted_map:
                if normalized_server.startswith(prefix) or (tool.type == "web_search" and native == "googleSearch"):
                    mapped_envelope = native
                    break
            if mapped_envelope:
                ungrouped.append({mapped_envelope: {}})
                continue
            raw_container = tool.extra.get("raw_container")
            container_index = tool.extra.get("container_index")
            declaration_index = tool.extra.get("declaration_index")
            if preserve_source and isinstance(raw_container, dict) and isinstance(container_index, int) and isinstance(declaration_index, int):
                container = grouped.setdefault(container_index, deepcopy(raw_container))
                declarations = container.setdefault("functionDeclarations", [])
                while len(declarations) <= declaration_index:
                    declarations.append({})
                declaration = deepcopy(declarations[declaration_index]) if isinstance(declarations[declaration_index], dict) else {}
                declaration["name"] = tool.name
                if tool.description is not None:
                    declaration["description"] = tool.description
                declaration["parameters"] = deepcopy(tool.input_schema)
                declarations[declaration_index] = declaration
                continue
            ungrouped.append(self._format_tool(tool, preserve_source=preserve_source))
        return [grouped[index] for index in sorted(grouped)] + ungrouped

    def _format_tool(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        raw = tool.extra.get("raw")
        if preserve_source and isinstance(raw, dict):
            return deepcopy(raw)
        declaration: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            declaration["description"] = tool.description
        declaration["parameters"] = deepcopy(tool.input_schema)
        return {"functionDeclarations": [declaration]}

    def _format_tool_call(self, call: ToolCall, *, preserve_source: bool, emit_opaque_state: bool = True) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        # Sanitize the raw replay down to the legal union-part shape.
        payload.pop("functionCall", None)
        payload.pop("function_call", None)
        payload.pop("thoughtSignature", None)
        payload.pop("thought_signature", None)
        function_call = {"name": call.name or "", "args": tool_arguments_object(call.arguments)}
        # `id` is Gemini-3+ only; synthetic correlation ids never reach the
        # wire (Gemini 2.x rejects them, pairing is name-based).
        if call.id and not call.extra.get("synthetic_id"):
            function_call["id"] = call.id
        payload["functionCall"] = function_call
        signature = call.signature or (call.raw.get("thoughtSignature") if isinstance(call.raw, dict) else None) or (call.raw.get("thought_signature") if isinstance(call.raw, dict) else None)
        if signature and emit_opaque_state:
            # D8: the signature replays with the call part for compatible
            # providers (Gemini 3 rejects unsigned first-per-step calls).
            payload["thoughtSignature"] = signature
        return payload

    def _format_tool_result(self, result: ToolResult, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(result.raw) if preserve_source and isinstance(result.raw, dict) else {}
        payload.pop("functionResponse", None)
        payload.pop("function_response", None)
        result_content = {"error": result.content} if result.is_error else result.content
        response = {"name": result.name or result.tool_call_id or "", "response": tool_result_object(result_content)}
        # `id` is Gemini-3+ only and never fabricated: emit only when the
        # source wire carried one, or when a genuine foreign correlation id
        # exists (never the name-derived fallback, never synthetics).
        had_wire_id = result.extra.get("had_function_response_id") is True
        genuine_foreign_id = (
            result.tool_call_id
            and result.tool_call_id != result.name
            and not result.extra.get("synthetic_tool_call_id")
        )
        if had_wire_id or genuine_foreign_id:
            response["id"] = result.tool_call_id
        payload["functionResponse"] = response
        part_signature = result.extra.get("thought_signature")
        if part_signature:
            # Part-level signatures ride the outer part (Gemini contract),
            # same as functionCall parts.
            payload["thoughtSignature"] = part_signature
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        source = request.extensions.get(self.name, {}) if preserve_source else {}
        generation = deepcopy(source.get("generationConfig") or {})
        safety = deepcopy(source.get("safetySettings") or [])
        tool_config = deepcopy(source.get("toolConfig") or {})
        params = deepcopy(request.generation_params)
        canonical_safety = params.pop("safety_settings", None)
        if canonical_safety is not None:
            safety = deepcopy(canonical_safety)
        if request.modalities:
            generation["responseModalities"] = [str(value).upper() for value in request.modalities]
        mapping = {
            "max_output_tokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
            "top_p": "topP",
            "top_k": "topK",
            "temperature": "temperature",
            "candidate_count": "candidateCount",
            "seed": "seed",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
        }
        for canonical, wire in mapping.items():
            if canonical in params:
                generation[wire] = params.pop(canonical)
        thinking_config = generation.get("thinkingConfig") or generation.get("thinking_config")
        if preserve_source and isinstance(thinking_config, dict) and "thinkingBudget" in thinking_config and "thinkingLevel" in thinking_config:
            # Same-protocol verbatim replay of a documented-illegal pair:
            # warn without rewriting (the provider's own 400 is clearer
            # than a silent mutation of the client's shape).
            add_conversion_warning(
                request,
                code="reasoning_control_conflict",
                message="thinkingConfig carries both thinkingBudget and thinkingLevel; the API rejects the pair (replayed verbatim — the provider will arbitrate)",
                field="generationConfig.thinkingConfig",
                target_protocol=self.name,
            )
        tool_config_replay = tool_config if isinstance(tool_config, dict) else None
        if preserve_source and isinstance(tool_config_replay, dict) and isinstance(tool_config_replay.get("retrievalConfig"), dict):
            # Vertex-only retrieval config riding a verbatim replay: no
            # canonical home — disclosed so the loss is visible on rebuild.
            add_conversion_warning(
                request,
                code="unsupported_optional_control",
                message="toolConfig.retrievalConfig has no canonical representation; kept verbatim on the same-protocol path only",
                field="toolConfig.retrievalConfig",
                target_protocol=self.name,
            )
        structured = params.pop("structured_output", None)
        if isinstance(structured, dict) and not preserve_source:
            # Same-protocol keeps generationConfig verbatim from extensions
            # (schema key spelling, mime, everything) — the canonical
            # rebuild is a CROSS-PROTOCOL concern only.
            if structured.get("strict") is False:
                add_conversion_warning(
                    request,
                    code="structured_output_strictness_strengthened",
                    message="Gemini enforces its response schema; explicit strict=false was strengthened",
                    target_protocol=self.name,
                    field="structured_output.strict",
                )
            formatted_structure = format_structured_output(structured, self.name)
            if formatted_structure is not None:
                generation.update(
                    {
                        key: value
                        for key, value in formatted_structure.items()
                        if value is not None
                    }
                )
            elif structured.get("type") not in {"json_schema", "json_object"}:
                add_conversion_warning(
                    request,
                    code="unsupported_optional_control",
                    message=f"structured output type {structured.get('type')!r} has no Gemini representation; dropped",
                    target_protocol=self.name,
                    field="structured_output",
                )
            # responseSchema and responseJsonSchema are mutually exclusive
            # (the docs mandate omitting the counterpart) — never both.
            if "responseJsonSchema" in generation:
                generation.pop("responseSchema", None)
            elif "responseSchema" in generation and "responseMimeType" in generation and generation.get("responseMimeType") != "application/json":
                generation.pop("responseSchema", None)
        reasoning = params.pop("reasoning", None)
        if not preserve_source:
            # Cross-protocol mapping only; same-protocol passthrough keeps
            # the preserved original verbatim.
            reasoning_emissions = format_reasoning_controls(reasoning, self.name, request)
            generation.update(reasoning_emissions.get("generation_config", {}))
        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            if preserve_source and tool_config:
                # Same-protocol: the original toolConfig (any mode +
                # allowlist combination) is restored verbatim.
                pass
            else:
                tool_config = format_tool_choice(tool_choice, self.name)
        retain_supported_generation_params(
            request,
            params,
            supported=set(),
            target_protocol=self.name,
        )
        if not preserve_source:
            # Unmapped generationConfig controls (logprobs, mediaResolution,
            # speechConfig, imageConfig, ...) dropping at foreign targets is
            # RECORDED — never silent (they never enter canonical params, so
            # the retain-warn above cannot see them).
            source_generation = request.extensions.get(self.name, {}).get("generationConfig")
            if isinstance(source_generation, dict):
                handled = set(generation) | {
                    "maxOutputTokens", "stopSequences", "topP", "topK", "temperature",
                    "candidateCount", "seed", "frequencyPenalty", "presencePenalty",
                    "responseMimeType", "responseSchema", "responseJsonSchema",
                    "thinkingConfig", "thinkingLevel",
                }
                for key in source_generation:
                    if key not in handled:
                        add_conversion_warning(
                            request,
                            code="unsupported_optional_control",
                            message=f"generationConfig.{key} has no cross-protocol representation; dropped",
                            target_protocol=self.name,
                            field=f"generationConfig.{key}",
                        )
        return generation, safety, tool_config

    def _format_usage(self, usage: Usage | None) -> dict[str, int] | None:
        if usage is None:
            return None
        payload = {
            "promptTokenCount": usage.input_tokens,
            "candidatesTokenCount": usage.output_tokens,
            "totalTokenCount": usage.total_tokens,
        }
        if usage.reasoning_tokens:
            payload["thoughtsTokenCount"] = usage.reasoning_tokens
        if usage.cache_read_tokens:
            payload["cachedContentTokenCount"] = usage.cache_read_tokens
        tool_prompt = (usage.extra or {}).get("tool_use_prompt_tokens")
        if tool_prompt:
            payload["toolUsePromptTokenCount"] = int(tool_prompt)
        return payload


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _operation_from_context(context: ProtocolContext | None, default: str) -> str:
    supported = {OPERATION_CHAT, OPERATION_COUNT_TOKENS, "generate", "stream_generate"}
    if context and isinstance(context.provider_options, dict):
        operation = normalize_operation(context.provider_options.get("operation"))
        if operation in supported:
            return operation
    if context and isinstance(context.metadata, dict):
        operation = normalize_operation(context.metadata.get("operation"))
        if operation in supported:
            return operation
    return default


def _response_operation(response: dict[str, Any], context: ProtocolContext | None) -> str:
    requested = _operation_from_context(context, OPERATION_CHAT)
    if requested == OPERATION_COUNT_TOKENS:
        return OPERATION_COUNT_TOKENS
    if "totalTokens" in response and "candidates" not in response:
        return OPERATION_COUNT_TOKENS
    return requested if requested in {"generate", "stream_generate"} else OPERATION_CHAT


def _decode_sse_data(raw_event: Any) -> Any:
    from .streaming import decode_sse_data

    return decode_sse_data(raw_event)


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}


def _parse_gemini_generation_params(generation: dict[str, Any], tool_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini generation and tool controls."""

    params: dict[str, Any] = {}
    mapping = {
        "maxOutputTokens": "max_output_tokens",
        "stopSequences": "stop_sequences",
        "topP": "top_p",
        "topK": "top_k",
        "temperature": "temperature",
        "candidateCount": "candidate_count",
        "seed": "seed",
        "frequencyPenalty": "frequency_penalty",
        "presencePenalty": "presence_penalty",
    }
    for wire, canonical in mapping.items():
        if wire in generation:
            params[canonical] = deepcopy(generation[wire])
    response_mime = generation.get("responseMimeType")
    has_schema = generation.get("responseJsonSchema") is not None or generation.get("responseSchema") is not None
    # responseFormat (the newer structured-output envelope) parses alongside
    # the classic keys — the schema/mime inside it carry the same semantics.
    response_format = generation.get("responseFormat")
    if isinstance(response_format, dict) and isinstance(response_format.get("text"), dict):
        text_cfg = response_format["text"]
        if response_mime is None and isinstance(text_cfg.get("mimeType"), str):
            response_mime = text_cfg["mimeType"]
        if not has_schema and isinstance(text_cfg.get("schema"), dict):
            generation = dict(generation)
            generation["responseSchema"] = text_cfg["schema"]
            has_schema = True
    if has_schema or (isinstance(response_mime, str) and response_mime == "application/json"):
        # ONLY application/json means structured output (text/x.enum and
        # text/plain are distinct modes, never folded into JSON).
        params["structured_output"] = canonical_structured_output(
            {
                "type": "json_schema" if has_schema else "json_object",
                "schema": deepcopy(generation.get("responseJsonSchema") or generation.get("responseSchema")),
            },
            "gemini",
        )
    elif isinstance(response_mime, str) and response_mime not in ("text/plain", "text/x.enum"):
        # Non-default mime types ride extensions for same-protocol replay
        # and are recorded as unsupported cross-protocol.
        params["response_mime_type"] = response_mime
    elif response_mime == "text/x.enum":
        params["response_mime_type"] = response_mime
    thinking = generation.get("thinkingConfig")
    if isinstance(thinking, dict):
        budget = thinking.get("thinkingBudget")
        if budget == -1:
            # -1 = DYNAMIC thinking (the model decides) — a distinct mode,
            # never an absent control and never an inverted effort.
            params["reasoning"] = {"enabled": True, "dynamic": True, "include_thoughts": thinking.get("includeThoughts")}
        else:
            params["reasoning"] = {
                # thinkingBudget: 0 is Gemini's documented OFF switch — canonical
                # folds it to enabled: False (never an "enabled with 0" inversion).
                **({"enabled": False} if budget == 0 else {}),
                **({} if budget == 0 else {"budget_tokens": budget}),
                "include_thoughts": thinking.get("includeThoughts"),
            }
        # thinkingLevel is a thinkingConfig member (Gemini-3 effort lever:
        # minimal/low/medium/high — the documented vocabulary only; level
        # strings always map to effort, NEVER to the dynamic mode flag
        # (dynamic is thinkingBudget:-1's semantic alone).
        level = thinking.get("thinkingLevel") or thinking.get("thinking_level")
        if isinstance(level, str) and level.strip():
            normalized_level = level.strip().lower()
            params["reasoning"] = {"effort": normalized_level, **params["reasoning"]}
        params["reasoning"] = {k: v for k, v in params["reasoning"].items() if v is not None}
    if tool_config:
        params["tool_choice"] = _parse_gemini_tool_choice(tool_config)
    return params


def _parse_gemini_tool_choice(tool_config: dict[str, Any]) -> Any:
    """Normalize Gemini function-calling mode and allow-list."""

    config = tool_config.get("functionCallingConfig") or tool_config.get("function_calling_config") or {}
    if not isinstance(config, dict):
        return deepcopy(tool_config)
    mode = str(config.get("mode") or "AUTO").lower()
    names = deepcopy(config.get("allowedFunctionNames") or config.get("allowed_function_names") or [])
    if mode == "none":
        return {"mode": "none"}
    if mode == "validated":
        # Documented distinct mode (schema-adherence enforcement); a silent
        # downgrade to AUTO would lose the guarantee the client asked for.
        return {"mode": "validated"}
    if mode == "any" and len(names) == 1:
        return {"mode": "named", "name": names[0]}
    if mode == "any":
        return {"mode": "required", "allowed_names": names}
    # AUTO with an allowlist is a legal distinct shape (free choice within
    # the list) — the list survives, never silently dropped.
    if names:
        return {"mode": "auto", "allowed_names": names}
    return {"mode": "auto"}


def _parse_gemini_media_source(value: Any, *, inline: bool) -> MediaSource:
    """Normalize Gemini inlineData and fileData sources."""

    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="base64" if inline else "url",
        media_type=payload.get("mimeType") or payload.get("mime_type"),
        url=payload.get("fileUri") or payload.get("file_uri"),
        data=payload.get("data"),
        file_id=payload.get("fileId") or payload.get("file_id"),
        raw=deepcopy(value),
        extra=_without(payload, {"mimeType", "mime_type", "fileUri", "file_uri", "data", "fileId", "file_id"}),
    )


def _media_block_type(media_type: Any) -> str:
    """Return the canonical media category for a MIME type."""

    value = str(media_type or "").lower()
    if value.startswith("image/"):
        return "image"
    if value.startswith("audio/"):
        return "audio"
    if value.startswith("video/"):
        return "video"
    return "file"


def _coerce_media_source(value: Any) -> MediaSource:
    """Coerce legacy dictionaries into a canonical media source."""

    if isinstance(value, MediaSource):
        return value
    if isinstance(value, str):
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="base64" if payload.get("data") else "url",
        media_type=payload.get("mimeType") or payload.get("mime_type") or payload.get("media_type"),
        url=payload.get("fileUri") or payload.get("file_uri") or payload.get("url"),
        data=payload.get("data"),
        file_id=payload.get("fileId") or payload.get("file_id"),
        raw=deepcopy(value),
    )


def _format_gemini_media(block: ContentBlock, *, preserve_source: bool, warnings: list | None = None) -> dict[str, Any] | None:
    """Format canonical media as a Gemini content part (None = dropped)."""

    source = _coerce_media_source(block.source)
    payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
    if source.data:
        if source.media_type:
            payload["inlineData"] = {"mimeType": source.media_type, "data": source.data}
        else:
            # inlineData.mimeType is required: mime-less inline data drops
            # with a recorded warning rather than an invented type.
            if warnings is not None and not any(
                w.code == "media_dropped" and w.message == "inline media without a mimeType has no Gemini representation; dropped" for w in warnings
            ):
                warnings.append(
                    ConversionWarning(
                        code="media_dropped",
                        message="inline media without a mimeType has no Gemini representation; dropped",
                        field="content[media]",
                        source_protocol=None,
                        target_protocol="gemini",
                    )
                )
            return None
    else:
        # URL/fileId-only media: fileUri is the wire member (external HTTPS
        # included; fileId is not a documented FileData member — it stays in
        # canonical identity for chat-side file_id mapping, never emitted).
        file_data: dict[str, Any] = {"fileUri": source.url or ""}
        if not file_data["fileUri"] and not source.data:
            # Nothing representable: an empty fileUri is an illegal shape —
            # disclosed, never a silent empty part.
            if warnings is not None and not any(
                w.code == "media_dropped" and w.message == "media without inline data or a fileUri has no Gemini part shape; dropped" for w in warnings
            ):
                warnings.append(
                    ConversionWarning(
                        code="media_dropped",
                        message="media without inline data or a fileUri has no Gemini part shape; dropped",
                        field="content[media]",
                        source_protocol=None,
                        target_protocol="gemini",
                    )
                )
            return None
        if source.media_type:
            file_data["mimeType"] = source.media_type
        payload["fileData"] = file_data
    return payload
