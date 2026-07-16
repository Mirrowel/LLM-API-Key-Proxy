# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Shared canonical semantics for cross-protocol generative conversion.

Protocol modules own wire parsing and formatting. This module owns only meanings
that must be identical across those modules: logical operations, completion
reasons, instruction placement, source-aware passthrough, and warning records.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Optional

from .types import (
    ContentBlock,
    ConversionWarning,
    ProtocolContext,
    ReasoningBlock,
    ToolCall,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    serialize_value,
)


STOP_REASON_STOP = "stop"
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_TOOL_USE = "tool_use"
STOP_REASON_CONTENT_FILTER = "content_filter"
STOP_REASON_ERROR = "error"
STOP_REASON_INCOMPLETE = "incomplete"
STOP_REASON_UNKNOWN = "unknown"


_STOP_REASON_ALIASES = {
    "stop": STOP_REASON_STOP,
    "end_turn": STOP_REASON_STOP,
    "stop_sequence": STOP_REASON_STOP,
    "completed": STOP_REASON_STOP,
    "length": STOP_REASON_MAX_TOKENS,
    "max_tokens": STOP_REASON_MAX_TOKENS,
    "max_output_tokens": STOP_REASON_MAX_TOKENS,
    "tool_calls": STOP_REASON_TOOL_USE,
    "function_call": STOP_REASON_TOOL_USE,
    "tool_use": STOP_REASON_TOOL_USE,
    "content_filter": STOP_REASON_CONTENT_FILTER,
    "safety": STOP_REASON_CONTENT_FILTER,
    "blocklist": STOP_REASON_CONTENT_FILTER,
    "prohibited_content": STOP_REASON_CONTENT_FILTER,
    "recitation": STOP_REASON_CONTENT_FILTER,
    "failed": STOP_REASON_ERROR,
    "error": STOP_REASON_ERROR,
    "incomplete": STOP_REASON_INCOMPLETE,
}


_TARGET_STOP_REASONS = {
    "openai_chat": {
        STOP_REASON_STOP: "stop",
        STOP_REASON_MAX_TOKENS: "length",
        STOP_REASON_TOOL_USE: "tool_calls",
        STOP_REASON_CONTENT_FILTER: "content_filter",
        STOP_REASON_ERROR: None,
        STOP_REASON_INCOMPLETE: "length",
        STOP_REASON_UNKNOWN: None,
    },
    "anthropic_messages": {
        STOP_REASON_STOP: "end_turn",
        STOP_REASON_MAX_TOKENS: "max_tokens",
        STOP_REASON_TOOL_USE: "tool_use",
        STOP_REASON_CONTENT_FILTER: "refusal",
        STOP_REASON_ERROR: None,
        STOP_REASON_INCOMPLETE: "max_tokens",
        STOP_REASON_UNKNOWN: None,
    },
    "gemini": {
        STOP_REASON_STOP: "STOP",
        STOP_REASON_MAX_TOKENS: "MAX_TOKENS",
        STOP_REASON_TOOL_USE: "STOP",
        STOP_REASON_CONTENT_FILTER: "SAFETY",
        STOP_REASON_ERROR: "OTHER",
        STOP_REASON_INCOMPLETE: "MAX_TOKENS",
        STOP_REASON_UNKNOWN: "OTHER",
    },
    "responses": {
        STOP_REASON_STOP: "completed",
        STOP_REASON_MAX_TOKENS: "incomplete",
        STOP_REASON_TOOL_USE: "completed",
        STOP_REASON_CONTENT_FILTER: "incomplete",
        STOP_REASON_ERROR: "failed",
        STOP_REASON_INCOMPLETE: "incomplete",
        STOP_REASON_UNKNOWN: "incomplete",
    },
}


def canonical_stop_reason(value: Any) -> Optional[str]:
    """Normalize a provider/client completion reason into a stable meaning."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    return _STOP_REASON_ALIASES.get(normalized, STOP_REASON_UNKNOWN)


def format_stop_reason(value: Optional[str], target_protocol: str) -> Optional[str]:
    """Return the target protocol's public completion reason."""

    if value is None:
        return None
    return _TARGET_STOP_REASONS.get(target_protocol, {}).get(value, value)


def is_same_protocol(
    context: ProtocolContext | None,
    protocol_name: str,
    source_protocol: str | None = None,
) -> bool:
    """Return whether source-owned raw fields are safe to replay.

    A missing context is not proof of ownership. Parsed unified objects carry
    their source protocol, so direct parse/build calls still preserve native
    fields without making manually constructed or foreign objects unsafe.
    """

    effective_source = context.source_protocol if context and context.source_protocol else source_protocol
    return effective_source == protocol_name


def source_extensions(
    extra: dict[str, Any],
    context: ProtocolContext | None,
    protocol_name: str,
    source_protocol: str | None = None,
) -> dict[str, Any]:
    """Return source extensions only for a same-protocol destination."""

    return deepcopy(extra) if is_same_protocol(context, protocol_name, source_protocol) else {}


def may_emit_opaque_provider_state(
    context: ProtocolContext | None,
    *,
    preserve_source: bool,
) -> bool:
    """Return whether opaque signatures may leave canonical/cache state.

    Opaque state is suppressed unless real execution explicitly proves provider
    compatibility through a compatible-domain flag or identical provider IDs.
    """

    if not preserve_source:
        return False
    if context is None:
        return False
    if context.provider_state_compatible:
        return True
    return bool(
        context.source_provider
        and context.target_provider
        and context.source_provider == context.target_provider
    )


def instruction_messages(request: UnifiedRequest) -> list[UnifiedMessage]:
    """Return ordered canonical system/developer instructions.

    Older parsers store a separate ``system`` block list. That field remains a
    compatibility carrier until all non-generative consumers migrate, but it is
    promoted only when no explicit system message already exists.
    """

    instructions = [message for message in request.messages if message.role in {"system", "developer"}]
    if request.system and not any(message.role == "system" for message in instructions):
        instructions.insert(0, UnifiedMessage(role="system", content=deepcopy(request.system)))
    return instructions


def conversation_messages(request: UnifiedRequest) -> list[UnifiedMessage]:
    """Return messages excluding system/developer instruction turns."""

    return [message for message in request.messages if message.role not in {"system", "developer"}]


def instruction_blocks(request: UnifiedRequest) -> list[ContentBlock]:
    """Flatten ordered instruction turns for protocols with one system field."""

    blocks: list[ContentBlock] = []
    for message in instruction_messages(request):
        blocks.extend(deepcopy(message.content))
    return blocks


def add_conversion_warning(
    request: UnifiedRequest,
    *,
    code: str,
    message: str,
    field: str | None,
    target_protocol: str,
) -> None:
    """Record a deliberate omission of an optional conversion hint."""

    request.warnings.append(
        ConversionWarning(
            code=code,
            message=message,
            field=field,
            source_protocol=request.source_protocol,
            target_protocol=target_protocol,
        )
    )


def retain_supported_generation_params(
    request: UnifiedRequest,
    params: dict[str, Any],
    *,
    supported: set[str],
    target_protocol: str,
) -> dict[str, Any]:
    """Return supported optional controls and record every deliberate omission."""

    kept: dict[str, Any] = {}
    for key, value in params.items():
        if key in supported:
            kept[key] = value
            continue
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message=f"{target_protocol} has no safe mapping for optional control '{key}'",
            field=key,
            target_protocol=target_protocol,
        )
    return kept


def canonical_tool_choice(value: Any, source_protocol: str) -> dict[str, Any] | None:
    """Normalize tool-choice spellings used by the four generative APIs."""

    if value is None:
        return None
    if isinstance(value, str):
        mode = value.lower()
        if mode in {"auto", "none", "required"}:
            return {"mode": mode}
        if mode in {"any", "any_required"}:
            return {"mode": "required"}
        return {"mode": "named", "name": value}
    if not isinstance(value, dict):
        return {"mode": "auto"}

    value_type = str(value.get("type") or value.get("mode") or "auto").lower()
    if source_protocol == "gemini":
        # Gemini normalization normally happens in its protocol module because
        # the value is nested in toolConfig; accept the canonical intermediate.
        if "allowed_names" in value:
            return deepcopy(value)
    if value_type in {"auto", "none"}:
        return {"mode": value_type}
    if value_type in {"required", "any"}:
        return {"mode": "required", "allowed_names": deepcopy(value.get("allowed_names") or [])}
    if value_type in {"function", "tool", "named"}:
        function = value.get("function") if isinstance(value.get("function"), dict) else {}
        name = value.get("name") or function.get("name")
        return {"mode": "named", "name": name}
    return {"mode": "auto"}


def format_tool_choice(value: Any, target_protocol: str) -> Any:
    """Format canonical tool choice for a destination protocol."""

    choice = value if isinstance(value, dict) and "mode" in value else canonical_tool_choice(value, target_protocol)
    if not choice:
        return None
    mode = choice.get("mode", "auto")
    name = choice.get("name")
    allowed_names = deepcopy(choice.get("allowed_names") or [])
    if target_protocol == "openai_chat":
        if mode == "named":
            return {"type": "function", "function": {"name": name or ""}}
        return "required" if mode == "required" else mode
    if target_protocol == "anthropic_messages":
        if mode == "named":
            return {"type": "tool", "name": name or ""}
        return {"type": "any" if mode == "required" else mode}
    if target_protocol == "responses":
        if mode == "named":
            return {"type": "function", "name": name or ""}
        return "required" if mode == "required" else mode
    if target_protocol == "gemini":
        if mode == "none":
            config: dict[str, Any] = {"mode": "NONE"}
        elif mode == "required":
            config = {"mode": "ANY"}
            if allowed_names:
                config["allowedFunctionNames"] = allowed_names
        elif mode == "named":
            config = {"mode": "ANY", "allowedFunctionNames": [name] if name else []}
        else:
            config = {"mode": "AUTO"}
        return {"functionCallingConfig": config}
    return deepcopy(value)


def canonical_structured_output(value: Any, source_protocol: str) -> dict[str, Any] | None:
    """Normalize JSON/object schema controls without source wrappers."""

    if not isinstance(value, dict):
        return None
    if source_protocol == "openai_chat":
        output_type = value.get("type")
        if output_type == "json_schema" and isinstance(value.get("json_schema"), dict):
            schema = value["json_schema"]
            return {
                "type": "json_schema",
                "name": schema.get("name"),
                "schema": deepcopy(schema.get("schema")),
                "strict": schema.get("strict"),
            }
        if output_type == "json_object":
            return {"type": "json_object"}
        return deepcopy(value)
    if source_protocol == "responses":
        return {
            "type": value.get("type") or "json_schema",
            "name": value.get("name"),
            "schema": deepcopy(value.get("schema")),
            "strict": value.get("strict"),
        }
    if source_protocol == "anthropic_messages":
        format_value = value.get("format") if isinstance(value.get("format"), dict) else value
        return {
            "type": format_value.get("type") or "json_schema",
            "name": format_value.get("name"),
            "schema": deepcopy(format_value.get("schema")),
            "strict": format_value.get("strict"),
        }
    if source_protocol == "gemini":
        normalized = deepcopy(value)
        if normalized.get("type") == "json_schema" and normalized.get("schema") is not None:
            normalized.setdefault("strict", True)
        return normalized
    return deepcopy(value)


def format_structured_output(value: Any, target_protocol: str) -> Any:
    """Format a canonical structured-output requirement for a destination."""

    if not isinstance(value, dict):
        return None
    output_type = value.get("type") or "json_schema"
    if target_protocol == "openai_chat":
        if output_type == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                key: deepcopy(item)
                for key, item in {
                    "name": value.get("name") or "response",
                    "schema": value.get("schema") or {},
                    "strict": value.get("strict"),
                }.items()
                if item is not None
            },
        }
    if target_protocol == "responses":
        if output_type == "json_object":
            return {"type": "json_object"}
        return {
            key: deepcopy(item)
            for key, item in {
                "type": "json_schema",
                "name": value.get("name") or "response",
                "schema": value.get("schema") or {},
                "strict": value.get("strict"),
            }.items()
            if item is not None
        }
    if target_protocol == "anthropic_messages":
        if output_type == "json_object":
            return {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        return {
            "format": {
                key: deepcopy(item)
                for key, item in {
                    "type": "json_schema",
                    "name": value.get("name"),
                    "schema": value.get("schema") or {},
                    "strict": value.get("strict"),
                }.items()
                if item is not None
            }
        }
    if target_protocol == "gemini":
        return {
            "responseMimeType": "application/json",
            "responseJsonSchema": deepcopy(value.get("schema")) if output_type != "json_object" else None,
        }
    return deepcopy(value)


def message_tool_calls(message: UnifiedMessage) -> list[ToolCall]:
    """Return de-duplicated calls from message and content representations."""

    calls: list[ToolCall] = []
    seen: set[tuple[Optional[str], Optional[str], str]] = set()
    for call in [*message.tool_calls, *[block.tool_call for block in message.content if block.tool_call]]:
        if call is None:
            continue
        key = (call.id, call.name, json.dumps(serialize_value(call.arguments), sort_keys=True, separators=(",", ":")))
        if key not in seen:
            calls.append(call)
            seen.add(key)
    return calls


def message_reasoning(message: UnifiedMessage) -> list[ReasoningBlock]:
    """Return de-duplicated reasoning from message and content representations."""

    blocks: list[ReasoningBlock] = []
    seen: set[tuple[Optional[str], Optional[str], bool]] = set()
    for reasoning in [*message.reasoning, *[block.reasoning for block in message.content if block.reasoning]]:
        if reasoning is None:
            continue
        key = (reasoning.text, reasoning.signature, reasoning.redacted)
        if key not in seen:
            blocks.append(reasoning)
            seen.add(key)
    return blocks


def message_tool_results(message: UnifiedMessage) -> list[ToolResult]:
    """Return tool results embedded in a canonical message."""

    return [block.tool_result for block in message.content if block.tool_result is not None]


def resolve_tool_result_names(messages: Iterable[UnifiedMessage]) -> list[UnifiedMessage]:
    """Enrich result records with function names from preceding calls.

    Chat and Anthropic identify results by call ID, while Gemini requires the
    function name on its response part. Keeping both in the canonical record
    allows either direction without provider-specific history lookups.
    """

    message_list = list(messages)
    names: dict[str, str] = {}
    ids_by_name: dict[str, list[str]] = {}
    result_index_by_name: dict[str, int] = {}
    call_index = 0
    for message in message_list:
        for call in message_tool_calls(message):
            if not call.id:
                call.id = f"call_{call_index}"
                call.extra["synthetic_id"] = True
            call_index += 1
            if call.id and call.name:
                names[call.id] = call.name
                ids_by_name.setdefault(call.name, []).append(call.id)
        for result in message_tool_results(message):
            if not result.name and result.tool_call_id in names:
                result.name = names[result.tool_call_id]
            if result.name and (not result.tool_call_id or result.tool_call_id == result.name):
                candidates = ids_by_name.get(result.name, [])
                result_index = result_index_by_name.get(result.name, 0)
                if result_index < len(candidates):
                    result.tool_call_id = candidates[result_index]
                    result.extra["synthetic_tool_call_id"] = True
                    result_index_by_name[result.name] = result_index + 1
    return message_list


def normalize_tool_result_messages(messages: Iterable[UnifiedMessage]) -> list[UnifiedMessage]:
    """Split embedded provider result blocks into canonical tool-role turns."""

    normalized: list[UnifiedMessage] = []
    for message in messages:
        if not any(block.tool_result for block in message.content):
            normalized.append(message)
            continue
        groups: list[tuple[bool, list[ContentBlock]]] = []
        for block in message.content:
            is_result = block.tool_result is not None
            if groups and groups[-1][0] == is_result:
                groups[-1][1].append(block)
            else:
                groups.append((is_result, [block]))
        for group_index, (is_result, blocks) in enumerate(groups):
            split = deepcopy(message)
            split.content = blocks
            split.role = "tool" if is_result else message.role
            split.tool_calls = [block.tool_call for block in blocks if block.tool_call]
            split.reasoning = [block.reasoning for block in blocks if block.reasoning]
            split.tool_call_id = blocks[0].tool_result.tool_call_id if is_result and blocks[0].tool_result else None
            if group_index:
                split.raw = None
                split.extra = {}
            normalized.append(split)
    return normalized


def ordered_message_blocks(message: UnifiedMessage) -> list[ContentBlock]:
    """Return one ordered block sequence without duplicate promoted fields."""

    blocks = deepcopy(message.content)
    reasoning_keys = {
        (block.reasoning.text, block.reasoning.signature, block.reasoning.redacted)
        for block in blocks
        if block.reasoning
    }
    call_keys = {
        (block.tool_call.id, block.tool_call.name, json.dumps(serialize_value(block.tool_call.arguments), sort_keys=True))
        for block in blocks
        if block.tool_call
    }
    missing_reasoning: list[ContentBlock] = []
    for reasoning in message.reasoning:
        key = (reasoning.text, reasoning.signature, reasoning.redacted)
        if key not in reasoning_keys:
            missing_reasoning.append(ContentBlock(type="reasoning", reasoning=deepcopy(reasoning)))
            reasoning_keys.add(key)
    if missing_reasoning:
        blocks = [*missing_reasoning, *blocks]
    for call in message.tool_calls:
        key = (call.id, call.name, json.dumps(serialize_value(call.arguments), sort_keys=True))
        if key not in call_keys:
            blocks.append(ContentBlock(type="tool_call", tool_call=deepcopy(call)))
            call_keys.add(key)
    return blocks


def coalesce_assistant_message(messages: Iterable[UnifiedMessage]) -> UnifiedMessage:
    """Collapse item-oriented provider output into one assistant message."""

    merged = UnifiedMessage(role="assistant")
    for message in messages:
        if message.role not in {"assistant", "model"}:
            continue
        merged.content.extend(ordered_message_blocks(message))
    merged.reasoning = message_reasoning(merged)
    merged.tool_calls = message_tool_calls(merged)
    return merged


def canonical_tool_arguments(value: Any) -> Any:
    """Decode JSON tool arguments while retaining invalid source text."""

    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def tool_arguments_object(value: Any) -> dict[str, Any]:
    """Return object arguments required by Anthropic and Gemini."""

    normalized = canonical_tool_arguments(value)
    if isinstance(normalized, dict):
        return normalized
    raise ValueError("tool arguments must be a JSON object")


def tool_arguments_text(value: Any) -> str:
    """Return compact JSON arguments required by Chat and Responses."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def tool_result_text(value: Any) -> str:
    """Return the string result representation required by Chat/Responses."""

    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def tool_result_object(value: Any) -> dict[str, Any]:
    """Return the object result representation required by Gemini."""

    normalized = canonical_tool_arguments(value)
    if isinstance(normalized, dict):
        return normalized
    return {"result": serialize_value(normalized)}
