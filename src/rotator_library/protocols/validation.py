# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Destination capability validation for generative protocol conversion."""

from __future__ import annotations

from typing import Any

from .canonical import (
    STOP_REASON_ERROR,
    canonical_tool_arguments,
    is_same_protocol,
    message_tool_calls,
    message_tool_results,
)
from .types import MediaSource, ProtocolContext, ProtocolError, UnifiedRequest, UnifiedResponse


_CONTENT_CAPABILITIES: dict[str, set[str]] = {
    "openai_chat": {"text", "image", "audio", "file", "document", "reasoning", "tool_call", "tool_result"},
    "anthropic_messages": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result"},
    "responses": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result"},
    "gemini": {"text", "image", "audio", "video", "file", "document", "reasoning", "tool_call", "tool_result"},
}

_RESPONSE_MODALITIES: dict[str, set[str]] = {
    "openai_chat": {"text", "audio"},
    "anthropic_messages": {"text"},
    "responses": {"text", "audio"},
    "gemini": {"text", "audio", "image"},
}


def validate_generative_request(
    request: UnifiedRequest,
    target_protocol: str,
    context: ProtocolContext | None,
) -> None:
    """Reject meaning-changing cross-protocol losses before provider transport.

    Same-protocol requests may retain future native content through their raw
    payload. Cross-protocol requests must use a known canonical meaning so a
    source-native object can never be emitted as a malformed foreign object.
    """

    if is_same_protocol(context, target_protocol, request.source_protocol):
        return
    if request.previous_response_id and target_protocol != "responses":
        raise ProtocolError(
            "A provider-bound previous_response_id cannot be translated safely",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": "previous_response_id"},
        )
    provider_bound_responses_fields = [
        field
        for field in ("background", "conversation", "prompt")
        if request.generation_params.get(field) not in (None, False)
    ]
    if provider_bound_responses_fields and target_protocol != "responses":
        raise ProtocolError(
            "Provider-bound Responses controls cannot be translated safely",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"fields": provider_bound_responses_fields},
        )
    if request.generation_params.get("safety_settings") and target_protocol != "gemini":
        raise ProtocolError(
            "Gemini safety settings have no equivalent in the selected provider protocol",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": "safety_settings"},
        )
    unsupported_modalities = set(request.modalities) - _RESPONSE_MODALITIES.get(target_protocol, {"text"})
    if unsupported_modalities:
        raise ProtocolError(
            f"{target_protocol} cannot produce required response modalities: {sorted(unsupported_modalities)}",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": "modalities", "unsupported": sorted(unsupported_modalities)},
        )
    supported = _CONTENT_CAPABILITIES.get(target_protocol, set())
    block_groups = [("system", request.system)] + [
        (f"message:{message_index}", message.content)
        for message_index, message in enumerate(request.messages)
    ]
    for group_name, blocks in block_groups:
        for block_index, block in enumerate(blocks):
            if block.type in supported:
                if block.type in {"image", "audio", "video", "file", "document"} and not _has_media_identity(block.source):
                    raise ProtocolError(
                        f"Cannot represent {block.type} content without URL, data, or file identity",
                        protocol=target_protocol,
                        pass_name="validate_request",
                        payload={"group": group_name, "content_index": block_index, "content_type": block.type},
                    )
                continue
            raise ProtocolError(
                f"Cannot represent required content type '{block.type}' in {target_protocol}",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"group": group_name, "content_index": block_index, "content_type": block.type},
            )
    for tool_index, tool in enumerate(request.tools):
        if tool.type != "function":
            raise ProtocolError(
                f"Cannot safely translate tool type '{tool.type}' into {target_protocol}",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"tool_index": tool_index, "tool_type": tool.type, "tool_name": tool.name},
            )
        if not tool.name or not isinstance(tool.input_schema, dict):
            raise ProtocolError(
                "Function tools require a name and object input schema",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"tool_index": tool_index, "tool_name": tool.name},
            )
    for message_index, message in enumerate(request.messages):
        for call_index, call in enumerate(message_tool_calls(message)):
            arguments = canonical_tool_arguments(call.arguments)
            if not call.name or (target_protocol != "gemini" and not call.id):
                raise ProtocolError(
                    "Tool calls require a name and correlation ID",
                    protocol=target_protocol,
                    pass_name="validate_request",
                    payload={"message_index": message_index, "call_index": call_index},
                )
            if target_protocol in {"anthropic_messages", "gemini"} and not isinstance(arguments, dict):
                raise ProtocolError(
                    f"{target_protocol} requires tool arguments to be a JSON object",
                    protocol=target_protocol,
                    pass_name="validate_request",
                    payload={"message_index": message_index, "call_index": call_index},
                )
        for result_index, result in enumerate(message_tool_results(message)):
            if not result.tool_call_id:
                raise ProtocolError(
                    "Tool results require correlation identity",
                    protocol=target_protocol,
                    pass_name="validate_request",
                    payload={"message_index": message_index, "result_index": result_index},
                )
            if target_protocol == "gemini" and not result.name:
                raise ProtocolError(
                    "Gemini tool results require the originating function name",
                    protocol=target_protocol,
                    pass_name="validate_request",
                    payload={"message_index": message_index, "result_index": result_index},
                )
    _validate_tool_choice(request, target_protocol)


def validate_generative_response(response: UnifiedResponse, target_protocol: str) -> None:
    """Reject failed provider responses that a target success envelope cannot express."""

    if response.stop_reason == STOP_REASON_ERROR and target_protocol != "responses":
        raise ProtocolError(
            f"{target_protocol} cannot represent a failed provider response as a successful completion",
            protocol=target_protocol,
            pass_name="validate_response",
            payload={"stop_reason": response.stop_reason},
        )


def _validate_tool_choice(request: UnifiedRequest, target_protocol: str) -> None:
    """Require named and allow-listed choices to reference declared tools."""

    choice: Any = request.generation_params.get("tool_choice")
    if not isinstance(choice, dict):
        return
    available = {tool.name for tool in request.tools if tool.name}
    if choice.get("mode") == "named":
        name = str(choice.get("name") or "").strip()
        if not name:
            raise ProtocolError(
                "Named tool choice requires a tool name",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"field": "tool_choice"},
            )
        if name not in available:
            raise ProtocolError(
                f"Named tool choice references unavailable tool {name!r}",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"field": "tool_choice", "name": name},
            )
    allowed_names = {str(name) for name in choice.get("allowed_names") or []}
    missing = allowed_names - available
    if missing:
        raise ProtocolError(
            f"Tool choice references unavailable tools: {sorted(missing)}",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": "tool_choice", "names": sorted(missing)},
        )


def _has_media_identity(source: object) -> bool:
    """Return whether a canonical or legacy media source is transportable."""

    if isinstance(source, MediaSource):
        return bool(source.url or source.data or source.file_id)
    if isinstance(source, str):
        return bool(source)
    if isinstance(source, dict):
        return bool(
            source.get("url")
            or source.get("data")
            or source.get("file_id")
            or source.get("fileUri")
            or source.get("file_uri")
        )
    return False
