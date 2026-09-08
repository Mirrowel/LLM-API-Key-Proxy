# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Destination capability validation for generative protocol conversion."""

from __future__ import annotations

from typing import Any

from .canonical import (
    STOP_REASON_ERROR,
    add_conversion_warning,
    canonical_tool_arguments,
    is_same_protocol,
    message_tool_calls,
    message_tool_results,
)
from .types import MediaSource, ProtocolContext, ProtocolError, UnifiedRequest, UnifiedResponse


_CONTENT_CAPABILITIES: dict[str, set[str]] = {
    "openai_chat": {"text", "image", "audio", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    # Refusal turns degrade to text with refusal stop semantics (D7
    # equivalent construct) rather than blocking the whole request.
    "anthropic_messages": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    "responses": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    "gemini": {"text", "image", "audio", "video", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
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
    # Cross-protocol extension fields have no foreign representation by
    # definition (each target replays only its own extensions verbatim):
    # every extra key dropping at a foreign target is disclosed (D7).
    for extra_key in sorted(request.extra):
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message=f"{extra_key} has no {target_protocol} representation; dropped cross-protocol",
            field=extra_key,
            target_protocol=target_protocol,
        )
    # Metadata dictionaries ride only where a target has a metadata field.
    if request.metadata and target_protocol == "gemini":
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message="metadata has no Gemini representation; dropped cross-protocol",
            field="metadata",
            target_protocol=target_protocol,
        )
    # Gemini-source envelope fields and unmapped generationConfig controls
    # dropping at FOREIGN targets are recorded here (symmetric with the
    # gemini-target build-time warnings; parse cannot know the target).
    # Bound envelope fields (cachedContent/labels/...) ride request.extra —
    # covered by the generic extra disclosure above.
    if request.source_protocol == "gemini" and target_protocol != "gemini":
        source_generation = request.extensions.get("gemini", {}).get("generationConfig")
        if isinstance(source_generation, dict):
            mapped = {
                "maxOutputTokens", "stopSequences", "topP", "topK", "temperature",
                "candidateCount", "seed", "frequencyPenalty", "presencePenalty",
                "responseMimeType", "responseSchema", "responseJsonSchema",
                "thinkingConfig", "thinkingLevel", "responseModalities",
            }
            for key in source_generation:
                if key not in mapped:
                    add_conversion_warning(
                        request,
                        code="unsupported_optional_control",
                        message=f"generationConfig.{key} has no cross-protocol representation; dropped",
                        field=f"generationConfig.{key}",
                        target_protocol=target_protocol,
                    )
    # Opaque function-call signatures (Gemini thought signatures) dropping
    # at foreign boundaries are disclosed — never silent (Gemini 3 rejects
    # unsigned current-turn calls on the way back).
    if request.source_protocol == "gemini" and target_protocol != "gemini":
        for message in request.messages:
            for call in message_tool_calls(message):
                if getattr(call, "signature", None):
                    add_conversion_warning(
                        request,
                        code="opaque_state_dropped",
                        message="function-call thought signature has no cross-protocol representation; dropped (same-protocol replay keeps it)",
                        field="tool_call.signature",
                        target_protocol=target_protocol,
                    )
    if target_protocol != "anthropic_messages":
        # Block-level cache hints are Anthropic provider policy: drops at
        # foreign targets are disclosed (same-protocol replay keeps them
        # verbatim via raw). Source-agnostic — any source may carry hints,
        # on message blocks, system blocks, or tool definitions.
        hinted = False
        for block in request.system or []:
            if (isinstance(block.extra, dict) and "cache_control" in block.extra) or (
                isinstance(block.raw, dict) and "cache_control" in block.raw
            ):
                hinted = True
                break
        if not hinted:
            for tool in request.tools or []:
                if isinstance(tool.extra, dict) and "cache_control" in tool.extra:
                    hinted = True
                    break
                raw_definition = getattr(tool, "raw", None)
                if isinstance(raw_definition, dict) and "cache_control" in raw_definition:
                    hinted = True
                    break
        if not hinted:
            for message in request.messages:
                for block in message.content:
                    if isinstance(block.extra, dict) and "cache_control" in block.extra:
                        hinted = True
                        break
                    if isinstance(block.raw, dict) and "cache_control" in block.raw:
                        hinted = True
                        break
                if hinted:
                    break
        if hinted:
            add_conversion_warning(
                request,
                code="unsupported_optional_control",
                message="cache_control hints have no cross-protocol representation; dropped (same-protocol replay keeps them)",
                field="cache_control",
                target_protocol=target_protocol,
            )
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
    # Tool types each destination understands (custom tools are a Chat-native
    # variant; server/hosted tools are Anthropic-native — their identity is
    # the versioned type and no cross-protocol mapping exists).
    if target_protocol == "openai_chat":
        supported_tool_types = {"function", "custom"}
    elif target_protocol == "anthropic_messages":
        supported_tool_types = {"function", "server"}
    elif target_protocol == "responses":
        # Responses' native ToolParam union: hosted tools are first-class
        # (namespace/apply_patch/tool_search/shell/web_search_preview are
        # current union members; preview spellings stay accepted).
        supported_tool_types = {
            "function",
            "custom",
            "web_search",
            "web_search_preview",
            "file_search",
            "code_interpreter",
            "image_generation",
            "computer_use_preview",
            "mcp",
            "local_shell",
            "shell",
            "namespace",
            "apply_patch",
            "tool_search",
        }
    elif target_protocol == "gemini":
        # Gemini hosts googleSearch/codeExecution/urlContext natively and
        # maps web_search server tools onto googleSearch; other hosted
        # families (bash, text_editor, computer) have no Gemini home.
        supported_tool_types = {"function", "server", "web_search"}
        for tool in request.tools:
            if tool.type == "server":
                server_type = str(tool.extra.get("server_tool_type") or tool.extra.get("gemini_hosted_tool") or "")
                if not server_type.startswith(
                    (
                        "web_search",
                        "googleSearch",
                        "google_search",
                        "codeExecution",
                        "code_execution",
                        "urlContext",
                        "url_context",
                        "googleMaps",
                    )
                ):
                    raise ProtocolError(
                        f"Cannot safely translate hosted tool '{server_type or tool.name}' into {target_protocol}",
                        protocol=target_protocol,
                        pass_name="validate_request",
                        payload={"tool_type": tool.type, "tool_name": tool.name},
                    )
            elif tool.type == "web_search":
                # Responses hosted web_search maps onto googleSearch too.
                continue
    else:
        supported_tool_types = {"function"}
    for tool_index, tool in enumerate(request.tools):
        if tool.type not in supported_tool_types:
            raise ProtocolError(
                f"Cannot safely translate tool type '{tool.type}' into {target_protocol}",
                protocol=target_protocol,
                pass_name="validate_request",
                payload={"tool_index": tool_index, "tool_type": tool.type, "tool_name": tool.name},
            )
        if tool.type in {"function", "custom"} and (not tool.name or not isinstance(tool.input_schema, dict)):
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
    # Pairing check for protocols that REQUIRE every tool call to be
    # answered before the conversation continues (Anthropic 400s orphans).
    # Calls on the FINAL message may legitimately await their results.
    if target_protocol == "anthropic_messages":
        answered: set[str] = set()
        for message in request.messages:
            for result in message_tool_results(message):
                if result.tool_call_id:
                    answered.add(result.tool_call_id)
        last_message_index = len(request.messages) - 1
        for message_index, message in enumerate(request.messages):
            if message_index == last_message_index:
                continue
            for call in message_tool_calls(message):
                if call.id and call.id not in answered:
                    raise ProtocolError(
                        "Tool calls require a following tool result before the conversation continues",
                        protocol=target_protocol,
                        pass_name="validate_request",
                        payload={"message_index": message_index, "call_id": call.id},
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
