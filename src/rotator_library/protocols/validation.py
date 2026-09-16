# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Destination capability validation for generative protocol conversion."""

from __future__ import annotations

from typing import Any, Mapping, Optional

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
    # G14: openai-compatible is a collective standard, not OpenAI's alone —
    # video input parts exist across the compatible ecosystem (Qwen-VL,
    # vLLM, Gemini's own compat surface), so chat carries video INPUT.
    "openai_chat": {"text", "image", "audio", "video", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    # Refusal turns degrade to text with refusal stop semantics (D7
    # equivalent construct) rather than blocking the whole request.
    "anthropic_messages": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    "responses": {"text", "image", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    "gemini": {"text", "image", "audio", "video", "file", "document", "reasoning", "tool_call", "tool_result", "refusal"},
    # Ollama vision models take base64 images on the message; there is no
    # audio/video/file content part on the native wire (disclose-drop).
    "ollama": {"text", "image", "reasoning", "tool_call", "tool_result"},
}

_RESPONSE_MODALITIES: dict[str, set[str]] = {
    "openai_chat": {"text", "audio"},
    "anthropic_messages": {"text"},
    # Officially text-only by default; image output exists via the
    # image_generation hosted tool's output items. Audio output is NOT a
    # Responses capability (the official audio guide routes audio through
    # Chat Completions) — the prior {text, audio} claim over-reached.
    "responses": {"text", "image"},
    "gemini": {"text", "audio", "image"},
    "ollama": {"text"},
}

# G8 hosted-tool identity on the Gemini wire: folded identity -> native
# single-key envelope. Mirrors the gemini formatter's mapping (retrieval is
# ordered BEFORE search so the prefix never folds it). A row's declared
# ``hosted_tools`` limits which of these a model actually serves.
_GEMINI_HOSTED_ENVELOPES: tuple[tuple[str, str], ...] = (
    ("websearch", "googleSearch"),
    ("enterprisewebsearch", "enterpriseWebSearch"),
    ("exaaisearch", "exaAiSearch"),
    ("parallelaisearch", "parallelAiSearch"),
    ("googlesearchretrieval", "googleSearchRetrieval"),
    ("googlesearch", "googleSearch"),
    ("codeexecution", "codeExecution"),
    ("urlcontext", "urlContext"),
    ("googlemaps", "googleMaps"),
    ("filesearch", "fileSearch"),
    ("computeruse", "computerUse"),
    ("mcpservers", "mcpServers"),
)


def _gemini_hosted_envelope(server_type: str) -> Optional[str]:
    """Map a hosted-tool identity to its Gemini native envelope name."""

    folded = str(server_type or "").lower().replace("_", "")
    for prefix, native in _GEMINI_HOSTED_ENVELOPES:
        if folded.startswith(prefix):
            return native
    return None


def _declared_modalities(capabilities: Optional[Mapping[str, Any]], wire_protocol: str) -> set[str]:
    """The accepted response modalities for a target (G8).

    Undeclared keeps the protocol table exactly; a declared
    ``output_modalities`` narrows to the {"text"} base and widens it with the
    declared list (text parts exist on every Gemini surface — the base is the
    always-representable floor).
    """

    allowed = set(_RESPONSE_MODALITIES.get(wire_protocol, {"text"}))
    if isinstance(capabilities, Mapping):
        declared = capabilities.get("output_modalities")
        if declared is not None:
            if isinstance(declared, str):
                # A single-modality declaration is legal shorthand, never an
                # iterable of characters.
                declared = [declared]
            allowed = {"text"} | {str(value).strip().lower() for value in declared}
    return allowed


def validate_generative_request(
    request: UnifiedRequest,
    target_protocol: str,
    context: ProtocolContext | None,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> None:
    """Reject meaning-changing cross-protocol losses before provider transport.

    Same-protocol requests may retain future native content through their raw
    payload. Cross-protocol requests must use a known canonical meaning so a
    source-native object can never be emitted as a malformed foreign object.

    ``capabilities`` is the resolved per-model capability record (G8 gemini
    split) threaded by the execution seam: it narrows/widens the Gemini
    response-modality table via ``output_modalities`` and limits the hosted
    tool allowlist via ``hosted_tools``. ``None``/absent keys = undeclared =
    exactly the previous behavior.
    """

    # G11: wire-level dispatch is FAMILY-level — sibling variants of one
    # format (responses stateless/stateful/websocket) share tables and
    # same-wire semantics; only the registry name differs.
    from .canonical import family_wire_name

    wire_protocol = family_wire_name(target_protocol)

    if is_same_protocol(context, target_protocol, request.source_protocol):
        return
    # Cross-protocol extension fields have no foreign representation by
    # definition (each target replays only its own extensions verbatim):
    # every extra key dropping at a foreign target is disclosed (D7).
    # Exception: keys the TARGET's own build warns about richer (Gemini
    # bound envelope fields) — one disclosure per fact, never two.
    gemini_bound_keys = {"cachedContent", "labels", "serviceTier", "store"}
    for extra_key in sorted(request.extra):
        if wire_protocol == "gemini" and extra_key in gemini_bound_keys:
            continue
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message=f"{extra_key} has no {target_protocol} representation; dropped cross-protocol",
            field=extra_key,
            target_protocol=target_protocol,
        )
    # Metadata dictionaries ride only where a target has a metadata field.
    if request.metadata and wire_protocol == "gemini":
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
    if request.source_protocol == "gemini" and wire_protocol != "gemini":
        source_generation = request.extensions.get("gemini", {}).get("generationConfig")
        if isinstance(source_generation, dict):
            mapped = {
                "maxOutputTokens", "stopSequences", "topP", "topK", "temperature",
                "candidateCount", "seed", "frequencyPenalty", "presencePenalty",
                "responseMimeType", "responseSchema", "responseJsonSchema",
                "thinkingConfig", "thinkingLevel", "responseModalities",
                # The text sub-config of responseFormat maps onto the canonical
                # structured-output control; only its audio/image siblings have
                # no representation (disclosed below, one warning per fact —
                # never the whole-envelope drop warning this key used to raise).
                "responseFormat",
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
            response_format_cfg = source_generation.get("responseFormat")
            if isinstance(response_format_cfg, dict):
                for sibling in ("audio", "image"):
                    if isinstance(response_format_cfg.get(sibling), dict):
                        add_conversion_warning(
                            request,
                            code="unsupported_optional_control",
                            message=f"generationConfig.responseFormat.{sibling} has no cross-protocol representation; dropped",
                            field=f"generationConfig.responseFormat.{sibling}",
                            target_protocol=target_protocol,
                        )
    # Opaque function-call signatures (Gemini thought signatures) dropping
    # at foreign boundaries are disclosed — never silent (Gemini 3 rejects
    # unsigned current-turn calls on the way back).
    if request.source_protocol == "gemini" and wire_protocol != "gemini":
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
    if wire_protocol != "anthropic_messages":
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
    if request.previous_response_id and wire_protocol != "responses":
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
    if provider_bound_responses_fields and wire_protocol != "responses":
        raise ProtocolError(
            "Provider-bound Responses controls cannot be translated safely",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"fields": provider_bound_responses_fields},
        )
    if request.generation_params.get("safety_settings") and wire_protocol != "gemini":
        raise ProtocolError(
            "Gemini safety settings have no equivalent in the selected provider protocol",
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": "safety_settings"},
        )
    # G14 (#28.8): an output modality the target cannot produce is a
    # disclosed downgrade, not a request-killing 400 — the request still
    # goes, producing what the target CAN produce. G8: a declared
    # output_modalities row narrows the protocol default to {"text"} and
    # widens it with the model's own list.
    unsupported_modalities = set(request.modalities) - _declared_modalities(capabilities, wire_protocol)
    if unsupported_modalities:
        request.modalities = [m for m in request.modalities if m not in unsupported_modalities]
        add_conversion_warning(
            request,
            code="unsupported_output_modality",
            message=f"{target_protocol} cannot produce requested response modalities {sorted(unsupported_modalities)}; downgraded to {request.modalities or ['text']}",
            field="modalities",
            target_protocol=target_protocol,
        )
    supported = _CONTENT_CAPABILITIES.get(wire_protocol, set())
    block_groups = [("system", request.system)] + [
        (f"message:{message_index}", message.content)
        for message_index, message in enumerate(request.messages)
    ]
    for group_name, blocks in block_groups:
        dropped: list[int] = []
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
            # G14 (#28.8): unrepresentable content is a warning-logged drop,
            # never a hard reject — the rest of the request survives.
            dropped.append(block_index)
            add_conversion_warning(
                request,
                code="unsupported_content_dropped",
                message=f"content block of type '{block.type}' has no {target_protocol} representation; dropped",
                field=f"{group_name}.content[{block_index}]",
                target_protocol=target_protocol,
            )
        if dropped:
            surviving = [block for index, block in enumerate(blocks) if index not in dropped]
            # A dialect-legal placeholder keeps emptied messages/anchors
            # intact when every block was unrepresentable.
            blocks[:] = surviving or [blocks[0].__class__(type="text", text="")]
    choice = request.generation_params.get("tool_choice") if isinstance(request.generation_params, dict) else None
    if (
        isinstance(choice, dict)
        and choice.get("mode") == "validated"
        and wire_protocol != "gemini"
    ):
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message="Gemini VALIDATED tool-choice mode (schema-adherence enforcement) has no cross-protocol representation; approximated as auto",
            field="tool_choice",
            target_protocol=target_protocol,
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
    elif wire_protocol == "gemini":
        # Gemini hosts googleSearch/codeExecution/urlContext natively and
        # maps web_search server tools onto googleSearch; other hosted
        # families (bash, text_editor, computer) have no Gemini home.
        supported_tool_types = {"function", "server", "web_search"}
        declared_hosted = capabilities.get("hosted_tools") if isinstance(capabilities, Mapping) else None
        if isinstance(declared_hosted, str):
            # A single-tool declaration is legal shorthand, never an
            # iterable of characters.
            declared_hosted = [declared_hosted]
        declared_envelopes = (
            {_gemini_hosted_envelope(name) or str(name) for name in declared_hosted}
            if declared_hosted is not None
            else None
        )
        dropped_hosted: list[int] = []
        for tool_index, tool in enumerate(request.tools):
            if tool.type == "web_search":
                # A foreign Responses hosted search maps onto googleSearch;
                # it participates in the declared-hosted limit exactly like
                # a server-typed tool does.
                envelope = "googleSearch"
            elif tool.type == "server":
                server_type = str(tool.extra.get("server_tool_type") or tool.extra.get("gemini_hosted_tool") or "")
                envelope = _gemini_hosted_envelope(server_type)
                if envelope is None:
                    raise ProtocolError(
                        f"Cannot safely translate hosted tool '{server_type or tool.name}' into {target_protocol}",
                        protocol=target_protocol,
                        pass_name="validate_request",
                        payload={"tool_type": tool.type, "tool_name": tool.name},
                    )
            else:
                continue
            if declared_envelopes is not None and envelope not in declared_envelopes:
                # G8: the full union stays representable, but the model's row
                # does not declare this tool — dropping it (disclosed) is
                # honest; passing it silently would be a guaranteed 400.
                dropped_hosted.append(tool_index)
                add_conversion_warning(
                    request,
                    code="unsupported_optional_control",
                    message=f"hosted tool {envelope!r} is not declared available on this model; dropped",
                    field=f"tools[{tool_index}]",
                    target_protocol=target_protocol,
                )
        if dropped_hosted:
            dropped_set = set(dropped_hosted)
            request.tools = [tool for index, tool in enumerate(request.tools) if index not in dropped_set]
    else:
        supported_tool_types = {"function"}
    for tool_index, tool in enumerate(request.tools):
        if request.source_protocol == "gemini" and wire_protocol != "gemini":
            hosted_kind = tool.extra.get("gemini_hosted_tool") or tool.extra.get("gemini_unmodeled_tool")
            if hosted_kind:
                # Gemini hosted tools (googleSearch/computerUse/mcpServers/...)
                # are provider-executed envelopes with no foreign dialect: an
                # honest, named rejection beats a fabricated client-callable
                # function (or a silently empty hosted envelope).
                raise ProtocolError(
                    f"Cannot safely translate hosted Gemini tool '{hosted_kind}' into {target_protocol}",
                    protocol=target_protocol,
                    pass_name="validate_request",
                    payload={"tool_index": tool_index, "tool_type": tool.type, "tool_name": hosted_kind},
                )
        if tool.type not in supported_tool_types:
            raise ProtocolError(
                f"Cannot safely translate tool '{tool.name or tool.type}' (type '{tool.type}') into {target_protocol}",
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
            if not call.name or (wire_protocol != "gemini" and not call.id):
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
            if wire_protocol == "gemini" and not result.name:
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


# Canonical embeddings controls shared by the embeddings protocols. Every
# name here has a first-class meaning on at least one wire; per-target
# mapping/rejection happens in the adapters, never here.
_EMBEDDINGS_DIMENSION_KEYS = ("dimensions", "output_dimensionality")
_EMBEDDINGS_ENCODING_FORMATS = {"float", "base64"}


def validate_embeddings_request(
    request: UnifiedRequest,
    target_protocol: str,
    context: ProtocolContext | None = None,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> None:
    """Reject malformed embeddings requests before provider transport (G9).

    Mirrors ``validate_generative_request``'s error contract: every failure
    is a ``ProtocolError`` whose message names the field, so the route's
    grounded ladder renders the client protocol's own 400 envelope — and
    because this runs before any credential attempt (client contract at
    builder time, destination contract in ``build_request``), rotation
    never burns keys on a request no credential can fix.

    Shape checks only (wire-neutral):
    - ``model`` present;
    - ``input`` present and non-empty — a non-empty string, a non-empty
      array of strings, or token arrays whose items are well-formed
      non-negative integers;
    - ``dimensions`` (or its gemini spelling ``output_dimensionality``)
      is a positive integer when present;
    - ``encoding_format`` is ``float`` or ``base64`` when present.

    Destination-specific losses (gemini's text-only content, the missing
    base64 representation, per-item control uniformity) are NOT checked
    here; adapters own those and disclose or reject them at build time.
    """

    def _fail(message: str, field: str, **payload: Any) -> None:
        raise ProtocolError(
            message,
            protocol=target_protocol,
            pass_name="validate_request",
            payload={"field": field, **payload},
        )

    if not str(getattr(request, "model", "") or "").strip():
        _fail("Embeddings requests require a model", "model")

    value = getattr(request, "input", None)
    if isinstance(value, str):
        if not value.strip():
            _fail("Embeddings input must be a non-empty string", "input")
    elif isinstance(value, list):
        if not value:
            _fail("Embeddings input array must not be empty", "input")
        if all(isinstance(item, str) for item in value):
            for index, item in enumerate(value):
                if not item.strip():
                    _fail("Embeddings input items must be non-empty strings", "input", index=index)
        else:
            # Token input(s): a flat array of non-negative ints (ONE
            # tokenized input) or an array of token arrays. Strings never
            # mix with tokens on any wire.
            has_flat_tokens = any(isinstance(item, int) and not isinstance(item, bool) for item in value)
            has_token_arrays = any(isinstance(item, list) for item in value)
            if has_flat_tokens and has_token_arrays:
                _fail("Embeddings token inputs must not mix tokens and token arrays", "input")
            for index, item in enumerate(value):
                if isinstance(item, list):
                    if not item:
                        _fail("Embeddings token arrays must not be empty", "input", index=index)
                    for token_index, token in enumerate(item):
                        # bool is an int subclass — True must never pass as token 1.
                        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                            _fail(
                                "Embeddings token arrays must contain non-negative integers",
                                "input",
                                index=index,
                                token_index=token_index,
                            )
                elif isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    _fail(
                        "Embeddings input must be a string, an array of strings, or token arrays",
                        "input",
                        index=index,
                    )
    else:
        _fail("Embeddings requests require an input (string or array)", "input")

    params = getattr(request, "generation_params", None)
    params = params if isinstance(params, dict) else {}
    for key in _EMBEDDINGS_DIMENSION_KEYS:
        dimensions = params.get(key)
        if dimensions is None:
            continue
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            _fail("Embeddings dimensions must be a positive integer", key)
        # One control, one home: a second spelling alongside it would be
        # sent twice with contradictory values on a rebuild.
        for other in _EMBEDDINGS_DIMENSION_KEYS:
            if other != key and params.get(other) is not None:
                _fail("Embeddings dimensions are declared twice (dimensions/output_dimensionality)", key)
        break
    encoding_format = params.get("encoding_format")
    if encoding_format is not None and str(encoding_format).strip().lower() not in _EMBEDDINGS_ENCODING_FORMATS:
        _fail(
            "Embeddings encoding_format must be 'float' or 'base64'",
            "encoding_format",
        )


def validate_generative_response(response: UnifiedResponse, target_protocol: str) -> None:
    """Reject failed provider responses that a target success envelope cannot express."""

    from .canonical import family_wire_name

    wire_protocol = family_wire_name(target_protocol)
    if response.stop_reason == STOP_REASON_ERROR and wire_protocol != "responses":
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
