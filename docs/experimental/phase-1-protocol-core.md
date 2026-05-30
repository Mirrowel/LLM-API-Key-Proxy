# Phase 1: Protocol Core

## Goal

Introduce the native protocol foundation without changing live request execution yet. This phase creates robust, documented primitives that later phases can wire into `RequestExecutor`, provider declarations, adapters, field-cache rules, Responses, and streaming transports.

## Non-Goals

- Do not replace LiteLLM execution yet.
- Do not add `/v1/responses` routes yet.
- Do not migrate providers to native protocols yet.
- Do not implement field-cache persistence yet.
- Do not rewrite current Anthropic compatibility routes yet.
- Do not change existing request behavior unless a test-only import path requires a harmless export.

## Current Code Context

- No `src/rotator_library/protocols/` package exists yet.
- Provider auto-discovery in `src/rotator_library/providers/__init__.py` is the model for protocol discovery.
- `RequestContext` currently holds execution/session/logging fields and can later receive protocol metadata, but Phase 1 should avoid mutating it unless necessary.
- `ProviderTransforms` is hardcoded and will remain active until later adapter migration.
- `TransactionLogger` currently logs initial request, transformed request, response, and stream chunks; Phase 2 will add transform-pass tracing, but Phase 1 types should be trace-friendly.
- `anthropic_compat` already has useful conversion knowledge, especially thinking/tool block handling, but Phase 1 should not delete or replace it.

## Files To Add

- `src/rotator_library/protocols/__init__.py`
- `src/rotator_library/protocols/types.py`
- `src/rotator_library/protocols/base.py`
- `src/rotator_library/protocols/registry.py`
- `src/rotator_library/protocols/openai_chat.py`
- `src/rotator_library/protocols/anthropic_messages.py`
- `src/rotator_library/protocols/gemini.py`
- `src/rotator_library/protocols/responses.py`
- `src/rotator_library/protocols/litellm_fallback.py`
- `tests/test_protocol_registry.py`
- `tests/test_protocol_openai_chat.py`
- `tests/test_protocol_anthropic_messages.py`
- `tests/test_protocol_gemini.py`
- `tests/test_protocol_responses.py`

## Possible Files To Touch

- `src/rotator_library/__init__.py` only if a public lazy export is needed.
- `src/rotator_library/core/__init__.py` only if shared exports are cleaner there.
- Avoid modifying `RequestExecutor` in Phase 1 unless tests reveal a strict import issue.

## Data Model Plan

- `ProtocolRole`: role names should remain strings in payloads, but internal dataclasses can use simple `str` fields to avoid over-constraining custom protocols.
- `ContentBlock`: typed block with `type`, optional text/image/source/tool fields, and `extra` dict for provider-specific data.
- `UnifiedMessage`: `role`, `content`, `name`, `tool_call_id`, `tool_calls`, `extra`.
- `ToolDefinition`: protocol-neutral tool schema with `name`, `description`, `input_schema`, `extra`.
- `ToolCall`: `id`, `name`, `arguments`, `type`, `index`, `extra`.
- `ToolResult`: `tool_call_id`, `content`, `is_error`, `extra`.
- `ReasoningBlock`: `type`, `text`, `signature`, `redacted`, `extra`.
- `Usage`: input/output/total tokens, cache read/write, reasoning tokens, raw usage, cost details.
- `CostDetails`: provider reported cost, estimated cost, currency, source, metadata.
- `UnifiedRequest`: model, messages, tools, system, stream flag, generation params, response format, previous response ID, metadata, raw payload, extra.
- `UnifiedResponse`: id, model, messages/output, stop reason, usage, metadata, raw payload, extra.
- `UnifiedStreamEvent`: event type, delta/message/tool/usage/error metadata, raw event, extra.
- `ProtocolContext`: provider, model, source protocol, target protocol, request ID, session ID, credential stable ID, transport, transaction metadata, provider options.
- `ProtocolResult` is probably unnecessary in Phase 1; keep methods direct and simple.

## Protocol Interface Plan

- `ProtocolAdapter` abstract/base class with override-friendly methods.
- Required class attributes: `name`, `aliases`, `supported_transports`.
- Methods:
  - `parse_request(raw_request, context) -> UnifiedRequest`
  - `build_request(unified_request, context) -> dict`
  - `parse_response(raw_response, context) -> UnifiedResponse`
  - `format_response(unified_response, context) -> dict`
  - `parse_stream_event(raw_event, context) -> UnifiedStreamEvent`
  - `format_stream_event(unified_event, context) -> Any`
  - `extract_usage(raw_or_unified, context) -> Usage | None`
  - `supports_transport(transport_name) -> bool`
- Defaults should preserve unknown data in `extra` rather than raising.
- Errors should be `ProtocolError` with protocol name, pass name, and optional payload preview.

## Registry Plan

- `PROTOCOL_PLUGINS: dict[str, type[ProtocolAdapter]]`
- `register_protocol(cls)` for explicit class registration.
- `get_protocol(name)` returns an instance or class consistently.
- `list_protocols()`.
- Auto-discover modules under `src/rotator_library/protocols/`, skipping private modules and infrastructure modules.
- Support aliases so `openai`, `chat_completions`, and `openai_chat` can resolve to the same adapter.
- Prevent duplicate names unless the class is identical or explicit replacement is requested later.
- Keep registry import safe and lightweight.

## Initial Protocol Behavior

### OpenAI Chat

- Parse chat completions request messages, system/developer/user/assistant/tool roles, text content, multimodal content arrays, tools, tool calls, tool_choice, response_format, stream, temperature/top_p/max_tokens/stop.
- Parse responses with choices, assistant messages, tool_calls, reasoning/reasoning_content, and usage details.
- Parse stream chunks into `UnifiedStreamEvent` while preserving raw delta.
- Format back to OpenAI Chat without losing unknown fields in `extra`.

### Anthropic Messages

- Parse separate `system`, messages with content blocks, thinking/redacted_thinking, tool_use, tool_result, images, documents where currently supported.
- Map Anthropic usage to `Usage`.
- Preserve thinking signatures in `ReasoningBlock.extra`.
- Build/format enough for round-trip tests, not full replacement of `anthropic_compat` yet.

### Gemini

- Parse `contents`, roles, parts, text, inline data, file data, functionCall, functionResponse, thought/thoughtSignature where present.
- Parse `generationConfig`, `safetySettings`, `tools`, stream flag metadata.
- Map usage metadata prompt/candidates/thoughts/cache where possible.
- Preserve provider-specific safety and generation fields.

### Responses

- Parse `input`, `instructions`, `previous_response_id`, tools, metadata, stream.
- Parse `output` items, message content, reasoning items, function/tool calls.
- Parse common response stream events into unified stream events.
- Do not add storage or routes yet.

### LiteLLM Fallback

- Wrap existing OpenAI-compatible dicts into unified request/response with raw preservation.
- Exist mainly as a named explicit protocol path for later logging.

## Transaction Logging Implications

- Phase 1 does not integrate runtime transform logs yet.
- All dataclasses need `to_dict()`/`from_dict()` or safe serialization helpers so Phase 2 can log every pass cleanly.
- `ProtocolError` should include pass names that Phase 2 can reuse.
- Avoid mutating raw payloads in protocol parse methods unless explicitly building a new provider request.

## Docstrings And Comments Required

- Every public protocol class explains which external API shape it models and which parts are intentionally partial/base behavior.
- `ProtocolAdapter` docstring explains override contract and why protocols are bases.
- Registry comments explain auto-discovery and skip rules.
- Conversion helpers comment on lossy or approximate mappings.
- Future-extension comments note where WebSocket, field-cache rules, provider overrides, and target transports will attach.

## Tests

- Registry auto-discovers built-in protocols.
- Aliases resolve correctly.
- Duplicate registration behavior is deterministic.
- Base protocol default methods preserve raw payloads.
- OpenAI Chat request round-trip with system/developer/user/assistant/tool messages, tool definitions, tool calls, reasoning content, and usage with cache/reasoning token details.
- Anthropic Messages request round-trip with system field, text blocks, thinking/redacted_thinking blocks, tool_use/tool_result, and usage cache fields.
- Gemini request round-trip with contents and parts, functionCall/functionResponse, thoughtSignature, generationConfig/safetySettings, and usageMetadata.
- Responses request/response parse with instructions, input string and input message list, previous_response_id, output messages, reasoning items, and function/tool call items.
- Stream event parse smoke tests for each protocol where practical.
- Serialization tests ensure unified types are JSON-serializable.

## Commit Checkpoints

1. Add protocol dataclasses, errors, and serialization helpers.
2. Add base adapter and registry auto-discovery.
3. Add LiteLLM fallback and OpenAI Chat protocol with tests.
4. Add Anthropic Messages protocol with tests.
5. Add Gemini protocol with tests.
6. Add Responses protocol with tests.
7. Run focused protocol test set and fix issues.
8. Phase review with `explore` and `explore-heavy`, then fixes if needed.

## Risk And Rollback

- Keep Phase 1 isolated so rollback is just removing the new package/tests.
- Avoid touching executor behavior to prevent regressions.
- If auto-discovery causes import cycles, switch to explicit built-in imports inside registry while preserving the public registry API.
- If tests become too large for one file, split into `tests/protocols/` only if ignore rules are handled with force-add when committing.
- Since `.gitignore` ignores most `tests/*`, remember to force-add new test files when committing.
