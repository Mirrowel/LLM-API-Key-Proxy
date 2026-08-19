# Native Protocol Architecture

Protocols are reusable bases, not rigid gospel. Providers can subclass, wrap, copy, or override protocol behavior when a provider deviates from an otherwise standard protocol.

## Why Protocols First

The current code relies heavily on LiteLLM and provider-specific transforms. That works, but it makes new protocols hard to reason about and makes debugging transformations difficult. The experimental goal is to make a provider mostly declarative:

```text
provider = protocol + auth + adapters + field cache rules + model options + quota behavior
```

If a provider needs custom behavior, it should override a narrow protocol method instead of forcing an entirely bespoke request path.

## Auto-Discovery

Protocols should follow the provider plugin style:

- protocol modules live under `src/rotator_library/protocols/`.
- modules register concrete protocol classes by name.
- a registry exposes names such as `openai_chat`, `anthropic_messages`, `gemini`, `responses`, and `litellm_fallback`.
- third-party or local protocol modules can be added later with minimal registry changes.

## Core Types

The unified representation should be explicit enough to cover all existing providers and the external reference protocols without losing important data.

Suggested types:

- `UnifiedRequest`
- `UnifiedResponse`
- `UnifiedStreamEvent`
- `UnifiedMessage`
- `ContentBlock`
- `ToolDefinition`
- `ToolCall`
- `ToolResult`
- `ReasoningBlock`
- `Usage`
- `CostDetails`
- `ProtocolMetadata`

These types should retain unknown provider-specific metadata in explicit extension dictionaries instead of dropping it. Robustness matters more than a narrow perfect schema.

## Protocol Interface

The base protocol should provide default methods that can be overridden:

- `parse_request(raw_request, context) -> UnifiedRequest`
- `build_request(unified_request, context) -> raw_provider_request`
- `parse_response(raw_response, context) -> UnifiedResponse`
- `format_response(unified_response, context) -> raw_client_response`
- `parse_stream_event(raw_event, context) -> UnifiedStreamEvent`
- `format_stream_event(unified_event, context) -> raw_stream_payload`
- `extract_usage(raw_or_unified, context) -> Usage | None`
- `supports_transport(transport_name) -> bool`

Provider-specific overrides should receive context that includes provider name, model, credential identity, source protocol, target protocol, request ID, and session tracking information.

## Initial Protocols

### OpenAI Chat

Must support:

- chat completions request/response.
- stream chunks.
- tools and tool calls.
- function-call legacy shapes.
- reasoning fields from OpenAI-compatible providers.
- cached token and reasoning token usage details.

### Anthropic Messages

Must support:

- messages request/response.
- system content extraction.
- text, image, thinking, redacted thinking, tool_use, tool_result blocks.
- stream lifecycle events.
- count_tokens path later if needed.

### Gemini

Must support:

- generateContent and streamGenerateContent shapes.
- content parts.
- functionCall/functionResponse.
- thought signatures.
- safety settings passthrough without unsafe auto-injection.
- Google/Gemini usage metadata.

### Responses

Must support:

- OpenAI Responses request/response.
- `previous_response_id`.
- output items.
- event streams.
- storage-friendly response objects.
- future WebSocket transport.

### LiteLLM Fallback

Must preserve existing behavior for providers/protocols not yet native. This path should be explicit and transaction-logged as a fallback, not hidden.

## Transport Separation

Protocol formatting must not be tied only to HTTP SSE. Define a transport boundary so the same unified stream events can be emitted through:

- non-streaming HTTP JSON.
- HTTP SSE.
- future WebSocket.

The Responses phase should leave clear extension points for WebSocket even if WebSocket is implemented later.

## Error Handling

Protocols should preserve provider error bodies where safe, but format client-facing errors consistently. Parsing errors should include transform-pass names and request IDs to make transaction logs useful.

## Docstrings And Comments

Protocol code should include docstrings explaining:

- which external API shape it models.
- what fields are intentionally preserved in metadata.
- where provider overrides are expected.
- future expansion hooks.

Comments should explain non-obvious transformations, especially lossy conversions between protocols.
