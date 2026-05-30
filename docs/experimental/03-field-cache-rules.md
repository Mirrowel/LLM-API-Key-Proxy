# Field Cache Rules

Field caching is required for providers that need values from previous responses or stream events to be returned on later requests. Examples include reasoning content, thought signatures, prompt cache keys, provider session IDs, and response IDs.

## Goals

- Let custom providers configure what fields to extract and where to inject them.
- Avoid hardcoding every provider-specific memory behavior.
- Preserve strict scoping so values never leak across provider, model, credential, classifier scope, or session.
- Support both non-streaming responses and streaming events.
- Support rules per provider and per model.

## Rule Shape

Illustrative JSON shape:

```json
{
  "name": "reasoning_content",
  "source": "response",
  "path": "choices.*.message.reasoning_content",
  "scope": "session",
  "mode": "last",
  "inject": {
    "target": "request",
    "path": "messages[-1].reasoning_content"
  }
}
```

This is a design sketch, not the final schema.

## Sources

- `request`
- `response`
- `stream_event`
- `unified_request`
- `unified_response`
- `unified_stream_event`

## Targets

- raw provider request path.
- unified request field.
- protocol metadata.
- provider-specific extension field.

## Scopes

- `provider`
- `model`
- `credential`
- `session`
- `conversation`
- combinations of the above when needed.

The default for conversation-affecting fields should be at least provider+model+session scoped.

## Modes

- `last`: only the latest matching value.
- `all`: all matching values within the scope.
- `last_user_turn`: latest value associated with the last user turn.
- `last_assistant_turn`: latest value associated with the last assistant turn.
- `per_tool_call`: keyed by tool call ID.

## Backing Store

Use the existing provider cache infrastructure first. Do not require SQLite.

Potential cache keys should include:

```text
provider / model / credential-or-scope / session-id / rule-name
```

Private/classifier scoped credentials must not share cached fields with global credentials.

## Examples

### DeepSeek Reasoning Content

Extract reasoning content from assistant responses and inject it into the next provider request when the provider expects continuity.

### Gemini Thought Signatures

Extract thought signatures from Gemini response parts and return them with matching future content parts.

### Responses Previous Response ID

Store response IDs and output items so `previous_response_id` can load prior context.

### Prompt Cache Keys

Carry `prompt_cache_key` or equivalent provider cache routing values forward when a provider benefits from stable cache routing.

## Tests

Required test categories:

- extraction from response.
- extraction from stream event.
- injection into next request.
- `last` versus `all` behavior.
- scope isolation.
- missing path is a no-op.
- malformed path produces useful validation error.
- redaction in transform logs.

## Key Decision

Field cache rules are a protocol/provider extension system, not a replacement for `SessionTracker`. Session tracking decides continuity and credential affinity; field cache rules preserve provider-specific protocol state.
