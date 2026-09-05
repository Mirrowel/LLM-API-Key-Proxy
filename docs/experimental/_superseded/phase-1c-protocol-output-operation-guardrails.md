# Phase 1c: Protocol Output Correctness And Native Operation Guardrails

## Goal

Close the Phase 1/1b third-pass findings that still affect protocol correctness and later native provider behavior.

## Scope

- Fix OpenAI Chat formatted usage so it emits public OpenAI-compatible usage fields rather than unified internal usage fields.
- Fix Responses formatted usage so it emits Responses-compatible usage fields rather than unified internal usage fields.
- Promote OpenAI legacy `function_call` into unified `ToolCall` while preserving legacy round-trip shape.
- Add Ollama response formatting that respects mutated unified responses instead of stale raw payloads.
- Enforce protocol operation compatibility in native execution before transport calls.

## Non-Goals

- Do not solve every Phase 5 provider issue here beyond the protocol guardrail needed by native execution.
- Do not enable native streaming for priority providers.
- Do not rewrite protocol registry discovery unless a focused test exposes a direct failure.
- Do not touch unrelated dirty files or user-facing reports.

## Implementation Plan

1. OpenAI Chat usage formatting.
   - Add a helper that converts `Usage` to `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details`, `completion_tokens_details`, and `cost_details`.
   - Ensure `input_tokens`, `output_tokens`, `raw`, and `extra` do not leak into formatted OpenAI usage.

2. Responses usage formatting.
   - Add a helper that converts `Usage` to `input_tokens`, `output_tokens`, `total_tokens`, `input_tokens_details`, `output_tokens_details`, and `cost_details`.
   - Preserve cache-write information only as a safe extension inside `input_tokens_details` when present.

3. Legacy OpenAI `function_call`.
   - Parse legacy `function_call` into a unified `ToolCall` when modern `tool_calls` are not present.
   - Preserve the raw legacy field in `extra` so formatting emits `function_call` instead of incorrectly upgrading to `tool_calls`.

4. Ollama response formatting.
   - Implement `OllamaProtocol.format_response()` for chat, generate, and embeddings shapes.
   - Merge non-core `extra` and usage/timing fields while honoring mutated unified messages/output/data.

5. Native operation enforcement.
   - In `NativeProviderExecutor.execute()` and `stream()`, reject unsupported protocol operations before parse/build/transport.
   - Keep errors sanitized and explicit so bad provider/protocol pairings fail early.

## Tests

- `tests/test_protocol_openai_chat.py`
- `tests/test_protocol_responses.py`
- `tests/test_protocol_ollama_mcp.py`
- `tests/test_native_provider_executor.py` or `tests/test_request_executor_native_routing.py`
- Full protocol suite plus native routing smoke tests before review.

## Acceptance Criteria

- OpenAI Chat formatted responses expose OpenAI-compatible usage fields.
- Responses formatted responses expose Responses-compatible usage fields.
- Legacy OpenAI `function_call` is represented in unified tool calls and round-trips as legacy `function_call`.
- Ollama formatted responses reflect mutated unified state.
- Native execution rejects unsupported protocol operations before network transport.
- Focused tests pass and both `explore` and `explore-heavy` reviewers report no blockers/highs/mediums.
