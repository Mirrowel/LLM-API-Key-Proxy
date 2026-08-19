# Phase 5c: Provider-Native Output Protocol And Contract Correction

## Goal

Close the Phase 5/5b third-pass provider-native findings while preserving the current safety stance: native streaming remains disabled for priority providers until each live stream path is proven, and LiteLLM remains fallback only for unsupported cases.

## Scope

- Return native provider responses in the originating client protocol instead of the provider protocol.
- Add explicit native endpoint/header/operation-selection hooks to `ProviderInterface`.
- Ensure Claude Code native requests always include required Anthropic `max_tokens`.
- Harden Claude Code API-key header behavior without breaking bearer/OAuth-style credentials.
- Fix Antigravity alias normalization so duplicated aliases do not collapse incorrectly and low/high thinking aliases can still affect request metadata.
- Centralize explicit-native streaming fail-closed behavior.
- Keep priority-provider native streaming disabled by default.
- Add focused tests for all blocker/high/medium findings.

## Non-Goals

- Do not enable live native streaming for Claude Code, Codex, Copilot, or Antigravity.
- Do not restore retired Antigravity device-profile/fingerprint behavior.
- Do not replace credential rotation, usage tracking, or routing.
- Do not implement all rich Codex Responses item conversions in this phase.
- Do not commit user-facing reports.

## Implementation Plan

1. Client target protocol in native context.
   - Add `client_protocol_name` to `NativeProviderContext`.
   - Default direct/library native use to provider protocol for backwards compatibility.
   - Set `client_protocol_name="openai_chat"` from the chat-completions executor path.
   - Format non-streaming native responses using the client protocol after provider protocol parsing.
   - Format native stream events using the client protocol when a client protocol is set.

2. Provider interface native contract.
   - Add default `get_native_endpoint()` and `get_native_headers()` methods that raise clear `NotImplementedError`.
   - Add `supports_native_operation()` defaulting to operation support through provider declarations.
   - Add `should_use_native_protocol()` defaulting to true only when a provider declares a native protocol and operation is supported.
   - Update executor checks to call hooks rather than relying on `hasattr()`.

3. Claude Code hardening.
   - `prepare_native_request()` ensures Anthropic `max_tokens` is present, using existing request value or `CLAUDE_CODE_MAX_TOKENS` default.
   - Header selection supports bearer credentials and API-key credentials:
     - `CLAUDE_CODE_AUTH_HEADER=bearer|x-api-key|auto`
     - auto uses `x-api-key` for `sk-ant-*` style keys and bearer otherwise.
   - Preserve `anthropic-version` and content type.

4. Antigravity alias/thinking metadata.
   - Fix alias-to-upstream mapping to use public alias map directly, not a lossy reverse map.
   - Preserve thinking-level hints for `gemini-3-pro-low` / `gemini-3-pro-high` in request metadata before the upstream model is normalized.
   - Keep model discovery output stable and prefixed.

5. Native streaming fail-closed helper.
   - Centralize explicit-native streaming unsupported handling so auto and explicit modes use the same fail-closed decision.
   - Keep priority providers non-streaming native only.

6. Tests.
   - Native executor: provider protocol response -> OpenAI chat client response.
   - Request executor: routed native Claude/Codex/Antigravity chat completions return OpenAI Chat shape.
   - Claude Code: missing `max_tokens` gets defaulted and auth header modes behave correctly.
   - Antigravity: low/high aliases normalize safely and preserve request metadata.
   - ProviderInterface: native hooks exist and unsupported operation checks are explicit.
   - Streaming: explicit native streaming fail-closed path uses centralized helper.

## Acceptance Criteria

- `/v1/chat/completions` native routes return OpenAI Chat response shape regardless of provider-native protocol.
- Claude Code native Messages requests include `max_tokens`.
- Antigravity model normalization does not lose low/high alias intent.
- Provider native endpoint/header/operation hooks are explicit on `ProviderInterface`.
- Explicit native streaming fails closed through one helper path.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
