# Phase 5b Plan: Priority Providers From Skeletons To Mock-Live Native Integrations

## Goal

Correct the Phase 5 audit gap. Phase 5 added native-provider seams and priority provider declarations, but the validation pass found the priority providers are still mostly skeletons. Phase 5b will make Claude Code, Codex, Copilot, and Antigravity usable through the native execution path under mock HTTP tests, while preserving Gemini CLI's existing custom execution path and adding parity declarations where they are safe.

## Non-Goals

- Do not remove LiteLLM fallback.
- Do not replace Gemini CLI's existing custom provider implementation with the generic native executor in this phase.
- Do not invent device fingerprinting or brittle environment/device-profile behavior for Antigravity.
- Do not add real credential acquisition flows for Copilot/Codex/Claude Code; this phase consumes credentials supplied by the existing credential system.
- Do not use external files outside the project root.
- Do not introduce SQLite or new persistence.
- Do not touch unrelated dirty `ARCHITECTURE.md`, `STRUCTURE.md`, `.opencode/`, `docs/issues/`, or old phase reports.
- Do not commit user-facing phase reports.

## Current Code State

- `NativeProviderExecutor` can parse/build/adapter/cache/HTTP/format native requests and streams.
- `RequestExecutor` can select native execution when routing target execution is `native` or auto-detected by provider protocol declaration.
- `_build_native_provider_context()` currently asks every provider for endpoint and headers with operation `"chat"`, which is too generic.
- Priority provider files exist for Claude Code, Codex, Copilot, and Antigravity, but tests mostly cover declarations and helper methods.
- Provider-prefixed model names can leak into upstream native payloads unless each provider normalizes them.
- Native streaming exists, but provider support flags are conservative and not all priority providers have stream coverage.
- Gemini CLI has substantial existing custom logic and must not be silently bypassed by auto-native routing.

## Implementation Plan

1. Add provider-native operation resolution.
   - Add default methods to `ProviderInterface`: `get_native_operation()`, `normalize_native_model()`, and optional `prepare_native_request()`.
   - Preserve current behavior by default.
   - Update `RequestExecutor._build_native_provider_context()` to ask providers for operation, endpoint, headers, normalized model, and prepared request metadata.

2. Make model normalization explicit and tested.
   - Claude Code strips `claude_code/`.
   - Codex strips `codex/`.
   - Copilot strips `copilot/`.
   - Antigravity strips `antigravity/` and maps public aliases to internal upstream names.
   - Gemini CLI remains custom-path first.

3. Make provider endpoints operation-aware.
   - Claude Code uses `messages` and `/v1/messages`.
   - Codex uses `responses` and `/v1/responses`.
   - Copilot uses `chat` and `/chat/completions`.
   - Antigravity uses Gemini generate/stream-generate endpoints.

4. Add minimal provider request preparation.
   - Normalize model before protocol parsing.
   - Allow provider `prepare_native_request()` to deep-copy and adjust request payloads before protocol parsing.
   - Trace this pass without adding credentials.

5. Strengthen provider auth/header behavior.
   - Keep current supplied-credential model.
   - Add tested header sets for each provider.
   - Do not add secrets to JSON config.

6. Native streaming support declarations.
   - Enable only where tested safe.
   - Prove native streaming selection through `RequestExecutor` tests.
   - Keep unsupported providers on existing fallback/custom paths.

7. Mock-live RequestExecutor integration tests.
   - Prove Claude Code, Codex, Copilot, and Antigravity use the native executor with the correct protocol, operation, endpoint, headers, model normalization, and response formatting under fake HTTP.
   - Cover streaming for providers that opt in.

8. Provider model discovery hardening.
   - Test fallback and successful discovery for each priority provider.
   - Avoid duplicate prefixes and invalid aliases.

9. Gemini CLI parity review.
   - Keep custom execution path.
   - Verify declarations align with Gemini protocol and field-cache paths.
   - Explicitly prevent accidental auto-native routing if required.

10. Documentation and comments.
    - Update provider docstrings away from “skeleton” once behavior is mock-live.
    - Explain model normalization, Antigravity safety boundaries, and Gemini CLI custom-path deferral.

## Tests

- `tests/test_claude_code_provider.py`
- `tests/test_codex_provider.py`
- `tests/test_copilot_provider.py`
- `tests/test_antigravity_provider_restore.py`
- `tests/test_gemini_cli_protocol_declarations.py`
- `tests/test_provider_protocol_declarations.py`
- `tests/test_native_provider_executor.py`
- `tests/test_native_provider_streaming.py`
- `tests/test_request_executor_native_routing.py`
- Relevant protocol, field-cache, and routing regressions.

## Acceptance Criteria

- Priority providers are no longer declaration-only skeletons; each has mock-live native `RequestExecutor` coverage or an explicit tested reason for custom-path deferral.
- Native operation and endpoint selection are provider-aware, not hardcoded to `"chat"`.
- Provider-prefixed model names are normalized before native upstream calls.
- Native streaming is enabled only where tested safe.
- Gemini CLI remains on its existing custom path unless explicitly routed otherwise.
- LiteLLM fallback remains available for uncovered providers/protocols.
- Focused tests and dual-agent review pass.
