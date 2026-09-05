# Detailed Phase Roadmap

This document expands the 10-phase roadmap before implementation begins. It exists to prevent later work from narrowing to only Phase 1 details and losing the full feature set.

Each phase still requires a fresh conversation plan immediately before implementation. This document is the durable baseline; phase-specific plans can adapt after current-code inspection.

## Phase 1: Protocol Core

Purpose: create native protocol foundations without changing live execution.

Primary deliverables:

- `src/rotator_library/protocols/` package.
- Auto-discovered protocol registry modeled after provider discovery.
- Override-friendly `ProtocolAdapter` base class.
- Unified request, response, message, content, tool, reasoning, usage, cost, stream event, and context dataclasses.
- Base protocols for OpenAI Chat, Anthropic Messages, Gemini, OpenAI Responses, and LiteLLM fallback.
- JSON-safe serialization helpers for transaction tracing and fixtures.
- Protocol errors that identify protocol name, pass name, and payload preview.

Key requirements:

- Protocols are bases, not rigid implementations.
- Providers can subclass, wrap, copy, or override protocol methods.
- Unknown provider fields must be preserved in `extra`/metadata instead of dropped.
- Runtime behavior should not change yet.
- LiteLLM fallback remains explicit and named.

Tests:

- Registry discovery and alias resolution.
- Base preservation behavior.
- Round-trip and parse/format fixtures for OpenAI Chat, Anthropic Messages, Gemini, and Responses.
- Stream event smoke coverage.

Review focus:

- Ensure protocol abstractions are not too narrow for current providers or external reference protocols.
- Ensure types are trace-friendly for Phase 2.

## Phase 2: Transform Pass Logging

Purpose: make every transformation state inspectable for debugging.

Primary deliverables:

- Transform trace model with ordered pass records.
- Transaction logger integration for request, response, stream, and error transform states.
- Redaction-at-log-boundary helpers.
- JSONL stream transform logs.
- Pass names shared by protocols, adapters, provider overrides, field-cache rules, and fallback execution.

Required pass coverage:

- `raw_client_request`
- `parsed_unified_request`
- `after_session_inference`
- `after_field_cache_injection`
- `after_request_adapters`
- `after_provider_override`
- `provider_request`
- `litellm_fallback_request`
- `raw_provider_response`
- `parsed_unified_response`
- `after_field_cache_extraction`
- `after_response_adapters`
- `after_client_protocol_format`
- `final_client_response`
- `raw_provider_stream_event`
- `parsed_unified_stream_event`
- `after_stream_field_cache_extraction`
- `after_stream_adapters`
- `formatted_client_stream_event`

Key requirements:

- Log snapshots must not mutate live request/response objects.
- Logs must preserve enough context to debug provider-specific behavior.
- Secret redaction should cover API keys, OAuth tokens, auth headers, cookies, and common secret field names.
- Logging must be usable by future admin/debug endpoints but remain file-based for now.

Tests:

- Trace pass ordering.
- Redaction behavior.
- Request/response/stream JSON serialization.
- Transform failure logging with pass name and protocol/provider context.

Review focus:

- Verify every planned pass is reachable from the architecture.
- Verify log output is useful without leaking credentials.

## Phase 3: Adapter And Field Cache System

Purpose: let custom providers configure what to transform, cache, return, and reinject without hardcoding every provider.

Primary deliverables:

- Adapter registry.
- Adapter chain execution for request, response, and stream events.
- Field-cache rule schema and validation.
- JSON-path-like extraction/injection helpers.
- Cache scope builder using provider, model, credential or classifier scope, session, conversation, and rule name.
- Provider/model-level rule configuration hooks.
- Initial built-in adapter rules for reasoning/thinking-related fields.

Field cache capabilities:

- Sources: request, response, stream event, unified request, unified response, unified stream event.
- Targets: raw provider request path, unified request field, protocol metadata, provider extension field.
- Scopes: provider, model, credential, session, conversation, and combinations.
- Modes: `last`, `all`, `last_user_turn`, `last_assistant_turn`, `per_tool_call`.
- Missing paths are no-ops unless strict validation is enabled.

Examples that must be supported by design:

- DeepSeek-style reasoning content.
- Anthropic thinking and redacted thinking signatures.
- Gemini thought signatures.
- Prompt cache keys.
- Provider session IDs.
- Responses `previous_response_id` metadata.

Key requirements:

- Field cache rules complement `SessionTracker`; they do not replace it.
- Rules must never leak across provider/model/credential/session boundaries.
- Providers can define default rules, and model config can override them.
- Transform logging must capture before/after extraction and injection passes.

Tests:

- Extract from response and stream events.
- Inject into later requests.
- Scope isolation.
- Mode behavior.
- Malformed path validation.
- Redacted transform logs.

Review focus:

- Verify custom provider authoring becomes declarative for common stateful fields.
- Verify no cross-session or cross-credential leaks.

## Phase 4: Responses API And WebSocket-Ready Transport Shape

Purpose: add high-priority OpenAI Responses API support while designing stream transports for future WebSocket support.

Primary deliverables:

- `/v1/responses` route.
- `GET /v1/responses/{response_id}` route.
- `DELETE /v1/responses/{response_id}` route.
- Optional Codex alias route if supported by provider work.
- JSON/file response storage, not SQLite.
- TTL cleanup for stored responses.
- `previous_response_id` loading and session anchor integration.
- SSE Responses streaming.
- Transport interfaces that can later support WebSocket without rewriting protocol logic.

Transport shape:

- Non-streaming HTTP JSON transport.
- HTTP SSE transport.
- Future WebSocket transport extension point.
- Unified stream events from Phase 1 should flow through transports.

Key requirements:

- `previous_response_id` must become strong session evidence where safe.
- Stored response objects must be scoped and cleaned up.
- Responses usage and output items must be normalizable in Phase 9.
- Transform logging must show Responses parse/build/storage-relevant states.

Tests:

- Create response.
- Retrieve response.
- Delete response.
- Previous response continuation.
- SSE event formatting.
- TTL cleanup.
- Transport interface extension smoke tests.

Review focus:

- Verify compatibility with OpenAI Responses expectations and external reference Responses behavior.
- Verify WebSocket is not blocked by the design.

## Phase 5: Provider Protocol Overhaul

Purpose: make providers use native protocols where practical and add priority providers after protocol foundations exist.

Primary deliverables:

- Provider protocol declaration mechanism.
- Provider hooks for protocol selection, adapter rules, field cache rules, and model options.
- Claude Code provider implementation or integration path.
- Codex provider implementation or integration path.
- Copilot provider implementation or integration path.
- Restored Antigravity provider if current/reference behavior supports it.
- Gemini CLI parity review and targeted fixes only where the external reference gateway has real improvements.

Provider priorities:

- Claude Code.
- Codex.
- Copilot.
- Antigravity.
- Gemini CLI review.

Antigravity comparison requirements:

- Compare the external reference implementation to `src/rotator_library/providers/_retired/antigravity_provider.py`.
- Restore only valid behavior.
- Avoid obsolete device-profile or fragile logic unless required by the current service.

Key requirements:

- Providers can override protocol methods when the base is close but not exact.
- New providers should avoid monolithic transform logic where adapter rules suffice.
- Native path must be testable independently of live credentials.
- LiteLLM fallback should be explicit if used.

Tests:

- Provider registration.
- Protocol selection.
- Auth header/token behavior via mocks.
- Request/response/stream translation.
- Quota/checker parsing.
- Field cache rules.
- No accidental LiteLLM fallback when native path should apply.

Review focus:

- Verify provider logic follows protocol architecture instead of reintroducing bespoke protocol code everywhere.
- Verify Gemini CLI improvements do not regress existing behavior.

## Phase 6: Routing And Fallback Groups

Purpose: add ordered model/provider fallback while keeping credential rotation inside each candidate.

Primary deliverables:

- Fallback group config parser.
- Ordered candidate planner for provider/model fallback chains.
- Retryable/non-retryable fallback decisions.
- Streaming fallback rules that only fallback before visible output.
- Optional target-group structure after fallback groups work.
- Optional selectors after ordered fallback works.

Fallback behavior:

- If requested model is in a group, try it first.
- Continue through remaining candidates only for retryable failures or configured fallback conditions.
- Preserve classifier/private credential scopes.
- Preserve session namespace isolation.
- Each candidate delegates to current credential selection and usage tracking.

Future target group selectors:

- `in_order`.
- `random`.
- `usage`.
- `cost`.
- `latency`.
- `performance`.

Key requirements:

- Fix or replace stale `fallback_groups` expectations without breaking current model resolution.
- Do not replace `UsageManager` or `SelectionEngine`.
- Transform logging should show chosen candidate and fallback attempt history.

Tests:

- Ordered fallback after retryable failure.
- Non-retryable failure stops.
- Requested model promotion.
- Exhausted chain reports useful error.
- Streaming fallback before output only.
- Scope/session isolation.

Review focus:

- Verify fallback integrates above credential rotation, not inside it.
- Verify behavior matches user preference for fallback chains over target-group complexity.

## Phase 7: Retry/Cooldown/Failover Cleanup

Purpose: streamline retry behavior and make provider/model cooldown real.

Primary deliverables:

- Replace or activate the currently inert provider `CooldownManager` path.
- Provider/model cooldown keys.
- Consecutive failure tracking.
- Exponential backoff.
- Retry-after precedence over computed cooldown.
- Success reset behavior.
- Retry history records for logging and future debug surfaces.
- Integration with fallback groups from Phase 6.

Cooldown layers:

- Credential-level cooldown remains in `UsageManager`.
- Provider/model cooldown applies only to evidence of provider-wide or model-wide failure.
- Credential quota exhaustion must not automatically cool an entire provider.
- Model cooldown should not block healthy models on the same provider.
- Provider cooldown should be reserved for provider-wide failure evidence.

Retry history fields:

- candidate provider/model.
- credential stable ID or masked identity.
- protocol path used.
- attempt number.
- status: success, failed, skipped, cooled_down.
- error category and raw classifier result.
- retryable decision.
- cooldown decision and duration.
- timing and latency.

Key requirements:

- Preserve the current strong retry-after parser, especially Google/Gemini compound duration parsing.
- Preserve streaming safety: no retry after visible output unless explicitly safe.
- Make cooldown state observable by transaction logs and future endpoint surfaces.

Tests:

- `start_cooldown` or successor has production callers.
- Provider cooldown blocks provider-wide only.
- Model cooldown blocks one model only.
- Credential cooldown does not become provider cooldown.
- Retry-after overrides exponential backoff.
- Backoff escalates with repeated failures.
- Success clears counts.
- Retry history is recorded.
- Fallback respects cooldown skips.

Review focus:

- Verify no healthy credential/model is suppressed too broadly.
- Verify retry/fallback/streaming interactions are deterministic.

## Phase 8: Streaming Library Upgrade

Purpose: harden streaming as a reusable library capability, not just a proxy route behavior.

Primary deliverables:

- Transport-aware stream event pipeline using unified stream events.
- Explicit upstream iterator close/cancel on client disconnect.
- TTFB timeout before first emitted output.
- TTFT metrics.
- Throughput stall detector.
- Stream usage extraction through protocol normalizers.
- Stream transform logging for raw, unified, adapted, and formatted states.
- SSE transport improvements and WebSocket-ready transport boundaries.

Streaming behavior:

- Retry is allowed before client-visible output when the error is retryable.
- Retry is not allowed after visible output unless a future protocol explicitly proves safety.
- Partial streams should not record accepted session response anchors.
- Completed streams should record response anchors and normalized usage.
- Disconnects should close upstream resources promptly.

Metrics:

- time to first byte.
- time to first token/content.
- tokens per second where token counts are available.
- chunk count.
- stall duration.
- completion status.

Key requirements:

- Implement with Python-native async patterns, not runtime-specific socket internals from the external reference gateway.
- Keep current conservative retry safety.
- Preserve current usage recording behavior and then improve it via Phase 9 normalizers.

Tests:

- Disconnect closes upstream async iterator.
- TTFB timeout retries before output.
- Stall detection trips after grace period.
- No retry after visible content.
- Retry before first output.
- Stream transform logging writes expected pass records.
- SSE formatting remains compatible.

Review focus:

- Verify library-level design is not tied only to FastAPI route wrappers.
- Verify future WebSocket can reuse the same unified event stream.

## Phase 9: Usage, Quota, And Cost Accuracy

Purpose: make usage and cost accounting protocol-aware while preserving the current usage engine.

Primary deliverables:

- Protocol-aware usage normalizer interface.
- Usage normalizers for OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini, OAuth/custom providers, and LiteLLM fallback.
- Structured cost details model integrated with protocol usage.
- Provider-reported cost extraction.
- SSE cost comment parsing where providers emit cost metadata.
- Reasoning/thinking token normalization.
- Cache read/write token normalization.
- Meter/checker abstraction for proactive provider quota checks where useful.

Usage fields:

- input tokens.
- output tokens.
- total tokens.
- reasoning/thinking tokens.
- cache read tokens.
- cache write tokens.
- provider-reported cost.
- estimated cost.
- cost source.
- raw provider usage metadata.

Cost precedence:

- Provider-reported cost wins when present and trusted.
- Structured provider cost fields are preferred over text parsing.
- SSE `: cost` comments can supply provider-reported cost.
- Estimated cost remains fallback.

Current systems to preserve:

- `UsageManager` facade.
- windowed tracking.
- fair cycle.
- custom caps.
- quota groups.
- classifier/private scope separation.
- JSON usage persistence.

Key requirements:

- Do not replace the usage engine.
- Normalize before recording when native protocol path is used.
- Existing LiteLLM response usage should continue working.
- Cost details should be transaction-loggable and eventually available to APIs/TUI.

Tests:

- OpenAI Chat usage details.
- Responses usage details.
- Anthropic cache read/write tokens.
- Gemini usage metadata including thoughts/cache when available.
- Reasoning token extraction.
- Provider-reported cost precedence.
- SSE cost comment extraction.
- Existing usage aggregation tests still pass.

Review focus:

- Verify accounting is more accurate without disrupting selection and quota logic.
- Verify provider-specific raw usage remains available for debugging.

## Phase 10: Config Polish

Purpose: support powerful protocol/routing/provider configuration without SQLite.

Primary deliverables:

- Optional JSON config file support.
- Env var pointing to JSON config path.
- Env overrides JSON.
- Validation with actionable errors.
- Config sections for protocols, adapters, field-cache rules, fallback groups, providers, model overrides, quota checkers, and stream settings.
- Documentation/examples for custom provider setup using existing protocols.

Config priorities:

- `.env` remains enough for simple setups.
- JSON supports complex nested config that is painful in env vars.
- Env overrides allow quick local changes.
- No SQLite/Postgres requirement.

Potential JSON sections:

- `protocols`.
- `providers`.
- `models`.
- `adapters`.
- `field_cache`.
- `fallback_groups`.
- `quota_checkers`.
- `streaming`.
- `logging`.

Key requirements:

- Validation errors must name the config section and field.
- Bad config should fail early when possible.
- Config should support per-provider and per-model overrides.
- Config should not require rewriting existing `.env` usage immediately.
- Config docs should show custom provider examples using OpenAI Chat, Anthropic Messages, Gemini, Responses, and LiteLLM fallback.

Tests:

- Env-only config.
- JSON-only config.
- Env overrides JSON.
- Bad config failure messages.
- Provider/model adapter rule config.
- Fallback group config.
- Field-cache rule config.

Review focus:

- Verify custom provider setup becomes practical and documented.
- Verify no accidental database dependency is introduced.

## Cross-Phase Review Contract

Every phase ends with two review agents:

- `explore` for code/file-level verification.
- `explore-heavy` for deeper architecture and reference comparison.

Review prompts must include the relevant phase plan, external reference areas, current proxy behavior to preserve, tests run, and transaction logging expectations. If either agent fails or runs out of context, restart with narrower scope.

## Cross-Phase Testing Contract

Every implementation phase must add or update tests before being considered complete. New test files may need `git add -f` because the repository ignores most `tests/*` by default.

## Cross-Phase Documentation Contract

Implementation code must include docstrings and comments for public extension points, non-obvious transformations, lossy conversions, future WebSocket seams, provider override points, and config decisions.
