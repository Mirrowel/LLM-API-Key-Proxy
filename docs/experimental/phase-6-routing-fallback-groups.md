# Phase 6 Plan: Routing And Fallback Groups

## Goal

Add explicit ordered routing/fallback groups and a provider-native routing seam that can choose between the current LiteLLM execution path, provider custom logic, and the Phase 5 `NativeProviderExecutor`. Fallback groups are the priority deliverable: a model can resolve to an ordered chain of concrete targets, and the executor can try the next target on eligible failures without replacing the existing credential rotation, usage tracking, session tracking, or retry-after parsing.

## Non-Goals

- Do not replace `UsageManager`, `SelectionEngine`, `SessionTracker`, or retry-after parsing.
- Do not add full target-group selector language before fallback groups work.
- Do not implement multi-user/security isolation.
- Do not implement a complete config-file system; Phase 10 owns config polish.
- Do not route every provider natively by default.
- Do not fallback after visible streamed output has been emitted.
- Do not silently fallback to LiteLLM for native providers unless the fallback chain explicitly includes a LiteLLM target.
- Do not make Claude Code/Codex/Copilot/Antigravity live-production claims beyond mocked/tested routing behavior.

## Current Code Context

- `RequestExecutor` is the live retry/rotation path.
- `RequestExecutor` already gets provider plugin instances, usage managers, credentials, request context, transaction logger, and LiteLLM provider params.
- Phase 5 `NativeProviderExecutor` is isolated and opt-in.
- Providers can declare `protocol_name`, adapter names, field-cache rules, native endpoints, and native headers.
- Model resolution currently maps a single requested model to one provider/model pair.
- Existing retry loops rotate credentials inside one provider; Phase 6 must add model/provider target fallback around that without breaking per-provider credential rotation.
- Existing transaction trace can log routing decisions.
- Existing streaming handler can rotate credentials before output, but stream fallback after output must remain disallowed.

## Files To Add

- `src/rotator_library/routing/__init__.py`
- `src/rotator_library/routing/types.py`
- `src/rotator_library/routing/config.py`
- `src/rotator_library/routing/resolver.py`
- `src/rotator_library/routing/policy.py`
- `src/rotator_library/routing/executor.py` or `attempts.py`
- `tests/test_routing_config.py`
- `tests/test_fallback_resolver.py`
- `tests/test_fallback_policy.py`
- `tests/test_request_executor_fallback_groups.py`
- `tests/test_request_executor_native_routing.py`
- `tests/test_streaming_fallback_policy.py`

## Files Likely To Touch

- `src/rotator_library/core/types.py` to add optional routing/fallback fields to `RequestContext`.
- `src/rotator_library/client/request_builder.py` to resolve fallback groups before execution.
- `src/rotator_library/client/models.py` if model resolution needs a group-aware helper.
- `src/rotator_library/client/executor.py` to wrap existing per-provider retry loops in target attempts and call native executor for eligible targets.
- `src/rotator_library/client/rotating_client.py` only if passing route configuration into components requires it.
- `src/rotator_library/providers/provider_interface.py` only if a small native-support method is needed.
- `src/rotator_library/transaction_logger.py` only if adding route correlation helpers is cleaner.

## Routing Types

`RouteTarget`:

- `name`: stable target identifier.
- `provider`: provider key.
- `model`: concrete model name, with provider prefix optional but normalized.
- `protocol`: optional override; defaults to provider declaration.
- `execution`: `auto`, `native`, `custom`, or `litellm_fallback`.
- `priority`: optional numeric order.
- `weight`: future target-group selector support, ignored for ordered fallback.
- `conditions`: optional metadata for later selectors.
- `metadata`.

`FallbackGroup`:

- `name`.
- `targets`: ordered `RouteTarget` list.
- `failover_on`: error categories that permit next-target fallback.
- `stop_on`: error categories that must stop.
- `streaming_policy`: e.g. `pre_output_only`.
- `max_targets`: optional guard.
- `metadata`.

`RoutingDecision`:

- `requested_model`.
- `group_name` optional.
- `targets`.
- `selected_target_index`.
- `reason`.

`RouteAttemptResult`:

- target, success/failure, error classification, emitted_output flag, usage summary.

## Configuration Plan

- Phase 6 supports env-first fallback definitions with a small parser.
- Optional JSON config can be parsed if simple, but deeper config merging belongs to Phase 10.
- Env examples:
  - `FALLBACK_GROUPS=sonnet_chain,codex_chain`
  - `FALLBACK_GROUP_SONNET_CHAIN=claude_code/claude-sonnet-4-5,copilot/claude-sonnet-4-5,anthropic/claude-3-5-sonnet-latest`
  - `FALLBACK_GROUP_CODEX_CHAIN=codex/gpt-5.1-codex,openai/gpt-5.1`
  - `MODEL_ROUTE_CLAUDE_SONNET=group:sonnet_chain`
  - `MODEL_ROUTE_CODEX=group:codex_chain`
- Also allow programmatic construction in tests.
- Config parser must validate:
  - group names are unique.
  - targets have provider/model.
  - empty chains are invalid.
  - cycles through group aliases are rejected.
  - provider names are not silently guessed when ambiguous.
- No secrets in route config.

## Target Resolution

- If requested model maps to a fallback group, return all group targets in order.
- If requested model is already `provider/model`, return a single target.
- If requested model maps through existing model definitions, preserve existing behavior.
- Target model should be normalized with provider prefix before entering `RequestExecutor`.
- Record transform trace pass `routing_decision` with requested model, group, target count, selected target names, execution modes, and no credentials.

## Fallback Policy

Try next target only on eligible failures:

- rate limit/quota/capacity.
- provider unavailable/server error.
- transient connection errors.
- native provider unsupported operation when explicit fallback target exists.

Do not fallback on:

- auth errors for all credentials unless the group target explicitly marks auth fallback safe.
- validation/permanent request errors.
- pre-request callback failures.
- client cancellation.
- streaming errors after visible output.

Respect existing per-provider credential rotation before moving to the next target:

- one target attempt should let current `RequestExecutor` exhaust eligible credentials according to existing retry logic.
- after a target is exhausted, the fallback layer can advance to the next target.

Keep error accumulator context across targets so final failure includes all target errors.

## Native, Custom, And LiteLLM Execution Choice

`execution=auto`:

- if plugin has `has_custom_logic()`, use current plugin `acompletion()`.
- else if provider has native protocol declaration and native endpoint/header methods, use `NativeProviderExecutor`.
- else use current LiteLLM call path.

`execution=native`: require native support or fail this target with a classified unsupported-operation error.

`execution=custom`: require `has_custom_logic()`.

`execution=litellm_fallback`: force current LiteLLM path and emit `native_provider_litellm_fallback` / `routing_litellm_fallback` trace metadata.

Tests must prove native-declared providers do not silently use LiteLLM unless the target says `litellm_fallback`.

## RequestExecutor Integration

- Keep the existing `_execute_non_streaming()` and `_execute_streaming()` credential retry loops as the per-target implementation.
- Add a wrapper method that accepts a `RequestContext` with `routing_targets`.
- For each target:
  - clone/update context provider/model/usage manager key/credentials for that target.
  - record `routing_target_attempt_started`.
  - run the existing target execution path.
  - on success record `routing_target_attempt_succeeded` and return.
  - on failure classify and ask fallback policy if next target is allowed.
  - record `routing_target_attempt_failed`.
- Avoid mutating the original request context in-place.
- If no group exists, execute exactly as today.

## Native Provider Execution Integration

- Add a small execution branch inside the target attempt where the selected target resolves to native execution.
- Construct `NativeProviderContext` from:
  - provider plugin declarations.
  - selected model.
  - credential identifier/token.
  - request context session/scope/classifier.
  - transaction logger.
  - endpoint/headers from provider methods.
- Use Phase 5 `NativeProviderExecutor` for mocked/tested providers.
- Preserve usage recording through the existing credential context. If native responses do not provide full usage yet, record normalized available usage only and leave cost normalization for Phase 9.
- For custom providers with `has_custom_logic()`, keep existing plugin `acompletion()` branch.
- For undeclared providers, keep LiteLLM path.

## Streaming Integration

- Fallback to next target is allowed only before any visible chunk is yielded.
- Track `emitted_output` in the stream wrapper.
- If a stream errors before output, fallback policy may advance to next target.
- If a stream errors after output, propagate the error; do not switch models mid-stream.
- Record:
  - `routing_stream_target_attempt_started`
  - `routing_stream_target_attempt_failed`
  - `routing_stream_target_attempt_succeeded`
  - `routing_stream_fallback_blocked_after_output`.
- Keep current stream retry policy for same-target/same-provider errors before group fallback.

## Transaction Trace Requirements

- `routing_decision`
- `routing_target_attempt_started`
- `routing_target_attempt_failed`
- `routing_target_attempt_succeeded`
- `routing_fallback_selected`
- `routing_fallback_exhausted`
- `routing_litellm_fallback`
- `routing_native_execution_selected`
- stream equivalents where applicable.
- Trace metadata includes group, target index, provider, model, execution mode, classification, and reason. It must not include raw credentials.

## Target Groups As Optional Richer Layer

- Phase 6 focuses on ordered fallback groups.
- Add type seams for future target groups:
  - `TargetSelector`
  - `TargetGroup`
  - `TargetSelectionPolicy`
- Do not implement weighted/latency/cost selector behavior unless it is trivial and isolated.
- Document that Phase 6 fallback groups are deterministic ordered chains.

## Tests

Config tests:

- parse env fallback group.
- invalid empty group.
- duplicate group names.
- provider/model parsing.
- explicit `litellm_fallback` execution target.

Resolver tests:

- requested alias maps to group.
- provider/model remains single target.
- target order is preserved.
- existing model prefix behavior stays intact.

Policy tests:

- fallback on rate limit/server/transient.
- stop on auth/permanent/pre-request/cancel.
- streaming fallback allowed before output.
- streaming fallback blocked after output.

Request executor tests:

- first target success uses no fallback.
- first target rate limits all credentials then second target succeeds.
- final error includes both target failures.
- no fallback group preserves old behavior.
- transaction trace includes routing passes.

Native routing tests:

- native-declared provider target uses `NativeProviderExecutor`.
- native-declared provider does not use LiteLLM unless target execution is `litellm_fallback`.
- custom provider still uses plugin `acompletion()`.
- undeclared provider still uses LiteLLM.

Streaming tests:

- pre-output target failure falls through to next target.
- post-output failure does not fall through.
- trace records blocked fallback after output.

Regression:

- Phase 1 protocol tests.
- Phase 2 logging tests.
- Phase 3 adapter/cache tests.
- Phase 4 Responses tests.
- Phase 5 provider tests.
- `test_session_tracking.py`.
- `test_selection_engine.py`.

## Commit Checkpoints

1. Add routing types, config parser, resolver, and policy with tests.
2. Add request context routing fields and target cloning helpers with tests.
3. Integrate non-streaming fallback group wrapper around existing executor path with tests.
4. Integrate native/custom/LiteLLM execution mode selection with tests.
5. Integrate streaming pre-output fallback policy with tests.
6. Add trace pass coverage and run regressions.
7. Review with `explore` and `explore-heavy`, fix findings, and write the uncommitted Phase 6 report.

## Risks And Mitigations

- Fallback could double-spend or corrupt usage. Mitigation: each target uses existing credential context and usage manager; group fallback starts only after target failure.
- Fallback could mask permanent request bugs. Mitigation: policy stops on validation/permanent errors.
- Streaming fallback could corrupt client output. Mitigation: disallow fallback after visible output.
- Native routing could accidentally bypass LiteLLM behavior for existing providers. Mitigation: default no group/no native route is unchanged; native path is explicit/declared.
- Config could become too broad. Mitigation: small env parser now; richer JSON config later in Phase 10.
- Session affinity could leak across provider pools. Mitigation: keep `SessionTracker` namespace behavior and clone target context with provider/model-specific namespace where needed.
