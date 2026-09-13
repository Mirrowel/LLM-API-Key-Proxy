# Fix-Pass Master Plan (v2 — consolidated 2026-09-11)

**Provenance:** synthesized from `audit-remediation-plan.md` (audit notes #17-#51) + two deepened advisory validation passes + the exhaustive 13-part Round 2 trio sweep (`audit-sweep-findings.md`, ~200 additional findings). This document consolidates EVERYTHING into one sorted group structure — no cross-referencing needed between original and amendment sections. Per-finding evidence (file:line, probes, doc citations) lives in the two audit files; this plan carries the per-group work contract.

**House rules (user-set):**
- Plan files + replication tests only; **no implementation without explicit per-group user approval**.
- One commit per completed group (house commit style #954).
- Gate bar (#17): a group is done only when every finding in its scope is fixed or explicitly ignored with recorded rationale; paired review discipline applies (#1049).
- **R11 live round with user BEFORE execution begins** (per #35/#51). After the fix pass: **the entire audit runs again** (#22).
- `tests/` are gitignored — force-add test files in every group commit.

---

## 1. Execution order (final)

```
R11 (live round, with user)
→ WINDOW 1: G1 (errors) + G15 (secrets — small, independent files, security)
→ G2 (hooks skeleton)
→ G7 (routing/profiles/config)
→ WINDOW 2: G4 (stream core) → G13 (stream formatter)
→ G3 (raw-path containment)
→ G5 (accounting & cost)
→ G14 (gemini depth)
→ G10 (conversion & disclosure)
→ G8 (providers & interface)
→ G6 (sessions)
→ WINDOW 3: G11 (variants & stores) + G16 (responses synthesis)
→ G9 (provider porting; embeddings split-out decided at execution)
→ G12 (suite & docs — continuous + closing)
```

**Early riders (behavior-free, may land in the G1 window):** `ModelPricing` relocation out of `usage/costs.py` import chain; `request_builder.py:87` try-wrap for routing-config parse; `_API_KEY` substring-scan fix (5 files — see G8); `providers/__init__` triple `import logging` cleanup.

**Single-writer / contention matrix (enforced):**

| File | Owning groups (in order) |
|---|---|
| `client/executor.py` | G1 (error regions) → G4 (stream loop) → G3 (:871 gate) → G5 (:2661 cost) |
| `native_provider/executor.py` | G2 (pipeline) → G3 (:77-102 basis) → G5 (cost sites) |
| `client/stream_ops.py` + `client/streaming.py` | G4 outright |
| `protocols/streaming.py` | G13 single owner (G4 hands off) |
| `protocols/` framework (canonical/validation/base/types/registry/transforms) | G10 |
| `protocols/` per-dialect wire files | G14 (gemini) / G10 (chat+anthropic conversion halves) / G11 (responses halves) |
| `protocols/gemini.py` + `client/gemini.py` | G14 |
| `proxy_app/main.py` | G1 (ladders) → G8 (gemini routes) → G11 (responses routes); G15 (redaction sites) |
| `providers/__init__.py` | G8 (registration) → G11 (variant profiles) |
| `responses/*` | G16 → G11 |
| `usage/` (accounting/costs/persistence) | G5 → G15 (persistence secrets) |
| `session_tracking.py` | G6 |
| `config/experimental.py` | G7 single owner |

---

## 2. G1 — Error taxonomy + decision matrix (XL, P0 — first)

**Sources:** notes #20, #22, #23.1, #24 error map, #25.3, #21, #48-49; sweep R1/R2/R3/R9 + N1-N4. Detail: sweep file R1/R2/R9 sections.

**Problem in one paragraph:** three parallel classifiers disagree (`classify_error` dict/httpx branches, `structured_api_response_error` — which also has substring false-positives: "rate" matches "generate"/"accurate"), the routing layer runs a FOURTH classifier re-deriving from raw 400 codes that hard-stops quota errors, five inline executor decision copies ignore the tested helper, FAIL is swallowed by catch-alls (deterministic 400s hammer every credential, wrong 503), quota-at-400 is non-rotatable AND resets the 3-consecutive counter, auth stops the chain (user: must failover), configuration errors rotate (must stop), and the shell carries ×4 litellm→status mapping ladders.

**Deliverables:**
1. ONE classification pipeline: body-status priority (Gemini `error.status` string over HTTP code — fixes N1 status-shadowing at error_handler.py:814), message-text quota sniffing at 400 (parity with the 429 branch), native no-envelope 429 rotates (N2), substring classifier restricted to structured type/status/code fields (never free-form message), wire `*_error` vocabulary recognized on input, JSON-string error data parsed; unify with `structured_api_response_error`.
2. Routing-layer classification unified with the above: `_route_error_type` trusts `StructuredAPIResponseError.error_type` first; 400-derived candidates never hard-stop when an explicit quota/classification candidate exists; executor-local `_HARD_STOP/_RETRYABLE` copies deleted (single source in routing/policy).
3. Per-error-type decision matrix (user directive #25.3): rotate / retry-same / failover / stop / cooldown(+scope) — documented, implemented once, consumed by both loops + stream/non-stream. **Auth → failover, not stop.** `configuration_error` → stop. `not_found`/unknown policy decided + pinned. 408 rotates; anthropic 529/overloaded_error/billing_error classified; `retry in` phrasing + days in duration parser; google.rpc `details[]` (RetryInfo/QuotaFailure) carried to cooldowns.
4. FAIL semantics restored: raises escape both loops' catch-alls; 3-quota abort reachable; counter-reset compounding (N3) gone; envelope-vs-exception asymmetry (N4) unified; `on_request_complete` hook exceptions contained (never destroy completed responses + accounting); cooldown-exceeds-budget → structured 429 (not raw RoutingExecutionError 500 — dev-regression); stream_options:null → 400 not rotation storm; non-stream error sleeps honor the deadline (`_sleep_before_transient_action` both paths); RateLimitError branch uses the gating helper; dict-changed-size on shared usage-managers (list() snapshot ×3 sites); NoAvailableKeys stream/non-stream behavior unified.
5. Bare-ValueError family → protocol 400s / RoutingConfigError: `_validate_request`, ModelReferenceError (request-path addressing), `max_targets` (2 sites), `@mode` parsing, unknown adapter names (startup error not per-request KeyError). SMALL_COOLDOWN_RETRY_THRESHOLD validated env-int (4 sites).
6. Shell de-logic (#22): ×4 mapping ladders move into the library; management routes protocol-shaped; cost-estimate/quota-stats/token-count malformed-input 400s (non-dict + non-string-model guard family — chat/gemini/countTokens/gemini-count routes too); auth 401s render client-protocol shapes; `/v1/responses` non-dict 400 (the .get outside try); models/model-info routes error-handled; quota-stats no internal-text echo; app.state bare access → 503/protocol shape.
7. OpenAI error `type` vocabulary canonicalized (`invalid_request_error` family + `param` key; anthropic branch already correct); anthropic `proxy_timeout` → `timeout_error` (504).
8. Key policy: `python -m uvicorn` + **`UVICORN_HOST` env bypass** (click auto_envvar) + programmatic `uvicorn.run` + ASGI-server-blind — fail-closed when bind unprovable; `import proxy_app.main` never launches the TUI (module-level argv check moves under `__main__`; duplicate trigger merged); `is_localhost_bind` uses ipaddress loopback check.
9. Metadata finalize on failed non-stream transactions; route logger captures `event:`-framed multi-line chunks (shared with G4).

**Files:** `error_handler.py`, `core/errors.py`, `routing/policy.py` + `routing/types.py`, `client/executor.py` (error regions), `routing/executor.py` (consolidate dead FallbackAttemptRunner), `proxy_app/main.py`, `route_helpers.py`, `key_policy.py`, `native_provider/http.py`.
**Tests:** existing error/fallback/policy suites as the base; add: quota-at-400 rotation (gemini-compat), FAIL-escapes-both-loops, auth-failover, N1/N2 shapes, substring false-positive pins, UVICORN_HOST bypass pin, TUI-import pin, ladder-removal shell parity.

## 3. G15 — Secrets & logging hygiene (M, P0 — lands WITH G1; independent files)

**Sources:** sweep R1 (H1 raw logger), R9 (H1 usage.json keys + quota-stats full_path, M error records, M headers channel), R1 misc. Detail: sweep R1/R9.

**Deliverables:**
1. **usage.json stops persisting raw API keys** (`credentials.<id>.accessor` + `accessor_index` — store hashed/derived ids only; migrate existing files on load) AND quota-stats never serves them (`full_path` redacted the same way logs are).
2. RawIOLogger redaction set extended: `x-goog-api-key`, `api-key` (+ generic `*-api-key` suffix match); redaction unit test iterating every header the shell authenticates.
3. Error records sanitized: metadata `errors[]` + provider `error.log` scrub URLs/keys (litellm gemini `?key=` — extend text scrubber with bare `key=` params); `log_response`/`log_transform_pass` headers channel routed through the redactor.
4. Launcher/TUI key displays: main menu full-key print → mask; `[:20]`/`[:8]` remnants; prompts use masked defaults + password=True; quota_viewer config store 0o600.
5. WS auth accepts the same carriers as HTTP routes + close code 1008 with reason; CORS `expose_headers` for the capability header; CORS wildcard+credentials resolved (reflect-and-allowlist or drop credentials).
6. `ProviderLogger.log_extra` path-traversal safe; `ReauthCoordinator.get_status` no full credential paths.

**Files:** `usage/persistence/storage.py`, `usage/manager.py`, `client/quota.py`, `proxy_app/detailed_logger.py`, `launcher_tui.py`, `quota_viewer*.py`, `settings_tool.py`, `main.py` (WS/CORS), `provider_cache.py` logging paths, `utils/reauth_coordinator.py`.
**Tests:** add: exfiltration-pin (quota-stats never returns key material), redaction-matrix, usage.json migration.

## 4. G2 — Hookable native pipeline + field cache correctness (XL, P0 — second)

**Sources:** #23.3/#23.4, #29 hooks map + contract breaks, #30 design (high-priority core extensibility); sweep R5 (all), R2 M5/M8. Detail: sweep R5.

**Pipeline (current, to become declared stages with hook slots before/after each):** execute() = metadata-inject → parse → unified-state-inject → basis-select → finalizer (currently BEFORE adapters+cache — wrong) → request adapters → cache-inject → extract-request → validate (currently LAST — wrong) → transport. Stream: parse → events → adapters (non-terminal) → per-frame extract → done bypasses adapters + both extractions; no assembled-response extraction; no finally/aclose.

**Deliverables:**
1. **Declared pipeline object** — ordered stages, each named, typed, with hook slots; registration from provider class / config / global registry; hooks may override every part of the stage at that position (base URL, any parameter, request AND response — availability by position/medium; #30). Finalizer folds in as the LAST pre-send hook; `validate_request` declared; adapters become specialized hook consumers.
2. **Strip primitives** as first-class stage consumers (#23.3): declarative per-protocol/per-provider always-strip lists; delete-with-warning for unknown SSE fields.
3. **Field-cache correctness:** portable `model:<name>` pooling actually works (siblings vs can_inherit agree — plan 2.7 feature exists in name only); unified_request-target injection covers messages/system/tools (currently dropped while engine reports success); insert+auto inserts when absent (currently never); cache_replay vocabulary accepts the documented modes (turn_only/last:N/turns:N); compat JSON surface implemented (registry + config parsing + plan-example value fixed); JSON vs replay inject defaults unified (opposite today); profile in the cache key signature (cross-profile shape sharing); metadata-target injection cannot clobber routing keys; continuation-strip deep; hydrate allowlist widened; envelope+injection kill fixed (finalizer-last solves it); shared-key signature includes compatibility+transform; `enabled` in the weakening guard; config-group exclusion; same-inject-path different-name guard; scope floor consistent (D11) between JSON and replay; cache keys normalized (case/prefix/order).
4. **Transport hygiene:** transport-owned default timeouts (httpx 5s default today on the DEFAULT path); `: cost` comment frames parsed (provider cost preserved); provider heartbeats visible to the stall detector; SSE parsing spec-correct (no lstrip; unknown fields dropped+warned); finally/aclose everywhere; 3xx → retryable/redirect class (not invalid_request); 200-non-JSON contained; disconnect always closes upstream (no config escape).
5. **Containment:** field-cache rule errors (bound violations, corrupt rows, transform errors) → log + skip the rule, never fail the request; registries atomic (protocols/adapters/field-cache trio — failed registration leaves no half-state); FieldRename stream stage typed correctly (StreamEvent not dict) or removed; ModelOverride never fabricates a `model` key on gemini; transform empty-string → skip (symmetric); when_missing_only None-semantics consistent; sibling fan-out bounded; store clear() scoped (never nuke the whole backing cache); TTL cleanup; lock objects not id() ints.

**Files:** `native_provider/*`, `field_cache/*`, `adapters/*`, `protocols/base.py`, `protocols/operation.py`, `client/executor.py` native regions.
**Tests:** native/W7/W13 suites as base; add: pooling-works pin (engine-level, not registry-level), insert+auto pin, envelope-last pin, timeout defaults pin, cost-comment pin, containment matrix.

## 5. G7 — Routing, profiles, config load (L, P0/P1 — third)

**Sources:** #24, #26 (all), #25.1, #38 casing/names, #48 CONFIG, #50; sweep R3 (all). Detail: sweep R3.

**Deliverables:**
1. **The D13 JSON declaration schema actually loadable:** `profiles`/`transport_profiles`/`default_profile`/`model_protocols`/`cache_replay` added to `_PROVIDER_CONFIG_KEYS` + validators + binding onto plugin instances; `<NAME>_CONFIG` implemented (D16) or formally retracted; profile JSON schema per plan 2.8 (endpoint_paths map keyed by operation — plural — with singular fallback; `{model}`/`{operation}` placeholder rendering in the BASE get_native_endpoint; per-profile auth via profile-aware `get_native_headers`); dynamic providers get the profile surface + full protocol choice (env-only dynamics default native openai_chat per W11 — currently silently litellm).
2. **D13 revision (#26.4):** priority list (chat → responses → anthropic → gemini) when bare-name has no protocol match, with terminal warning; explicit `:profile` still fails loudly; docstring + code + config-reference + old-rule test re-pinned together.
3. **Fallback groups are fail-open-safe:** pre-flight abort when ANY later target lacks credentials → skip-and-record (raise only when NO target serviceable — D17 breaker); @execution split only against the known vocabulary (ollama `@digest` survives; request-path `@native` rejected with 400, never forwarded inside the model id); resolver promotion profile-aware (explicit-profile requests keep the failover chain; no duplicate same-model attempts); runtime `protocol_name` never silently overrides an explicit profile; providerless `gpt-4:free`-style ids never parsed as profiles; `parse_route_target` validates profiles via `valid_profile_name` (multi-colon agreement between both parsers); whitespace stripping per-segment; casing normalized-or-rejected at validation AND normalized at every identity sink (usage/cooldown/session/cache keys); malformed addressing → protocol 400s.
4. **Config load hardening:** routing-config parse inside the try + validate-reject-keep-last-good (broken config never loads, never partially applies, never fails every request); `load_experimental_config` cached; explicit-config-path typo does not silently mean "no config"; `client/gemini.py` second per-request config-load site eliminated; env group-key collisions detected post-normalization; group names reject `:`; empty `failover_on`/`stop_on` → default (not silent disable); wire-vocabulary aliases (`invalid_request_error` etc.) recognized in policy sets; `acompletion(input_protocol=...)` real parameter; model-route aliases preserve profiles; trace/attempt-history records carry profile; field-cache model dimension canonicalized (stripped form always); compat member refs profile-stripped; catch-all group documented or added as a knob; `max_targets` semantics decided (runtime cap vs load-time guard) + documented.
5. Cleanup: priority/weight/protocol/conditions/metadata on RouteTarget wired or removed; `selected_target_index` dead; `split_profile_from_provider` dead; `FallbackAttemptRunner` consolidation (with G1); `ModelResolver` docstring.

**Files:** `routing/*`, `client/request_builder.py`, `client/gemini.py`, `client/scopes.py`, `client/rotating_client.py`, `config/experimental.py`.
**Tests:** routing/profile/w11 suites as base; add: schema-loadable pin, priority-list pin, skip-unservicable-target pin, digest survival pin, casing matrix, config-reject-keep-last-good pin.

## 6. G4 — Stream core (L+, P0 — window 2, first)

**Sources:** #31, #32 (corrected), #40 stream bits, #41, #42; sweep R2/R6. Detail: sweep R6 + R2 stream items. NOTE: `protocols/streaming.py` findings live in G13 (single writer); G4 owns `stream_ops.py`/`client/streaming.py`/pipeline mechanics.

**Deliverables:**
1. **Conditional re-serialization** (#42): observe a copy, relay bytes; engage the formatter only when repair/conversion/normalization is needed — repair-capable tail always attached.
2. **Finish/usage repair everywhere incl. fast path** (#25.5): tools seen → `tool_calls`; else `stop`; missing usage → present-but-zeros; success only on completion evidence; held-finish flush on bare EOF (not just `[DONE]`); provider's own final-frame reason wins over synthesized (kill the tool-call override in `_with_final_reason` + aggregator).
3. n>1 fixed on all sources (ChatWire plain-delta sibling drop, hold-back single-slot, native singular parse, aggregator log-side concat); sibling survivor = most-valid candidate.
4. Usage merge hybrid (late `{"usage": {}}` must not zero — pin exactly the empty-object case); stream usage frame gated on the client's include_usage; cost null-chains hardened (all sites); `_event_visible` reads real attributes (TTFT/visible metrics currently always wrong); chat reasoning-only counts as visible for the fallback lock (duplicated prefixes on failover).
5. `x-proxy-conversion` body key removed (logs-only disclosure); `x-proxy-estimate` policy decided in the same pass.
6. Deadline enforcement in the consume loop; pending_chunks bounded; disconnect-race final event; `cancel_upstream=False` no longer abandons the generator; pipeline `adopt()` conditional.
7. Logging: wrapper sees `event:` frames (back boundary for anthropic/responses); assembled L1 protocol-aware (not always chat-shaped); `client_protocol_context` rebuilt per attempt (stale credential id, formatter state across retries); failed-with-output marked failed in traces.
8. Delete retired `wrap_stream` + duplicate helpers — **after migrating the timing suite (TTFB/stall/heartbeat/disconnect/metrics) to the live pipeline** (prerequisite for deletion); stall gate measures only time blocked on `__anext__` (downstream backpressure exempt); `decide_streaming_error_action` adoption rides G1; neutral error-events terminate the pipeline; session stream callback protocol-aware (arguments never erased — `call.function` on a structure that has none; rides G6 for readers).

**Files:** `client/stream_ops.py`, `client/streaming.py`, `native_provider/streaming.py`, `protocols/openai_chat.py` (`_audio_format`).
**Tests:** stream/parity/usage suites as base; add: live-pipeline timing suite, repair-rule matrix, empty-object-usage pin, n>1 multi-chunk matrix.

## 7. G13 — Stream formatter correctness (XL, P0 — window 2, second; single owner of protocols/streaming.py)

**Sources:** sweep R6 + R-CHAT + R-ANTHROPIC + R-GEMINI (the convergence finding: every depth round hit this file). Detail: sweep R6 + the three protocol sections.

**Deliverables:**
1. **Block identity:** explicit-index events no longer defeat the family-epoch path (text→tool→text = three blocks for ALL real sources — the "covering" test used index-less hand-built events: fabricated-fixture mode, re-pin with real shapes); Gemini id-less same-name tool fragments keyed cross-event (no split-into-N); chat target tool-call indexes 0-based per message (not source block indexes); gemini/responses index-less calls get stable distinct indexes (no all-0 SDK-merge corruption).
2. **Terminal discipline:** `.done` snapshots never re-emitted as deltas to chat (duplicate content/tool calls); NO `[DONE]` on Responses terminals (6 tests + manual-guide pin the bug — fix per W10 authorship rule); bare-EOF synthesis per #25.5 (repair, zeros, success-only-on-evidence — feeds G4's repair rules); error frames carry client-protocol type vocabulary.
3. **Multi-candidate:** D9 first-wins everywhere (anthropic stream concat is a violation); per-choice finishes survive (hold-back multi-slot).
4. **Signature gating unified:** provider-identity predicate (D8), one implementation shared by stream + non-stream; redacted_thinking to foreign providers gated; gemini media-part signature covered (third bypass site); foreign-source signature replay only same-protocol + same-provider.
5. **Per-protocol synthesis:** anthropic non-whitelisted blocks raw-replay same-protocol (kill phantom empty text; fix the fictional whitelist entry; mcp/container_upload survive); server_tool_use input deltas never fabricate client-callable tool_use (litellm #17798's bug); gemini response-level metadata (responseId/promptFeedback.blockReason/modelStatus) + part-level (videoMetadata/partMetadata/mediaResolution/audioTranscription) + future part types preserved same-protocol; chat streaming audio emitted (`delta.audio`); annotations re-emitted (chat→chat citations); obfuscation passthrough; role per choice; identity preserved (ids/fingerprint/tier lifted, not re-minted — feeds G4 conditional re-serialization); husk frames never open foreign lifecycles; gemini→chat tool calls finish `tool_calls`; usage-only husk frames don't fabricate candidates.
6. Transport-neutrality restored: heartbeats via the transport seam (not hardcoded SSE formatter); WS formatter/converter wired (G11 consumes).

**Files:** `protocols/streaming.py` (single writer), `protocols/openai_chat.py`/`anthropic_messages.py`/`gemini.py`/`responses.py` stream-parse halves as needed for identity/index cooperation.
**Tests:** protocol streaming matrix re-pinned on REAL wire shapes (the fabricated-fixture purge); add: three-block real-shape pin, no-DONE-on-responses pin, signature-gating matrix, metadata-preservation pins.

## 8. G3 — Raw fast path containment + opaque-state strip (L, P0)

**Sources:** #29 containment, #43 raw-gate, #44 (decision: strip on provider switch), #45.1; sweep R2/R5. Detail: sweep R2 (raw items) + R5.

**Deliverables:** provider-identity term in the raw-basis gate (both sites: client/executor.py:871 + native_provider/executor.py:77-102); foreign bound state stripped on switch (stays cached for return); Gemini-3 post-strip function calls carry `skip_thought_signature_validator` sentinel (also at ALL strip/miss/foreign-history sites — #45 directive); unsigned thinking degraded to text or dropped; gemini raw-path illegal `model` key stripped (with G14); `_wire_view_matches_unified` covers input/operation; same-protocol passthrough preserves request warnings + model stamp; session anchors: non-stream anthropic content blocks + gemini candidates (readers ride G6), stream id threading (with G4); count_tokens dead-strip deleted (honest disclosure comment survives).

**Files:** `client/executor.py` (:871 + native context), `native_provider/executor.py`, `field_cache/engine.py`, `client/anthropic.py`.
**Tests:** add: cross-provider signature-strip pin, sentinel-fallback pin, unsigned-thinking degrade pin, anthropic anchor pin.

## 9. G5 — Accounting & cost chain (M, P0 — bug-fixes only; rearchitecture deferred #25.6)

**Sources:** #48 MONEY, #49 cost ledger; sweep R9 (usage cluster). Detail: sweep R9.

**Deliverables:**
1. Anthropic wire double-subtract fixed (convention-aware: subtract only when `input_tokens ≥ read+write`; native path extracts from canonical usage).
2. Gemini accounting: thinking tokens + toolUsePromptTokenCount + authoritative totalTokenCount honored (adopt litellm's inclusion handling); split-usage REPLACE-merge → hybrid (input/cache preserved); re-pin the fabricated warm-cache fixture with realistic numbers.
3. Cost chain: null-fallbacks hardened (all sites incl. responses/service + accounting); `provider_plugin` plumbed everywhere (stream/native/responses); `skip_cost_calculation` honored on all surfaces; litellm total → bucket attribution (not all-output); falsy-zero cache price; plugin-pricing parity across paths.
4. Config pricing: partial env override merges (never wipes JSON siblings); invalid MODEL_PRICE env warns.
5. Fair-cycle: duration expiry actually resets the cycle (default-enabled 7-day window silently stops enforcing!); read-only checks never mutate (exhaustion flag); per-tier priority reset.
6. Windows/limits: model-scoped cooldowns visible to soonest-helpers; `applies_to="credential"` windows enforced or contract removed; `_get_usage_count` no lifetime fallback after rollover; monthly off-by-one + weekly/monthly reset-time; model-window limit sync; status-endpoint exhaustion reporting under credential mode.
7. Storage: usage.json poisoned-field crash chain contained (parse-time validation, quarantine-not-discard, atomic Windows move); quota sync limit-overwrite conditional.
8. Session-state unbounded growth (3000 requests → 3000 sessions persisted) — memory + save caps (rides G6 for the serialization half).

**Files:** `usage/accounting.py`, `usage/costs.py`, `usage/limits/fair_cycle.py`, `usage/persistence/storage.py`, `client/stream_ops.py` (cost), `native_provider/executor.py`, `responses/service.py`, `client/executor.py` (:2661).
**Tests:** usage suites as base (solid pins, keep green); add: wire-vs-canonical anthropic, null-cost matrix, fair-cycle reset pin, poisoned-file quarantine pin.

## 10. G14 — Gemini depth completion (L, P1 — after stream core)

**Sources:** #45, #46, #47; sweep R-GEMINI (all). Detail: sweep R-GEMINI. (Resolved non-items: safetyConfig/contextCompression = Live API only; thinkingLevel lowercase = reference spelling.)

**Deliverables:** partialArgs/willContinue streaming family (args preserved + no duplication; non-stream ValueError → repair); functionCall/functionResponse willContinue/scheduling/parts same-protocol preservation; countTokens nested form correct both directions (build emits envelope XOR contents; ingress parses SDK nested form); raw-path model-key strip (with G3); computerUse/mcpServers/enterpriseWebSearch/exaAiSearch/parallelAiSearch modeled (hosted families mapped or honestly rejected — no fabricated "gemini_tool" names); `skip_thought_signature_validator` at all sites (with G3); structured output: schema case-normalized for chat/responses targets + strict never fabricated + text/x.enum mode preserved; displayName preserved (inlineData/fileData); usage detail arrays + trafficType/serviceTier (non-stream + stream + toolUsePromptTokenCount); per-candidate extras disclosed cross-protocol; chat→gemini citations mapped (homes exist); VALIDATED allowlist preserved; hosted-envelope params mapped; logprobs both directions; responseTokenCount alias + tool-bucket fallback total; VIDEO output modality; candidateCount-on-stream guard; rpc details[] + "retry in" (rides G1); thinking-off model-aware on Gemini 3; fileData mimeType required-with-warning; toolConfig siblings preserved; urlContextMetadata lifted; GET /v1beta/models discovery ingress; false drop-warnings fixed (responseMimeType/responseFormat); empty-candidates → honest error.

**Files:** `protocols/gemini.py`, `client/gemini.py`, `proxy_app/main.py` (gemini routes), `providers/gemini_provider.py` (discovery).
**Tests:** gemini suites as base; add: partialArgs matrix, nested countTokens both directions, sentinel matrix, schema-case pin, discovery-ingress pin.

## 11. G10 — Cross-dialect conversion & disclosure (L+, P0)

**Sources:** #27, #28.4/#28.7/#28.8, #43 mapper drops; sweep R4 + R-CHAT/R-ANTHROPIC conversion items. Detail: sweep R4 + protocol sections. (Stream-side disclosure items live in G13; G10 owns the non-stream + framework halves.)

**Deliverables:**
1. Validation rule softened per #28.8 (unrepresentable = warning-logged drop, not hard reject) + modality/capability tables researched to reality (web-grounded: chat video-in?, responses audio-in — user reports both work).
2. Hosted tools cross-dialect: proper native mappings (gemini↔responses web-search family; responses web_search → anthropic versioned server tools) or honest rejection with disclosure — NEVER a client-callable function with an empty schema.
3. Base-class defaults: `build_request`/`format_response` never ship stale raw when canonical diverged (fail-closed or merge — super()-callers ship pre-mutation payloads today).
4. Reasoning controls: budget clamp sees the DEFAULTED max_tokens (guaranteed-400 today); minimal→anthropic clamped to low (xhigh/max pass through — inverse bug both ways); enabled-only + dynamic four-shape silent drops warned; gemini budget ceiling clamp; thinking×forced-tool-use incompatibility handled.
5. Tool choice: allowed_tools with tool-definition dicts (false-reject + silent widen both directions — extract names; parse the flat Responses spelling); unknown-target echoes never leak canonical shapes (tool choice + structured output); `resolve_tool_result_names` never mutates the caller's canonical.
6. Structured output: text→anthropic returns None (never fabricated empty json_schema — semantic inversion); grammar/custom-tool format carried on ToolDefinition (chat+responses emit; parameters never fabricated for custom tools).
7. Usage/warnings: chat emits `cache_write_tokens` (official spelling; tests re-pinned); usage-bucket exemptions per-target; warn helpers consolidated into canonical.add_conversion_warning (dedup keys aligned incl. field); incomplete→length disclosed; cancelled status mapped; finishReason drift batch (IMAGE_RECITATION/NO_IMAGE/UNEXPECTED_TOOL_CALL/MISSING_THOUGHT_SIGNATURE/ESCALATION).
8. Chat conversion halves: foreign annotation shapes guarded (nested-only emission); file parts preserve prompt_cache_breakpoint same-protocol + disclose cross-protocol; legacy functions/function_call canonicalized (cross-protocol tool loss); custom-tool grammar (with 6); moderation chunks + obfuscation passthrough; input-audio label→MIME; logprobs:false not flipped; empty-choices → honest error (not empty 200).
9. Anthropic conversion halves: mid-conversation system + per-message output_config preserved; tool_choice:none legal; document content-source correct; count_tokens accepts server tools + documents (or honest scoped error — with G14); citation shapes mapped both directions (positions preserved); stop_details on stream refusals; usage members preserved; caller/toolset_name disclosed; model_context_window_exceeded canonical value preserved; delta-type spelling unified.
10. Registry hygiene: register_protocol/transform atomic + collision-fail; LiteLLMFallback wired or unregistered; transport seam enforced or removed; ollama images canonical; non-generative validation + extra-vs-core precedence documented.

**Files:** `protocols/canonical.py`, `validation.py`, `types.py`, `base.py`, `registry.py`, `transforms.py`, the four wire adapters' conversion halves.
**Tests:** w2-w4/protocol suites as base (expect churn); add: cross-dialect drop-disclosure matrix, budget-vs-default pin, allowed_tools matrix, custom-tool grammar pin.

## 12. G8 — Providers & interface (M-L, P1)

**Sources:** #38, #35 env/config review, #41 quirks; sweep R8 (all). Detail: sweep R8.

**Deliverables:**
1. **Identity fixes:** NVIDIA three-way split unified (`providers.nvidia_nim` readable; `providers.nvidia` never bricks startup); mixed-case names normalized/rejected; openai_compatible key unified; example/abstract providers not registered routable; `_retired` imports actually fixed (6 modules).
2. **Discovery fixes:** Mistral envelope (bare array); Cohere host (.com) + pagination; gemini pagination; discovery URLs respect env base + catch HTTPStatusError; static get_models single-bad-entry tolerant; `import litellm` .env side-effect neutralized (registry CWD-stable).
3. **Auth:** sentinel guard on the litellm path (never `__proxy_no_auth__` as Bearer); discovery headers per auth_mode; `_API_KEY` scan fixed in FIVE files (main/launcher/credential_tool/model_filter_gui/settings_tool — anchored regex).
4. **Custom-path providers (porting prep with G9):** DeepSeek fabricated reasoning placeholder removed (verbatim pass-back or strip per official API; all-turns rule); escalation maps corrected (low→high etc. removed; Mistral hardcoded "high" removed — relocation not overwrite).
5. Endpoint resolvers raise on unknown operations; gemini+anthropic count_tokens native operations reachable (local estimate = fallback); profile endpoint placeholders rendered (with G7); supports_native_streaming operation-aware; quota bases unified (firmware/nanogpt split-brain; firmware baseline + unprefixed member); NanoGPT fallback scope narrowed; quota fetch timeouts; ProviderCache env-int crash + no-loop guard.
6. Env/JSON dynamic parity review (with G7): env-only default native; stream/non-stream consistency; env+JSON transport keys validated not import-crashed.
7. Config-reference regenerated from source (rides G12 for docs, source-of-truth fixes here).

**Files:** `providers/*`, `providers/__init__.py`, scan sites, `provider_cache.py`.
**Tests:** provider suites as base; add: nvidia identity matrix, discovery-envelope pins (mistral/cohere), sentinel-auth pin, scan-regex pin.

## 13. G6 — Session persistence + anchor harvest (L, P1)

**Sources:** #48 MEMORY, #49 dedupe/size; sweep R2/R9 session items. Detail: sweep R9 + R2.

**Deliverables:** save-side caps + compact serialization (16MB total-discard → graceful partial load); write amplification down; dedupe: hash-keyed references (same evidence stored once — robust + light); storage size: zstd + per-entry caps (session + field cache + provider cache); protocol-aware anchor readers (anthropic content blocks via G3, gemini candidates — responseId ≠ id; chat id ballast policy); stream anchors: arguments preserved (with G4), id threading, candidate-0 keying fixed; protocol-conditional anchor groups; `msg_` ids under proper group; compaction edges (shared-summary collision, generic markers, probes window); inference WARNING line volume policy; session count bounded (with G5); global-RLock hashing serialization reviewed (D17).

**Files:** `session_tracking.py`, `utils/zstd_io.py`, `field_cache/store.py`, `provider_cache.py`.
**Tests:** session suites as base; add: graceful-load pin, dedupe pin, compression round-trip, protocol-anchor pins.

## 14. G11 — Variants & stores (XL, P1 — window 3, with G16; executed in research-grounded phases)

**Sources:** #33, #34, #35 R8 split, #36 hybrid, #37 taxonomy, #40, #46, #47; sweep R7; operator rulings 2026-09-12/13 (variant opt-in model, first-class WS, two-URL gemini faces, storage interim + engine rework).

**Operator rulings that govern this group (2026-09-12/13):**
- Responses variants are **per-provider opt-in capabilities**: stateless is the automatic baseline every responses-capable provider has; **stateful and WS exist on a provider only when it declares them** (WS is exclusive to one provider in practice; stateful to few). "Provider can support part of the responses — and that is normal."
- **Gemini two faces by URL**: `v1beta/models/...` = native gemini format, `v1beta/openai/...` = Google's OpenAI-compat surface (Bearer auth). The gemini FORMAT is Google's several services (AI Studio, Vertex, Cloud) plus possible third parties — the protocol stays declarable by anyone; the two faces are transport profiles on Google-shaped providers. Default face: native.
- **WS is first-class**: parallel conversations on one connection (lanes/steering) are IN SCOPE, not deferred. **A failure never deletes conversation memory** (fork-eviction ban).
- **Storage is two-step**: durable JSON now (Phase B), then the full storage-engine rework as its own planned group (G17) — compressed storage, append-friendly medium (no whole-store recompression per write), access-time pruning, dedupe; covers every cache AND transaction logs.

**Phase A — The split (research trio first):** three responses siblings as registry entries (stateless / stateful / websocket) sharing a wire base class; provider declarations for variant opt-in (stateless automatic, stateful/WS declared); variant matching via base family (client "responses" matches all siblings → existing ambiguity/default-profile machinery decides); D13 priority stays family-level. Gemini two faces: transport_profiles {native: gemini@v1beta, openai: openai_chat@/v1beta/openai Bearer}, default native; `/v1beta/openai/chat/completions` + `/v1beta/openai/models` ingress bound to the openai profile; drop `/v1/` ingress. Research grounds: official stateful/stateless semantics, WS lane/steering grammar (feeds Phase C), Vertex-vs-AI-Studio endpoint/auth differences (same format, different auth surfaces).

**Phase B — Memory (research trio where needed):** hybrid continuation (#36, gomodel shape): local store first → provider passthrough on miss for same-dialect responses-native targets (provider ids honored, provider-continuation field-cache rules PRESERVED — `_disable_provider_continuation` derives from the chosen path, never from lineage existence); cross-protocol or switched provider → local replay; unknown ids → provider-GET fallback before 404; scope/capability checks stay local and ahead of any fallback (anti-injection); provenance (provider/route) stamped on stored rows. Store default → durable JSON (interim); `expires_at` validated at load (poison-row containment); store-never-fails (one safe-store helper at every site — deliver + log, never kill a finished answer); unbounded default store bounded; created_at not provider-stampable.

**Phase C — WS first-class (research trio: full official WS event/lane grammar):** parallel conversations per connection (lanes, steering, stream_id grammar); failure NEVER evicts conversation memory (per-lane parent tracking via the written-never-read LaneState); warmup scope from frame routing kwargs + validation + tool preservation; error frames spec-shaped (inner error.type, invalid_stream_id code); capture-on-error fires on WS turns; turn handler goes through the formatter; frame-size cap; lanes bounded; `MAX_CONNECTION_SECONDS` isfinite; the WebSocketStreamFormatter transport seam wired (G13's hand-off).

**Phase D — Polish (with G16):** bridge deleted entirely (fabricated-completed-on-truncation dies with it; PROXY_ROUTING_KEYS relocated); sequence-number domain unified per stream (burns fixed — one counter, not the module global); finally/aclose on the native stream loop; failure terminals correlate ids (no fresh minting); `[DONE]` asymmetry resolved; store_in_progress honored or removed; DELETE object literal; HTTP cancel route implemented for real (gomodel shape: stored row → provider cancel via provenance → store update; unknown → 404; scoped stays local; WS cancel stays rejected per current docs).

**Files:** `responses/*`, `protocols/responses.py`, `providers/gemini_provider.py`, `providers/__init__.py`, `proxy_app/main.py`, `routing/profiles.py`, `config/experimental.py`.
**Tests:** responses suites as base (bridge tests deleted by design); add: variant matrix, hybrid continuation pins, gemini openai-profile passthrough pin, store-never-fails pin, WS lanes/steering matrix.

## 15. G16 — Responses synthesis SDK-conformance (M, P1 — window 3, executes as G11 Phase D)

**Sources:** sweep R7 (synthesis cluster). Detail: sweep R7.

**Deliverables:** every proxy-synthesized Responses object SDK-valid (required fields: created_at/parallel_tool_calls/tool_choice/tools; usage detail objects not zero-omitted — official SDK model_validate passes on OUR failure paths); input_items correct envelope + normalized items + pagination params (first_id/last_id/has_more + after/limit/order/include); error objects use official code VOCABULARY (strings, not numeric status); capability token per-scope (chain access stable across turns — per-create rotation orphans); WS error frames + limit frame per spec; native top-level `error` stream event terminal (no double terminal); provider completed-without-object handled; poison-row containment (with G11-B); sequence burns (with G11-D); store_failed default documented true (config-reference fixed). No vendor SDKs are used anywhere — conformance is about OUR output shapes (operator Q&A 2026-09-13).

**Files:** `responses/streaming.py`, `responses/service.py`, `responses/websocket.py`, `responses/store.py`.
**Tests:** SDK-model_validate pins for every synthesized object family; capability-token chain pin; error-vocabulary pin.

## 15b. G17 — Storage engine rework (L, P1 — RESEARCH AND PLAN FIRST; operator directive 2026-09-13)

**Operator directive:** "Rework of all the cache needs to be done to compress everything stored, select a better medium. The cache will balloon too much (it already does). Maybe compress with zstd? Need a format that allows this natively, WITHOUT recompressing the whole thing again. And decent pruning — delete entries not accessed for X amount of time. Same applies to transaction logs."

**Scope:** every persisted store (provider cache, field cache, responses store, session state, quota/usage state, transaction logs). Mandatory research phase (trio, exhaustive, docs-grounded) BEFORE any planning decision: medium candidates (embedded DB e.g. stdlib sqlite vs compressed append-segment files vs hybrid), access-time-based pruning (LRU-by-touch semantics), compression strategy per medium (row-level zstd vs page-level vs segment compaction), dedupe (identical payloads recorded once — operator note from R9), migration from current JSON files, retention caps. Then a full plan presentation to the operator for approval BEFORE implementation. Existing CACHE-dedupe/size items from R9 fold in here.

**Position:** after G11 phases; before or alongside G8/G9 (which are read-heavy on cache surfaces) — exact slot decided at its planning presentation.

## 16. G9 — Provider porting pass + embeddings (XL, P1 — after skeleton)

**Sources:** #38 standing, #26.2, #41, #25.1/#25.2; sweep R2 (embeddings blocker). Detail: sweep R2 + R8.

**Deliverables:**
1. **Embeddings FIRST-CLASS (#25.2 — blocker priority):** operation-aware RequestContext (or dedicated execution entry); native gate declines embeddings to an embeddings-capable transport (never configuration_error 500); litellm branch calls `litellm.aembedding`; custom branch calls `plugin.aembedding`; interim guard until first-class: force litellm for non-generative; FIRST embedding tests (synthetic, all paths); batcher (first-char + usage multiplication) fixed or deleted with the feature decision; `request`-key collision guard on the embeddings route.
2. Every live provider reviewed/ported against the native+hooks runtime (G1+G2+G7+G8 as prerequisites); thinking-handlers keep/migrate/kill executed; #41 quirk table applied; native-path transform parity decided per transform (currently silently skipped — gemma system→user dead on native groq/openrouter).
3. litellm `drop_params` scoped + dropped-set disclosed (D7 surface on the fallback path).

**Files:** `client/executor.py`, `client/rotating_client.py`, `client/request_builder.py`, all provider files, `route_helpers.py` (embeddings route).
**Tests:** add: embeddings all-paths matrix, transform-native parity pins, provider porting checklists.

## 17. G12 — Suite + docs (L, P2 — continuous + closing)

**Sources:** #17, #19 leftovers, #28.5, #50, #51; sweep R10. Detail: sweep R10.

**Deliverables:**
1. 33 untracked test files: delete-by-default (user ruling) but each deletion paired with replacement pins where it was the only coverage (gate bar #17).
2. Full suite review: outdated tests deleted/rewritten; FakeCredentialContext ×3 unified; `_context()` ×9; sys.path boilerplate ×20; private reach-ins; flaky sleeps; real-network script removed; stale __pycache__; registry pollution teardown (register_* replace=True leaks across files).
3. **Wrong pins fixed (~20 explicit items — the suite institutionalizes known bugs):** Responses [DONE] ×6 + manual-guide; store-failure-RAISE ×2; anthropic input-on-delta fixtures; fabricated costDetails + bare-[DONE]-to-gemini; index-less three-block events; dotted message.delta; transport==sse in the WS formatter; drifted keep vocab; old D13 rule; redacted-thinking signature-shaped fixture; effort-minimal; wav + empty-ids; cache_creation_tokens; thinkingLevel case (resolved — keep lowercase); tautology + grep-the-source tests; custom-caps three-way contradiction (doc/code/tests).
4. Coverage additions named by the sweep: embeddings (rides G9), multi-chunk n>1, client-wire usage shapes, capture-set alias coverage (non-self-referential), R3 routing highs, registry-vs-engine pooling, insert+auto, transform-native integration.
5. Vocabulary scrub: 121 dangling ledger refs across 31 files → inline self-contained summaries (plan doc optional). 
6. Docs: config-reference regenerated from source (phantom knobs: REQUEST_LOGGING_ENABLED/ADVISORY_MODEL_PRICING/FAIR_CYCLE_EXHAUSTION_THRESHOLD/ROTATOR_LIBRARY_CONFIG/SKIP_OAUTH_INITIALIZATION; wrong defaults: GLOBAL_TIMEOUT/FAIR_CYCLE_DURATION ×3/MODEL_INFO_INTERVAL/RESPONSES_STORE_FAILED; HOST/PORT; hot-reload premise; capture-set; WS path; missing families); manual-test-guide fixed BEFORE R11 (body-key-as-header, test count, [DONE] instruction — it's the live-round instrument); `.env.example` (dead paragraphs/knobs out, missing in); ARCHITECTURE.md count_tokens one-liner; README/DOCUMENTATION structural fixes only now (full redo at end of PR life #51.2).

**Files:** `tests/*`, `docs/*`, `.env.example`.
**Tests:** the suite IS the deliverable; acceptance = every fix-pass group's pins green + zero known-defect pins remain (grep-able list).

---

## 18. Ledger annotations (corrections that govern over the raw remediation file)

1. `x-proxy-conversion` is a response-**body** extension key, not an HTTP header — G4 scope; decide `x-proxy-estimate` policy in the same pass.
2. Live pipeline HAS direct functional tests; the gap is specifically **timing tests** (they drive the retired handler).
3. Session anchors ARE capped in memory; the defect is uncapped save + load-only caps + all-or-nothing 16MB discard.
4. `main.py:1777` dead-add_credential/NameError is CORRECT (reachable via TUI argv-mutation; second site :1802).
5. `routing/profiles.py` docstring AND implementation both state the OLD rule — G7 implements the revision.
6. quota-as-400 compounds: non-rotatable + FAIL-swallowed + counter-reset defeat + routing-layer re-derivation. All → G1.
7. wav fabrication is the missing/empty media_type case; URL-only audio loses its URL by the same line.
8. Usage zeroing is the late empty-object case, not missing usage.
9. `_API_KEY` scan exists in 5 files (settings_tool was the missed fifth).
10. `.git/info/exclude hides tests/` was machine-local/stale phrasing — no work item.
11. Sentinel-cost inversion: `skip_cost_calculation` honored non-stream only.
12. `_usage_record_has_values` dead — cleanup.
13. WS `response.cancel` is NOT a client event per current official docs — the earlier "implement cancel on WS" directive is DROPPED; HTTP cancel rides the background-mode decision.
14. Anthropic `output_tokens_details.thinking_tokens` wire spelling is CORRECT per current SDK — no change.
15. Gemini `thinkingLevel` lowercase is the reference spelling — resolved, no live-verify needed.

## 19. Explicitly deferred / out of scope

- Usage-manager + cost REARCHITECTURE (post-PR, #25.6) — G5 is bug-fixes only.
- Stateful chat completions — NEVER (#40).
- Standalone images/audio APIs — NOT wanted (#28.2). Embeddings first-class remains wanted (#25.2 — in G9).
- README/DOCUMENTATION.md full redo — end of PR life (#51.2).
- Fallback-groups feature review + group kinds + GitHub fallback wishes — separate later review (#26.1); fix focus stays the no-groups path.
- Config-file-first migration (env stays primary this pass; #26.3 recorded for post-PR).
- Per-provider concrete implementations (Codex, Claude Code, Antigravity, Copilot, Gemini CLI) — user implements manually (#795); antigravity is the G2 hooks stress test.
- Responses background mode — real lifecycle work (task + polling); rejection stays honest until then.
