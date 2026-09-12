# Audit Remediation Plan — Input Ledger

**Provenance:** Full audit of `experimental` vs `dev`, 2026-09-10, rounds R1-R10 (shell, executor, routing, protocol core, native execution/adapters/field cache, streaming, Responses+WS, providers, cross-cutting, tests/docs) plus per-protocol depth rounds (chat, anthropic, gemini) and reference comparisons (plexus + gomodel). Every load-bearing finding personally verified against source by the primary engineer; agent findings marked where not re-read.

**Status:** RAW INPUT — notes #17-#51 transcribed verbatim below (the durable audit queue). Next step: group into workstreams + write the final plan state; then execute group-by-group in build mode with a commit per group. House rules for this pass: plan files + replication tests only; no implementation without explicit user approval per group. R11 (live round with user) still pending; after the fix pass the ENTIRE AUDIT RUNS AGAIN.

---

## #17 — Gatekeeper closure bar (standing rule, 2026-09-07)
A stage passes ONLY when ALL reviewer findings — BLOCKER through NIT — are fixed or explicitly ignored after stated consideration with rationale recorded. "SATISFIED with mediums/lows open" is NOT a pass. Outcome bar: theoretically perfect protocols + translations.

## #19 — Audit anomaly queue (5-mapper inventory, verify per round)
- R1 (shell): main.py misindented format_client_protocol_error continuation in chat_completions JSONDecodeError handler; stray `# noqa: F811` on nested transaction_logger_factory in WS route; launcher_tui empty env_oauth_providers loop; ?key= query-param auth may leak into access logs (detailed_logger redacts headers only); DEFAULT_PROXY_API_KEY hardcoded (intentional/user-approved).
- R2 (executor): RoutingExecutionError local-only type; ProcessedChunk docstring stale.
- R3 (routing): dead FallbackAttemptRunner/FallbackExhaustedError (never called in src); `:free`/`:tag` colon-IDs vs grammar.
- R4 (protocols): types.py:399-409 stray duplicate docstring; format_stop_reason falls through returning raw value when docstring claims table's unknown entry; near-duplicate SSE decoders (shims — refuted as duplication; real one is _without ×4); protocols/__init__ missing Annotation+BuiltinToolCall exports; _warn_*_once dedup helpers duplicated despite canonical.add_conversion_warning; internal W/D/H ledger vocabulary in production comments opaque without docs.
- R5 (native/cache): cache_replay + transport_profiles declared but zero in-tree provider users (verify deliberate seam).
- R6 (streaming): client/streaming.py self-deprecated but load-bearing, stream_ops reaches into its private helpers (_sse_data_payloads, _usage_record_from_sse_cost_chunk); stream_retry_policy.py pure shim; _MIXED_REASONING_CONTENT_MODELS hardcoded model allowlist (google/diffusiongemma-26b-a4b-it + nvidia_nim alias).
- R7 (responses): bridge.py naming now historical — verify no stale docstrings.
- R8 (providers): nvidia DiffusionGemma feature rides in refactor PR (user-approved); providers/__init__ inline `import logging` twice; DynamicOpenAICompatibleProvider adds Content-Type on GET /models; README stale link to providers/google_oauth_base.py (moved to _retired).
- R9 (cross-cutting): session_sticky_entry_ttl_seconds 3600→300 default + max(1)→max(0) where 0 disables stickiness — silent behavior change; core/types.py ProcessedChunk "Used by StreamingHandler" drift.
- R10 (meta): .git/info/exclude hides tests/ locally (~30 files incl. quota JSONs never to be committed); untracked docs/experimental/protocol-interoperability-second-opinion.md duplicates _superseded copy — stale draft outside archive; docs/refinement-notes.md scope creep (user-requested, keep).

## #20 — R1 shell & entry verified findings
INTRODUCED:
- [H] key_policy.py:81-83 `python -m uvicorn` / programmatic uvicorn.run bypass default-key policy on public binds (argv0=__main__.py) — fix: sniff sys.argv for module-launch markers or import-time uvicorn frames.
- [M] main.py:1707-1711 cost-estimate malformed JSON → 500 FastAPI envelope; quota-stats POST + token-count non-dict same class.
- [M] W6/W9 incomplete: litellm→status mapping ladders duplicated ×4 routes (count_tokens missing RateLimit/Timeout/ServiceUnavailable → renders 429/504 as 500; gemini maps all non-invalid → 500); responses route inline-duplicates SSE_HEADERS; get_responses_service duplicates lifespan construction + silently fabricates fresh store; WS env parsing in shell; cost-estimate fallback shaping in shell.
- [L] /v1/responses :813 non-dict JSON body → AttributeError → 500.
- [L] main.py:1777-1784 dead duplicate add_credential block (NameError if reached).
- [L] launcher_tui.py:763 reset dialog prints key[:20] + padding length (TUI frozen — log only).

PRE-EXISTING (fix-pass candidates): [M] route_helpers.py:94-113 double log_final_response on stream error (500 record then bogus 200/{} in finally; non-chat logs empty body even on success); [M] auth 401s = {"detail"} envelopes on ALL routes (contract: client-protocol shape); [L-M] /v1/chat/completions stream missing SSE_HEADERS; [L dormant] batch_manager first-character slice; [nit] module-level cls+banner on import.

VERDICTS: D1@shell PASS; thin-shell PARTIAL; W6 error lifecycle PASS generative / FAIL management+auth; key policy PARTIAL (-m bypass, env-var shadow re-prompt trap); startup.py faithful PASS; lazy-startup PASS.

## #21 — R2 client layer & executor verified findings
VERIFIED GOOD: execution-mode dispatch (auto=custom-first>native>litellm, explicit fail-closed); fallback chain mechanics (ordered targets, attempt history, visible-output fail-closed incl. malformed-chunk conservative); cooldown gating pre-output only; single accounting; credential redaction.

FINDINGS:
- [M, pre-existing dev-identical] ErrorAction.FAIL raises swallowed by `except Exception: pass` in BOTH loops (exec :1596→:1609, stream raise sites→:2258) → deterministic 400-class litellm errors hammer every credential; non-stream 3-consecutive-quota abort defeated; StructuredAPIResponseError escapes correctly.
- [M] stream_ops.py:732 CostCalculator() built WITHOUT provider_plugin (non-stream passes it) → plugin-priced models get different approx_cost stream vs non-stream.
- [M] _safe_field_cache_override (exec :3340-3381) compares everything EXCEPT `enabled` → same-name override with enabled=False silently disables provider isolation rule; docstring "never crashes" contradicted by raise.
- [M introduced] embeddings session-hook (request_builder :411→:146) receives raw SHARED kwargs — hook mutation corrupts executed payload; contract says provider-protocol view.
- [M-L latent] _should_use_native_protocol (exec :3048-3060) profile-blind vs profile-aware _build_native_provider_context → auto hard-fails instead of litellm-fallback when profile operation unsupported; unguarded hook call.
- [L] NoAvailableKeysError non-streaming → "unknown" → no failover (streaming failovers via server_error backstop).
- [Nit] proxy_error/proxy_busy frames labeled server_error in traces (outcome correct).
- SMELLS: clone usage_manager_key defaults to bare provider (scoped group key dropped); fallback-identity warning per-clone not per-request; shallow kwargs copies share nested structures across retries/targets; aembedding pops none of the internal plumbing keys; broken routing config fails every request ungracefully.

FLASH CLAIM REJECTED: "proxy_error frames marked success / chain never advances" — FALSE (`or "server_error"` backstop exists).

## #22 — Fix-pass directive: proxy = thin example consumer
The proxy is a THIN EXAMPLE CONSUMER of the library — product intent is "library handles everything feasible; proxy stays minimal". All R1 findings of logic living in proxy_app (mapping ladders ×4, WS env parsing, cost-estimate fallback shaping, SSE header duplication, get_responses_service duplication) are INTENT violations, not style nits. Post-audit workflow: fix pass first, then THE ENTIRE AUDIT RUNS AGAIN.

## #23 — Requirements added (R2-redo)
1. ERROR-CONTRACT EXTENSIVE REVIEW: per-protocol error shapes/classification exhaustively verified via web search + official docs — classification buckets, retry-after/timer extraction, status-code mapping, per dialect. Dedicated round.
2. RECONSTRUCTOR: delivered as standalone script vs intent — on-demand quick-access per-transaction shortcut (plan promised generated reconstruct.bat per transaction) using the LIBRARY's own mechanisms. Re-examine in fix pass.
3. RAW-PATH STRIPPING WIRING: pluggable always-strip surface (protocol-invalid fields, per-provider strips) — declarative rules + hooks on the raw fast path.
4. HOOKS/CALLBACKS SYSTEM: first-class hook surface for providers + operators — override/rewrite system prompt, prompt parts, tools; inject tools; resolve tools; transform-heavy providers. Antigravity provider (user implements manually) is the stress test.

## #24 — R3 routing & profiles verified findings
- [M by read] request_builder.py:111-121 _with_request_scope reconstructs each scoped RouteTarget WITHOUT `profile` → config fallback target `provider:profile/model` silently loses explicit protocol. Fix: profile=target.profile.
- [M] execution_profile consumed ONLY in _build_native_provider_context — litellm/custom path silently ignores explicit profile.
- [M-L] resolver raises bare ValueError from RouteTarget.__post_init__ on bad @mode — escapes RoutingConfigError handler → unhandled 500-class.
- SMELLS: config parser doesn't validate profile names; no per-segment whitespace strip; provider case handling inconsistent; MODEL_ROUTE alias `-`/`_` asymmetry; no group target dedup (promoted group can retry failed target); priority/weight dead surface; per-request re-parse + fail-open; compat parse doesn't strip profiles from config member refs; split_profile_from_provider dead; FallbackAttemptRunner divergent error conversion.
- VERIFIED GOOD: `:free`/`:tag` survival; bare-name fast-path-or-error; @+profile composition; IDENTITY NORMALIZATION — all runtime identity sinks receive bare provider, profile recorded as provenance; no cycles.
- ERROR-SURFACE MAP: gaps — no active parse_quota_error implementations; quotaResetTimeStamp ISO unparsed; HTTP-date Retry-After skipped; Anthropic {"type":"error"} without error key uncaught; unsupported_operation has no producer; configuration_error missing from http_status map (→502) and rotates instead of stopping; three near-identical classification ladders; timers not extracted at classification time; dict-branch never populates quota_reset_timestamp.

## #25 — R2-redo directives + confirmed work items
1. acompletion(input_protocol=...) real parameter replaces the stamped _input_protocol key.
2. EMBEDDINGS FIRST-CLASS: same treatment as completions — full protocols, same machinery, adapted. Major work item; no reference providers to test against — synthetic test strategy needed.
3. AUTH ERRORS MUST NOT STOP the fallback chain. Expand error review to per-error-type routing/credential behavior (rotate vs failover vs stop vs cooldown scope) — full decision matrix.
4. FIELD-CACHE ON FAST PATH: verified WORKING (injection before basis selection; basis switch recorded as unified_state_injection overlay). No action.
5. CLIENT-FACING NORMALIZATION LAYER: missing finish AND/OR usage is NOT fatal — repair and pass forward (tools emitted + missing finish → tool_calls; no tools + missing finish → stop; missing usage → carry what exists, never error). ALL paths incl. fast path ("fast path = minimal work, NOT zero work"). Today missing-finish-no-usage wrongly treated as error — needs the repair pass.
6. USAGE MANAGER + COST CALCULATION: major rework NEEDED, separate from this PR, AFTER main work.
7. DESIGN CHOICES TO REVISIT: acompletion naming; stamped-key → parameter; isolation key derived twice; embeddings outside protocol machinery; overlay-diff drift detection indirect.

## #26 — R3 directives
1. FALLBACK GROUPS/ROUTING FEATURES = NOT the point of this PR — separate later review of ALL group kinds + GitHub issues for fallback wishes; audit focus = no-groups path.
2. LITELLM = LEGACY, to be removed. Explicit-profile-on-litellm finding downgraded.
3. CONFIG DIRECTION (post-PR): move off env fully → real config file primary; env compatibility only; default = no .env; .env = small overrides + keys at most. Broken config: validate, REJECT, keep last-good (never fail-open, never partially apply).
4. CONVERSION RULE — FINAL (revises D13): bare-name protocol resolution uses PRIORITY LIST when nothing matches client dialect: openai_chat → responses → anthropic_messages → gemini (first offered wins). Eliminates "endpoint does not exist" for bare names. Terminal WARNING when priority-default used. Explicit profiles still FAIL LOUDLY. Implement in resolve_profile + warning surface.
5. Accepted evaluations: fail-open wrong; priority/weight dead (wire or remove); duplicate runners consolidate; bare ValueError escape fix.
6. @native = routing-config syntax only; first '@' partitions suffix; @auto inferred by absence; clients never send @.
7. Plan mode long — notes schedule all file writes; notes are the durable fix queue.

## #27 — R4 protocol core verified findings
PERSONALLY VERIFIED:
- [M-H semantic corruption] Gemini hosted tool (googleSearch) → Anthropic target silently becomes CLIENT-CALLABLE function tool {"name":"googleSearch","input_schema":{}} — model emits tool_use the client must answer instead of server execution.
- [M silent drop] Redacted thinking → Chat target vanishes silently (join drops textless blocks; "reasoning" excluded from unknown-block warning set).
- [M] Stream formatter drops media + citations silently for 3 of 4 targets (anthropic media → phantom empty text block; responses fabricates empty output_text.done). Non-stream discloses; stream has NO validation backstop — structural root cause.

AGENT-FOUND (consistent): video→3 targets hard reject (honest); redacted→gemini fabricates empty thought part; hosted→Responses hard reject despite native web_search equivalents (asymmetric); Responses emits encrypted_content ungated (D8 inconsistency); chat URL-only audio source loses URL; logprobs cross-protocol silent; foreign builtin→Responses synthesizes non-catalog item types; multi-candidate→anthropic stream merges without disclosure; responses modalities validation/build disagree; anthropic+responses _format_content unknown-type cross-protocol silent drops.

HYGIENE: ledger references in production comments dangle (defining doc untracked); types.py stray docstring; missing exports; format_stop_reason latent fall-through; _warn_*_once dedup divergence; _without ×4.

NON-GENERATIVE DORMANT (intentional): ollama/audio/embeddings/images/mcp registered but unreachable (config allowlist restricts to generative four; /v1/embeddings bypasses protocol layer). Activation = relax allowlist + endpoints. Ties to embeddings-first-class.

RESIDUALS: W3-1 untraced instruction repositioning CONFIRMED (+test pin); W3-3 gemini raw-basis model key drifted parenthetical (canonical strips, raw carries, comment inaccurate); W4 stream-summary vehicle absent client-side CONFIRMED; W4 summary→includeThoughts + adjacent-turn merge CONFIRMED; W2 server_tool_use FIXED/LANDED.

## #28 — R4 directives + dispositions (revised)
1. OLLAMA: add as supported protocol (wire beyond dormant adapter).
2. [REMOVED BY USER] standalone images + audio API first-class — NOT wanted. Embeddings first-class REMAINS.
3. Session detection check queued (done in R9 — see #48).
4. R4 dispositions: (a) hosted-tool cross-dialect mistranslation → FIX (map properly or reject with disclosure); (b) redacted/encrypted reasoning at foreign targets → vanish BUT disclose; recoverable via field cache when it passed through us; (c) NEVER fabricate invalid reasoning fields (kill gemini empty-thought).
5. VOCABULARY SCRUB (whole repo): replace ALL internal ledger references in comments with inline self-contained summaries; plan doc becomes optional.
6. R4 evaluations accepted: stream formatter disclosure discipline; consolidate warn helpers; embeddings-first-class YES, images/audio NO.
7. MODALITY/CAPABILITY RESEARCH: user reports chat models DO take video input, Responses DOES take audio — tables likely WRONG/conservative. Extensive web-grounded research per protocol; extend tables to reality.
8. VALIDATION RULE CHANGE: unrepresentable content cross-protocol = WARNING-LOGGED DROP, not hard protocol-formatted rejection (D7 reject rung softens to disclosed drop; applies to validation.py + tests).

## #29 — R5 native execution + adapters + field cache verified findings
PERSONALLY VERIFIED:
- [M-H CONTRACT] finalizer (prepare_native_request) runs BEFORE request adapter chain + field-cache injection; contract says envelope/finalizer LAST (envelope-adapter conflict documented in builtin warning).
- [M CONTRACT] terminal `done` stream event bypasses stream adapters AND both field-cache extractions.
- [M CONTRACT] streams NEVER extract from the assembled response — per-frame partial only; "capture from finalized assembled response" unmet on streams.
- [H] failing field-cache rule RAISES (no containment) → any rule error kills the request. Caching must never fail a request.
- [H] _shared_cache_signature omits compatibility+transform → portable rule sharing cache_key with bound rule launders bound state across models (engine-layer hole; merge-layer guard checks these).
- [M VERIFIED] parse_quota_error DEAD on native path (error_type branch pre-empts + body plumbing mismatch: .text/.body vs dict) — Google-RPC body timers unreachable for native errors. THE antigravity-relevant hook.

AGENT-VERIFIED: D11 provider+model floor unenforced on raw plugin rules; config-group exclusion unimplemented; different-name same-inject-path rule bypasses weakening guard; SSE decoder emits id:/retry: garbage chunks; native streams not unconditionally closed on disconnect; no transport-owned timeouts; post-injection request extraction can echo injected values back; stream path records no transport overlays; paths.py inconsistencies; replay.py silently swallows malformed; inherited_from dropped from traces; ProviderCacheFieldStore TTL never evicts + corrupt JSON fatal; adapter transaction_logger=None suppresses traces; TypeError-swallow; 3xx → invalid_request.

HOOKS MAP: native-path extension = declared protocol + adapter chain + prepare_native_request + validate_request (zero implementers) + field-cache/cache_replay + session hints (zero implementers) + caller callbacks (canonical-field overlays only). WISHLIST: declarative strip PARTIAL (no strip primitive); prompt override ABSENT; tool inject PARTIAL (code-only); tool resolve ABSENT; envelope unwrap PARTIAL; quota parsing DEAD on native. ProviderTransforms SKIPPED entirely on native (thinking handlers dead for flipped providers). apply_sync dead; model_override fallback dead.

VERIFIED GOOD: W7 staging both directions; raw fast-path deepcopy discipline; injection before send; fail-closed operation/streaming checks; SSE decode robustness; header hygiene; retry-safety.

## #30 — R5 directives
[HIGH PRIORITY — CORE DESIGN] HOOKABLE PIPELINE: before/around EACH native-execution pass (all ~9 stages both directions) a callback/hook spot. Provider/plugin/filter takes the stage payload, modifies it, passes it back; pipeline continues. UNIVERSAL for providers/plugins/filters. HOOKS CAN OVERRIDE BASICALLY EVERY PART OF THE STAGE — base URL, any parameter, request AND response; what's editable depends on hook position and medium. Field-cache injection NOT last — a post-injection hook can veto/edit cached fields. Adapters become a specialized consumer; finalizer/validate_request fold in. Needs concrete stage inventory + registration surface (provider class / config / global registry).

## #31 — R6 streaming e2e verified findings
PERSONALLY VERIFIED:
- [BUG n>1] stream_ops.py:286-287 — meaningful-usage path returns ONLY events[0] (sibling choices' content dropped when usage rides last content chunk; hold-back filter also drops delta-less siblings' finish reasons; terminal branch correct).
- [BUG anchors] _merge_streamed_tool_arguments fragment+snapshot concatenation → invalid JSON anchors (duplicated in two copies).
- [BUG narrow] cost fallback chain defeated by explicit null (dict.get returns stored None).

AGENT-VERIFIED: TTFB/stall/heartbeat/disconnect tests drive RETIRED StreamingHandler — live pipeline's duplicated helpers have ZERO direct coverage; decide_streaming_error_action exported+tested but executor re-implements inline; bare EOF = success+partial usage+synthesized terminal (aligns with recoverability ruling; Q: success-mark on possibly-truncated — log-only); disconnect race discards final event; error-rotation mark_failure carries no tokens (quota under-count); usage merging is REPLACE not merge (late empty zeroes); request deadline not enforced in stream loop; JSONDecodeError swallowed; trace_metrics dead knob; stale credential id in traces; failed-with-output traced as success; dead divergent copies; _collect_anchors hardcodes choice 0; id-less tool calls skipped; _MIXED_REASONING_CONTENT_MODELS dead gate; route logger misses event: frames; non-JSON frame drift.

CHECKLIST VERDICT: ALL 8 preserved-machinery items VERIFIED ENFORCED on the live pipeline (TTFB/stall/heartbeat/cancel-disconnect default TRUE/metrics/cost precedence+frames/completion gate/sane defaults).

## #32 — R6 directives
TAIL WARNINGS DECIDED: conversion summaries go to LOGS ONLY, never to the client. Remove the x-proxy-conversion client-visible response-header on non-streaming responses (fix pass; update attach_conversion_summary consumers + tests).

## #33 — R7 Responses API + WS verified findings
PERSONALLY VERIFIED:
- [BUG narrow] sequence-number domain mixing on native stream path (raw frames verbatim vs synthesized response.failed from module-global counter → per-stream monotonicity violation).
- [M hygiene] native stream loop has no finally/aclose — teardown GC-deferred.
- REFUTED: "no timeouts on native path" — enforcement lives in executor pipeline (heartbeats literally flow through the service loop).

AGENT-VERIFIED: parseable row with non-numeric expires_at raises outside corrupt-row guard; RESPONSES_STORE_IN_PROGRESS no-ops on native path; store failure converts COMPLETED stream into response.failed (policy decision); conversation-id dropped on legacy seam; WS _turn hand-builds frames bypassing formatter; no frame-size cap; lanes unbounded; LaneState.latest_response_id dead; warmup pins scope=public (misleading 404s); warmup previous_response_id unvalidated; created_at gaming; InMemory TTL lazy-only; reset_stream_sequence footgun; legacy disconnect skips metadata finalize; main.py route catch-all returns str(e); no agenerate fallback AttributeError path.

VERIFIED GOOD: lifecycle validation on all four entries; cross-scope blocked at every lineage node + fail-closed 404s with hints; double-continuation prevention; capability system solid (token, hash-only, constant-time, migration exact-match); store_failed both paths; WS aclose chain complete; per-turn scope re-derivation; error frame shapes; response.cancel clean rejection; SSE formatter grammar; write-before-yield; no per-request service state.

## #34 — R7 directives
1. STORE-FAILURE POLICY: deliver the response + log the storage failure; never convert completed stream/create into response.failed over a store error. THE STORE MUST NEVER FAIL — resilient best-effort persistence.
2. RESPONSE.CANCEL: implement for real on WS.
3. BRIDGE REMOVAL: delete the Responses↔chat bridge entirely (to_chat_kwargs, from_chat_response, legacy stream_events surface, parse_chat_sse_chunk consumer, hasattr dead branch) + fix stale "chat bridge" docstring + "bridge_context_expanded" trace label on native path.
4. STORES DEEP-DIVE requested (after audit). Fact: responses store defaults to MEMORY; durable opt-in. User wants persistent-by-default considered.
5. HEADER SYSTEM explained (X-Proxy-Session-Domain = proxy extension; per-scope capability token).
6. PRESENTATION LESSON: no dense arrow-chain sentences — numbered plain steps.
PERSONAL-STUDY ADDITIONS: all agent findings confirmed by direct read; NEW: stale bridge vocabulary mislabels native paths; warmup public-pinning reason identified (WS carries no scope headers — fix = derive from frame body routing kwargs); WS _turn hand-builds frames.

## #35 — Audit parts map + R8 directives
R1-R8 DONE (see respective notes); R9/R10 done subsequently (#48/#50); per-protocol depth rounds done (R-CHAT #40/#41, R-ANTHROPIC #43, R-GEMINI #44/#45); R11 LIVE ROUND pending with user; then fix pass; then ENTIRE AUDIT AGAIN.
R8 DIRECTIVES: [HIGHEST PRIORITY] SPLIT RESPONSES INTO 3 SIBLING PROTOCOLS in the registry — stateless / stateful(id-continuation) / websocket — providers declare which variants they speak. Env-dynamic providers must NOT be chat-limited — full protocol choice; REVIEW whole env+JSON config surface as a system.

## #36 — R7 research verdict + decisions locked
Responses continuation = HYBRID, PLEXUS-FIRST (local store + replay + scope/capability + streamed-turn storage) TAKING from gomodel: same-protocol native-target passthrough (provider ids honored, encrypted-reasoning/cache continuity preserved — FIX current design throwing provider continuity away on EVERY continuation incl. same-provider; foreign-shaped ids → provider passthrough, ours stay local; GET falls back to provider for untracked provider-ids). previous_response_id serves id-continuation clients (delta+id, NOT full history). WS: connection-local cache covers continuation. WS-for-Responses = separate convertible protocol, high priority.

## #37 — Responses taxonomy
Always split Responses into 3 variants: (1) STATELESS (full history resend, no ids); (2) STATEFUL (id-continuation — server-side state at whoever terminates; hybrid per #36); (3) WEBSOCKET (connection state, local turn cache, ZDR). Label which variant is being touched.

## #38 — R8 providers & interface verified findings
PERSONALLY CONFIRMED:
- [HIGH] mixed-case JSON provider keys: registration lowercases, runtime lookup exact-case → config silently lost/ValueError; name regex allows uppercase.
- [HIGH] auth_mode:none sentinel __proxy_no_auth__ leaks as Bearer on litellm-fallback path + hardcoded Bearer in model discovery; native path correct.
- [M] 9 flipped providers' get_native_endpoint ignores `operation` (safe only via supports_operation gate).
- [M] gemini count_tokens endpoint unreachable via generic routing (get_native_operation never returns it); local estimate instead.
- [M] 7 providers hardcode /models URL (env override ignored for discovery) + catch only RequestError (HTTPStatusError escapes).
- [M] main.py:385 `"_API_KEY" in key` substring scan → *_API_KEY_BACKUP registered as LIVE credential (same in launcher + credential_tool).
- [M] config-reference documents ROTATOR_LIBRARY_CONFIG + `adapters` — NEITHER exists (code: LLM_PROXY_CONFIG_FILE; adapter_names). model_protocols acceptance table (D13 promise) zero src hits.

AGENT-VERIFIED: interface calls undeclared hooks (hasattr-guarded); example+openai_compatible over-registered as routable (one crashes); query-key blocklist misses bare token/key; dynamic providers bypass bind_runtime_config guard (naming mismatch); anthropic path convention split (/v1/messages vs /messages); base-URL accessor naming split; quota-base vs chat-base split-brain (firmware/nanogpt); dead tier wiring + divergent quota-group conventions; hardcoded thinking-pattern lists rot-prone; NanoGPT 3-call discovery + unlocked singleton mutation; quota jobs new client per run + firmware reaches into usage_manager; malformed interval env crashes instantiation; empty-string env registers dynamic; OPENAI_BASE_URL ignored; per-request runtime-config re-parse (disk I/O); half-applied protocol_name override; shared mutable class-attr dicts.

VERIFIED GOOD: all ten declarations correct; discovery mechanics (underscore-skip, nvidia_nim remap, transport-key rejection); JSON validation fail-loud; snapshot freeze test-pinned; dynamic endpoint derivation; no credential leakage beyond sentinel bug.
STANDING: existing providers built against litellm — review/port pass ordered; thinking-handlers dead on native part of it.

## #39 — R-CHAT scope record
openai_chat depth round: wire surface both directions, streaming grammar, triple role (client+provider+litellm wire), route/error ladder, neutral round-trip + conversion semantics.

## #40 — R-CHAT verified findings + decision
USER DECISION: NO stateful chat completions — stored-completions CRUD will NEVER be implemented. Chat stays stateless-only; store rides raw/extra untouched.

(R-CHAT detailed findings live in the round presentation + #41; key code findings: native singular parse drops n>1 siblings on chat streams — native_provider/executor.py:306; cross-protocol audio unknown-MIME silently mislabeled wav — openai_chat.py:1577-78 contradicting docstring; missing tool result → literal "null"; held finish flushes only on [DONE]; stream media silent for chat clients while non-stream warns; usage frame sent without include_usage; include_usage:false overridden on one path; four finish-reason disciplines across sources; aggregator loses n>1 + crashes on index-less fragments + blind concat; session anchors hardcode choice 0; malformed message arrays → raw TypeErrors.)

## #41 — R-CHAT comparison verdicts
STANDARD vs us: single ingress ✓; same-protocol unknown-field survival ✓; same-protocol streaming = RAW BYTE RELAY + observational parsing (both references) — we always re-serialize (divergence; #42 principle adopted); cross-protocol explicit+lossy ✓ AND BEAT (we warn, they drop silently); reasoning_content relay ✓ exceed; usage trailing frame ✓ (plexus respects client's include_usage — our gate-on-client matches); error envelope ✓.
BORROW: gomodel n>1 end-to-end preservation (per-choice Split + per-choice finish Terminate + full assembly — maps exactly to our 3 n>1 breaks); plexus finish-repair shape (synthetic stop chunk, single-latch terminal dedup, incomplete→length/content_filter); gomodel embedded-error-in-200 (we have ✓); provider attribution in errors (consider); raw escape-hatch routes (consider later).
PER-PROVIDER QUIRKS (for porting pass): o-series max_tokens→max_completion_tokens + drop temperature (CHECK our canonical handling); Bailian reverse; Bedrock extras; vLLM reasoning rename; DeepSeek reasoning padding; Anthropic max_tokens required.

## #42 — Design principle (user-endorsed, extends D4 to streams)
Same-protocol streaming should NOT convert to neutral for relaying — observation decoupled from transport (gomodel ObservedSSEStream model: parse a copy for observers, relay bytes; re-serialize ONLY when transformative features need it: repair rules, usage normalization, cost merging, conversion). Fix-pass: conditional stream re-serialization with repair-capable tail always attached but pass-through-fast when already well-formed.

## #43 — R-ANTHROPIC verified findings
PERSONALLY CONFIRMED:
- [BUG] count_tokens prior-turn-thinking strip is DEAD CODE (filters type=="reasoning" blocks the chat projection never emits); comment asserts protection that doesn't exist.
- [BUG-adjacent] raw fast path forwards FOREIGN signatures cross-provider (executor.py:871 gates on protocol equality ONLY); canonical rebuild strips correctly; neither reference allows this.
- [GAP] non-stream anthropic responses anchor ONLY on id (no content-block branch; tool_use ids never anchored).
- [BUG-adjacent] unsigned thinking emitted toward anthropic targets (signature=None foreign reasoning → guaranteed 400; should degrade to text or drop).

MAPPER-VERIFIED: request-side cross-protocol drops UNDISCLOSED (container_upload, mcp blocks, builtin request blocks, service_tier/betas — the one silent path); cache_control text/system silently dropped (only image/document warn); absent stop_reason → end_turn fabricated WITHOUT warning; tool_use id-or-empty-string; tool_arguments_object bare ValueError; stream block index 0; top_k float bypass; merge loses message extra; dead _format_message; stream error frames non-Anthropic type strings; route logger misses event: frames; anthropic-version absent + path split for config-declared providers; native count_tokens unreachable (R8 repeat); msg_ ids under responses group.

SPEC-DIFF: COVERED (system array+cache_control, thinking enabled/adaptive+display+clamps, signatures both directions, redacted_thinking, server-tool catalog versioned, tool_choice+disable_parallel, max_tokens default 4096, full stop_reason vocab, usage cache fields+epoch, full stream grammar, citations). RIDES raw/extra (canonically unaware): compaction family, fallback block, output_config, search_result, container/browser/caller families, mcp_servers, role:system in messages, inference_geo. ERROR VOCAB: conflict_error(409), timeout_error(504), request_id, ratelimit header family, inline tool error codes.

COMPARISON: standard = same-protocol byte-faithful + model-rewrite-only (match non-stream; streams = #42); DECISION LOCKED: strip foreign opaque state on provider switch (#44); borrow: gomodel max_tokens auto-raise, sampling-param drop, forced-tool downgrade; plexus cache-sticky headers (future); gomodel error taxonomy + provider attribution + Retry-After passthrough.

## #44 — R-GEMINI scope + strip decision
DECISION LOCKED (user): foreign opaque state (thinking signatures etc.) is PER-PROVIDER — STRIP on provider switch, like any other provider-dependent field (fix executor.py:871 raw-fast-path gate: provider identity must match, else strip bound state; state stays cached for return).

## #45 — R-GEMINI corrections
1. skip_thought_signature_validator = the protocol's LEGAL FALLBACK when no signature can be restored (must be present in some form). Wire as fallback exactly where we strip or cannot restore signatures (esp. post-strip Gemini-3 function calls).
2. API-version surface: ingress accepts BOTH v1beta and v1 paths; upstream speaks v1beta only (host overridable, version fixed — minor config gap). Google's openai-compat surface intentionally not mirrored as a plain route — superseded by #46/#47.

## #46 — Gemini variants directive
Support BOTH native Gemini AND its OpenAI-compatible version — split gemini provider surface into 2+ variants like Responses' split. Lands as transport profiles (native: gemini protocol; openai: openai_chat at /v1beta/openai).

## #47 — Gemini surface decisions locked
1. v1 (old/stable) NOT needed — drop /v1/models/{model}:action ingress decorators. v1beta-or-latest + openai-compat only.
2. Traffic model: 99% goes native v1beta OR straight-through gemini:openai WITHOUT conversion (D4 raw fast path end-to-end, byte-faithful).
3. ENDPOINT→PROFILE ROUTING: our /v1beta/openai/* ingress routes to gemini:openai profile (client pointed at google's compat URL, pointed at us, works unchanged — mirror Google's URL shapes on ingress). Implement: gemini transport_profiles {native: gemini@v1beta, openai: openai_chat@/v1beta/openai Bearer}; default native; ingress gains /v1beta/openai/chat/completions + /v1beta/openai/models bound to the openai profile; error classification by body error.status.

## #48 — R9 cross-cutting verified findings
[MONEY] [BUG-high, verified] ANTHROPIC WIRE DOUBLE-SUBTRACT — accounting.py:234 subtracts cache buckets from already-exclusive input_tokens (input=2006+cache_creation=2095 → recorded 0); wire-shaped native payloads hit it; canonical-shaped fine. Memory corrected. [BUG ×3 null-cost family] streaming.py:1058 + responses/service.py:1789 + accounting.py:290/305 (request_cost_usd missing from breakdown keys). [CONTRACT] cost chain: provider_plugin pricing ONLY on non-stream executor:2668; stream/native/responses = bare CostCalculator; skip_cost_calculation ignored on those surfaces; responses synthetic provider "responses". [BUG] custom-caps percentage cooldowns encoded but never decoded → every % = quota_reset. [OK] gemini+openai shapes; exactly-one-record on all 3 paths; sticky TTL 3600→300 documented (only changelog callout missing). [SMELL] litellm total as output_cost; reported 0.0 falsy; stream _reduce wholesale-replace.

[MEMORY] [HIGH, verified] PERSISTENCE SELF-DESTRUCT — history_signatures unbounded in memory AND save; 16MB/4096 caps LOAD-ONLY (>16MB file → total discard on restart); full-state re-serialization every 5s. [POINTER ANSWER] ONLY responses previous_response_id harvested end-to-end; GEMINI = ZERO anchors (responseId≠id, candidates unread); ANTHROPIC = id-only misfiled; chat tool_call ids request-side only; STREAMING never records response ids; chat chatcmpl ids burn strong slots as ballast. Expansion: protocol-aware response readers + stream id threading + protocol-conditional groups. [MED] compaction edges: shared-summary collision (two conversations sticky-merged); generic markers + score>0; probes messages[:2] only. [MED] per-request WARNING inference log still firing. [OK] scoring/namespace/eviction/generation-dedup/rebuild-rejections; Windows move non-atomic (os.replace).

[ERRORS+LOGS] [HIGH, verified] QUOTA-AS-400 NON-ROTATABLE (error_handler.py:821 — no status string → invalid_request; exactly the gemini-compat case; fix = message-text quota sniffing + body-status priority). [HIGH] FAIL-swallow both paths (deterministic 400s rotate all creds → wrong 503 proxy_all_credentials_exhausted; Structured + NoAvailableKeys escape). [HIGH] decide_streaming_error_action dead; 5 inline copies; DRIFT: stream omits 5xx retry-same (same 502: stream rotates, non-stream retries same key). [MED] unknown rotates creds but never fails over; not_found same. [MED] dict-vs-httpx 400 context-window disagreement. [MED] non-stream FAILED transactions never finalize metadata. [VERIFIED-SOLID] W12 (tiers/zstd/never-fail/redaction/capture-set+alias-test/retention/correlation/CancelledError-guard). [RECONSTRUCTION] no per-transaction launcher exists; manual .py, 3 stages, injections marked-not-replayed.

[CONFIG] [HIGH] config-reference 7 wrong names/defaults + phantoms + 5 missing families; .env.example de-facto accurate (1 self-contradiction). [HIGH] routing parse errors fail every request (request_builder.py:87 outside try). [MED-HIGH] no cache on load_experimental_config (per-request parse; lru precedent at executor:3253). [MED] six divergent env-validation policies; warnings+unknown_sections never consumed. [LATENT, verified] experimental.py:24 module-level costs import → litellm 9.8s latent edge (fix = relocate ModelPricing dataclass). [OK] lazy boundaries measured fast; per-manager isolation fine.

## #49 — R9 directives + bad-decisions ledger
RECONSTRUCTION STANDARD RAISED: 1:1 with real state — injections, adapters, ALL deterministic stages; full-fidelity replay + quick-access launcher per transaction.
BAD DECISIONS LEDGER: 5 inline error-decision copies; three vocabularies (quota_exhausted/exceeded misspelling at seam); load-only caps; cost chain cloned incompletely; hand-written config-reference; decoder-shim era; content-priority ordering (media dropped from OUTPUT); tests pinning deviations; docstring-code contradictions (wav, finalizer order, Windows atomic).
CACHE DEDUPE (medium): same thing recorded twice must not store twice — hash-keyed references, robust + light.
CACHE STORAGE SIZE (medium): compress (zstd already dep), compact encodings, per-entry caps — session persistence + field cache + provider cache.

## #50 — R10 tests+docs verified findings
[TESTS] 1,333 collected; checklist: 10 COVERED / 3 PARTIAL (no multi-chunk n>1 stream test; cost parity tokens-only; no native-field-names-to-client). [HIGH] live NeutralStreamPipeline timing loop ZERO direct tests (all on retired handler; stream_ops duplicates the loop). [HIGH, verified] 33 UNTRACKED test files (81 vs 114) incl. 28/30 refactor/.
[PINNING WRONG] gemini SSE-without-alt; fabricated zero-usage success; fabricated call_N ids; responses route bridge-only; FAIL-masking untested both ways; wav + empty-string ids unpinned/unrefuted.
[HYGIENE] zero skips; FakeCredentialContext ×3; _context() ×9 signatures; sys.path boilerplate ×20; private reach-ins (session_tracking worst); flaky sleeps; stale __pycache__; real-network script in tests/.
[DOCS ROT] README: dead gemini_cli surface (~60 claims), 180-vs-300, endpoint table missing responses/gemini/quota-stats, ~/.gemini false. DOCUMENTATION.md: litellm framing, gemini_cli section, 86400, zero experimental coverage. config-reference: R9's 7 + HOST/PORT + CUSTOM_CAP JSON misleading + dead GEMINI_CLI_OAUTH_1 + D13 stale rule. manual-test-guide: WS path WRONG (/v1/responses/ws vs /v1/responses), /quota wrong name (actual /v1/quota-stats). .env.example: rotation-mode self-contradiction + 86400 + 3 dead GEMINI_CLI knobs + X-Proxy-Output-Protocol paragraph (removed feature); MISSING: FIELD_CACHE_COMPAT_GROUPS, <NAME>_CACHE_REPLAY, FALLBACK_GROUP failover/stop/policy overrides, SMALL_COOLDOWN_RETRY_THRESHOLD, FAIR_CYCLE_QUOTA/RESET, ROTATION_TOLERANCE, JSON keys transport_profiles/default_profile/cache_replay. ARCHITECTURE.md most accurate (1 flag + count_tokens stale line). STRUCTURE.md: decrypt_share_link only.
[DANGLING REFS] 121 lines across 31 files (D1-D17 minus D2/D3/D5/D6/D10/D16 + W2/3/5/6/7/10-13); routing/profiles.py module docstring states OLD D13 rule while its function implements the revision (self-inconsistent) — same stale rule in config-reference.
[DOCSTRING CONTRACTS] wav (confirmed), resilient_io Windows claim, timeout_config 180-vs-300; count_tokens: agent claim corrected — strip is DEAD CODE, disclosure comment is the true half, ARCHITECTURE.md repeats the false half.
[MISSING OPERATOR DOCS] profiles/grammar how-to, cache_replay, WS mode explainer (wrong path in config-reference), gemini /v1beta/openai plan, /v1/quota-stats, session-domain flow, request overrides.

## #51 — R10 residue + closing rulings
1. 33 untracked test files are OLD, probably DELETE — full test-suite cleanup pass ("a lot of tests are outdated") — review every test file, delete/rewrite stale.
2. README + root docs NOT updated = FINE — redone at END of this PR's life (no fix-pass time beyond structural needs).
CARRY: manual-test-guide 2 wrong paths (live-round instrument — fix); 121 dangling ledger refs + routing/profiles.py D13 docstring (ties to vocabulary scrub); .env.example X-Proxy-Output-Protocol paragraph + dead/missing knobs; ARCHITECTURE.md count_tokens one-liner.
STATUS: R1-R10 COMPLETE; R11 live round pending WITH USER; then fix pass; then FULL AUDIT AGAIN.
