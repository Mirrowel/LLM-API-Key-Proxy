# Protocol Runtime — Final Plan

**Status:** Authoritative. Supersedes every other document in `docs/experimental/` (all moved to `_superseded/`).  
**Created:** 2026-09-05. **Revised:** v1.1 (adversarial review), v1.2 (design session: D10-D16, W11-W13, deferred-intents ledger), v1.3 (evaluator round 2: finish-line re-sequencing, profiles workstream, D8/D11 precedence rules, cache_replay compile-down spec, capture-on-error class set, W11 declaration requirements).  
**Inputs:** second-opinion decision log (D1-D9), phase roadmaps, the 2026-09-05 audit of HEAD `f4ac60a`, two paired adversarial review rounds (four agents total, all findings source-verified), a full superseded-corpus re-audit, and the design-session decisions D10-D16.  
**Change authority:** Product decisions in this document were confirmed individually by the user. Architectural changes require a new user-confirmed decision, recorded here.

> **Naming note:** the reference implementation compared throughout is the private local codebase under `stuff/`; its name is intentionally not written into this document (repo hygiene rule).

---

## 1. Product Goal

Any supported client protocol can use any configured upstream provider, and vice versa, without writing pairwise converters. Conversion goes through one neutral core: 4 generative protocol adapters = 8 converter halves total; a 5th protocol costs one new file, not eight new converters. One exception by design (D13): on a **multi-profile** provider addressed by its bare name, only the protocol-matching profile is served (fast path or explicit error) — conversion there requires the explicit `provider:profile/model` form.

Every required meaning is converted faithfully, degraded deterministically, or rejected with a protocol-formatted error. Nothing required is silently dropped, reordered, merged, or turned into an empty success.

**The rewrite's reason for existing:** eliminate the LiteLLM dependency from everything normally used. Native protocol execution is the default for all generative traffic; LiteLLM remains an explicit, logged fallback and the path for non-generative operations. Provider plugins shrink from format wrappers to declarations + narrow tweaks. Transaction logging becomes light by default and fully reconstructable on demand. Modern reasoning models get automatic reasoning continuity via declarative cache-and-replay.

The proxy stays what it is on `dev`: thin FastAPI shell (`proxy_app`) + all logic in `rotator_library`, with the existing credential discovery (`<NAME>_API_KEY[_N]`), dynamic providers from `<NAME>_API_BASE`, usage/rotation/cooldown machinery untouched.

## 2. Target Architecture

### 2.1 Three layers, two roles

```text
client protocol  <->  neutral canonical  <->  upstream protocol
```

- The **client protocol** is bidirectional: the protocol of the request is the protocol of the response. Always. No independent output-protocol selection exists (D1).
- The **upstream protocol** is bidirectional and fixed per target at startup (D2). A provider may instead declare **Neutral** as its execution representation (D3).
- The **neutral canonical** is the only interoperability medium. One representation for requests, responses, and stream events.

### 2.2 One adapter file per protocol, both directions

```text
src/rotator_library/protocols/
  openai_chat.py          parse_request / format_response   (client role)
  anthropic_messages.py   build_request / parse_response    (upstream role)
  gemini.py               parse_stream_event / format_stream_event
  responses.py
```

Nobody implements half a protocol. "Client" and "upstream" are roles at request time, not split code.

**Scope note:** the registry also auto-discovers non-generative operation adapters (`openai_images`, `openai_embeddings`, `openai_audio`, `ollama`, `mcp`, `litellm_fallback`). The v1 contract and acceptance matrix cover the **generative four only**; non-generative adapters remain operation-scoped and stay on LiteLLM.

### 2.3 Request pipeline

```text
client payload
  -> client parser                 -> neutral request
  -> routing/credential selection  (existing systems)
  -> target's declared representation:
       wire protocol  -> generic protocol builder -> provider finalizer (envelope, alias, defaults)
       neutral        -> provider owns execution, returns neutral
  -> upstream endpoint
```

Same client protocol == upstream protocol and no semantic edits required: **raw-preserving fast path** — original payload is the transport basis, neutral parsed only as a sidecar for routing/session/validation/accounting/state (D4). Intentional stripping only via explicit, traceable rules.

Provider finalizers never translate protocols, never see the client's wire format, never branch on client protocol (D3).

**LiteLLM/custom execution:** LiteLLM-backed providers speak Chat as their *native wire format* — their chat-shaped output is parsed once into neutral, exactly like any other provider protocol. `has_custom_logic()` providers keep their verified custom path (D10); custom-first ordering is preserved.

### 2.4 Response pipeline

```text
upstream response
  -> [provider response adapters — provider shape, before parsing]  (non-streaming)
  -> upstream protocol parser (or provider returns neutral)
  -> neutral response
  -> client protocol formatter
  -> client
```

Same protocol and no adapter/semantic change: raw response passthrough with sidecar observation.

**Adapter contracts (two, explicit):**
- Non-streaming: response adapters run on the **provider wire shape**, before `parse_response`.
- Streaming: stream adapters run on the **neutral event**, after `parse_stream_event` (rationale: field-cache extraction and envelope stripping on streams operate per-event; provider frames are SSE-wrapped, not discrete payloads). Different input contracts by design — W7 pins both.

### 2.5 Streaming

Same boundary as non-streaming (derived rule): neutral stream events are authoritative whenever conversion is required; raw passthrough for safe same-protocol streams. **No forced Chat intermediate** on any native-provider stream path between provider parsing and client formatting. Retry, errors, usage, cost, session completion, and field cache all observe neutral events. The neutral stream model carries candidate, output-item, content-block, tool, reasoning, refusal, citation, usage, completion, error, and opaque-state lifecycles.

**Transport neutrality is a hard constraint:** SSE is one formatter, not the type. The stream pipeline must preserve the transport-formatter boundary (`streaming/transport.py`, `responses/streaming.py` formatter seams, `protocols/base.py:supports_transport`) so a WebSocket transport can be added later without protocol rewrites. W5b and W6 are forbidden from re-entrenching SSE-only assumptions.

**Preserved stream machinery (must survive the W5 rewrite):** TTFB/stall timeout enforcement (`STREAM_TTFB_TIMEOUT_SECONDS`/`STREAM_STALL_TIMEOUT_SECONDS` at `config/experimental.py:228-229`), heartbeat and cancel-upstream-on-disconnect behavior, `StreamMetrics`/`StreamMonitor`, provider-reported cost precedence including SSE `: cost` / `event: cost` frame parsing, and the phase-11 completion-evidence gate (bare iterator EOF is a transport fact; response identity is recorded only on explicit provider completion signals — on neutral streams too).

### 2.6 Execution representation

One symmetric representation per target for v1: receives X, returns X (X = one of the four wire protocols, or Neutral). Asymmetric request/response representations are a planned post-v1 extension, out of scope (D5).

### 2.7 Opaque provider state and cached fields (D8, D11, D12, W13)

Two-tier provenance: runtime-injected state comes only from the scoped field cache with recorded origin provenance, restored only to compatible contexts; client-supplied opaque fields pass verbatim only on the same-protocol raw path. Routing identity is never provenance.

Cache scoping (D11): keys require **provider + model**; credential and session are optional refinements — deliberately relaxed for the current single-operator model, tightened later when multi-user and robust session detection arrive (the key structure accepts those refinements additively). No fail-closed rule on unknown credentials: injection proceeds, provenance is always recorded. The existing `classifier` scope stays as delivered — it is the seed of the multi-user isolation system (§2.10). Session is the conversation — there is no separate `conversation` scope.

Compatibility classes (D12) govern inheritance:

```text
Bound fields (encrypted reasoning, thinking/thought signatures):
  locked to provider+model (+credential/session when known). Identity-match only.
  Unknown credential -> no injection (fail-closed, see above).

Portable fields (plaintext reasoning — GLM/Kimi/DeepSeek style):
  pooled by compatibility group.
    model:<name>        same model, any provider (open models across providers)
    curated groups      global registry shipped with the proxy
                        (open-models group, closed-models group, kept updated)
    config groups       user-declared ad-hoc groups (e.g. glm+kimi)
  no matching group     -> no inheritance (default-deny)
```

Config shape (JSON `field_cache` / `providers` sections; env `<NAME>_CONFIG` carries the same block):

```json
{ "compatibility_groups": { "open-reasoning": ["glm-5.3", "kimi-k3"] } }
```

Precedence: a config group's exclusion wins over `model:` identity; curated groups are the fallback baseline.

Cross-provider remains forbidden for bound fields; portable fields inherit within declared compatibility only. On rotation away from a compatible context, stale bound state is not injected (strip-and-retry semantics — the state stays cached for when the compatible context returns).

Multi-format storage (B2): a cached field is stored in the format it arrived and can be injected in any format the provider accepts via **transform-on-inject** — chat-completions reasoning in, Anthropic thinking-block shape out. Transformable semantics convert; bound fields pass through same-format only. The transform registry lives in the **protocols layer** (per-protocol reshape tables), not in field_cache. Unified-shape cache sources/targets are completed as part of W13.

### 2.8 Multi-profile providers (D13, W-PROF)

A provider whose endpoint accepts multiple wire protocols declares **transport profiles** — one provider entry, one credential pool, one quota/cooldown/usage identity; profiles are routing aliases that tag routing, cache provenance, and (later) per-profile stats.

**Declaration schema** (JSON `providers` section; same block via `<NAME>_CONFIG`):

```json
{ "myprovider": {
    "api_base": "https://api.example.com",
    "profiles": {
      "chat":      { "protocol": "openai_chat", "endpoint_paths": {"chat": "/v1/chat/completions"} },
      "responses": { "protocol": "responses",   "endpoint_paths": {"responses": "/v1/responses"} }
    },
    "default_profile": "chat",
    "model_protocols": { "some-model": ["responses"] }
} }
```

- Endpoints derive per profile from the protocol-conventional path over the shared `api_base`, overridable via `endpoint_paths` (D2's same-origin rules apply per profile).
- `model_protocols` is the **optional acceptance table**: maps models → accepted/preferred protocols. Default: all models accept all offered protocols. Populated only from authoritative sources — never invented when upstream `/v1/models` doesn't advertise it — and kept separate from the advertised/dynamic model list. It narrows, never expands.
- The profile rides on `RouteTarget.profile` and `RequestContext`, and is recorded in cache provenance and W12 boundary metadata.

**Addressing grammar:**

```text
provider/model             bare name:
                            single-protocol provider -> normal behavior
                            multi-profile provider -> fast-path-or-error: only the
                              profile matching the client's protocol; no match ->
                              error "endpoint does not exist"

provider:profile/model     explicit profile: that protocol, converting as asked.
                            Composes with the existing @execution suffix:
                            provider:profile/model@native
```

Rules: `:` is legal only in the provider segment; provider and profile names are validated to never contain `:`; parsing splits on the first `/` (existing split sites: `routing/config.py:parse_route_target`, `request_builder.py:_provider_from_model`), then the provider spec on `:`.

**Identity normalization (W-PROF inventory requirement):** `provider:profile` must normalize to `provider` at every identity-keyed site — usage-manager scope (`executor.py:768`), cooldown keys, classifier scope, session namespace, field-cache provider provenance. The workstream delivers the complete inventory before touching anything.

### 2.9 Cache-and-replay (D14, W13)

One declarative parameter on **every** provider kind (code-built, env-defined, JSON-defined):

```text
cache_replay:
  - field: reasoning            # wire path in the existing field-cache DSL
                                # (e.g. choices.0.message.reasoning_content) OR a
                                # registered semantic alias ("reasoning") that
                                # expands per protocol
    mode: turn_only | all | last:N | turns:N | per_tool_call
    inject: auto | always
```

- `cache_replay` **compiles down to `FieldCacheRule`s** — it is sugar over the engine, not a parallel system. Semantic aliases are a per-protocol path-expansion table registered in the protocols layer.
- **Mode mapping:** `turn_only` = everything since the last user message (new engine mode, `since_last_user`); `all` = existing `all`; `last:N` = the last N cached values (new parameterized mode); `turns:N` = the last N turns' worth (new); `per_tool_call` = existing mode, retained — required for Gemini thought-signature per-part keying. The existing five modes remain expressible in raw rules; `FieldCacheMode`'s closed `Literal` becomes structured/parameterized (schema change, part of W13).
- **Capture** always from the **finalized assembled response** (streams assemble into exactly that; never a user-facing knob).
- **Inject precedence (uniform, operator-controlled):** `auto` (default) = add only when the field is absent — a client-supplied value is always preserved; `always` = overwrite unconditionally, **any field class including bound** — this happens only because the operator chose it per field, and that choice is the point (D8's never-overwrite rule governs automatic runtime behavior only, never explicit configuration).
- Storage/keying per §2.7: provider+model + session context; compatibility classes govern inheritance; transform-on-inject re-shapes across formats (destination format = the injecting target's declared protocol).

### 2.10 Multi-user-ready construction (D17)

The runtime is designed for a real multi-user proxy (hundreds to thousands of users) and degrades cleanly to a single operator. Concretely: no code path assumes one caller; every shared pool (cache, sessions, sticky state, logging) is keyed so a user/domain refinement can be added without schema changes; the classifier scope and isolation domains are preserved as the isolation seams; per-user enforcement layers (API keys, quotas, rate limits, per-user accounting) are deliberately deferred (§11) and must remain additive when they arrive.

## 3. Authoritative Decisions

| # | Decision |
|---|---|
| D1 | Client request protocol = client response protocol. Remove `X-Proxy-Output-Protocol`, `output_protocol_name`, provider `default_output_protocol`. Routes determine protocol both directions; library callers declare client protocol explicitly. |
| D2 | One upstream protocol per target/profile, bound at startup. Multi-protocol services = multiple profiles sharing provider identity/credentials/quota (§2.8). |
| D3 | Each target declares the representation it receives: its one upstream wire protocol (normal) or Neutral (explicit executor contract). Providers never handle arbitrary client protocols. |
| D4 | Raw-preserving same-protocol fast path (request and response), neutral sidecar for proxy concerns. Intentional stripping only by explicit traceable rule; rebuild allowed but must preserve unrelated source-native fields and prove fidelity. |
| D5 | Symmetric execution representation for v1. Asymmetry = post-v1 nice-to-have, not in scope. |
| D6 | Streaming follows the same conversion boundary (derived): neutral events authoritative cross-protocol; raw passthrough same-protocol. |
| D7 | Ordered best-effort conversion hierarchy: exact mapping → equivalent construct → deterministic inference/coercion → proxy-managed emulation → honest fallback → explicit error. Every semantic classified: Required (convert/emulate/reject, never drop), Optional tuning (omit + recorded summary), Opaque state (scoped cache), Unknown extension (same-protocol only). |
| D8 | Opaque-state provenance: cache-sourced with origin for injection; client-supplied = same-protocol passthrough only, never overwritten by *automatic* runtime behavior (explicit `inject: always` configuration overrides — operator choice); routing identity is never provenance. |
| D9 | Multi-candidate (`n`/`candidateCount`): direct mapping when both sides support it; request coerced to upstream max + summary; multiple candidates into a single-response protocol = first candidate wins, rest dropped, recorded summary. No concatenation. |
| D10 | Native-by-default execution (2026-09-05). All generative traffic on providers with a declared protocol runs native; LiteLLM is an explicit, logged fallback only; non-generative operations stay on LiteLLM. `has_custom_logic()` providers keep their custom path (custom-first) until manually migrated. No accidental fallback — test-guarded. Implemented as declarations + gate change (W11), not a bare config flip. |
| D11 | Field-cache scoping (2026-09-05): provider+model required; credential/session optional refinements, deliberately relaxed for single-operator use, tightened additively with multi-user; no fail-closed rule on unknown credentials (injection proceeds, provenance recorded); `classifier` scope retained as the multi-user isolation seed. Session = conversation; `conversation` scope dropped. |
| D12 | Cache compatibility classes (2026-09-05): bound vs portable; default-deny inheritance; curated global groups + config groups; every entry records provenance. |
| D13 | Multi-profile providers + `provider:profile/model` grammar (2026-09-05): profiles share one provider identity (normalized at every identity-keyed site); bare name on multi-profile = fast-path-or-error; conversion only by explicit profile; optional per-provider model-protocol acceptance table (authoritative sources only). Implemented by W-PROF. |
| D14 | Cache-and-replay (2026-09-05): declarative per-provider field replay — assembled-response capture, modes `turn_only`/`all`/`last:N`/`turns:N`/`per_tool_call`, inject `auto`/`always` (uniform operator-controlled precedence; `auto` default preserves client values, `always` overwrites any class), automatic context keying, multi-format transform-on-inject. Pre-merge (W13). |
| D15 | Transaction logging levels (2026-09-05): L1 default = four boundary payloads + light metadata; L2/L3 opt-in; smart capture-on-error on by default — persists buffered L2/L3 as a zstd archive only when the error class is request-related (set enumerated in W12 over `core/errors.py` buckets); all artifacts zstd; offline reconstruction via quick-access script; retention cap by default. |
| D16 | Dynamic-provider env parity (2026-09-05): simple env knobs + `<NAME>_CONFIG` (path or inline JSON). Individual env knobs override same-named CONFIG values; both feed one validation pipeline. |
| D17 | Multi-user-ready construction (2026-09-05): the rewrite is built FOR multi-user use (hundreds/thousands of users) and works with one. No single-user hardcoding; identity/domain seams preserved (classifier scope, isolation domains, optional cache refinements); per-user enforcement (keys, quotas, rate limits) arrives later as additive layers, not rewrites. Structural principle for this rewrite; enforcement itself stays deferred (§11). |

Full D1-D9 reasoning history: `_superseded/protocol-interoperability-second-opinion.md`. D10-D16 were decided in the 2026-09-05 design session and are recorded here authoritatively.

## 4. Scope Boundaries

**Out of scope:** concrete provider implementations (Codex, Claude Code, Antigravity, Copilot, Gemini CLI, and any new provider — each implemented manually against captured upstream behavior; mocked provider tests are not evidence); credential/rotation system changes; GitHub CI; asymmetric representations; client-response protocol override (removed by D1); non-generative protocol adapters beyond keeping them importable and operation-scoped.

**In scope:** `src/rotator_library/protocols/`, `native_provider/`, `field_cache/`, `adapters/`, `client/` protocol paths, `responses/` service, `session_tracking.py` response-anchor shapes, provider *interface* contract, `anthropic_compat/` retirement, `proxy_app/main.py` shell restoration, W-PROF/W11/W12/W13 as specified, and tests for all of the above using synthetic providers only.

**Post-merge follow-ups (specified now, implemented after the main work):** the model library (per-model providers, capabilities, compatibility class; seeded/refreshed from external catalogs — models.dev primary, OpenRouter and Catwalk as additional sources; pricing data flows the same way); per-profile stats surface; target-group selectors.

## 5. Removals, Reverts, and Legacy Cores

### 5.1 Decision-1 machinery (remove entirely)

| Location | What |
|---|---|
| `client/protocol_selection.py` | `OUTPUT_PROTOCOL_HEADER`, `request_output_protocol()`, `resolve_client_output_protocol()`. `require_same_protocol_stream` is only a generative-four gate — D1 holds by construction after removal. Delete or reduce to a trivial assertion. |
| `client/rotating_client.py:600-657` | `agenerate`'s `output_protocol` param, `resolve_output_protocol` call, `require_same_protocol_stream` call, `kwargs["_output_protocol"]` injection at :623 |
| `client/request_builder.py:237-241, 301-309, 369` | `_output_protocol` kwarg pop (the only consumer), provider-default override, context field |
| `client/executor.py` (6 sites) | `get_protocol(context.output_protocol_name)` reads (:695, :1172, :1773); forced `"openai_chat"` at :764 (→ W1b/W5); :1758, :1761 pass-throughs |
| `client/anthropic.py:107-129` | `resolve_output_protocol` + `output_protocol=` + foreign-payload early return |
| `client/gemini.py:81` | `ProtocolContext(output_protocol=...)` vocabulary (rename to client/upstream) |
| `core/types.py:95` | `RequestContext.output_protocol_name` field |
| `protocols/types.py:479, 497` | `ProtocolContext.output_protocol` field + serialization entry (rename across all four adapters) |
| `config/experimental.py` (5 sites) | `default_output_protocol` config key/field/validation |
| `providers/provider_interface.py:277, 339-343` | `default_output_protocol` attr + getter |
| `native_provider/context.py:31, 58`, `native_provider/executor.py:44, 167` | output-protocol fields/fallbacks |
| `responses/service.py:203-211, 261, 292, 315-330, 425-433, 451-457, 461-469` | resolver calls, `output_protocol="responses"` in the native call (:261), `_output_protocol` kwarg set on the bridge path (:292), the entire cross-protocol response-conversion block, stream converter branch |
| `proxy_app/main.py` (8 call sites) | `resolve_client_output_protocol` imports/uses |
| tests | Regenerate mechanically at execution time (`rg -n "output_protocol\|default_output_protocol\|OUTPUT_PROTOCOL_HEADER" src/ tests/`); current broad count ~127 refs / 15 files. Known primary: `test_protocol_client_surfaces.py` (21), `test_protocol_streaming_matrix.py` (37), `test_native_protocol_runtime_matrix.py` (17 — incl. :175 asserting the pre-D1 contract: delete, don't update), `test_request_builder_routing.py:175-202`, `test_provider_runtime_config.py:39-163`, `test_responses_routes.py:86`, `test_anthropic_transform_tracing.py:96,116`. W1a note: runtime-matrix :175 was *converted* to D1 form (`test_runtime_formats_every_client_protocol_from_every_provider_protocol`, 16 real cells) rather than deleted — accepted deviation, stronger than deletion. |

**Leak guard (W1a acceptance):** the injected literal key `_output_protocol` has producers (`rotating_client.py:623`, `responses/service.py:292`) and exactly one consumer (`request_builder.py:237`). Acceptance: `rg "_output_protocol" src/` returns zero, plus a bridge-path test asserting no underscore-prefixed proxy keys in payloads.

### 5.2 Provider slop (experimental-only guesswork — delete)

| Item | Disposition |
|---|---|
| `providers/antigravity_provider.py` (NEW, 256 ln) | delete |
| `providers/claude_code_provider.py` (NEW, 147 ln) | delete |
| `providers/codex_provider.py` (NEW, 117 ln) | delete |
| `providers/copilot_provider.py` (NEW, 96 ln) | delete |
| `tests/test_{codex,claude_code,copilot}_provider.py`, `test_antigravity_provider_restore.py`, `test_provider_field_cache_contracts.py` | delete |
| `tests/test_native_protocol_runtime_matrix.py:16-17` | re-base onto an inline synthetic provider in the same change (W8) |

### 5.3 Modified provider files

| File | Change | Disposition |
|---|---|---|
| `providers/__init__.py` | dynamic-provider expansion | **KEEP** — generic path |
| `providers/provider_interface.py` (+212) | native contract hooks | **KEEP** — generic runtime surface |
| `providers/nvidia_provider.py` (+39, `enable_thinking`) | model tweak | **KEEP + review note** (user decision) |
| `providers/provider_cache.py` (+58) | generic cache infra hardening | **KEEP** |
| `providers/utilities/__init__.py`, `base_quota_tracker.py` | gemini re-export fallout | keep as-is |
| `_retired/*` import-path fixes | mechanical | keep |
| gemini_cli/google files in `_retired/` | deliberate retirement | keep retired |

### 5.4 Legacy converter cores (retire — user-confirmed)

| Item | Disposition |
|---|---|
| `anthropic_compat/` + `client/anthropic.py` no-`agenerate` facade branch | **DELETE**; unify `/v1/messages` (incl. legacy fallback and count_tokens translation) on the protocol runtime |
| `responses/bridge.py` non-`agenerate` fallback (`service.py:275-300`, forced `_output_protocol` at :292) | **DELETE** the fallback branch (the `agenerate` branch at :261 keeps, minus its output-protocol kwarg per 5.1). Done in W1a. The stream-path `stream_events` legacy surface is **retained through W5** (landed) and its bridge-based implementation retires in **W6+W9** — retiring it is route-level consolidation onto the executor's neutral pipeline, and the transport-neutral `ResponsesStreamEvent` seam plus the phase-8c timing/heartbeat suite depend on it (2026-09-05 re-sequencing, review-confirmed). |

## 6. Defect Closure Requirements (audit of 2026-09-05, HEAD f4ac60a)

| # | Defect | Status | Required end-state |
|---|---|---|---|
| 1 | `encrypted_content` lost: continuation rebuild is summary-only (`protocols/responses.py:351-355`); zero refs in src | **NOT FIXED** | encrypted reasoning survives same-protocol round-trips and continuation expansion byte-for-byte |
| 2 | Forced Chat SSE intermediate on native stream paths (`client/executor.py:761-764`), re-parsed at `:1770-1775`; usage/cost/session observe chat shapes | **NOT FIXED** | native paths: neutral events authoritative through retry/usage/cost/session; single parse per provider frame; LiteLLM/custom paths parse their chat output into neutral once |
| 3 | No candidate identity in `UnifiedResponse`; cross-protocol alternatives coalesced (`openai_chat.py:199`); gap is Chat `n` ↔ canonical candidate-count (Gemini mapping exists, `gemini.py:455,567`) | **PARTIAL** | D9 implemented: identity in neutral, direct mapping, first-wins degradation with summary |
| 4 | Instruction hoisting in all 4 builders (`openai_chat.py:148` etc.) and callback merge (`client/executor.py:843`) | **NOT FIXED** | original interleaving preserved wherever destination can express it; hoisting only where destination mandates a single instruction field, then as ordered merge per D7 |
| 5 | Refusal/built-in-tool output/annotations unrepresented → empty successes cross-protocol (`responses.py:382-398, 495-496`) | **PARTIAL** | all three represented in neutral; converted, honestly degraded, or rejected per D7 — never an empty success |
| 6 | Same-protocol rebuild lossy: filename stripped, nested shapes lost, fabricated `name`/`parameters` (`responses.py:535-541`) | **PARTIAL** | D4 fast path makes same-protocol lossless modulo explicit overlays; zero fabricated fields |
| 7 | Reasoning controls silently dropped cross-family (`openai_chat.py:552`); `.warnings` has zero consumers | **PARTIAL** | documented budget↔effort↔enable↔visibility table; every drop emits a recorded conversion summary |
| 8 | Stream block identity hardcoded (`streaming.py:371-378`); reasoning-summary deltas unhandled on parse (`responses.py:246`) | **NOT FIXED** | block/item/candidate identity end-to-end; `text A → tool → text B` stays three blocks; summary deltas parse canonically |
| 9 | Routing-match authorizes opaque-state passthrough (`canonical.py:162-166`, `native_provider/executor.py:199-205`); `input_provider` routing-derived (`request_builder.py:372`); stale after fallback (`routing/attempts.py:35-49`) | **PARTIAL** | D8 two-tier provenance; no routing-derived identity |
| 10 | Responses native stream raises after start (`responses/service.py:502-507`, unguarded loop `:471-501`); gemini backstop OpenAI-shaped | **PARTIAL** | every post-start failure ends in a valid terminal frame of the client protocol, all four families |
| 11 | Non-streaming response adapters run on client-format payload after `format_response` (`native_provider/executor.py:113→117`) | **PARTIAL** | non-streaming: adapters on provider shape before parsing; streaming: pinned on neutral events; both fixture-locked (W7) |
| 12 | `main.py` ~2164 lines: stream wrapper + chat aggregation, per-route error plumbing, hand-rolled count_tokens payloads, duplicated SSE headers | **PARTIAL** | thin shell: auth, parse, delegate, construct response; all protocol behavior in library |

**Named third-pass re-verify items (fold into W10's re-verification, explicitly):** cooldown bypass when remaining cooldown exceeds deadline budget (phase 7); streaming `@`-mode parity with non-streaming (phase 6 — W11 touches this logic); requested-model-in-group promotion (phase 6); cache-write token double-counting (phase 9); `format_response` emitting protocol-native usage field *names*, not unified names (phase 1 — W10 fixture 13 asserts field names).

## 7. Implementation Workstreams

**Execution order (v1.3 — merge gate is the LAST step):**

```text
W1a -> (W1b + W5a-e) -> W2 -> W3 -> W4 -> W6+W9 -> W7 -> W8
     -> W12 (logging; delivers L1 metadata schema first) -> W11 (native flip)
     -> W-PROF (profiles) -> W13 (cache-and-replay) -> W10 (FINAL GATE:
        matrix re-run + config-reference + manual test + user merge)
```

**Process gate (every workstream):** paired subagent review before done — base/flash for mapping checks, heavy+flash identical-prompt pair for critical workstreams; findings verified by the primary agent against source before acceptance. Unstaged edits only; commits happen when the user says so.

**W1a — Decision-1 realignment (surfaces).** Remove §5.1 machinery; `client_protocol_name` = input protocol everywhere; vocabulary rename. *Acceptance: `rg "_output_protocol" src/` = 0; leak-guard test passes; no underscore-prefixed proxy keys reach `parse_request`.*

**W1b — Executor internals (with W5, not before).** The forced-chat field at `executor.py:764` is load-bearing for the chat-shaped operational layer; land only with W5's neutral flow. *Acceptance: post-W1b fault test — native stream, anthropic client: terminal frame + usage + session anchors recorded.*

**W2 — Neutral model completeness.** Candidate identity + per-candidate stop; refusal, annotations/citations, built-in tool items as canonical capabilities; content-block index; output modality identity; provenance fields on cached state *(re-scoped 2026-09-06: owned by W13/D12 — field-cache entries record origin provenance there)*; `session_tracking._anchors_from_response` learns neutral shapes (chat parser as legacy fallback; landed with W5). *Deferred to W4 (2026-09-06, W2 review): stream-side emission of refusal/builtin/annotation blocks (non-streaming builders emit or reject; stream formatters not yet taught the new block types); Anthropic `server_tool_use`/`web_search_tool_result` blocks parsing into builtin_tool records.* *Acceptance: types carry them; parsers extract; builders emit or reject per D7; fixture: anthropic-client native stream records response anchors.*

**W3 — Same-protocol fidelity (D4).** Raw-preserving transport; overlay semantics for canonical edits; encrypted_content, filenames, nested shapes, unknown extensions survive; overlay trace record lists every applied overlay. *Acceptance: byte-level fidelity fixtures modulo explicit traced overlays.* **DELIVERED** (`72080c8`, `0db37d7`, close-out; PASS × 2 reviewers). Accepted residuals, recorded: (1) instruction repositioning is normalized by design — a proxy-side move of a system message ships the original order untraced (the one mutation class exempt from the traced-overlay contract); (2) raw byte forwarding of STREAMS stays on the W5 neutral-event pipeline (event sequences fixture-locked; §2.5's "raw passthrough for streams" means raw client-protocol frames, not provider bytes); (3) the production gemini raw basis carries the facade-injected routing `model` key (canonical builder emits it too; real-API strictness question → W10 gemini verification pair); (4) W12 carrier caveat: `request_transport_overlays` covers basis selection + model only — post-basis adapter edits, request-level cache injections, AND response-side wire-adapter edits (which run pre-parse on the raw provider response since W7) are traced in their own passes and must not be assumed reconstructable from the overlay list alone.

**W4 — Cross-protocol conversion (D7/D9).** Instruction ordering nuance; reasoning-control mapping table + recorded summaries; refusal/built-in-tools/annotations; Chat `n` ↔ canonical candidate-count; first-wins; media/image, structured output, generation controls, tool-choice, canonical stop/status mapping (restored test contract); Gemini safety-settings passthrough guarantee. *Carry-ins from W2 review (2026-09-06): stream-side emission of refusal/builtin/annotation blocks; Anthropic `server_tool_use`/`web_search_tool_result` parsing into builtin records; cross-protocol audio → chat responses synthesizing input-direction parts (true `message.audio` synthesis); annotations-on-refusal-blocks conversion summary at responses target.* *Acceptance: every D7 class defined per protocol pair, fixture-covered; summaries content-asserted.* *Accepted residuals (review 2026-09-06, recorded): stream-side builtin drops at non-responses targets have no summary vehicle (stream summary surface is by-design absent; revisit at W5/W10 if a stream carrier emerges); `summary:auto|detailed` map equivalently to `includeThoughts:true` at Gemini (equivalent construct, single disclosure); adjacent-turn merge at Anthropic stays (alternation-mandated level-2 equivalent); responses-source dual warnings for include_thoughts+summary accepted (one client control, two views).*

**W5 — Streaming alignment (D6):** W5a neutral event flow (kills forced chat + double parse); W5b `StreamingHandler` rewrite (~1157 lines — largest item; preserves transport boundary, TTFB/stall, heartbeat/cancel, metrics); W5c session anchors + completion-evidence gate on neutral streams; W5d usage/cost from neutral events incl. provider-reported cost + SSE cost frames; W5e field-cache stream-rule migration. *Acceptance: no chat-shaped intermediate on native paths; defects 2+8 fixtures; usage/cost/anchor parity for gemini + anthropic clients.*

**W6+W9 — Error lifecycle + thin shell (merged).** Guard responses frame loop (terminal frame, not raise); gemini backstop protocol-correct; one rewrite of `main.py:743-920` moving wrapper/aggregation/error plumbing/SSE headers to the library; count_tokens route cleanup + projection note; TUI key masking (`launcher_tui.py:455`). *Acceptance: fault-injection tests — protocol-valid terminal frames, all four families; main.py = shell duties.*

**W7 — Adapter staging.** Non-streaming response adapters before `parse_response`; streaming pinned on neutral events; context protocol matches payload shape; delete `test_native_protocol_runtime_matrix.py:407`. *Acceptance: both contracts fixture-locked.*

**W8 — Slop removal.** Execute §5.2; re-base the runtime matrix onto inline synthetic providers in the same change. *Acceptance: provider dir matches dev except keep-list; no mocked provider tests; suite green.*

**W12 — Transaction logging redesign (D15) — before W11, because W11's acceptance needs its metadata schema.** Migration, not rewrite (the four boundaries already exist as files in the per-transaction layout).
- **First deliverable: the L1 boundary metadata schema** (timestamps, provider, model, credential id, per-attempt protocol identities, execution mode + fallback flags, status, timings, routing decisions, session/scope correlation, per-attempt fast-path flag, injection-overlay digest/values, reconstruct shortcut).
- **L1 (default):** four boundary payloads — in / out / back (streams: raw chunks + assembled) / return — + light metadata.
- **L2 (opt-in):** intermediates; **L3 (opt-in):** verbose per-frame. Transform-tracing tests configure L2 explicitly; L1 metadata carries a pass-name digest.
- **Smart capture-on-error (on by default):** buffered L2/L3 persisted **as zstd archive only** when the error class is request-related. Capture set enumerated over `core/errors.py` structured buckets and bound to them as the single source: **capture** = `invalid_request`, `context_window`, `not_found`, `server_error` (server bugs may be payload-triggered), 413-payload-too-large; **no capture** = `rate_limit`, `authentication`, `forbidden`, transport/timeout classes. Alias-coverage test for the classification (known historical alias gaps). Dispositions for ambiguous codes documented in `config-reference.md`.
- **Reconstruction:** `tools/reconstruct_traces.py` + generated `reconstruct.bat` per transaction; re-runs parse → neutral → build → finalize offline using recorded routing choices, fast-path flags, and recorded injection overlays (W3's carrier). Guarantee scoped to *transforms given recorded inputs*; live-state decisions stay metadata. Determinism test: reconstructed == live for a fixture set, including one provider request containing a cache-injected field.
- **Invariants:** boundary redaction incl. camelCase (fix `responses/service.py:1336-1353` helper); never-fail-requests; correlation fields; failure records (stage before/after provider + error type + redacted snapshot); JSONL stream safety + sampling; **retention knob + default cap** (`TRANSACTION_LOG_RETENTION_*`).
- `zstandard` dependency added.

*Acceptance: L1-only run writes KB-scale transactions with bounded retention; capture-on-error archives only the enumerated set; reconstruction reproduces fixtures byte-identically incl. injected fields; redaction suite green.*

**W11 — Native-by-default execution (D10) — declarations + gate change, not a config flip.**
- Per-provider **declaration table** added for the flip set (openai, openrouter, groq, mistral, cohere, chutes, nanogpt, firmware, nvidia_nim → `openai_chat`; gemini → `gemini`): `protocol_name`, `native_streaming_supported`, endpoint mapping, operation resolution per provider.
- **Gate re-spec:** `_should_use_native_streaming` (`executor.py:2869-2888`, currently fail-closed because the wrapper expects LiteLLM-shaped chunks) and `_should_use_native_protocol` (`:2827-2839`) get post-W5 semantics: declared-protocol providers execute native in `auto`.
- **Env-only dynamic providers:** default native `openai_chat`; endpoint derivation rule for a bare `<NAME>_API_BASE` (protocol-conventional paths: `/chat/completions`, `/models`), overridable.
- Custom-first preserved; LiteLLM via explicit fallback mode only, **logged with fallback identity** on W12 boundary metadata; **no-accidental-fallback test guard**; accounting continuity on both paths.
*Acceptance: parity fixtures native-vs-litellm per flipped provider (equivalent response semantics, identical usage figures); guard tests green; fallback visibly tagged.*

**W-PROF — Multi-profile providers (D13).** Grammar parsing at `parse_route_target`/`_provider_from_model` (+ `@execution` composition); `RouteTarget.profile` + `RequestContext` carrier; declaration schema + `default_profile` + optional `model_protocols` acceptance table (§2.8); **identity-normalization inventory delivered first** (usage scope `executor.py:768`, cooldown keys, classifier scope, session namespace, field-cache provenance — all normalize `provider:profile` → `provider`); endpoint derivation per profile. *Acceptance: profile fixtures — bare name fast-path-or-error per D13; explicit profile converts; identity normalization verified (quota/cooldown/session keys identical across profiles); `:` validation; acceptance-table narrowing fixture.*

**W13 — Cache-and-replay (D11/D12/D14) — after W-PROF (uses profile-aware provenance) and W2+W5.** `cache_replay` compiles to `FieldCacheRule`s; semantic-alias table in protocols layer; `FieldCacheMode` schema change for parameterized modes (`last:N`, `turns:N`, `since_last_user`); `per_tool_call` retained; inject precedence per §2.9 (client-supplied bound fields never overwritten; `always` rejected for bound class at validation); compatibility-group registry (curated + config) with provenance; multi-format entries + transform-on-inject (protocols-layer registry); unified-shape sources/targets completed; scoping per D11 incl. bound-field fail-closed on unknown credential. *Acceptance fixtures: (1) reasoning continuity — GLM-style plaintext cached turn 1, injected turns 2..N per mode; (2) bound field never leaves its provider (cross-provider), survives rotation back; (3) transform — chat-shape reasoning injected as anthropic thinking block; (4) groups — glm↔kimi inherit, glm↔gpt never; (5) scoping — same model two providers shares portable pool; unknown credential → injection proceeds with provenance recorded; (6) client-present portable reasoning + `inject: always` → cache wins; client-present value + `auto` → client value preserved (uniform, all classes).*

*Delivered 2026-09-06 (commits 3c67c4d, be53a9d, cb87527, c5ad681 + follow-ups): compilation, D11 scoping (provider+model required, credential/session optional with `_none`-bucket provenance), D12 classes (bound identity-match only; portable groups default-deny; operator per-field choices honored incl. `always` on bound — uniform precedence), transform-on-inject (portable-only, compile-time name validation), env parity `<NAME>_CACHE_REPLAY`, guarded name-collision merging with documented precedence (json > env > class > provider declaration), isolation-class widening denials in the guard, engine-enforced strict session binding for continuation rules, classifier fail-closed (D17 seed), per-operation profile provenance.* **Explicit deferrals (recorded, not silent):** (a) *`turns:N`* is values-bounded (last N cached values), not strict conversation-turn detection — the engine has no turn boundary concept; honest semantics documented in `replay.py`; a true `since_last_user` mode needs the turn-container vocabulary and lands with the first provider that requires it; (b) *semantic-alias path table* (per-protocol path expansion like `$reasoning`) deferred to W10's `config-reference.md` companion — paths are literal-only today, documented there; (c) fixtures (2) and (4) are covered abstractly (cross-model deny + primary-key hit + group inheritance tests) rather than with concrete glm/kimi names — the model library that seeds real groups is post-merge scope; (d) `inject: always` on bound class is NOT rejected (§2.9's rejection clause was superseded by the operator-trust-boundary ruling — v1.4).

**W10 — FINAL GATE (moved last).** Pre-gate: **protocol verification pairs** — one review pair per protocol (openai_chat, anthropic_messages, responses, gemini), each owning one protocol and ALL its translations, using web search exhaustively against official protocol documentation to find quirks, niches, and validation gaps; findings are fixed until both the pair and the primary engineer consider the protocol correct. Then: 16-cell matrix re-run (4 client × 4 upstream; streaming + non-streaming; diagonal cells prove D4 fast path; synthetic providers only; profile cells included post-W-PROF) + fidelity suites. **Authored from the defect list and D7/D9 classes, never from current behavior**; delete tests asserting defective behavior. Per-cell assertion standard: (a) exact ordered frame/event sequences; (b) block/item/candidate index continuity; (c) absence-of-fabrication; (d) D7 degradations assert recorded summaries by content; (e) byte-level equality for diagonal cells. Fixture checklist (cell not green without applicable categories): 1 same-protocol byte fidelity + overlay traceability; 2 instruction ordering; 3 refusal/built-in tools/annotations; 4 D9 multi-candidate; 5 streaming three-block identity; 6 reasoning-summary deltas; 7 mid-stream terminal frames all families; 8 anchors+usage/cost for non-chat clients; 9 D8 two-tier incl. post-fallback; 10 D7 rejection payloads; 11 mid-stream retry integrity; 12 media/structured-output/generation-controls/tool-choice/stop-status; 13 usage-detail parity **asserting protocol-native field names**. Also: deliver `config-reference.md` (all knobs incl. cache_replay, profiles, acceptance tables, logging levels/retention/capture set, dynamic-provider vars incl. `<NAME>_CONFIG`); reconcile the 68 drift failures; re-verify the named third-pass items (§6); manual test guide; `git add -f` note. *Acceptance: all cells green; no open Critical/High; user passes manual test; **user merges to `dev`**.*

*DELIVERED 2026-09-07/08.* **Verification pairs:** six surfaces signed at the full-closure bar (every finding BLOCKER→NIT fixed or explicitly ignored with rationale; no approving with open mediums/lows): openai_chat (5 rounds, flash + light), anthropic_messages (4 rounds), responses (5 rounds incl. encrypted-content harvest + custom-tool replay), gemini (6 rounds incl. hosted-tool envelope maps, multi-candidate stream finishes), WebSocket Responses variant (own pair, 5 rounds, ZDR store-failed matrix), neutral canonical model (own pair, 5 rounds). ~120 gatekeeper findings fixed with regression fixtures; the fixture-shaped-fix failure mode (probes passing only on fabricated wire shapes) was caught twice and closed with real-wire pins. **Matrix:** the 4×4 parametrized interoperability + streaming-matrix suites cover all 16 cells in both roles; diagonal = W3 byte-fidelity fixtures; the fixture checklist categories are distributed across the W2-W5/gatekeeper fixture families. **Drift reconciliation:** asyncio_mode=auto (`pytest.ini`) + per-file triage — 1321 passed / 0 failed / 0 regressions (39 fair-cycle + 12 capacity + 1 timestamps tests retired as pre-decomposition supersession; refactor files mechanically updated to current constructor/state shapes with assertions preserved or strengthened). **config-reference.md** delivered (all subsystems). **manual-test-guide.md** delivered (the user's pre-merge acceptance pass). Remaining: the USER's manual test + the USER's merge to `dev`.

## 8. Acceptance & Finish Line

1. All workstream acceptance criteria met (§7), in the stated order, with W10 as the final gate.
2. No open Critical/High defect; every Medium either fixed or explicitly accepted by the user in writing here.
3. Full local suite green (baseline 1046+; slop tests deleted, contract fixtures added; 68 drift failures reconciled; named third-pass items re-verified).
4. Documentation acceptance: docstrings on public extension points; lossy conversions documented at the conversion site; future-seam comments (WebSocket, target-group selectors, multi-user); `config-reference.md` delivered at W10.
5. User personally tests through real entry points (primary evidence, not agent self-report). Manual test guide delivered with W10.
6. User merges `experimental` → `dev` after W10 — the merge is the definition of finished.

## 9. Current-State Evaluation (2026-09-05)

| Requirement | Verdict | Notes |
|---|---|---|
| Four routes converge on shared runtime | DELIVERED | all generative routes enter executor |
| Protocol identities per execution | DELIVERED, MISALIGNED | output violates D1 → W1 |
| Native execution from canonical | DELIVERED, GAPS | defects 1,3-7 |
| Config-defined providers | DELIVERED, MISALIGNED | `default_output_protocol` violates D1 |
| Field-cache foundation | DELIVERED, GAPS | provenance gap (defect 9); D11/D12 → W13 |
| Canonical streaming module | PARTIAL | forced-chat seam (defect 2), identity (defect 8) |
| Error normalization | PARTIAL | responses raise-hole (defect 10) |
| Same-protocol preservation | MISALIGNED/PARTIAL | not the D4 fast path; fabrication bug |
| Phases 6-11 machinery | DELIVERED per phase reports | re-verify at W10 (named items §6) |
| Thin shell | PARTIAL | defect 12 |
| Provider plugins single-protocol | NOT STARTED as policy | §5.2 |
| One conversion core | VIOLATED | §5.4 deletions user-confirmed |
| Native-by-default (D10) | NOT STARTED | W11 |
| Leveled/reconstructable logging (D15) | NOT STARTED | W12 |
| Cache-and-replay (D14) | NOT STARTED | W13 |
| Multi-profile grammar (D13) | NOT STARTED | W-PROF |

**How the previous rounds failed (guard against repeat):** beta-ready verdict with Critical defects open; matrix tests asserting structure not semantics; mocked provider contracts as evidence; one test asserting a defect as contract; a synthesis losing the LiteLLM intent; (round 2, caught in time) a plan whose finish-line fired mid-sequence and two decisions without owners or schemas. Guards: W10-final authorship rule + assertion standard + fixture checklist; provider mocks never count; primary-source evidence for every claim; deferred-intents ledger; every decision has an owning workstream; merge gate is last.

## 10. Superseded Documents

All other `docs/experimental/*.md` in `docs/experimental/_superseded/` — history, not contracts. Corpus re-audited twice (2026-09-05); remaining folds are in this document.

## 11. Deferred-Intents Ledger

Post-v1 or later, explicitly: WebSocket transport (seams preserved §2.5); target-group selectors (seam at `routing/types.py:142`); multi-user/admin proxy; admin/debug trace read endpoints; Responses cancel endpoint; hard cost caps; periodic TTL sweep for responses store; strict conversion mode; quota-checker config section; protocols-as-subclassable extension surface; asymmetric execution (D5); retry-history enrichment fields (W12 metadata natural home); model library (W14 — post-merge, catalogs: models.dev/OpenRouter/Catwalk); per-profile stats. Formally retired for v1: `conversation` cache scope (session is the conversation); declared-but-unwired unified cache sources/targets as standalone work (completed under W13 with real semantics).

## 12. Reference-Implementation Comparisons (sanitized summary)

Adopted: per-attempt path visibility (W12 metadata); leveled debug capture + capture-on-error (W12, refined to error-class-aware smart default, zstd-archive-only); multi-catalog model metadata with ETag refresh (W14); per-model API-type restriction → optional acceptance tables (§2.8).

Deliberate divergences: explicit profile grammar + fast-path-or-error vs silent prefer-then-convert; declarative cache-and-replay with compatibility classes (no equivalent there; their state survives only by client round-trip, stripped on failover); offline reconstruction + zstd (theirs is intra-request chunks→assembled, no compression); env/JSON-first providers vs DB/UI records; evidence-based session tracking vs in-memory hash stickiness.
