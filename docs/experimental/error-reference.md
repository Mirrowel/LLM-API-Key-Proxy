# Error Reference — the four generative protocols (canonical)

> Consolidated 2026-09-11 from three independent research agents (flash/light/seek; source files preserved in Temp `research/G1-errors/`). Confidence marks: **[V]** VERIFIED (official doc / SDK source read directly), **[R]** REPORTED (issues/blogs/litellm-only), **[I]** INFERRED. This is the single source of truth for G1 (error taxonomy + decision matrix) and every error-rendering surface.

**The one rule everything falls out of:** HTTP status is the weakest signal. Strength order: structured body fields (`error.status` Gemini > `error.code` OpenAI > `error.type` Anthropic) → HTTP status → free-text message sniffing (last resort, most dangerous — see the substring traps below).

---

## 1. OpenAI Chat Completions

### 1.1 Envelope [V]
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```
All four keys; `param`/`code` nullable. No complete official `error.code` enum exists — treat vocabularies as open.

### 1.2 Status/type/code table [V unless noted]

| HTTP | type | code (when present) | Meaning | Retry? |
|---|---|---|---|---|
| 400 | `invalid_request_error` | `context_length_exceeded` (`param: messages`) | context overflow | No (deterministic) |
| 400 | `invalid_request_error` | `string_above_max_length`, etc. | request-shape errors | No |
| 401 | auth family | `invalid_api_key` | bad key / wrong org / IP not authorized | No (failover) |
| 403 | `permission_error` | — | geo/region block, key lacks permission | No (failover) |
| 404 | — | `model_not_found` [R litellm] | unknown model | No (failover target) |
| 429 | `rate_limit_error` | — | true throughput rate limit | Yes, follow Retry-After |
| 429 | `insufficient_quota` | `insufficient_quota` (type AND code both [R]) | quota/billing — "retrying won't restore access" [V] | **No** |
| 429 | `insufficient_quota` | `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded` | billing family — discriminate by code [V] | **No** (quota class) |
| 500 | `server_error` | — | server error | Yes, brief wait |
| 503 | — | — | "engine currently overloaded" | Yes, backoff |
| 503 | — | — | "Slow Down" (rate ramp) | hold ≥15 min, ramp |

**Trap:** both true rate limits AND billing land on 429 + SDK class `RateLimitError` — only `error.code`/`error.type` separates them. Historical 402 quota spelling exists [R]; treat `insufficient_quota` markers at 402/429 as billing.

### 1.3 Rate-limit headers [V]
- `Retry-After`: **seconds** (integer). Official: only for temporary 429s, never fixes billing.
- `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` + `-project-tokens` variants: reset values are **Go-duration strings** (`1s`, `6m0s`, `23h18m29.144s`) — NOT integers, NOT timestamps. Tolerate unknown siblings (`*_usage_based` [R]).
- `x-should-retry: true|false` header **overrides** status-based retry decisions [V].

### 1.4 Streaming [V-negative + R]
No documented mid-stream error event. Real failure modes: a `data:` frame carrying `{"error": {...}}` inside the 200 stream [R, multiple reports], or abrupt close without `[DONE]`. Either way: any data frame whose top level is an `error` object = terminal stream error [I]. `finish_reason: content_filter` is a **successful 200** — never an error.

### 1.5 SDK behavior [V]
Retries by default (2×): connection errors, 408, 409, 429, ≥500. `MAX_RETRY_AFTER_DELAY=120s` (bigger Retry-After disables retry). Timeout 600s total/5s connect. SDK exception hierarchy: `APIError → {APIConnectionError → APITimeoutError, APIStatusError → BadRequest/Authentication/PermissionDenied/NotFound/Conflict/UnprocessableEntity/RateLimit/InternalServerError}`.

## 2. OpenAI Responses

### 2.1 HTTP errors
Same envelope as chat. Context overflow: **HTTP 400**, `type: invalid_request_error`, `code: context_length_exceeded`, `param: input` [R bifrost #4413]. Note: `code: null` bodies exist — message/type carry the signal [R].

### 2.2 ResponseError.code enum (response object, NOT HTTP) [V — openai-python 2.54]
`server_error, rate_limit_exceeded, invalid_prompt, data_residency_mismatch, bio_policy, vector_store_timeout, invalid_image, invalid_image_format, invalid_base64_image, invalid_image_url, image_too_large, image_too_small, image_parse_error, image_content_policy_violation, invalid_image_mode, image_file_too_large, unsupported_image_media_type, empty_image_file, failed_to_download_image, image_file_not_found`

**Traps:** `rate_limit_exceeded` (Responses code, `_exceeded`) ≠ `rate_limit_error` (chat type, `_error`) — different strings in different positions. `context_length_exceeded` is NOT in this enum — it only appears as the 400 body code. `response.incomplete` (`incomplete_details.reason: max_output_tokens | content_filter`) is NOT an error.

### 2.3 Stream terminals — FOUR outcomes, not one [V]
1. `response.completed` — success.
2. `response.incomplete` — early stop (length/filter). Not an error.
3. `response.failed` — carries `response.error` (ResponseError codes above).
4. out-of-band `error` event: `{type: "error", code?, message, param?, sequence_number}` [V SDK type; R live captures]. Terminal for the whole stream.
Plus: stream can end with NO terminal event [R langchain #39039 — consumers mistaking truncation for success]. The `error` event can arrive with **empty message** on context overflow [R bifrost #4413/#4418].

### 2.4 WebSocket-mode errors [V]
`previous_response_not_found` (retry with full input, `previous_response_id: null`), `websocket_connection_limit_reached` (60-min cap; reopen).

### 2.5 Rate limits
Same headers/SDK engine as chat.

## 3. Anthropic Messages

### 3.1 Envelope [V]
```json
{"type": "error", "error": {"type": "not_found_error", "message": "..."}, "request_id": "req_..."}
```
`request-id` header on EVERY response (same value). Unknown `error.type` values arrive over time — handle gracefully [V].

### 3.2 Complete error.type enum [V — docs.claude.com/en/api/errors]

| HTTP | error.type | Notes |
|---|---|---|
| 400 | `invalid_request_error` | ALSO: self-set spend limit ("You have reached your specified API usage limits") — quota-at-400 trap [V] |
| 401 | `authentication_error` | |
| **402** | **`billing_error`** | Anthropic-only status |
| 403 | `permission_error` | |
| 404 | `not_found_error` | |
| **409** | **`conflict_error`** | resolve-then-retry |
| **413** | **`request_too_large`** | 32 MB messages; Cloudflare may front it |
| 429 | `rate_limit_error` | rate OR tier spend-cap OR workspace limit. **Spend-cap 429 has NO retry-after** + `error.details.error_code: enforced_spend_limit_reached` [V] |
| 500 | `api_error` | |
| **504** | **`timeout_error`** | dedicated timeout type |
| **529** | **`overloaded_error`** | Anthropic-global capacity; retry w/ backoff |

### 3.3 Rate headers [V]
`retry-after` = seconds (absent on spend-cap 429s). `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` — reset values are **RFC 3339 timestamps** (unlike OpenAI's durations). `anthropic-ratelimit-unified-*` newer [R]. Headers reflect the most-restrictive active limit. `x-should-retry` present too [R].

### 3.4 Context spellings (no dedicated code) [R wire shapes; V in litellm mapper]
Message forms: `prompt is too long: 247583 tokens > 200000 maximum`; `input length and max_tokens exceed context limit: 198157 + 21333 > 200000`. Match BOTH spellings (current repo misses `prompt is too long` [V gap]).

### 3.5 Streaming [V]
Mid-stream, after HTTP 200: `event: error` + `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` — most commonly overloaded; **no `message_stop` precedes it**. Known SDK bug surfaces these as bare 200-status errors — the proxy must map `error.type → status` itself (`overloaded_error→529`, `rate_limit_error→429`, `api_error→500`) [R anthropic-sdk-python #1258]. `stop_reason: refusal` on message_delta = successful 200.

### 3.6 SDK classes [R GitHub]
Adds `OverloadedError` (529), `DeadlineExceededError` (504), `ServiceUnavailableError` (503), `RequestTooLargeError` (413), `ConflictError` (409) — finer than OpenAI's. Default 2 retries, honors retry-after.

## 4. Google Gemini v1beta

### 4.1 Envelope: google.rpc.Status [V]
```json
{"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED",
  "details": [{"@type": "...google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaDimensions": {"model": "..."}, "quotaValue": "10"}]},
              {"@type": "...google.rpc.RetryInfo", "retryDelay": "18.403470473s"},
              {"@type": "...google.rpc.ErrorInfo", "metadata": {"quotaResetDelay": "156h14m36.752463453s"}},
              {"@type": "...google.rpc.Help", "links": [...]}]}}
```
**`error.status` (string) is authoritative over both the numeric body `code` and HTTP status** [V AIP-193 + wire captures]. Stringified-JSON-in-message double envelopes observed [R gemini-cli #9248].

### 4.2 Code table (observed on Gemini) [V]

| code | status | HTTP | Meaning |
|---|---|---|---|
| 3 | INVALID_ARGUMENT | 400 | malformed; context overflow ("exceeds the maximum number of tokens") |
| 4 | DEADLINE_EXCEEDED | 504/408 | timeout |
| 5 | NOT_FOUND | 404 | bad model |
| 7 | PERMISSION_DENIED | 403 | key scope; leaked-key block ("Your API key was reported as leaked") |
| 8 | RESOURCE_EXHAUSTED | 429 | ALL quota/rate: per-minute, per-day (RPD resets midnight Pacific), spend-based rolling-10-min, free-tier |
| 13 | INTERNAL | 500 | |
| 14 | UNAVAILABLE | 503 | "The model is overloaded. Please try again later." |
| 16 | UNAUTHENTICATED | 401 | "API key not valid." |
| 1 | CANCELLED | 499 | transient [R]; retry-eligible |

### 4.3 details[] members [V protos + R wire]
- **RetryInfo.retryDelay** — authoritative wait: `"42s"`, `"18.403470473s"` (fractional), `"290.979975ms"` (ms suffix!), `"156h14m36.752463453s"` (compound), dict `{"seconds":"123"}`; prose `Please retry in Ns.` inside message.
- **QuotaFailure.violations[]** — `quotaId`, `quotaDimensions{location,model}`, `quotaValue`. **Transient-vs-terminal rule [R gemini-cli, evidence-backed]:** `quotaId` PerMinute/PerSecond + short delay → wait + retry same key; PerDay/Daily or hour-scale delay → fail over. Cooldown scope key: `(quotaId, model)`.
- **ErrorInfo** — `reason`, `domain`, `metadata{quotaResetDelay}`.
- **Help** — docs links, no machine signal. BadRequest/PreconditionFailure possible per proto.

### 4.4 Quota at wrong statuses [V/R]
Gemini-compat surfaces wrap 429 in 400/500/503 with body `code:429`/`status:RESOURCE_EXHAUSTED` [V litellm handles 5xx-wrapped-429; R compat-400 reports]. **Rule: any status whose body carries RESOURCE_EXHAUSTED/QuotaFailure/RetryInfo/body-code-429 classifies as quota — never invalid_request.** litellm's own mapper only promotes via narrow string matches and even has a bare-`"403"` substring bug (#34954) — do not inherit either behavior.

### 4.5 Streaming [V]
Default framing is a **JSON array**; `?alt=sse` gives SSE; **no `[DONE]`** — termination = final `finishReason: STOP` or close. Mid-stream errors arrive as `{"error": {...}}` chunks at HTTP 200 [V litellm handles; R captures] — scan every chunk. `finishReason` values that are NOT errors (200-success family): STOP, MAX_TOKENS, SAFETY, RECITATION (retry-eligible [R]), LANGUAGE, BLOCKLIST, PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL, IMAGE_*. `promptFeedback.blockReason` = whole prompt refused, still 200. **Silent suppression:** 200 + STOP + empty parts + consumed usage [R] — treat as error-ish (existing EmptyResponseError path is right).

### 4.6 Auth/transport [V/R]
Key via `x-goog-api-key` or `?key=`. google-genai SDK retries 408/429/500/502/503/504 up to 5×. SDK classes: `ClientError`(4xx)/`ServerError`(5xx); legacy `google.api_core.exceptions` (`TooManyRequests`, `ResourceExhausted`…).

## 5. Cross-cutting

### 5.1 Retry-signal units — one parser, four formats [V]
| Source | Format |
|---|---|
| OpenAI/Anthropic `Retry-After` | integer seconds |
| OpenAI `x-ratelimit-reset-*` | Go durations (`6m0s`, `23h18m29.144s`) |
| Anthropic `anthropic-ratelimit-*-reset` | RFC 3339 timestamps |
| Gemini `retryDelay`/`quotaResetDelay` | protobuf durations (fractional s, ms, compound h-m-s, dict) |
Plus HTTP-date `Retry-After` (spec-legal, rare). Branch on header NAME, never assume a unit. `x-should-retry` overrides everything.

### 5.2 Refusals are never errors [V]
chat `finish_reason: content_filter`; Responses refusal parts / `response.incomplete`; Anthropic `stop_reason: refusal`; Gemini SAFETY/RECITATION/BLOCKLIST/…/promptFeedback. Never rotate/cool for these. Distinguish from HTTP-level content-policy 400s (real errors, non-retryable) [V litellm ContentPolicyViolationError].

### 5.3 Billing/quota vs throughput — the split that matters [V]
Quota/billing (never tight-retry; cooldown per provider signal; rotate): OpenAI `insufficient_quota` family; Anthropic 402 `billing_error`, spend-cap 429 (no retry-after + `enforced_spend_limit_reached`), self-set-spend 400; Gemini PerDay/Daily quotas + hour-scale delays. Throughput (retry same key if short, else rotate): true rate 429s with short Retry-After; Gemini PerMinute quotas.

### 5.4 Context-window spellings (all → context_window, deterministic stop) 
chat `code: context_length_exceeded`; Responses same as 400 body code; Anthropic "prompt is too long" / "input length and max_tokens exceed context limit"; Gemini INVALID_ARGUMENT "exceeds the maximum number of tokens". Match code/patterns, never message-alone substrings (exclusions: `string_above_max_length`, invalid-user-string are NOT context errors [V litellm]).

### 5.5 Timeouts [V]
Anthropic 504 `timeout_error` (dedicated); Gemini DEADLINE_EXCEEDED/408/504; OpenAI has no API-side timeout (client `APITimeoutError`); gateways emit 408/504. litellm normalizes Anthropic 408 AND 504 → `Timeout`. All timeout-class → retryable/rotate.

### 5.6 Transport taxonomy [V installed source]
httpx: `TransportError → {TimeoutException → {Connect/Read/Write/PoolTimeout}, NetworkError → {Connect/Read/Write/CloseError}, ProtocolError → {Local/Remote}, ProxyError, UnsupportedProtocol}` (bare `httpx.Timeout` is a config class — never caught). openai/litellm hierarchy as §1.5; notable litellm classes: `ContextWindowExceededError`(400), `ContentPolicyViolationError`(400), `RejectedRequestError`(400), `BadGatewayError`(502), `ServiceUnavailableError`(503), `MidStreamFallbackError`(→503, wraps original + partial content + is_pre_first_chunk — the pre-first-token signal), `BudgetExceededError` (litellm-internal, carries category/rate_limit_type — NOT a vendor 429), `APIConnectionError` carries a FAKE status_code=500 — treat by class, not status. litellm maps Anthropic 529 → InternalServerError (signal lost — recover from status/body ourselves) [V].

### 5.7 Embedded errors in 2xx [V gomodel/litellm]
OpenRouter and others put `{"error": ...}` in 200 bodies (gomodel `ParseEmbeddedProviderError`: numeric error.code → status). Gemini streams embed errors at 200 (§4.5). Chat SSE data frames may carry error objects (§1.4). Detection belongs at every 2xx boundary.

### 5.8 Reference-implementation lessons
- **gomodel:** one flat internal enum + per-dialect renderer; embedded-error-in-2xx detection; raw error bodies capped (64 KiB) for audit, never serialized to clients; `Retry-After` passthrough.
- **plexus:** pluggable per-provider cooldown parsers behind a registry seam (`CooldownParserRegistry`); explicit default cooldown (10 min) when no timing info; 402 counted retryable (credits can be topped up mid-flight).
- **litellm (negative lessons):** substring matching on free text misroutes (bare "403" bug; our own "rate"⊂"generate"); 529 signal destroyed by collapsing to InternalServerError; quota-at-400 only survives via narrow string lists.

### 5.9 Decision matrix (grounded — feeds G1)

| Wire signal | Internal class | Action |
|---|---|---|
| 401 / UNAUTHENTICATED / authentication_error / "API key not valid" / leaked-key | authentication | **failover** (next credential; next provider when exhausted) + reauth if OAuth |
| 403 / PERMISSION_DENIED / permission_error | forbidden | rotate credential; failover on provider exhaustion |
| 404 / NOT_FOUND / model_not_found | not_found | **failover to next target** (user-approved D2); stop if single target |
| 409 / conflict_error / ALREADY_EXISTS | conflict | retry-once-same [V SDK default], then failover |
| 413 / request_too_large | invalid_request (size) | stop (client payload too large) |
| 422 | invalid_request | stop |
| 429 true-rate (short Retry-After / PerMinute quotaId) | rate_limit | retry-same if under small-cooldown threshold, else rotate; cooldown = Retry-After |
| 429/402/400 quota-billing family (insufficient_quota codes; billing_error; spend-cap 429 no-retry-after + enforced_spend_limit_reached; self-set-spend 400; PerDay quotaId; hour+ delays) | quota_exceeded | cooldown from provider signal (quotaResetDelay/reset timestamp, else default); rotate; never tight-loop |
| 400 with quota markers in body (RESOURCE_EXHAUSTED / QuotaFailure / body-code-429) at ANY status | quota_exceeded | same as above — never invalid_request |
| 400 context family | context_window_exceeded | stop, fail fast |
| 400 other invalid | invalid_request | stop (FAIL semantics — no retry hammering) |
| 408 / 504 / timeout_error / DEADLINE_EXCEEDED / APITimeoutError / httpx TimeoutException | timeout | retry-same (bounded), then rotate; counts to deadline |
| 500 / INTERNAL / api_error | server_error | retry-same w/ backoff, then rotate/failover |
| 502 / 503 / UNAVAILABLE / "model is overloaded" / 529 overloaded_error | overloaded (server_error family) | retry-same w/ backoff, then rotate/failover |
| 5xx wrapping body-429/RESOURCE_EXHAUSTED | quota_exceeded | quota handling |
| transport (ConnectError/NetworkError/ProtocolError/DNS/TLS) | api_connection | retry-same, then rotate |
| mid-stream error frame (any protocol, pre-first-token) | per underlying class | failover allowed (nothing shown yet); post-token → terminal frame to client |
| embedded error in 2xx body | per underlying class | detect at boundary, classify by body |
| refusal family (content_filter/refusal/SAFETY/RECITATION/blockReason) | NOT an error | return as normal completion |
| 200 + empty (STOP + empty parts / zero candidates) | empty_response (server_error-ish) | rotate (transient) |

### 5.10 Envelope rendering vocabulary (per dialect, output side)
- chat: `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `rate_limit_error`, `insufficient_quota`, `server_error`, `api_error`-family + `param` key; machine `code` separate or null.
- anthropic: `*_error` full enum incl. billing_error/conflict_error/request_too_large/timeout_error/overloaded_error; `request_id` echoed.
- gemini: `{error: {code: <int HTTP-ish>, message, status: <SCREAMING_CASE>}}` — statuses UNAUTHENTICATED/PERMISSION_DENIED/NOT_FOUND/INVALID_ARGUMENT/RESOURCE_EXHAUSTED/DEADLINE_EXCEEDED/INTERNAL/UNAVAILABLE.
- responses: HTTP bodies like chat; stream `response.failed` carries ResponseError codes; `error` event shape per §2.3.
