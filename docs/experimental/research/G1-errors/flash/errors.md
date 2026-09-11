# G1 Error Landscape — Four Generative LLM Protocols

Research deliverable for the error-handling overhaul of LLM-API-Key-Proxy (feeds `docs/experimental/fix-pass-plan.md` §2 "G1").
Compiled 2026-09-11. Confidence marks: **VERIFIED** (official doc / SDK source on disk), **REPORTED** (issue tracker, blog, litellm-only), **INFERRED**.

Repo cross-references: `src/rotator_library/error_handler.py` (current classifier), `src/rotator_library/core/errors.py` (protocol renderers), `stuff/GoModel` and `stuff/plexus` (sibling gateways' taxonomies), litellm at `C:\Python312\Lib\site-packages\litellm`.

---

## 1. OpenAI Chat Completions

### 1.1 Error envelope

Top-level `error` object with `message`, `type`, `param`, `code` (all four documented in the GoModel port as well — stuff/GoModel/internal/core/errors.go:77-85). `code` and `param` are nullable. openai-python `ErrorObject` types `code` as `Optional[str]` (C:\Python312\Lib\site-packages\openai\types\shared\error_object.py:11).

### 1.2 Official code/status table (developers.openai.com/api/docs/guides/error-codes)

All rows below **VERIFIED** against https://developers.openai.com/api/docs/guides/error-codes (and the same table mirrored on platform.openai.com/docs/guides/error-codes):

| HTTP | error.code / notable string | error.type | Meaning | Retry? |
|---|---|---|---|---|
| 400 | — | `invalid_request_error` | Invalid `service_tier` argument; `error.param` names the field | No |
| 401 | invalid_api_key | `invalid_request_error`/`authentication_error` family | Invalid auth, wrong org, not org member, IP not authorized | No |
| 403 | — | `permission_error` family | Country/region not supported | No |
| 429 | `credit_balance_exhausted` | `insufficient_quota` | Org has no prepaid credits | **No — retrying billing/spend/quota errors won't restore access** (official page says this explicitly) |
| 429 | — | `rate_limit_error` | Rate limit reached for requests | Yes, follow `Retry-After` |
| 429 | `slow_down` | `rate_limit_error` | "Slow down" — request rate increased too quickly | Yes, follow `Retry-After`; hold flat ≥15 min then ramp |
| 429 | `organization_spend_limit_exceeded` | `insufficient_quota` | Org spend limit reached | No |
| 429 | `project_spend_limit_exceeded` | `insufficient_quota` | Project spend limit reached | No |
| 429 | `organization_usage_limit_exceeded` | `insufficient_quota` | OpenAI-assigned usage limit reached | No (request raise) |
| 500 | — | `server_error` family (`InternalServerError`) | Server error | Retry after brief wait |
| 503 | — | — | "The engine is currently overloaded, please try again later" | Retry |
| 503 | — | — | "Slow Down" (sudden rate increase) | Reduce rate; retry |

Key billing nuance (page quote): *"For billing-related errors, inspect `error.code` to identify the specific cause. The broader `error.type` can still be `insufficient_quota`."* — i.e. **429 + `error.type: insufficient_quota` is quota, NOT rate limit.** This matters for the proxy's `quota_exceeded` vs `rate_limit` split: OpenAI marks quota as `insufficient_quota` type (often with HTTP 429). Quota-vs-rate discrimination by `code`/`type`, not just `"quota" in body`.

**Historical 402**: older docs used HTTP 402 for quota ("You exceeded your current quota"); current vocabulary is 429 `insufficient_quota`/`credit_balance_exhausted`. Treat any `insufficient_quota` marker at 402/429 as terminal-billing. (REPORTED — legacy community reports)

### 1.3 Context-window exhaustion

Real wire shape (community-confirmed, thousands of reports): HTTP **400**, `type: invalid_request_error`, `param: "messages"`, `code: "context_length_exceeded"`, message "This model's maximum context length is N tokens. However, you requested M tokens…". (REPORTED — https://community.openai.com/t/help-needed-tackling-context-length-limits-in-openai-models/617543 and https://github.com/langchain-ai/langchain/issues/16781; the code string is *not* in the openai-python 2.54 typed constants — grep of C:\Python312\Lib\site-packages\openai found zero occurrences — so it is message-level vocabulary, not an SDK enum.)

Repo note: `error_handler.py:39-54` `_CONTEXT_WINDOW_ERROR_PATTERNS` covers this; `ContextWindowExceededError` exists in litellm (exceptions.py:504, subclass of BadRequestError).

### 1.4 Rate limit headers (VERIFIED — platform.openai.com/docs/guides/rate-limits)

| Header | Meaning |
|---|---|
| `x-ratelimit-limit-requests` / `-limit-tokens` | Max per window |
| `x-ratelimit-remaining-requests` / `-remaining-tokens` | Remaining |
| `x-ratelimit-reset-requests` / `-reset-tokens` | Time until reset — **duration strings** like `1s`, `6m0s` (Go duration spelling, not seconds-int, not HTTP-date) |
| `x-ratelimit-limit-project-tokens` / `-remaining-project-tokens` / `-reset-project-tokens` | Project-scoped token limit variant |

Plus standard `Retry-After` (seconds integer) on 429s — official error-codes page instructs "follow the `Retry-After` header when it's present". Repo's `get_retry_after` reads both (error_handler.py:690-711) — matches. Units caveat: `x-ratelimit-reset-*` is a Go-style duration (`6m0s`), which the repo's `_parse_duration_string` handles; it is NOT a Unix timestamp (the repo's header branch at error_handler.py:698-711 assumes a Unix timestamp — that branch works for some Azure/OpenRouter variants, not for OpenAI's own header spelling).

### 1.5 Streaming errors

- HTTP errors stream nothing — you get the status code before any chunk. Mid-stream, the protocol has **no documented `error` chunk type** for Chat Completions; a server-side failure ends the stream abruptly (dropped connection / `data: [DONE]` may never arrive). (VERIFIED-negative: OpenAI streaming docs document no error event; REPORTED by gateway implementations — GoModel's `sse_validation.go` treats non-JSON/garbled SSE as transport failure.)
- litellm detects mid-stream embedded errors in upstream chunks and raises `MidStreamFallbackError` (litellm/exceptions.py:1086, subclass of `ServiceUnavailableError`) — the proxy already special-cases this in `error_handler.py:853-866`.
- `finish_reason: content_filter` is a **successful 200 response** delta, not an error — content filter refusals are never errors in the OpenAI wire format. (VERIFIED — platform docs; see cross-cutting §5.6.)

### 1.6 Safe-to-retry (official)

429 with `Retry-After` (rate limits only — not billing codes), 500, 503, plus connection errors and 408/timeout transport conditions. Official page: "Retrying billing, spend, or quota errors won't restore API access." openai-python auto-retries: connection errors, 408, 409, 429, ≥500 (openai/_base_client retry logic).

---

## 2. OpenAI Responses API

### 2.1 Error envelope on failed requests

Same `{"error": {...}}` envelope as Chat Completions (shared `ErrorObject`). HTTP statuses match §1.2's table. (VERIFIED — openai-python shared models.)

### 2.2 Responses-specific error object (`ResponseError`)

**VERIFIED** — C:\Python312\Lib\site-packages\openai\types\responses\response_error.py:10-38. `code` enum:

```
server_error, rate_limit_exceeded, invalid_prompt, data_residency_mismatch,
bio_policy, vector_store_timeout,
invalid_image, invalid_image_format, invalid_base64_image, invalid_image_url,
image_too_large, image_too_small, image_parse_error,
image_content_policy_violation, invalid_image_mode, image_file_too_large,
unsupported_image_media_type, empty_image_file, failed_to_download_image,
image_file_not_found
```

Fields: `code`, `message`. Note `rate_limit_exceeded` (not `rate_limit_error`) is the Responses `ResponseError.code` spelling — distinct from the chat `type` `rate_limit_error`.

**`context_length_exceeded` is NOT a Responses `ResponseError.code`** in SDK 2.54 — context exhaustion on Responses surfaces as a normal HTTP 400 `invalid_request_error`. Any Responses error.code like `context_length_exceeded` seen in the wild is REPORTED/injection-shaped, not current API vocabulary. (Relevant to fix-pass-plan G1 deliverable 7's "official code vocabulary" canonicalization.)

### 2.3 Stream error events (VERIFIED — platform.openai.com/docs/api-reference/responses-streaming/error)

- **`error` event**: terminal SSE event; fields `code`, `message`, `param`, `sequence_number`, `type: "error"`. Emitted when an error occurs mid-stream.
- **`response.failed` event**: carries the failed `response` object; the response contains `error: ResponseError` (§2.2 codes) and `status: "failed"`. `sequence_number`, `type: "response.failed"`.
- **`response.incomplete` event**: NOT an error — response ended early (length/content-filter). Carries `response.incomplete_details.reason` (`max_output_tokens`, `content_filter`). (VERIFIED — same streaming reference.)
- Related failure events: `response.mcp_list_tools.failed`, `response.mcp_call.failed` — item-scoped, response can still continue.
- **WebSocket mode errors** (VERIFIED — developers.openai.com/api/docs/guides/error-codes): `previous_response_not_found` (retry with full input, `previous_response_id: null`), `websocket_connection_limit_reached` (60-min connection limit; reopen).

Decision-relevant: a mid-stream `error` event is **terminal** for the stream; a `response.failed` gives you a machine `ResponseError.code` to map onto the proxy taxonomy (`server_error` / `rate_limit_exceeded` → retryable; `invalid_prompt` → not).

### 2.4 Rate limit headers

Identical `x-ratelimit-*` family to §1.4 (same org/project headers on `/v1/responses`). (VERIFIED — same rate-limits guide applies platform-wide.)

---

## 3. Anthropic Messages API

### 3.1 Complete `error_type` enum (VERIFIED — docs.claude.com/en/api/errors)

Envelope: top-level `{"type": "error", "error": {"type": "<error_type>", "message": "..."}, "request_id": "req_..."}`. Docs: "The API always returns errors as JSON, with a top-level `error` object that always includes a `type` and `message` value. The response also includes a `request_id` field".

| HTTP | `error.type` | Meaning | Retry guidance (official) |
|---|---|---|---|
| 400 | `invalid_request_error` | Format/content issue; **also returned when org/workspace spend limit reached** | No |
| 401 | `authentication_error` | Malformed/revoked/expired key | No |
| **402** | **`billing_error`** | Billing/payment issue | No |
| 403 | `permission_error` | Key lacks permission for resource | No |
| 404 | `not_found_error` | Resource not found | No |
| **409** | **`conflict_error`** | Concurrent modification / unique-value conflict | Case-dependent |
| 413 | `request_too_large` | >32 MB Messages / 32 MB token-counting / 256 MB batch / 500 MB files | No |
| 429 | `rate_limit_error` | Rate limit, tier monthly spend cap, or Claude Code workspace spend limit. **Tier spend-cap 429 has NO `retry-after`** | Yes for rate limits; no for spend caps |
| 500 | `api_error` | Unexpected internal Anthropic error | Retry w/ exponential backoff |
| **504** | **`timeout_error`** | Request timed out while processing | Use streaming for long requests |
| **529** | **`overloaded_error`** | API temporarily overloaded | Retry w/ backoff |

Bold rows are the ones the current proxy classifier does NOT know: `billing_error` (402), `conflict_error` (409), `timeout_error` (504), `overloaded_error` (529) — G1 deliverable 3 explicitly calls for 529/overloaded_error/billing_error classification. Every response also carries a `request-id` HTTP header matching body `request_id` (`req_…`).

Spend-cap discrimination (VERIFIED — docs.claude.com/en/api/rate-limits): 429-with-spend-cap is `rate_limit_error` **without** `retry-after`; on Messages API `error.details.error_code` is `enforced_spend_limit_reached`. Self-set spend limit surfaces as **HTTP 400 `invalid_request_error`** with message "You have reached your specified API usage limits…" — a quota-shaped 400, exactly the Gemini-style trap (see §5.1).

### 3.2 Rate limit headers (VERIFIED — docs.claude.com/en/api/rate-limits)

| Header | Meaning |
|---|---|
| `retry-after` | **Seconds** ("The number of seconds to wait until you can retry the request. Earlier retries will fail."). Not sent with spend-cap 429. |
| `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset` | Request limits; reset in **RFC 3339 datetime** |
| `anthropic-ratelimit-tokens-limit` / `-remaining` / `-reset` | Token limits (input+output total); reset RFC 3339 |
| `anthropic-ratelimit-input-tokens-reset` | (legacy spelling seen in older responses) REPORTED |
| `anthropic-workspace-id` | Which workspace the request counted against |

Units: `retry-after` = seconds-integer; `*-reset` = RFC 3339 timestamps (NOT seconds). Headers reflect the **most restrictive active limit**. Fast mode uses separate `anthropic-fast-` headers.

### 3.3 Streaming errors (VERIFIED — docs.claude.com/en/api/errors "Error events" + docs.claude.com/en/api/streaming)

Errors can occur **after a 200** mid-SSE. Shape — a terminal `error` event in the normal event stream:

```
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

- The most common mid-stream error is `overloaded_error` (HTTP-529-equivalent after 200). (Also REPORTED — Portkey docs show it arriving as the first chunk with HTTP 200.)
- There is **no `message_stop` before it** — the error event replaces the successful terminal; a client waiting for `message_stop` must treat `event: error` as the terminal.
- Proxy consequence: `core/errors.py` `StreamedAPIError` exists for exactly this; the Anthropic protocol branch in `to_protocol_payload` (core/errors.py:105-115) correctly maps into `*_error` vocabulary. Add `overloaded_error`/`timeout_error`/`billing_error` to that map.
- `stop_reason: "refusal"` on a `message_delta` is a **successful** response (content refusal, not an error). (VERIFIED — errors page: "`stop_reason` values — including `max_tokens` and `refusal` — are part of a successful 200 response".)

### 3.4 Safe-to-retry (official)

SDKs auto-retry: connection errors, 429 (with `retry-after`), 5xx, and 529 — exponential backoff, twice by default. Not retryable: 400/401/403/404/413. 409 and 402 are not in the official retry list.

---

## 4. Google Gemini API (v1beta, generativelanguage.googleapis.com)

### 4.1 Envelope: `google.rpc.Status` (VERIFIED — cloud.google.com/apis/design/errors AIP-193; google/rpc/status.proto)

```json
{
  "error": {
    "code": 429,                    // numeric google.rpc.Code
    "message": "Resource has been exhausted (e.g. check quota).",
    "status": "RESOURCE_EXHAUSTED", // canonical status string
    "details": [ { "@type": "type.googleapis.com/google.rpc.RetryInfo", ... } ]
  }
}
```

**The `error.status` string is authoritative over the HTTP code** — this is the repo's N1 fix (fix-pass-plan G1 deliverable 1: "body-status priority (Gemini `error.status` string over HTTP code)"). Real example verified: `{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}` (litellm issue #18003 — REPORTED for wire capture, status vocabulary itself VERIFIED).

### 4.2 Canonical `google.rpc.Code` numbers ↔ status strings (VERIFIED — grpc status codes + google/rpc/code.proto)

| Code | Status | Typical Gemini meaning | Retry? |
|---|---|---|---|
| 0 | OK | — | — |
| 3 | INVALID_ARGUMENT | Malformed request (also context-window/too-many-tokens at REST) | No |
| 4 | DEADLINE_EXCEEDED | Deadline exceeded | Maybe |
| 5 | NOT_FOUND | Unknown model / resource | No |
| 6 | ALREADY_EXISTS | — | No |
| 7 | PERMISSION_DENIED | Key lacks access | No |
| 8 | RESOURCE_EXHAUSTED | **Quota or rate limit** (RPM/TPM/RPD/spend) | Yes w/ retryDelay |
| 12 | UNIMPLEMENTED | — | No |
| 13 | INTERNAL | Internal error | Retry |
| 14 | UNAVAILABLE | Service down / overloaded / 503 | Retry w/ backoff |
| 16 | UNAUTHENTICATED | Bad/missing API key | No |

Gemini HTTP mapping (observed on generativelanguage): 400 INVALID_ARGUMENT, 401 UNAUTHENTICATED, 403 PERMISSION_DENIED, 404 NOT_FOUND, 429 RESOURCE_EXHAUSTED, 500 INTERNAL, 503 UNAVAILABLE. **VERIFIED** for 429/503 via ai.google.dev/gemini-api/docs/troubleshooting ("`429 RESOURCE_EXHAUSTED` or `503 UNAVAILABLE`").

### 4.3 `details[]` members (VERIFIED — installed google/rpc/error_details.proto at C:\Python312\Lib\site-packages\google\rpc\error_details.proto)

- **`RetryInfo`** (proto line 92): field `retry_delay` = `google.protobuf.Duration` — JSON spelling `"42s"`, `"3.000000001s"` (decimal-seconds with trailing `s`). "Clients should wait until `retry_delay` amount of time has passed" (line 86). Gemini also emits `RetryInfo` with `"retryDelay": "562476.752463453s"` style fractional values — the repo's `_parse_duration_string` (error_handler.py:57-115) handles this plus compound `156h14m36.752463453s`.
- **`QuotaFailure`** (line 117): `violations[]` with `subject`, `quota_metric`, `quota_value`, `quota_unit`. **The live Gemini API additionally returns `quotaId` (e.g. `GenerateContentPerMinutePerProject`) and `quotaValue` ("50") in violations** — that's what the repo's `_extract_quota_details` parses (error_handler.py:596-660) and litellm surfaces. (quotaId/quotaValue spellings: REPORTED — captured wire shapes + repo code; proto canonical fields VERIFIED.)
- **`ErrorInfo`** (line 51): `reason`, `domain`, `metadata` map. Gemini puts `quotaResetDelay` ("156h14m36.752463453s"), `quota_future*` keys in `metadata` (REPORTED — captured wire shapes; repo error_handler.py:579-588 parses `quotaResetDelay` with case variants).
- **`Help`** (line 339): `links[]` (documentation URLs).
- **`BadRequest`** (line 234): `field_violations[]` for INVALID_ARGUMENT detail.
- `PreconditionFailure` (line 214): `violations[]`.

### 4.4 Rate limits & quota semantics (VERIFIED — ai.google.dev/gemini-api/docs/rate-limits + troubleshooting)

- Limits per project (NOT per API key): RPM, TPM, RPD (TPD for some models); RPD resets midnight Pacific; spend-based limits on paid tiers.
- Exceeding any dimension → 429 RESOURCE_EXHAUSTED.
- Official retry guidance: exponential backoff on `429` and `5xx` (`503 UNAVAILABLE`); do NOT retry 400/403. Python SDK auto-retries transient errors up to 4 times, initial delay ~1s, max 60s.
- **No HTTP-header-based rate limit info** on the Gemini API — quota state arrives only in the 429 body's `details[]`. (VERIFIED-negative: no ratelimit header documented anywhere in Gemini docs.)

### 4.5 Streaming errors + framing

- **Framing (VERIFIED)**: `POST /v1beta/models/{model}:streamGenerateContent` (no `alt=sse`) returns a **JSON array** `[{chunk}, {chunk}, …]`; adding `?alt=sse` returns standard **SSE** `data: {…}` lines with no `[DONE]` sentinel — termination is signaled by the final chunk's `finishReason` (e.g. `"STOP"`) or stream close. (ai.google.dev API reference + https://github.com/musistudio/claude-code-router/issues/1315 for the failure mode when `alt=sse` is omitted.)
- **Mid-stream errors**: an error can arrive as a regular `{"error": {...}}` object inside the stream payload (JSON-array position or SSE line) after chunks were already received — litellm detects "embedded errors (e.g. 429 RESOURCE_EXHAUSTED) in streaming chunks" and raises `VertexAIError` (C:\Python312\Lib\site-packages\litellm\llms\vertex_ai\gemini\vertex_and_google_ai_studio_gemini.py:3109, 3243). **VERIFIED** (litellm source), matching wire behavior.
- **`finishReason` is not an error** — safety/refusal terminations arrive in a 200 stream: enum values from the installed proto (google/ai/generativelanguage_v1beta/types/generative_service.py:1125-1185): `STOP=1, MAX_TOKENS=2, SAFETY=3, RECITATION=4, LANGUAGE=6, BLOCKLIST=7, PROHIBITED_CONTENT=8, SPII=9, MALFORMED_FUNCTION_CALL=10, IMAGE_SAFETY=11, IMAGE_PROHIBITED_CONTENT=14, IMAGE_RECITATION=17`. `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII` are content-policy stops (non-retryable as-is, but a 200-success); `MALFORMED_FUNCTION_CALL` indicates a broken tool-call stream. Can be `finishReason` at top level **or** inside `promptFeedback.blockReason` for prompt-blocked requests.
- Empty candidates + no error → treat as provider failure (repo G14: "empty-candidates → honest error").

### 4.6 Timeouts

Google APIs: 408 (rare), 504, and `DEADLINE_EXCEEDED` (code 4) for server-side deadlines. Gemini SDKs use client-side timeouts with retry.

---

## 5. Cross-cutting / Edge cases

### 5.1 Quota masquerading as 400

- **Gemini-compat quota at HTTP 400**: the Gemini OpenAI-compatible surface and some proxies return quota-exhaustion bodies (`RESOURCE_EXHAUSTED` status / "quota" text) with **HTTP 400**. The repo's audit confirms this is the highest-severity G1 defect: "gemini `RESOURCE_EXHAUSTED` at 400 → invalid_request → no failover" (docs/experimental/audit-sweep-findings.md:96); the plan mandates "message-text quota sniffing at 400 (parity with the 429 branch)" (fix-pass-plan.md:62). Repo already sniffs at 429 (error_handler.py:953) — extend to 400.
- **Anthropic self-set spend limit at 400** `invalid_request_error` with message "You have reached your specified API usage limits" (VERIFIED — docs.claude.com/en/api/rate-limits). Same class of trap.
- **litellm behavior**: the Gemini/Vertex mapper only promotes quota-at-400 to `RateLimitError` if the *error string* matches `"429 Quota exceeded" / "Quota exceeded for" / "Resource exhausted" / "429 Unable to submit request because the service is temporarily out of capacity."` (C:\Python312\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py:1201-1220); otherwise HTTP 400 → `BadRequestError` (line 1251-1264). It also unwraps 429-bodies-inside-5xx envelopes (lines 1221-1242). So litellm gives no protection for a silent `{"code": 429, "status": "RESOURCE_EXHAUSTED"}` inside an HTTP 400 — the proxy must read `error.status`/`error.code` from the body itself (G1 deliverable 1).

### 5.2 529 / overloaded

Anthropic: 529 + `overloaded_error` (retryable, backoff). OpenAI expresses the same condition as **503** "The engine is currently overloaded" (no 529). Gemini: 503 UNAVAILABLE / "The model is overloaded." (litellm maps that string to `InternalServerError`, exception_mapping_utils.py:1243-1249). Decision: treat 529 ≈ 503-class (retry/rotate), never a hard stop. litellm's Anthropic mapper: 529 → `InternalServerError` (exception_mapping_utils.py:584-590); string `"overloaded_error"/"Overloaded"` → `InternalServerError` (line 521).

### 5.3 Billing

- OpenAI: 429 + `insufficient_quota` type / specific `error.code` (`credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded`) — never retry until fixed.
- Anthropic: dedicated 402 `billing_error`; plus spend-cap 429 without `retry-after` (`enforced_spend_limit_reached` in `error.details.error_code`).
- Gemini: quota exhaustion is RESOURCE_EXHAUSTED 429; billing-deadline issues surface as PERMISSION_DENIED 7 or specific quotaIds.
- Proxy mapping: `billing_error` → treat like quota_exceeded (rotate credential / stop retrying same credential), NOT rate_limit (no point waiting).

### 5.4 Context-window exhaustion spellings

| Protocol | Spelling | HTTP |
|---|---|---|
| OpenAI chat | `type: invalid_request_error`, `code: context_length_exceeded`, `param: messages` | 400 |
| OpenAI Responses | no dedicated code — plain 400 `invalid_request_error` (or SDK-side `BadRequestError`) | 400 |
| Anthropic | `invalid_request_error` with message "prompt is too long: N tokens > M maximum"; ALSO `prompt_too_long` code seen in newer payloads | 400 |
| Gemini | `INVALID_ARGUMENT` (3) message "…exceeds the maximum number of tokens…", sometimes `FAILED_PRECONDITION` (9) for specific limits | 400 |
| litellm | `ContextWindowExceededError` (subclass of BadRequestError) matched by string patterns incl. "context_length", "maximum context length" | 400 |

Anthropic "prompt is too long" + Gemini "exceeds the maximum" — REPORTED (wire shapes; consistent with the repo's pattern list error_handler.py:39-47). Repo `is_context_window_error_text` is the unified detector; litellm `ContextWindowExceededError` at exceptions.py:504.

### 5.5 Timeouts

- Anthropic: dedicated **504 `timeout_error`**; docs recommend streaming for long requests.
- OpenAI: no documented timeout status — client-side `APITimeoutError` (openai/_exceptions.py:112, subclass of `APIConnectionError`); servers may send 408/504 via gateways.
- Gemini: `DEADLINE_EXCEEDED` (code 4) or 408/504.
- litellm: Anthropic 408 AND 504 → `Timeout` (exception_mapping_utils.py:572-577, 605-609); transport `httpx.TimeoutException` → litellm `Timeout`.
- Repo: `proxy_timeout` currently maps to 504 in `core/errors.py:97` but for the Anthropic protocol must render as `timeout_error` (currently missing from the map at core/errors.py:106-114 → falls to `api_error`; G1 deliverable 7 fixes this).

### 5.6 Content filter / refusal — NOT errors

- OpenAI chat: 200 + `finish_reason: "content_filter"`.
- OpenAI Responses: 200 + `response.incomplete` with `incomplete_details.reason: "content_filter"`; image-specific refusal code `image_content_policy_violation` in ResponseError IS an error-object code but arrives within failed responses.
- Anthropic: 200 + `stop_reason: "refusal"` (message_delta).
- Gemini: 200 + `finishReason: SAFETY/RECITATION/BLOCKLIST/PROHIBITED_CONTENT/SPII` or `promptFeedback.blockReason`.
- Server-side content-policy errors that ARE errors: OpenAI 400 with `content_filter`/`ContentPolicyViolationError` (litellm class at exceptions.py:588), litellm maps Gemini "The response was blocked." → `ContentPolicyViolationError` with status 400 (exception_mapping_utils.py:1184-1200).
- Rule: never rotate/cool a credential for in-band refusals; distinguish `finishReason=SAFETY` (success path) from HTTP-level content policy 400 (request-level, non-retryable).

### 5.7 Transport exception taxonomy

httpx hierarchy (used by the proxy's native path): `httpx.TransportError` → {`TimeoutException` (→ `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `PoolTimeout`), `NetworkError` (→ `ConnectError`, `ReadError`, `WriteError`, `CloseError`)}. `httpx.HTTPStatusError` wraps non-2xx when `raise_for_status` is used. Proxy classifies `(TimeoutException, ConnectError, NetworkError)` → `api_connection` (error_handler.py:1029-1034).

openai-python (base of litellm's classes — C:\Python312\Lib\site-packages\openai\_exceptions.py:83-162):
```
APIError
├── APIResponseValidationError
├── APIConnectionError          (no status; retry)
│   └── APITimeoutError
└── APIStatusError              (has .status_code, .response)
    ├── BadRequestError (400)
    ├── AuthenticationError (401)
    ├── PermissionDeniedError (403)
    ├── NotFoundError (404)
    ├── UnprocessableEntityError (422)
    ├── RateLimitError (429)
    └── InternalServerError (>=500)
```

litellm additions (C:\Python312\Lib\site-packages\litellm\exceptions.py): `ContextWindowExceededError` (:504, from BadRequestError), `RejectedRequestError` (:546), `ContentPolicyViolationError` (:588), `ServiceUnavailableError` (:633, 503), `BadGatewayError` (:681, 502), `BudgetExceededError` (:960, plain Exception), `InvalidRequestError` (:990, from BadRequestError), `MidStreamFallbackError` (:1086, from ServiceUnavailableError), `AuthenticationError` (:129), `RateLimitError` (:413), `APIConnectionError` (:819), `Timeout` (:inherits openai.APITimeoutError), `NotFoundError` (:173), `PermissionDeniedError` (:374), `InternalServerError` (:729), `OpenAIError` (:905, base — too broad, deliberately excluded from the proxy's server_error check per error_handler.py:1127).

litellm per-provider mapper sources for which exceptions get raised from which providers (exception_mapping_utils.py): Anthropic region ~lines 510-620 (408→Timeout, 529→InternalServerError, 504→Timeout, 502→BadGateway, 503→ServiceUnavailable, 400/413→BadRequest, 429→RateLimit, 403→PermissionDenied, 404→NotFound, 401→Authentication); Gemini/Vertex region ~lines 1100-1290 (quota strings→RateLimit, "API key not valid."→Authentication, "The response was blocked."→ContentPolicyViolation, "The model is overloaded."→InternalServerError, else status-based). `anthropic_interface/exceptions/exception_mapping_utils.py:15-24` holds litellm's Anthropic HTTP→`error_type` reverse map (incl. 529→overloaded_error, 413→request_too_large) — useful as the canonical Anthropic vocabulary table in code.

### 5.8 Sibling gateways' taxonomies (repo-local references)

**GoModel** (stuff/GoModel/internal/core/errors.go:19-36): flat enum — `provider_error` (5xx→502 default), `rate_limit_error` (429), `invalid_request_error` (4xx→400), `authentication_error` (401; **403 folds into it** at ParseProviderError:259-266), `not_found_error` (404), `permission_error` (403), `internal_error` (500, gateway-internal only). Renders dialect-specific (Anthropic envelope via `anthropicapi.ErrorFromGateway` — error_support.go:33-39). Notable handling: **errors embedded in 2xx bodies** (`ParseEmbeddedProviderError`, errors.go:318-324 — OpenRouter puts `{"error": …}` in 200s; numeric `error.code` becomes HTTP status, else 502); **OpenRouter `metadata.raw` preference** (errors.go:441-454); error model carries `param` + `code` (mirrors OpenAI), `model_not_found` code on 404 (errors.go:236-238). Lesson for G1: single enum + per-dialect renderer + embedded-error-in-2xx detection.

**Plexus** (TS gateway, stuff/plexus/packages/backend/src/services/dispatch/): `failover-policy.ts` — retryable = 402 (insufficient credits) + configured statuses (429/5xx); OAuth retryable = 5xx, 429, 402, plus message tokens (timeout/ECONNREFUSED/ETIMEDOUT/network/socket/temporary/unavailable); network tokens matched on `error.code`/message. `cooldown-parsers.ts` — provider-specific parsers: openai-codex "Try again in ~N min", OpenRouter `error.metadata.retry_after_seconds` + `metadata.headers['Retry-After']` + regex fallbacks. Dispatcher (dispatcher.ts:743-769): for 429/503 check `Retry-After` header first, then provider parsers; **default cooldown 10 min** when non-429 errors carry no timing. Anthropic mid-stream `overloaded_error` tested as chunk `{"type":"error","error":{"type":"overloaded_error",…}}` (transformers tests). Lesson for G1: provider-pluggable cooldown parsers + 402-as-retryable decision + explicit default cooldown.

### 5.9 Decision-matrix-relevant summary (for G1 deliverable 3)

| Condition | Rotate credential | Retry same | Cool down | Terminal (no retry) |
|---|---|---|---|---|
| 429 rate limit (true rate) | yes | yes if retry-after < threshold | Retry-After / ratelimit-reset | — |
| 429 quota (`insufficient_quota`, spend-cap 429, `enforced_spend_limit_reached`) | yes (per-key quota) | no | hours-days (quotaResetDelay / reset timestamp) | if account-level billing |
| 402 billing (anthropic) | yes | no | — | billing must be fixed |
| 400 quota-disguised (gemini-compat, anthropic self-spend-limit) | yes | no | from details[] | — |
| 400 real invalid request | no | no | — | yes |
| 400 context window | no | no | — | yes |
| 401 auth | yes (+reauth) | no | — | — |
| 403 | yes | no | — | — |
| 404 model | decide+pin (G1) | — | — | likely |
| 409 | case-dependent | — | — | — |
| 408/504/timeout_error/DEADLINE_EXCEEDED | yes | yes | — | — |
| 500 / 500 INTERNAL / api_error | yes | yes w/ backoff | — | — |
| 502 / 503 / UNAVAILABLE / overloaded (529) | yes | yes w/ backoff | — | — |
| 5xx-embedded-429 body | yes | no | from body | — |
| Stream `error` event (anthropic overloaded_error, responses error) | yes (if before first token: full retry; mid-stream: cannot re-emit cleanly) | — | — | stream terminal |
| finish_reason/refusal/SAFETY (200) | NO — not an error | — | — | — |

---

## Source index

Official (VERIFIED):
- https://developers.openai.com/api/docs/guides/error-codes (also platform.openai.com/docs/guides/error-codes)
- https://platform.openai.com/docs/guides/rate-limits
- https://platform.openai.com/docs/api-reference/responses-streaming/error
- https://docs.claude.com/en/api/errors (error table, shapes, request-id, streaming-error note)
- https://docs.claude.com/en/api/rate-limits (headers, spend caps, fast mode)
- https://docs.claude.com/en/api/streaming
- https://ai.google.dev/gemini-api/docs/troubleshooting (retry strategy)
- https://ai.google.dev/gemini-api/docs/rate-limits (per-project limits, RPD reset)
- https://cloud.google.com/apis/design/errors (AIP-193, google.rpc.Status)
- https://grpc.github.io/grpc/core/md_doc_statuscodes.html (code↔status table)

SDK sources on disk (VERIFIED):
- C:\Python312\Lib\site-packages\openai\_exceptions.py, types\shared\error_object.py, types\responses\response_error.py
- C:\Python312\Lib\site-packages\google\rpc\code.proto, error_details.proto (RetryInfo/QuotaFailure/ErrorInfo/Help/PreconditionFailure/BadRequest)
- C:\Python312\Lib\site-packages\google\ai\generativelanguage_v1beta\types\generative_service.py (FinishReason enum)
- C:\Python312\Lib\site-packages\litellm\exceptions.py, litellm_core_utils\exception_mapping_utils.py, anthropic_interface\exceptions\exception_mapping_utils.py, llms\vertex_ai\gemini\vertex_and_google_ai_studio_gemini.py

Repo (VERIFIED, file:line):
- src/rotator_library/error_handler.py, src/rotator_library/core/errors.py
- docs/experimental/fix-pass-plan.md (§2 G1), docs/experimental/audit-sweep-findings.md
- stuff/GoModel/internal/core/errors.go, internal/server/error_support.go
- stuff/plexus/packages/backend/src/services/dispatch/failover-policy.ts, dispatcher.ts, services/runtime/cooldown-parsers.ts, transformers tests

REPORTED (web):
- context_length_exceeded wire shapes: community.openai.com/t/617543, github.com/langchain-ai/langchain/issues/16781
- Anthropic mid-stream overloaded with HTTP 200: docs.portkey.ai (catch-anthropic-errors)
- Gemini RESOURCE_EXHAUSTED wire example: github.com/BerriAI/litellm/issues/18003
- streamGenerateContent JSON-array-vs-SSE failure: github.com/musistudio/claude-code-router/issues/1315
- Anthropic legacy `anthropic-ratelimit-input-tokens-reset` header spelling
