# G1 Error Landscape — Four Generative LLM Protocols (light)

Scope: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google Gemini v1beta
(generativelanguage `streamGenerateContent`/`generateContent`), plus cross-cutting
(rate-limit/retry, streaming, quota-at-400, billing, context window, refusal-vs-error,
timeouts, transport/litellm taxonomy, reference implementations).
Feeds `docs/experimental/fix-pass-plan.md` §G1; contrasts with the current classifiers in
`src/rotator_library/error_handler.py` and `src/rotator_library/core/errors.py`.

Confidence legend — every bullet ends with one:
**VERIFIED** = official docs or installed SDK/litellm source read directly.
**REPORTED** = issue trackers, blogs, third-party docs, litellm-only behavior.
**INFERRED** = reasoned from the above, needs a pin/test before relying on it.

Local source versions pinned: `litellm 1.100.0`, `openai 2.54.0`, `httpx 0.28.1`,
`google-generativeai` (legacy) `0.8.5` (all VERIFIED via `importlib.metadata` /
direct file reads on 2026-09-11). `anthropic` SDK and `google-genai` (new) SDK are
**not installed** here; their shapes are taken from GitHub source + official docs.

---

## 1. OpenAI — Chat Completions (`/v1/chat/completions`)

### 1.1 Non-stream error envelope (VERIFIED)

```json
{ "error": { "message": "...", "type": "invalid_request_error",
  "param": "messages", "code": "context_length_exceeded" } }
```

- `type` is the coarse family; `code` is the fine-grained machine string; `param`
  names the offending parameter (may be `null`, may be a path like
  `messages[1].tool_calls`). VERIFIED — `C:\Python312\Lib\site-packages\openai\_exceptions.py:48-80`
  (`APIError` parses `code`/`param`/`type` from the body dict); real-world bodies in
  §1.2.
- `error.type` values seen on the wire: `invalid_request_error`, `authentication_error`
  (not directly observed; SDK maps 401 → `AuthenticationError`), `permission_error`
  (403 → `PermissionDeniedError`), `not_found_error` (404), `rate_limit_error` (429),
  `server_error` (500), `insufficient_quota` (429 billing — **type and code are both
  the string `insufficient_quota`**, not `rate_limit_error`). VERIFIED for
  `insufficient_quota` — community-quoted body
  `{'message': 'You exceeded your current quota…', 'type': 'insufficient_quota',
  'param': None, 'code': 'insufficient_quota'}` (REPORTED, dozens of independent
  reports, e.g. https://community.openai.com/t/429-error-insufficient-quota/492350);
  SDK mapping table VERIFIED — https://github.com/openai/openai-python/blob/main/README.md
  (400→BadRequestError, 401→AuthenticationError, 403→PermissionDeniedError,
  404→NotFoundError, 422→UnprocessableEntityError, 429→RateLimitError, ≥500→InternalServerError).
- Special internal type observed once: `auth_subrequest_error` with code
  `internal_error` at HTTP 500 on fine-tuning (REPORTED —
  https://community.openai.com/t/api-error-code-500-fine-tuned-model/791933).
  Classifier must not assume a closed `type` vocabulary. [INFERRED recommendation]

### 1.2 Official error-code table (VERIFIED — https://developers.openai.com/api/docs/guides/error-codes)

| HTTP | `error.code` / message | Meaning / action |
|---|---|---|
| 400 | Invalid `service_tier` (`invalid_request_error`, `param: service_tier`) | Requested/resolved tier not allowed for project; `auto`/omitted can also hit it |
| 401 | Invalid Authentication / Incorrect API key / Not-a-member-of-org | Revoked key, wrong org/project key, IP allowlist (`IP not authorized` is also 401, not 403) |
| 403 | Country/region/territory not supported | Geo block, not a key-permission problem |
| 429 | `credit_balance_exhausted` | Prepaid balance is zero → add credits; retry never helps |
| 429 | (plain) Rate limit reached for requests | RPM/TPM/RPD/TPD/IPM whichever first; honor `Retry-After` |
| 429 | `organization_spend_limit_exceeded` | Org monthly spend cap; resumes at reset unless raised |
| 429 | `project_spend_limit_exceeded` | Project monthly spend cap; sibling projects unaffected |
| 429 | `organization_usage_limit_exceeded` | OpenAI-assigned (not self-configured) monthly usage cap; request increase |
| 500 | `server_error` "The server had an error…" / "An error occurred while processing…" | Retry briefly, then support with request id |
| 503 | "The engine is currently overloaded, please try again later" | Retry with backoff; check status page |
| 503 | "Slow Down" (PAYG shared models) | **Reduce to original rate ≥15 min, then ramp gradually** — backoff alone keeps you throttled |

- Billing/spend/quota errors: inspect `error.code`; broader `error.type` may remain
  `insufficient_quota`. "Retrying billing, spend, or quota errors won't restore API
  access." VERIFIED — same page.
- Legacy but still live: `type=insufficient_quota, code=insufficient_quota`,
  message "You exceeded your current quota, please check your plan and billing
  details." VERIFIED as currently returned (2024–2026 reports) though absent from the
  new docs table; REPORTED — see §1.1 links plus https://prismix.dev/guides/openai-insufficient-quota.
  **Critical for G1: both `insufficient_quota` and `rate_limit_exceeded` raise the
  same SDK class (`openai.RateLimitError`, HTTP 429) — only `error.code` tells
  billing apart from throughput.** REPORTED (prismix guide) + VERIFIED SDK mapping
  (all 429 → RateLimitError).

### 1.3 Context-window exhaustion spellings (VERIFIED via quoted bodies)

- `type: invalid_request_error, code: context_length_exceeded`, HTTP 400, `param`
  usually `messages`. Message forms: "This model's maximum context length is 8192
  tokens. However, you requested 9850 tokens (1750 in the messages, 8100 in the
  completion)." / "…your messages resulted in 116313 tokens." REPORTED (multiple
  community threads, consistent shape) — treat as VERIFIED shape, REPORTED wording
  variants.
- litellm maps to `ContextWindowExceededError(status_code=400)` via
  `ExceptionCheckers.is_error_str_context_window_exceeded` substrings (`exceed context
  limit`, `this model's maximum context length is`, `is longer than the model's
  context length`, `exceeds the maximum number of tokens allowed`, …). VERIFIED —
  `C:\Python312\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py`
  (`ExceptionCheckers` class). Note the exclusions: `string_above_max_length` and
  `invalid 'user' … string too long` are explicitly NOT context errors (same source).
- `code: context_length_exceeded` at HTTP 400 is relied on by third parties for
  auto-compaction (opencode #17746). REPORTED — https://github.com/anomalyco/opencode/issues/17746.
  G1 should match on `code`, not message substrings. [INFERRED]

### 1.4 Content-filter: REFUSAL is not an error (VERIFIED)

- Non-stream: model returns HTTP 200 with `finish_reason: content_filter`; message
  content is a refusal string. No error envelope. VERIFIED — openai-python has
  `ContentFilterFinishReasonError` (raised by `.parse()` helpers, not by the API) and
  `LengthFinishReasonError` — `C:\Python312\Lib\site-packages\openai\_exceptions.py:166-188`.
- litellm maps safety text (`content_policy_violation` in body, "request was rejected
  as a result of the safety system") to `ContentPolicyViolationError(status_code=400)`.
  VERIFIED — `_map_openai_exception` in `exception_mapping_utils.py`.
- G1 consequence: a 200 with `finish_reason ∈ {content_filter, length}` must never be
  classified as `server_error`/`api_connection`. [INFERRED]

### 1.5 Rate-limit headers (VERIFIED — https://developers.openai.com/api/docs/guides/rate-limits)

- `Retry-After: 56` — **seconds** (integer), present only on temporary 429s. "It does
  not mean that quota, billing, or other errors that require user action can be
  resolved by retrying."
- `x-ratelimit-limit-requests / -limit-tokens / -remaining-requests /
  -remaining-tokens / -reset-requests ("1s") / -reset-tokens ("6m0s")` plus
  project-scoped `x-ratelimit-limit-project-tokens / -remaining-project-tokens /
  -reset-project-tokens`. Reset values are **durations** (`12ms`, `4m12.172s`,
  `23h18m29.144s`), not timestamps. VERIFIED — same page + community header dumps
  (REPORTED corroboration: https://community.openai.com/t/how-can-we-check-rate-limit-openai-api/414613).
- Historical variant `x-ratelimit-*-tokens_usage_based` observed 2023 (REPORTED —
  https://community.openai.com/t/what-is-new-field-in-rate-limits-x-ratelimit-reset-tokens-usage-based/541210).
  Parser should tolerate unknown `x-ratelimit-*` siblings. [INFERRED]
- Limits are per-org **and** per-project, per-model; five metrics at once (RPM, TPM,
  RPD, TPD, IPM); shared limits across model families; `x-request-id` on every
  response (`_request_id` on SDK objects). VERIFIED — rate-limits page + openai-python
  README ("All object responses provide `_request_id` … from `x-request-id`").

### 1.6 Streaming (SSE, `text/event-stream`, `data: [DONE]` terminator)

- Chunk object is `chat.completion.chunk` with `choices[].delta` (never
  `choices[].message`); usage arrives only in a final empty-`choices` chunk when
  `stream_options.include_usage: true`; stream ends with literal `data: [DONE]`.
  REPORTED (formal-ai #604 documents the exact required shape; fastapi-sse-lab demo;
  OpenAI reference "Can also be empty for the last chunk if you set
  `stream_options: {\"include_usage\": true}`" —
  https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).
- **Mid-stream error shape**: server emits an SSE data chunk carrying
  `{'error': {'message': 'The server had an error while processing your request…',
  'type': 'server_error', 'param': None, 'code': None}}` **inside the 200 stream**
  (no new HTTP status possible). REPORTED — https://community.openai.com/t/api-error-in-streaming-mode/271014
  (multiple 2023 reports incl. `gpt-4-0613` + function-call). openai-python surfaces
  streaming failures as `APIError` ("Error occurred while streaming", legacy
  `api_requestor.py:687`, REPORTED). Treat any `data:` chunk containing a top-level
  `error` member as terminal stream error. [INFERRED]
- `stream_options: null` sent explicitly has caused rotation storms in this repo's
  history (fix-plan G1 §4); wire `null` ≠ omitted. [INFERRED from fix-plan]

### 1.7 SDK exception hierarchy + retry policy (VERIFIED — installed source)

- Hierarchy: `OpenAIError → APIError → {APIConnectionError → APITimeoutError,
  APIResponseValidationError, APIStatusError → {BadRequestError 400,
  AuthenticationError 401 (+OAuthError), PermissionDeniedError 403, NotFoundError 404,
  ConflictError 409, UnprocessableEntityError 422, RateLimitError 429,
  InternalServerError 5xx}}`, plus `LengthFinishReasonError`,
  `ContentFilterFinishReasonError`, `WebSocketConnectionClosedError`,
  `WebSocketQueueFullError`. VERIFIED — `openai/_exceptions.py:36-207`.
  `APIStatusError` carries `.status_code`, `.response`, `.request_id` (from
  `x-request-id`), `.body`, `.code/.param/.type`. VERIFIED — same file, lines 93–104.
- Default timeout: **600 s total, 5 s connect** (`DEFAULT_TIMEOUT =
  httpx.Timeout(timeout=600, connect=5.0)`); default retries **2**;
  `INITIAL_RETRY_DELAY 0.5 s`, `MAX_RETRY_DELAY 8.0 s`, `MAX_RETRY_AFTER_DELAY 120 s`
  (a larger `Retry-After` disables retry). VERIFIED — `openai/_constants.py` (full
  file, 15 lines).
- Retried by default: connection errors, **408, 409, 429, ≥500**; timeouts are
  retried twice. VERIFIED — `openai/_base_client.py::_should_retry` (read directly)
  + https://github.com/openai/openai-python/blob/main/README.md ("Certain errors are
  automatically retried 2 times… 408, 409, 429, >=500").
- `x-should-retry: true/false` response header **overrides** the status-code retry
  decision. VERIFIED — `_should_retry` source. Anthropic sends this header too
  (REPORTED — anthropic-sdk-typescript #357 shows `'x-should-retry': 'true'` on a
  429); a proxy should forward-or-honor it. [INFERRED]
- `Retry-After` parsing returns **seconds as float** (supports HTTP-date too, per
  docstring "number of seconds (not milliseconds)"). VERIFIED — `_parse_retry_after_header`
  docstring in `_base_client.py`.

---

## 2. OpenAI — Responses (`/v1/responses`,incl. streaming + WebSocket mode)

### 2.1 Non-stream error envelope

- Same `{"error": {"message","type","param","code"}}` envelope as Chat (the proxy's
  `StructuredAPIResponseError.to_protocol_payload` default branch already assumes
  this; `core/errors.py:129-135`). VERIFIED for 500 shape:
  `{'error': {'message': 'An error occurred while processing your request… request ID
  req_…', 'type': 'server_error', 'param': None, 'code': 'server_error'}}` —
  REPORTED (openai-python #2298; n8n forum 2025-11; Azure Q&A 2026-04 shows Azure's
  variant message with `type/code: server_error` at HTTP 500).
- 400 example: `{'message': 'No tool output found for function call call_xxx.',
  'type': 'invalid_request_error', 'param': 'input', 'code': None}` — note
  **`code: None`**; message carries the machine signal. REPORTED —
  https://community.openai.com/t/need-help-no-tool-output-found-for-function-call-error-after-function-call-in-responses-api/1245970.
  G1 must handle `code: null` (match on message/type, not code alone). [INFERRED]

### 2.2 `Response.error` object vocabulary (response-level, NOT HTTP errors)

When generation fails the Response object itself carries `status: failed` +
`error: {code, message}`. Code vocabulary observed in the OpenAPI-derived community
reference (sync of Jan-2025 spec, REPORTED — community "Responses API streaming —
the simple guide", post by `_j`):
`server_error`, `rate_limit_exceeded`, `invalid_prompt`, `vector_store_timeout`,
`invalid_image`, `invalid_image_format`, `invalid_base64_image`, `invalid_image_url`,
`image_too_large`, … (list truncated in source; treat as open vocabulary).
- `error` reference type exists in the official Python reference index
  (`class ResponseError: "An error object returned when the model fails to generate
  a Response."` with `code`/`message`). REPORTED —
  https://developers.openai.com/api/reference/python/resources/responses.
- `status` enum: `completed | failed | in_progress | cancelled | queued | incomplete`.
  REPORTED — same + https://zenmux.ai/docs/api/openai/openai-responses.html
  (third-party mirror of the reference). `incomplete_details.reason` e.g.
  `max_output_tokens` ("reached max_output_tokens"). REPORTED (zenmux mirror).
- **G1 trap: `rate_limit_exceeded` (Response.error code, underscores) vs
  `rate_limit_error` (HTTP error type, singular "error") are different strings in
  different positions.** [INFERRED — high decision relevance]

### 2.3 Streaming error events (VERIFIED from official streaming-events reference)

- Terminal failure: `type: response.failed` carrying the failed `response` object
  (with `status: failed` + non-empty `error`). Terminal incomplete:
  `type: response.incomplete` (+ `incomplete_details`). VERIFIED —
  https://developers.openai.com/api/reference/resources/responses/streaming-events
  ("An event that is emitted when a response fails/incompletes").
- Out-of-band stream error: `type: error` event with `{code, message, param,
  sequence_number}` (distinct from `response.failed`). REPORTED — vercel/ai #16021
  quotes a live one:
  `{"type":"error","error":{"type":"server_error","code":"server_error","message":"An
  error occurred…","param":null},"sequence_number":2}` arriving right after
  `response.in_progress` (gpt-5.4, schema-triggered). Every stream event carries
  `sequence_number`. VERIFIED — streaming-events reference lists `sequence_number`
  on each event.
- Realtime sibling shape (same error vocabulary family): `response.done` with
  `response.status: failed` + `status_details.error: {type: server_error, code: null,
  message}`. REPORTED — MS Q&A 2026-03 (Azure Realtime `resp_…` failure).
- Community lifecycle summary (REPORTED, consistent with the reference): only read
  usage from `response.completed`; on `response.incomplete`/`response.failed` stop
  assembling and surface reason/error; `error` event → abort entire stream, optionally
  retry. https://community.openai.com/t/responses-api-streaming-the-simple-guide-to-events/1363122.

### 2.4 WebSocket-mode-only errors (VERIFIED — official error-codes page)

- `previous_response_not_found` — `previous_response_id` unresolvable; retry with
  full context + `previous_response_id: null`.
- `websocket_connection_limit_reached` — 60-minute connection cap; open a new
  connection. VERIFIED — https://developers.openai.com/api/docs/guides/error-codes
  ("WebSocket mode errors" section).

### 2.5 Responses rate-limit / retry semantics

- Same HTTP headers as Chat (`Retry-After`, `x-ratelimit-*`); SDK default 2 retries
  on 408/409/429/5xx. VERIFIED — §1.5/§1.7 (shared client).
- Practical 2026 guide (REPORTED — https://kissapi.ai/blog/openai-responses-api-rate-limit-handling-2026.html):
  429 recovery = honor `Retry-After`, token-aware queue, adaptive concurrency.
- Known server-side flake: File-Search + Responses returning 500 `server_error`
  with request id (REPORTED — openai-python #2298). Retryable. [INFERRED]

---

## 3. Anthropic — Messages (`/v1/messages`)

### 3.1 Non-stream error envelope (VERIFIED — https://platform.claude.com/docs/en/api/errors)

```json
{ "type": "error",
  "error": { "type": "not_found_error", "message": "…" },
  "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy" }
```

Plus `request-id` response header on **every** response (same value). SDKs expose it
(`message._request_id` in Python/TS; raw-response accessors elsewhere). VERIFIED —
same page ("Request ID" section with 8 language samples).

### 3.2 Official `error.type` enum + HTTP mapping (VERIFIED — same page)

| HTTP | `error.type` | Notes |
|---|---|---|
| 400 | `invalid_request_error` | Also used for "other 4xx not listed"; **org/workspace spend limits YOU set → 400** (message `You have reached your specified … usage limits`), except Claude-Code workspace limits which may return 429 |
| 401 | `authentication_error` | Malformed/revoked/expired key; on AWS also SigV4 problems |
| 402 | `billing_error` | Payment/billing problem (Anthropic-only status code) |
| 403 | `permission_error` | Key lacks permission (org/workspace settings) |
| 404 | `not_found_error` | Bad endpoint path or resource/model id |
| 409 | `conflict_error` | Concurrent modification / uniqueness conflict → resolve, then retry |
| 413 | `request_too_large` | Over per-endpoint byte cap (Messages 32 MB, Batch 256 MB, Files 500 MB); on direct API, Cloudflare returns it before Anthropic does |
| 429 | `rate_limit_error` | RPM/ITPM/OTPM exceeded, **tier monthly spend cap**, Claude-Code workspace spend limit, or acceleration-limit burst |
| 500 | `api_error` | Retry with backoff; contact support with request id if persistent |
| 504 | `timeout_error` | Prefer streaming for long requests |
| 529 | `overloaded_error` | Temporary overload (Anthropic's own 529, not standard) |

- `type` values may grow over time (versioning policy); handle unknown values
  gracefully. VERIFIED — "Error shapes" section.
- Distinguishing 429 sub-causes: tier spend-cap 429 has **no `retry-after`** and
  carries `error.details.error_code: enforced_spend_limit_reached`; self-set spend
  limit is a **400** (`You have reached your specified …`); Claude-Code workspace
  limit may be 429 **with** `retry-after`. VERIFIED — errors page + rate-limits page
  (https://platform.claude.com/docs/en/api/rate-limits, "Reaching your spend cap").
  **G1 must branch on `error.details.error_code` + presence of `retry-after`, not on
  429 alone.** [INFERRED — high relevance]
- Billing: 402 `billing_error` is distinct from 429-spend-cap. VERIFIED — errors
  page. SDK type stubs also list `BillingError`/`GatewayTimeoutError` response
  variants. REPORTED — https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md.
- All SDK error classes expose `.type` (the API error-type string) for programmatic
  classification without message parsing. REPORTED —
  https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/error-codes.md.

### 3.3 Context-window spellings (no dedicated code — message match required)

- Anthropic surfaces context exhaustion as **HTTP 400 `invalid_request_error`**
  with messages: `prompt is too long: 247583 tokens > 200000 maximum`, or `input
  length and max_tokens exceed context limit: 198157 + 21333 > 200000, decrease
  input length or max_tokens and try again`. REPORTED (claude-error-handbook;
  claude-code #5346 with verbatim bodies; hermes-agent #813). litellm's anthropic
  mapper keys on `"prompt is too long"` / `"prompt: length"` then
  `ContextWindowExceededError`. VERIFIED — `_map_anthropic_exception` in
  `exception_mapping_utils.py`.
- G1 must add `prompt is too long` and `exceed context limit`/`exceed…context` to the
  400 message sniff list (current `error_handler.py:40-47` patterns miss the
  `prompt is too long` spelling — gap confirmed by reading). [INFERRED]

### 3.4 Rate-limit headers + retry semantics (VERIFIED)

- Per-response: `anthropic-ratelimit-requests-limit/-remaining/-reset`,
  `anthropic-ratelimit-tokens-limit/-remaining/-reset` (combined, rounded to nearest
  1k), `anthropic-ratelimit-input-tokens-limit/-remaining/-reset`,
  `anthropic-ratelimit-output-tokens-limit/-remaining/-reset`; on 429:
  `retry-after` (**seconds**). Reset values are **RFC 3339 timestamps**
  (`2026-07-21T14:30:00Z`), unlike OpenAI's durations. VERIFIED —
  https://docs.aws.amazon.com/claude-platform/latest/userguide/rate-limits.html
  (header table) + anthropic-sdk-typescript #357 (live 429 header dump incl.
  `retry-after` absent + `x-should-retry: true`) + throttle.com profile (REPORTED
  corroboration).
- Newer: `anthropic-ratelimit-unified-*` headers for unified (Max-plan) budgets.
  REPORTED — openclaw #56047.
- Rate model: token-bucket per org per model class; dimensions RPM + ITPM
  (uncached input + cache-write count; cache-read excluded except Haiku 3.5) +
  OTPM (real-time output; `max_tokens` does not consume OTPM budget); short bursts
  can 429 below nominal RPM; sharp traffic ramps hit "acceleration limits" (429,
  fix = ramp gradually, hold steady). VERIFIED — rate-limits page.
- SDKs auto-retry transient failures (connection errors, rate limits, 5xx) with
  exponential backoff, **twice by default, honoring `retry-after`**;
  `max_retries`/`maxRetries` configures/disables. VERIFIED — errors page ("The
  official SDKs automatically retry…").
- 529 vs 429 ownership: 529 = Anthropic-global capacity (retry/backoff, consider
  Haiku/fallback model, queue); 429 = your org's RPM/TPM/spend/acceleration budget
  (throttle, backoff, inspect headers). VERIFIED — errors page + rate-limits page.

### 3.5 Streaming error frame (VERIFIED — https://platform.claude.com/docs/en/api/streaming)

```sse
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

- Mid-stream errors arrive as `event: error` **after HTTP 200**; "error handling
  doesn't follow these standard mechanisms." VERIFIED — errors page ("When receiving
  a streaming response… See Error events") + streaming page ("Error events" section).
- Known SDK bug (REPORTED — anthropic-sdk-python #1258): `_make_status_error`
  dispatches on the original HTTP 200, so mid-stream `overloaded_error` surfaces as
  bare `APIStatusError(status_code=200)` instead of `OverloadedError(529)`. A proxy
  must map SSE `error.type → status` itself (`overloaded_error→529,
  rate_limit_error→429, api_error→500`, …); the issue proposes exactly this table.
- `ping` events may interleave; unknown event types must be tolerated. VERIFIED —
  streaming page.
- Thinking-signature stream events (`signature_delta`) exist; dropping them corrupts
  extended-thinking turns, but that is G13 territory, noted here only. VERIFIED —
  streaming page ("Thinking delta").

### 3.6 SDK exception hierarchy (REPORTED — GitHub source, package not installed)

`AnthropicError → APIError → {APIStatusError → {BadRequestError 400,
AuthenticationError 401, PermissionDeniedError 403, NotFoundError 404,
ConflictError 409, RequestTooLargeError 413, UnprocessableEntityError 422,
RateLimitError 429, InternalServerError ≥500, OverloadedError 529,
DeadlineExceededError (504), ServiceUnavailableError 503}, APIConnectionError,
APITimeoutError, APIResponseValidationError}, RetryableError` (opt-into-retry
marker raisable from middleware). REPORTED —
https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_exceptions.py
(161 lines) + `_client.py` status→class dispatch (400/401/403/404/409/413/422/429/
529/≥500). Note: **504 maps to `DeadlineExceededError`, 503 to
`ServiceUnavailableError`** — finer than OpenAI's single `InternalServerError`.
- `APIStatusError` parity with OpenAI's (`.status_code`, `.response`,
  `.request_id`-capable raw response). REPORTED — same sources.
- Default `max_retries=2`, `DEFAULT_TIMEOUT` present in `_constants.py` (REPORTED —
  `from ._constants import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES` in `__init__.py`).
  Current repo `core/errors.py:107` maps `proxy_timeout → timeout_error` for the
  anthropic branch — consistent. VERIFIED (local file).

### 3.7 Timeouts

- 504 `timeout_error` is a first-class API error (not just a client-side socket
  timeout); guidance = stream or batch long requests, TCP keep-alive, SDKs enforce a
  10-minute non-streaming expectation. VERIFIED — errors page ("Long requests").
- litellm maps Anthropic 408→`Timeout`, 504→`Timeout(exception_status_code=504)`.
  VERIFIED — `_map_anthropic_exception`.

---

## 4. Google Gemini v1beta (Generative Language API)

### 4.1 Non-stream error envelope: `google.rpc.Status` over HTTP (VERIFIED)

```json
{ "error": { "code": 429, "message": "You exceeded your current quota…",
  "status": "RESOURCE_EXHAUSTED",
  "details": [
    { "@type": "type.googleapis.com/google.rpc.QuotaFailure",
      "violations": [{ "quotaMetric": "generativelanguage.googleapis.com/…",
        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel",
        "quotaDimensions": { "location": "global", "model": "…" },
        "quotaValue": "10" }] },
    { "@type": "type.googleapis.com/google.rpc.Help",
      "links": [{ "description": "Learn more about Gemini API quotas",
                  "url": "https://ai.google.dev/gemini-api/docs/rate-limits" }] },
    { "@type": "type.googleapis.com/google.rpc.RetryInfo",
      "retryDelay": "42s" } ] } }
```

- Real bodies VERIFIED via quoted production payloads (cline #5202: full
  QuotaFailure+Help+RetryInfo at 429; gemini-cli #9248: nested **stringified-JSON-in-
  `message`** double envelope `{"error":{"message":"{\n \"error\": {…}…}","code":429,
  "status":"Too Many Requests"}}`; gemini-cli #6986: per-day vs per-minute
  `quotaId` pair). REPORTED (issue quotes; shapes mutually consistent).
- **Body `error.status` (string) is authoritative; HTTP code is a lossy projection.
  G1 fix-plan N1 (body-status priority) is correct.** The repo already parses
  `RetryInfo.retryDelay` + `ErrorInfo.quotaResetDelay` + `QuotaFailure.quotaValue/
  quotaId` (`error_handler.py:536-660`) — keep and extend per §4.3. VERIFIED (local).
- `retryDelay` formats observed: `"42s"`, `"16s"`, `"34s"`, `"13s"`,
  `"18.403470473s"` (fractional), `"34.074824224s"`, `"562476.752463453s"`,
  compound `"156h14m36.752463453s"`, `"290.979975ms"` (ms!), plus prose `Please
  retry in 34.074824224s.` inside `message`. VERIFIED — issues above + local
  `_parse_duration_string` docstring (which already handles ms/compound/plain).
  Note: fractional seconds with >9 decimals and `ms` suffixes both occur — parser
  must accept both. [INFERRED — evidence-backed]

### 4.2 `google.rpc.Code` numbers + HTTP mapping (VERIFIED — googleapis `code.proto` + Cloud docs)

| # | Name | HTTP | Seen on Gemini |
|---|---|---|---|
| 0 | OK | 200 | — |
| 1 | CANCELLED | 499 | 499 `CANCELLED` observed transiently on Gemini ( REPORTED — python-genai #2506); NOT in SDK default retry set |
| 2 | UNKNOWN | 500 | fallback |
| 3 | INVALID_ARGUMENT | 400 | bad params, bad tool schema, unsupported feature |
| 4 | DEADLINE_EXCEEDED | 504/408 | timeouts |
| 5 | NOT_FOUND | 404 | bad model name (`native_error_not_found.json` fixture exists in gomodel — VERIFIED path `stuff/GoModel/tests/contract/testdata/gemini/native_error_not_found.json`) |
| 6 | ALREADY_EXISTS | 409 | — |
| 7 | PERMISSION_DENIED | 403 | key scope/sharing violations |
| 8 | RESOURCE_EXHAUSTED | 429 | **all** quota/rate-limit (per-minute, per-day, spend-based, free-tier) |
| 9 | FAILED_PRECONDITION | 400 | — |
| 10 | ABORTED | 409 | — |
| 11 | OUT_OF_RANGE | 400 | — |
| 12 | UNIMPLEMENTED | 501 | — |
| 13 | INTERNAL | 500 | `{"error":{"code":500,"message":"…","status":"INTERNAL"}}` pattern handled in current classifier (VERIFIED — `error_handler.py:851-866` MidStreamFallback branch) |
| 14 | UNAVAILABLE | 503 | `The model is overloaded. Please try again later.` ( VERIFIED wording via gemini-cli #6986 quote `{"error":{"code":503,"message":"The model is overloaded…","status":"UNAVAILABLE"}}`) |
| 15 | DATA_LOSS | 500 | — |
| 16 | UNAUTHENTICATED | 401 | `API key not valid.` (litellm vertex mapper keys on this string; VERIFIED — `_map_vertex_exception`) |

VERIFIED sources: https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto,
https://docs.cloud.google.com/translate/docs/reference/rpc/google.rpc,
https://chromium.googlesource.com/external/github.com/grpc/grpc/+/refs/tags/v1.21.4-pre1/doc/statuscodes.md
(number table), https://pkg.go.dev/google.golang.org/grpc/codes.

### 4.3 `details[]` members (VERIFIED)

- `google.rpc.RetryInfo{retryDelay}` — authoritative wait; consume as minimum +
  jitter (same discipline as `Retry-After`). VERIFIED — issue bodies + Cloud
  reference (https://docs.cloud.google.com/asset-inventory/docs/reference/rpc/google.rpc).
  String form `"Ns"` / fractional; dict form `{"seconds":"123"}` also seen by the
  repo parser (VERIFIED — `error_handler.py:565-576`).
- `google.rpc.QuotaFailure{violations[]: {quotaMetric, quotaId, quotaDimensions{location,model}, quotaValue}}`
  — quota identity. **Decision rule used by gemini-cli (REPORTED — #9248 proposal):
  `quotaId` containing `PerDay`/`Daily` (+long `retryDelay`, e.g. >5 min) = terminal
  for the session → fail over model/key; `PerMinute`/`PerSecond` (+short delay) =
  transient → wait + retry same key.** Real pairs: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
  vs `GenerateContentInputTokensPerModelPerMinute-FreeTier` (VERIFIED quotes in
  #6986). G1 cooldown scoping should key on `(quotaId, model, location)`.
  [INFERRED recommendation, evidence-backed]
- `google.rpc.Help{links[]}` — docs URL, no machine signal. VERIFIED.
- `google.rpc.ErrorInfo{reason, domain, metadata{quotaResetDelay,…}}` — metadata
  carries `quotaResetDelay` (compound durations); reason/domain stable identifiers.
  VERIFIED — Cloud reference + local `_extract_retry_from_json_body` (lines 578-588).
- `google.rpc.PreconditionFailure`, `BadRequest{fieldViolations}`, `DebugInfo`,
  `LocalizedMessage`, `RequestInfo{requestId}` may appear per the generic model;
  Gemini does not reliably send `RequestInfo`. REPORTED (Cloud reference lists them;
  Gemini bodies quoted never include `requestId`). Prefer logging the raw body.
  [INFERRED]

### 4.4 Quota-vs-rate: Gemini does not distinguish them on the wire

- Every 429 is `RESOURCE_EXHAUSTED` with `QuotaFailure`; per-minute RPM/TPM,
  per-day RPD (`Resets at midnight Pacific`), spend-based rolling-10-min caps
  (Tier1 $10 / Tier2 $50 / Tier3 $200), and free-tier daily caps all share the shape.
  Only `quotaId`/`retryDelay` magnitude tells transient from terminal. VERIFIED —
  https://ai.google.dev/gemini-api/docs/rate-limits ("How rate limits work",
  "Spend-based rate limits", "RPD quotas reset at midnight Pacific").
- Free-tier `quotaValue` examples: `10` (requests/min/model), `50`/day
  (`GenerateRequestsPerDayPerProjectPerModel`), `125000` input tokens/min,
  `250000` paid-tier input tokens/min. REPORTED (issue bodies above).
- **HTTP 400 carrying quota bodies**: Gemini's OpenAI-compat endpoint
  (`/v1beta/openai/…`) has been observed returning quota/RESOURCE_EXHAUSTED-style
  failures at HTTP 400 (forum thread "OpenAI compatibility endpoint, 400 invalid
  argument" is a different 400 — tool-response validation — but the fix-plan G1
  work item "quota-at-400 (gemini-compat)" plus litellm's 5xx-wrapping-429 handling
  show the class is real). Evidence strength: fix-plan asserts it (local doc);
  litellm VERIFIED handles the sibling case "HTTP 500/503 with
  `{\"error\":{\"code\":429}}`" → `RateLimitError` (`_map_vertex_exception`,
  `_get_body_error_code` branch — VERIFIED in installed source). **G1 rule: any
  400/5xx whose body contains `RESOURCE_EXHAUSTED`/`QuotaFailure`/`RetryInfo` or
  body-`code: 429` must classify quota/rate-limit, never `invalid_request`.**
  [INFERRED rule from VERIFIED sibling + REPORTED compat behavior]
- litellm ordering bug (REPORTED, fixed upstream? — BerriAI/litellm #34954): in
  installed `1.100.0`, `_map_vertex_exception` still has `elif "403" in error_str:`
  **before** the quota branches (VERIFIED in installed source by direct read), so a
  retry-delay like `18.403470473s` misroutes a 429 → `BadRequestError(403)` with
  `_should_retry=False` (~0.9% of Gemini 429s in the reporter's production). G1
  must never substring-match bare digits; gate on status code + `status` string.
  [INFERRED]

### 4.5 Generation results that are NOT errors (but look like them)

- `finishReason`: `STOP` (natural), `MAX_TOKENS` (truncated — partial text may be
  present; empty `parts` with `MAX_TOKENS` is a known SDK wart, REPORTED —
  deprecated-generative-ai-python #280), `SAFETY` / `RECITATION` (blocked; JS SDK
  treats exactly these two as `hadBadFinishReason`, VERIFIED via gist mirror of
  `@google/generative-ai` source), `OTHER`, `LANGUAGE`, `BLOCKLIST`,
  `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL` (newer enum members,
  VERIFIED — https://github.com/google-gemini/deprecated-generative-ai-js/blob/main/common/api-review/generative-ai.api.md),
  image set `IMAGE_PROHIBITED_CONTENT/IMAGE_RECITATION/IMAGE_OTHER/NO_IMAGE`
  (REPORTED — google-gemini-php #143), `FINISH_REASON_UNSPECIFIED`.
- `promptFeedback.blockReason`: `SAFETY` / `OTHER` (+ `blockReasonMessage`,
  `safetyRatings[]`). The whole **prompt** was refused — HTTP 200, no `error`
  member. VERIFIED — JS SDK `BlockReason` enum (same api.md source).
- `RECITATION` is retry-eligible (same prompt often passes on retry; no backoff
  needed). REPORTED — https://discuss.ai.google.dev/t/no-response-due-to-recitation-finishreason/3957.
- Silent suppression: HTTP 200 + `finishReason: STOP` + **empty `parts`** with
  consumed `usageMetadata` ("Internal Output Suppression", PII-shaped diffs).
  REPORTED — https://discuss.ai.google.dev/t/gemini-3-0-pro-preview-empty-responses/116226.
  G1/G4 should treat empty-parts-with-200 as an error-ish signal (current
  `EmptyResponseError→server_error/503` already does; VERIFIED — `error_handler.py:1051-1058`).
- Vertex mid-stream 429 swallowed: litellm #23707 (REPORTED) — stream ends
  `finish_reason: stop` + `[DONE]` with the 429 only in debug logs. A proxy must
  scan Gemini stream chunks for `{"error": …}` members even after 200. [INFERRED]

### 4.6 Streaming framing: `alt=sse` vs JSON array (VERIFIED)

- `POST …:streamGenerateContent?alt=sse` → SSE (`data: {GenerateContentResponse}…
  `, usage typically in final chunk). **Without `alt=sse` the response is one JSON
  array of `GenerateContentResponse` objects** (not SSE at all). VERIFIED —
  https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Streaming_REST.ipynb,
  sse-stream example (`?alt=sse` required), forum confusion thread
  (discuss.google.dev/t/streamgeneratecontent…146282), Gate.AI reference.
- Stream-level errors: a chunk shaped `{"error": {"code": 429/503, "message": …,
  "status": "RESOURCE_EXHAUSTED"/"UNAVAILABLE"}}` with no `candidates`.
  REPORTED — litellm #23707 quotes `{'error': {'code': 429, 'message': 'Resource
  exhausted…', 'status': 'RESOURCE_EXHAUSTED'}}` arriving mid-stream.
- 503 wording: `The model is overloaded. Please try again later.` (REPORTED, #6986
  quote). The current classifier's 503/`MODEL_CAPACITY_EXHAUSTED` branch
  (VERIFIED — `error_handler.py:999-1020`) should also match `UNAVAILABLE` +
  `overloaded`. [INFERRED]

### 4.7 Auth, safety-thresholds, misc 4xx (VERIFIED unless noted)

- Key via `x-goog-api-key` header or `?key=` query (litellm Gemini uses
  `x-goog-api-key`; VERIFIED — `llms/gemini/common_utils.py`). Leaked-key block
  message: `Your API key was reported as leaked. Please use another API key.`
  (REPORTED — troubleshooting guide content below).
- `API key not valid.` → auth error (litellm keys on it; VERIFIED —
  `_map_vertex_exception`). Unrestricted-key sunset (June 2026) announced in-forum
  (REPORTED — discuss.ai.google.dev/t/use-ai-studio…/111017 notice banner).
- SDK retry: new `google-genai` retries **408/429/500/502/503/504** up to 5
  attempts (initial ~1 s, max 60 s), configurable via
  `HttpRetryOptions{initial_delay, attempts, http_status_codes}`; default does
  **not** include 499 `CANCELLED` (REPORTED issue #2506 with `_RETRY_HTTP_STATUS_CODES`
  quote) — retry-eligible per G1 anyway. VERIFIED guidance —
  https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy
  (retryable = 408/429/5xx + network; permanent = 400/401-don't-retry-unchanged).
- Error classes: new SDK `APIError{code, message}` → `ClientError` (4xx) /
  `ServerError` (5xx) via `APIError.raise_error` (REPORTED —
  https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py);
  legacy SDK raises `google.api_core.exceptions.*` (`TooManyRequests 429`,
  `ServiceUnavailable 503`, `ResourceExhausted`, …) (REPORTED —
  https://googleapis.dev/python/google-api-core/latest/exceptions.html).
  `retry.if_transient_error` does **not** catch new-SDK errors (separate
  hierarchies; REPORTED — cookbook #1091).

---

## 5. Cross-cutting

### 5.1 Decision matrix (wire → internal type → action)

| Wire signal | Internal | Action (recommended) |
|---|---|---|
| 401 any protocol (`UNAUTHENTICATED`, `authentication_error`, `API key not valid`, leaked-key) | `authentication` | **fail over to next credential** (fix-plan user directive; never hard-stop the chain), queue re-auth if OAuth-shaped |
| 403 (`PERMISSION_DENIED`, `permission_error`, `PermissionDeniedError`, geo-block) | `forbidden` | rotate immediately; distinct from 401 in logs |
| 404 (`NOT_FOUND`, `not_found_error`, `model_not_found` code) | `not_found` | **stop** (deterministic; rotating keys cannot fix a bad model id). gomodel maps unknown-model 404 with `code: model_not_found` (VERIFIED — `stuff/GoModel/internal/core/errors.go:232-238`) |
| 409 (`conflict_error`, `ALREADY_EXISTS/ABORTED`) | `invalid_request`-adjacent / retry-once | OpenAI/Anthropic SDKs retry 409 by default (VERIFIED §1.7); Anthropic says resolve-then-retry |
| 413 (`request_too_large`, `RequestTooLargeError`) | `invalid_request` (size subclass) | stop; truncate/compress input. Anthropic-only type; do not merge into context-window (different fix) |
| 422 (`UnprocessableEntityError`) | `invalid_request` | stop (OpenAI SDK does NOT retry 422 — VERIFIED §1.7) |
| 429 throughput (`rate_limit_error`, `RateLimitError`, `Retry-After`/`retryDelay` short, no spend-limit markers) | `rate_limit` | wait min(retry-signal)+jitter, retry same key if < small-cooldown threshold; else rotate |
| 429 billing/quota-terminal (`insufficient_quota`, `credit_balance_exhausted`, `*_spend_limit_exceeded`, `enforced_spend_limit_reached`+no `retry-after`, `quotaId` PerDay/Daily, `retryDelay` hours, RPD at midnight-Pacific) | `quota_exceeded` | cooldown with provider-quoted reset; rotate; never tight-loop |
| 400-context (`context_length_exceeded` code; `prompt is too long`; `maximum context length`; `exceed…context`; Gemini `exceeds the maximum number of tokens allowed`) | `context_window_exceeded` | **stop** (fail fast; surface token counts). Never rotate — deterministic |
| 400-other (`invalid_request_error`, `INVALID_ARGUMENT`, validation) | `invalid_request` | stop; fix request. But FIRST re-check body for quota markers (§4.4) |
| 402 `billing_error` (Anthropic) | `authentication`?/dedicated billing | stop + surface billing CTA. Current repo has no billing type — gap. [INFERRED] |
| 500/502/503/504 (`server_error`/`api_error`/`INTERNAL`/`UNAVAILABLE`/`overloaded_error`(529)/`timeout_error`(504)) | `server_error` (keep 529/overloaded + 504/timeout sub-signals) | retry same key with backoff, then rotate/failover |
| 408 + client timeouts (`APITimeoutError`, `DeadlineExceededError`, httpx `TimeoutException`) | `api_connection`/`proxy_timeout` | retry same key; count toward deadline, not quota counters |
| Transport (`ConnectError`, `ReadError`, `RemoteProtocolError`, DNS, TLS) | `api_connection` | retry same key, then rotate |
| 200-with-refusal (`content_filter` finish, `refusal` parts, Gemini SAFETY/RECITATION/BLOCKLIST…, Anthropic `stop_reason: refusal` family) | **not an error** | return to client as normal completion |
| 200-empty (`STOP` + empty parts, zero candidates) | `server_error`(503-equiv, current `EmptyResponseError`) | rotate (transient suppressor). VERIFIED current behavior — `error_handler.py:1051-1058` |
| Bare 429 with no timing info | `server_error`-equiv transient (current `TransientQuotaError→503`) | rotate, do not set long cooldowns. VERIFIED current — `error_handler.py:1060-1067` |

### 5.2 Retry-After / retry-signal units (must not be confused)

- OpenAI: `Retry-After` **seconds**; `x-ratelimit-reset-*` **durations**
  (`12ms`, `4m12s`, `23h18m29.144s`). VERIFIED §1.5.
- Anthropic: `retry-after` **seconds**; `anthropic-ratelimit-*-reset` **RFC 3339
  timestamps**. VERIFIED §3.4. A parser must branch on header name, never assume one
  unit. [INFERRED]
- Gemini: `retryDelay` **durations** (`42s`, `18.403470473s`, `290.979975ms`,
  `156h14m36s`, dict `{"seconds":"123"}`); prose `Please retry in Ns.` VERIFIED §4.1.
- `x-should-retry: true/false` (Anthropic, possibly others) overrides status-code
  heuristics. VERIFIED §1.7 + REPORTED §3.4.
- litellm `RateLimitError` also carries `category` (vendor vs litellm-internal
  limiter) and `rate_limit_type` (requests/tokens/concurrent/budget/max_iterations)
  — VERIFIED (`exceptions.py:RateLimitErrorCategory/RateLimitType`). A proxy's own
  budget rejections should be distinguishable from vendor 429s the same way.
  [INFERRED]

### 5.3 Safe-to-retry per official guidance (VERIFIED summary)

- Retry: OpenAI 408/409/429-temporary/5xx; Anthropic connection-errors/429/5xx
  (+explicit 529 guidance); Gemini/Vertex 408/429/5xx (+network). Do NOT retry
  unchanged: 400/401/403/404/413/422, billing/spend/quota-terminal, validation.
  Sources: §1.7, §3.4, §4.7 (all VERIFIED).
- SDKs already retry twice by default (OpenAI, Anthropic) / 4–5 attempts (Gemini).
  Proxy-level retries **add** to SDK retries — account for both budgets or disable
  one layer. REPORTED best practice (m2ml rate-limit-retry skill: "SDK retries OR
  custom retries, not both"). G1 non-stream sleeps must honor the overall deadline
  (fix-plan item; VERIFIED as requirement, implementation pending).

### 5.4 Timeouts

- No universal HTTP code: Anthropic 504 `timeout_error` (+ SDK
  `DeadlineExceededError`); Gemini `DEADLINE_EXCEEDED` (504/408); OpenAI has no
  API-side timeout error — client `APITimeoutError` after 600 s default (VERIFIED).
  litellm normalizes to `Timeout(status_code=408 default, or 504 passthrough)`.
  VERIFIED — `exceptions.py:Timeout`, `_map_anthropic_exception` (408→Timeout,
  504→Timeout), `_map_exception_by_status` (408→Timeout, 504→Timeout).
- Current repo maps httpx `TimeoutException/ConnectError/NetworkError` →
  `api_connection`, and Anthropic `proxy_timeout → timeout_error` for protocol
  rendering (VERIFIED — `error_handler.py:1029-1034`, `core/errors.py:107`).
  Keep; add 408→rotate (fix-plan decision; 408 currently falls into generic 4xx
  `invalid_request` at `error_handler.py:991-997` — VERIFIED gap). [INFERRED]

### 5.5 Transport exception taxonomy (VERIFIED — installed source)

- httpx 0.28.1: `HTTPError → {RequestError → {TransportError → {TimeoutException →
  {ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout}, NetworkError →
  {ConnectError, ReadError, WriteError, CloseError}, ProtocolError →
  {LocalProtocolError, RemoteProtocolError}, ProxyError, UnsupportedProtocol},
  DecodingError}, ResponseError → {HTTPStatusError, StreamError → {StreamConsumed,
  StreamClosed, ResponseNotRead, RequestNotRead}}}`. VERIFIED — `dir(httpx)` listing
  + `TimeoutException`/`NetworkError`/`HTTPStatusError` handling in
  `error_handler.py:927-1034`. (`Timeout` bare name in httpx is the config class,
  not an exception — do not catch it. [INFERRED])
- litellm classes reaching this proxy (VERIFIED — `exceptions.py` class list):
  `RateLimitError(429)`, `ContextWindowExceededError(400)`,
  `ContentPolicyViolationError(400)`, `RejectedRequestError(400)`,
  `BadRequestError/InvalidRequestError/UnprocessableEntityError/ImageFetchError/
  UnsupportedParamsError/LiteLLMUnknownProvider`, `AuthenticationError(401)`,
  `PermissionDeniedError(403)`, `NotFoundError(404)`, `Timeout(408/504)`,
  `InternalServerError(500)`, `BadGatewayError(502)`,
  `ServiceUnavailableError(503)`, `APIConnectionError(500!)`, `APIError`,
  `BudgetExceededError(429, litellm-internal — carries category/rate_limit_type,
  NOT a vendor 429)`, `MidStreamFallbackError(→503, wraps original + generated
  content + is_pre_first_chunk)`, `GuardrailRaisedException`,
  `APIResponseValidationError/JSONSchemaValidationError`.
- litellm mapper provenance (VERIFIED — `exception_mapping_utils.py`):
  OpenAI-family → `_map_openai_exception` (rate-limit substring OR status;
  context-window substrings; `model_not_found`→NotFound; `content_policy_violation`
  →ContentPolicy; `invalid_encrypted_content`→BadRequest with affinity guidance);
  Anthropic → `_map_anthropic_exception` (string-first, then status ladder
  401/403/400/413/404/408/429/500|529/502/503/504); Vertex/Gemini →
  `_map_vertex_exception` (string-first incl. the buggy bare-`"403"` branch,
  then status ladder; 5xx-with-body-code-429 → RateLimitError); everything else →
  `_map_exception_by_status` (401/403/404/408/429/500/502/503/504/explicit, else
  4xx→BadRequest, 5xx→APIError). **Overloaded strings map to
  `InternalServerError`, not a dedicated class** (`"overloaded_error" in error_str
  → InternalServerError` in the Anthropic mapper — VERIFIED). G1's
  `anthropic 529/overloaded_error` item must therefore recover the signal from
  `status_code==529`/body text, not the litellm class. [INFERRED]
- Note `APIConnectionError.status_code = 500` in litellm (VERIFIED —
  `exceptions.py`), while this repo treats it as 503-equiv transient (VERIFIED —
  `error_handler.py:1118-1123`). Keep the repo's transient treatment; do not trust
  the 500 literally. [INFERRED]

### 5.6 Reference implementations (what plexus/gomodel do — VERIFIED by direct read)

- **gomodel** (`stuff/GoModel/internal/core/errors.go`): 7-type taxonomy
  (`provider_error 5xx→502 default / rate_limit_error 429 / invalid_request_error
  4xx / authentication_error 401 / permission_error 403 / not_found_error 404 (+
  `model_not_found` code mirroring OpenAI) / internal_error 500`); status-code-first
  `ParseProviderError` preserving upstream codes and `param`/`code`; embedded-error-
  in-2xx detection (`ParseEmbeddedProviderError`, e.g. OpenRouter bare
  `{"error":…}` with numeric `code` → status); raw bodies capped at 64 KiB for
  audit, never serialized to clients; Anthropic dialect renderer
  (`internal/anthropicapi/errors.go`: 413→`request_too_large`, 503→
  `overloaded_error`, else provider→`api_error`); dialect-aware top handler +
  `Retry-After`-preserving header passthrough (`internal/server/error_support.go`).
  Gaps vs G1 needs: no quota-vs-rate split, no `details[]` parsing, 403→
  authentication (deliberate merge — differs from this repo's forbidden split).
- **plexus** (`stuff/plexus/packages/backend/src/services/`): no wire-error
  taxonomy (relies on providers); error handling = `CooldownParserRegistry`
  (`services/runtime/cooldown-parsers.ts:20-79` — pluggable per-provider cooldown
  parsers; built-ins: `openai-codex` "Try again in ~N min", `openrouter`
  `retry_after_seconds` metadata / `Retry-After` header / regex) +
  `provider-cooldown.ts` (provider-type resolution incl. OpenRouter sniffing) +
  quota-checker subsystem (`services/quota/`: enforcer, middleware, scheduler,
  checker-registry — local quota, not provider-error parsing). Takeaway for G1:
  registry-pattern for per-provider retry-delay parsers is proven; Gemini
  `RetryInfo`/`QuotaFailure` parsers belong behind such a seam. [INFERRED]

### 5.7 G1-relevant gaps in the current classifiers (VERIFIED by reading both files)

1. `classify_error` 400 branch (`error_handler.py:978-990`) misses Anthropic
   `prompt is too long` / `exceed context limit` spellings (§3.3) — add.
2. Bare 408 falls into generic 4xx `invalid_request` (`:991-997`) — should rotate
   per fix-plan; add explicit 408 (and 504→timeout) arms.
3. No `billing_error` (402) / `request_too_large` (413) / `conflict` (409) /
   `not_found` (404) / `timeout_error` (504) / `overloaded` (529) recognition on
   input; `StructuredAPIResponseError.http_status` already emits `not_found:404`
   but no classifier produces it. Add vocabulary both directions.
4. `structured_api_response_error` substring matching (`"rate" in descriptor`,
   `core/errors.py:173`) false-positives on "generate"/"accurate" (fix-plan noted)
   — restrict to structured type/status/code fields (see fix-plan G1.1; independent
   confirmation: litellm #34954 shows the same bug class in litellm itself).
5. Asymmetry: `Response.error.code` vocabulary (`rate_limit_exceeded`,
   `invalid_prompt`, `vector_store_timeout`, …) not recognized; SSE `error` events
   and `response.failed` not in the stream-error path vocabulary.
6. `Retry-After` HTTP-date form unhandled (`error_handler.py:691-695` "skip for
   now"); Anthropic RFC-3339 reset timestamps unhandled — add both, keyed by header.
7. `retry in` phrasing + day-units in duration parser (fix-plan G1.3) — Gemini prose
   `Please retry in Ns.` confirms the need; add `retry in\s*([\d.]+)s`.
8. `details[]` parsed for cooldown seconds but `quotaId`/`quotaMetric`/model not
   carried to cooldown scope (only `quota_value`/`quota_id` on `ClassifiedError`;
   no per-(model,quotaId) scoping) — extend per §4.3 rule.

### 5.8 Citation index (primary sources)

- OpenAI error codes: https://developers.openai.com/api/docs/guides/error-codes
- OpenAI rate limits: https://developers.openai.com/api/docs/guides/rate-limits
- OpenAI Responses streaming events: https://developers.openai.com/api/reference/resources/responses/streaming-events
- OpenAI Chat streaming events: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
- OpenAI Python reference (errors/retries): https://developers.openai.com/api/reference/python
- openai-python source: `C:\Python312\Lib\site-packages\openai\_exceptions.py`,
  `_constants.py`, `_base_client.py::_should_retry/_parse_retry_after_header`
- Anthropic errors: https://platform.claude.com/docs/en/api/errors
- Anthropic rate limits: https://platform.claude.com/docs/en/api/rate-limits
- Anthropic streaming: https://platform.claude.com/docs/en/api/streaming
- Anthropic AWS header table: https://docs.aws.amazon.com/claude-platform/latest/userguide/rate-limits.html
- anthropic-sdk-python: `src/anthropic/_exceptions.py`, `_client.py`, `__init__.py`
  (via https://github.com/anthropics/anthropic-sdk-python/blob/main/…);
  SSE-200 bug: https://github.com/anthropics/anthropic-sdk-python/issues/1258
- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Gemini troubleshooting/retry: https://ai.google.dev/gemini-api/docs/troubleshooting
- Vertex retry strategy: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy
- google.rpc Code: https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto;
  Status/details: https://docs.cloud.google.com/asset-inventory/docs/reference/rpc/google.rpc
- google-genai errors: https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py;
  retry codes thread: https://github.com/googleapis/python-genai/issues/2506
- FinishReason/BlockReason enum: https://github.com/google-gemini/deprecated-generative-ai-js/blob/main/common/api-review/generative-ai.api.md
- Gemini streaming (`alt=sse`): https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Streaming_REST.ipynb
- Quota bodies: https://github.com/cline/cline/issues/5202,
  https://github.com/google-gemini/gemini-cli/issues/9248,
  https://github.com/google-gemini/gemini-cli/issues/6986
- litellm source: `C:\Python312\Lib\site-packages\litellm\exceptions.py`,
  `litellm_core_utils\exception_mapping_utils.py` (ExceptionCheckers,
  _map_openai/anthropic/vertex_exception, _map_exception_by_status),
  `llms\gemini\common_utils.py` (GeminiError, x-goog-api-key); litellm bugs:
  https://github.com/BerriAI/litellm/issues/34954 (`403`-substring),
  https://github.com/BerriAI/litellm/issues/23707 (swallowed mid-stream 429),
  https://github.com/BerriAI/litellm/issues/18003, https://github.com/BerriAI/litellm/issues/27470
- httpx 0.28.1: `dir(httpx)` hierarchy (§5.5)
- Responses error vocabulary: community spec-sync https://community.openai.com/t/responses-api-streaming-the-simple-guide-to-events/1363122;
  in-stream `server_error` case https://github.com/vercel/ai/issues/16021;
  `ResponseError` type https://developers.openai.com/api/reference/python/resources/responses
- `insufficient_quota`: https://community.openai.com/t/429-error-insufficient-quota/492350,
  https://prismix.dev/guides/openai-insufficient-quota
- Anthropic context spellings: https://github.com/anthropics/claude-code/issues/5346,
  https://github.com/NousResearch/hermes-agent/issues/813,
  https://github.com/anomalyco/opencode/issues/17746
- gomodel: `stuff/GoModel/internal/core/errors.go`, `internal/anthropicapi/errors.go`,
  `internal/server/error_support.go`
- plexus: `stuff/plexus/packages/backend/src/services/runtime/cooldown-parsers.ts`,
  `services/providers/provider-cooldown.ts`
- Current classifiers: `src/rotator_library/error_handler.py` (1262 lines),
  `src/rotator_library/core/errors.py` (245 lines)
