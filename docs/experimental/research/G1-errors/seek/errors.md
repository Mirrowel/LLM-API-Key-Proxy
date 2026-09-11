# G1 — Grounded error landscape for the four generative LLM protocols

Research unit output (1 of 3 independent workers). Feeds `docs/experimental/fix-pass-plan.md` §G1
("Error taxonomy + decision matrix"). Read-only reference to
`src/rotator_library/error_handler.py`, `src/rotator_library/core/errors.py`,
`stuff/plexus`, `stuff/GoModel`, `C:\Python312\Lib\site-packages\litellm`.

**Scope:** OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google Gemini v1beta
(generateContent). Every fact is tagged:

- **[V] VERIFIED** — official protocol doc or shipped SDK source (cited).
- **[R] REPORTED** — issue tracker / blog / litellm-only implementation (cited).
- **[I] INFERRED** — reasoned from cited adjacent facts; not directly documented.

Method: official docs fetched 2026-09-11; SDK sources read from the local install
(`openai 2.54.0`) and upstream GitHub (`anthropic-sdk-python`, `python-genai`); litellm read
locally (unversioned site-package; code lines cited); repo companions read directly.

Deliberate design rule that falls out of the evidence: **HTTP status is the weakest signal.**
Body `error.status`/`error.code`/`error.type` (Gemini `status`, OpenAI `code`) is stronger, and
free-text message sniffing is the last resort and the most dangerous (see §5.6, repo defect at
`error_handler.py:814` and the substring false-positive in `core/errors.py:173`).

---

## 1. OpenAI Chat Completions

### 1.1 Error envelope [V]

Every non-2xx API error is JSON:

```json
{ "error": { "message": "...", "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded" } }
```

- `error.message` (string, required), `error.type` (string, required), `error.param` (string|null),
  `error.code` (string|null). The SDK model is exactly this shape:
  `openai/types/shared/error_object.py` (`code: Optional[str]`, `message: str`, `param: Optional[str]`,
  `type: str`) [V]. `APIStatusError.__init__` reads the flat body via `construct_type` and sets
  `.code/.param/.type` from top-level body keys — so a provider that nests differently loses these
  fields (`openai/_exceptions.py`) [V].

### 1.2 Documented HTTP status → meaning [V]

From `platform.openai.com/docs/guides/error-codes` (mirrored on `developers.openai.com`):

| Status | Meaning |
|---|---|
| 401 | Invalid Authentication / Incorrect API key / must be a member of an organization / IP not authorized |
| 403 | Country, region, or territory not supported |
| 429 | **Rate limit reached for requests** (RPM/TPM/etc.) |
| 429 | **You exceeded your current quota** — out of credits / monthly/spend limit (billing, not rate) |
| 500 | Server had an error while processing the request |
| 503 | Engine currently overloaded, try again later |
| 503 | **Slow Down** — sudden request-rate increase; reduce rate and hold ~15 min |
| 404 | (implied; `model_not_found` appears as `invalid_request_error` + `model_not_found` in litellm) |

The same page lists Python SDK types: `APIConnectionError`, `APITimeoutError`,
`AuthenticationError`, `BadRequestError`, `ConflictError`, `InternalServerError`, `NotFoundError`,
`PermissionDeniedError`, `RateLimitError`, `UnprocessableEntityError` [V].

### 1.3 `error.code` vocabulary (partial — no official complete enum) [V/R]

OpenAI does **not** publish a complete `error.code` enum. Documented/confirmed:
- `context_length_exceeded` — 400, `type: invalid_request_error`, `param: messages`/`input` [V via
  community + litellm; example in `developers.openai.com` community threads].
- `insufficient_quota` — 429 billing/quota; the docs explicitly say *"For billing-related errors,
  inspect `error.code`… The broader `error.type` can still be `insufficient_quota`"* [V].
- `rate_limit_exceeded` — 429 rate limit [R community].
- `invalid_api_key` — 401 [R community].
- `string_above_max_length` — 400 param-length (a request-shape error, NOT context window) [R].
- `model_not_found` — 404 / `invalid_request_error` [R litellm `exception_mapping_utils.py:305`].

Representative 400 body:
```json
{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 129500 tokens ...","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}
```
[R — community / tokenswise; corroborated by litellm's string checker]

### 1.4 Rate-limit & retry semantics [V]

Response headers (`platform.openai.com/docs/guides/rate-limits`):

| Header | Sample | Unit |
|---|---|---|
| `x-ratelimit-limit-requests` | `60` | count |
| `x-ratelimit-limit-tokens` | `150000` | tokens |
| `x-ratelimit-remaining-requests` | `59` | count |
| `x-ratelimit-remaining-tokens` | `149984` | tokens |
| `x-ratelimit-reset-requests` | `1s` | **Go duration string, NOT seconds int** (e.g. `6m0s`) |
| `x-ratelimit-reset-tokens` | `6m0s` | Go duration string |
| `x-ratelimit-limit-project-tokens` / `remaining` / `reset` | `60000` | project-scoped tokens |

There is also a `retry-after` header (OpenAI errors doc says official SDKs honor it "for eligible
retries"; present on 429s) [V]. `x-ratelimit-reset-*` is a **duration string**; do not assume
integer seconds (contrast: the repo's `get_retry_after` does `int(x-ratelimit-reset)` →
`error_handler.py:698-711`, which will swallow `"6m0s"`).

SDK retry policy (`openai/_base_client.py`, `_constants.py`) [V]:
- `DEFAULT_MAX_RETRIES = 2`; retries with exponential backoff + jitter, `INITIAL_RETRY_DELAY=0.5s`,
  `MAX_RETRY_DELAY=8s`, capped `MAX_RETRY_AFTER_DELAY=120s` for `retry-after`.
- `_should_retry()`: honors `x-should-retry: true|false`; `retry-after` > 120s disables retry;
  otherwise retries connection errors, 408, 409, 429, and ≥500.
- `APITimeoutError` derives from `APIConnectionError` (no status) — treat as connection-class [V].

**Client-side non-error finish states** (critical for "content filter is not an error"):
`LengthFinishReasonError` and `ContentFilterFinishReasonError` are raised by the SDK when you ask
for parsed structured output and the model stopped with `finish_reason == "length"` /
`"content_filter"`. They are local parse errors, not API errors [V — `openai/_exceptions.py`].

### 1.5 Streaming error shape [I/R]

Chat Completions SSE has **no documented dedicated error event**. Failures before the first byte
are ordinary 4xx/5xx JSON. Mid-stream provider failures surface either as (a) a data frame shaped
`{"error": {...}}` followed by stream termination, or (b) an abrupt close with no terminal marker.
OpenAI's own spec only defines `data: [DONE]` as terminator. Treat "stream ended without `[DONE]`"
as a failure; litellm synthesizes `MidStreamFallbackError` for this family [R —
`litellm/exceptions.py:1086`]. Do not assume a chat error chunk exists [I].

---

## 2. OpenAI Responses

### 2.1 Non-stream error body [V/R]

Same flat envelope as Chat Completions: `{"error":{message,type,param,code}}`. A context overflow on
`/v1/responses` returns HTTP 400 with `code: "context_length_exceeded"`, `type:
"invalid_request_error"`, `param: "input"`:

```json
{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","param":"input"}}
```
[R — bifrost issue #4413 logs the raw upstream body; openai-python `ErrorObject` confirms the shape]

### 2.2 `Response.error` object (terminal, non-HTTP) [V]

When a Response finishes as `status: "failed"`, the Response object carries:

```python
class ResponseError:            # openai/types/responses/response_error.py
    code: Literal["server_error","rate_limit_exceeded","invalid_prompt",
                  "data_residency_mismatch","bio_policy","vector_store_timeout",
                  "invalid_image","invalid_image_format","invalid_base64_image",
                  "invalid_image_url","image_too_large","image_too_small",
                  "image_parse_error","image_content_policy_violation",
                  "invalid_image_mode","image_file_too_large",
                  "unsupported_image_media_type","empty_image_file",
                  "failed_to_download_image","image_file_not_found"]
    message: str
```

**Note the trap:** `context_length_exceeded` is **not** in the `ResponseError.code` enum — it only
appears as a 400 HTTP-body `code`. Do not classify solely from `response.error.code`.

`response.status` lifecycle values: `queued`, `in_progress`, `completed`, `incomplete`, `failed`.
`incomplete_details.reason` is `max_output_tokens` or `content_filter` — an **incomplete, not an
error** (content filter → status `incomplete`, no `error` object) [V for status; R for the exact
`incomplete_details.reason` enum from stream dumps; corroborated by SDK event list].

### 2.3 Streaming events [V]

`developers.openai.com/api/reference/resources/responses/streaming-events/` + openai SDK:

- `response.created` → `response.id`, `response.error` (null initially).
- `response.in_progress`.
- Output deltas: `response.output_text.delta`, `response.refusal.delta`,
  `response.function_call_arguments.delta`, `response.reasoning*`, etc.
- Terminal events: **`response.completed`**, **`response.incomplete`** (carries
  `incomplete_details`), **`response.failed`** (carries `response.error`), and out-of-band
  **`error`**.
- `ResponseErrorEvent` (`openai/types/responses/response_error_event.py`):
  `{type: "error", code?: str, message: str, param?: str, sequence_number: int}`.
- `ResponseFailedEvent` (`response_failed_event.py`): `{type: "response.failed", response, sequence_number}`.

Both a `response.failed` **and** a bare `error` event can be the terminal signal; a stream may also
end with no terminal event at all. Four terminal outcomes, not one [V spec; R langchain issue
#39039 documents real-world consumers dropping `response.failed`/`error` and treating a truncated
stream as success].

Known observed production bug: the `error` event for context overflow can arrive with an **empty
`message`**, while the equivalent non-stream path returns the full `context_length_exceeded`
message [R bifrost #4413; fix PR #4418].

### 2.4 WebSocket mode errors [V]

From `platform.openai.com/docs/guides/error-codes` ("WebSocket mode errors"):
- `previous_response_not_found` — `previous_response_id` unresolvable; retry with full input and
  `previous_response_id: null`.
- `websocket_connection_limit_reached` — 60-minute connection cap; open a new connection.

### 2.5 Rate limits/retry [V]

Shares the OpenAI SDK retry engine (§1.4): honors `retry-after`, retries 408/409/429/≥500, 2
retries by default. No Responses-specific rate-limit header scheme documented.

---

## 3. Anthropic Messages

### 3.1 Error envelope + request id [V]

`docs.anthropic.com/en/api/errors` (now `platform.claude.com/docs/en/api/errors`):

```json
{ "type": "error",
  "error": { "type": "not_found_error", "message": "The requested resource could not be found." },
  "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy" }
```

Every response also carries a `request-id` header; the same value is `request_id` in the body.
`type` values may grow over time (versioning policy) — do not treat the enum as closed [V].

### 3.2 HTTP status → `error.type` [V]

| HTTP | error.type | Notes |
|---|---|---|
| 400 | `invalid_request_error` | Format/content issue; **also used for other 4xx not listed**; **usage/spend-limit 400** (except Claude Code workspace limits, which can 429) |
| 401 | `authentication_error` | malformed/revoked/expired key; on AWS also SigV4/AWS creds |
| 402 | `billing_error` | payment/billing issue |
| 403 | `permission_error` | key lacks permission for resource |
| 404 | `not_found_error` | resource/endpoint |
| 409 | `conflict_error` | resource state conflict; resolve then retry |
| 413 | `request_too_large` | > per-endpoint bytes (Messages 32 MB, Batch 256 MB, Files 500 MB) |
| 429 | `rate_limit_error` | org rate limit, tier monthly spend cap, or Claude Code workspace spend limit. **Tier spend-cap 429 has NO `retry-after` and keeps failing until access resumes** |
| 500 | `api_error` | internal; retry w/ backoff |
| 504 | `timeout_error` | request timed out while processing |
| 529 | `overloaded_error` | API temporarily overloaded (high traffic across all users) |

SDK `ErrorType` alias (`anthropic-sdk-python/src/anthropic/types/shared/error_type.py`) [V]:
`invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`,
`rate_limit_error`, `timeout_error`, `overloaded_error`, `api_error`, `billing_error`.

SDK exception classes (`anthropic-sdk-python/src/anthropic/_exceptions.py`) [V]:
`BadRequestError(400)`, `AuthenticationError(401)`, `PermissionDeniedError(403)`,
`NotFoundError(404)`, `ConflictError(409)`, `RequestTooLargeError(413)`,
`UnprocessableEntityError(422)`, `RateLimitError(429)`, `ServiceUnavailableError(503)`,
`OverloadedError(529)`, `DeadlineExceededError(504)`, `InternalServerError`.
`APIStatusError` sets `.request_id` from the `request-id` header and `.type` from `error.type`;
`.workspace_id` from `anthropic-workspace-id` [V]. Note **529 is its own class** and must not be
folded into 503.

### 3.3 Rate-limit headers [V]

`docs.anthropic.com/en/api/rate-limits`:

| Header | Meaning / unit |
|---|---|
| `retry-after` | seconds to wait (integer seconds) |
| `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset` | RPM; reset **RFC 3339** timestamp |
| `anthropic-ratelimit-tokens-limit` / `-remaining` / `-reset` | tokens; remaining rounded to nearest 1k; reset RFC 3339 |
| `anthropic-ratelimit-input-tokens-limit` / `-remaining` / `-reset` | ITPM |
| `anthropic-ratelimit-output-tokens-limit` / `-remaining` / `-reset` | OTPM |
| `anthropic-priority-input-tokens-*` / `anthropic-priority-output-tokens-*` | Priority Tier only |

Units are heterogeneous: `retry-after` = **seconds**; `*-reset` = **RFC 3339 date-time** (NOT unix
seconds). The repo's `get_retry_after` only handles integer-seconds `retry-after` and unix
`x-ratelimit-reset` (`error_handler.py:688-711, 723-730`), so it silently drops Anthropic's reset
timestamps [I from code + doc].

Retry guidance [V]: all official SDKs auto-retry transient failures (connection errors, rate
limits, 5xx) with exponential backoff, **twice by default**, honoring `retry-after`; configurable
via max-retries. Rate limits are token-bucket (continuous replenishment, not fixed-window resets).
Long-context (1M beta) has separate ITPM/OTPM limits.

### 3.4 Streaming error event [V]

`docs.anthropic.com/en/api/messages-streaming` — an error after HTTP 200 arrives as a normal SSE
frame:

```sse
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

Event flow: `message_start` → (`content_block_start` + `content_block_delta`* +
`content_block_stop`)* → `message_delta`* → `message_stop`, plus any number of `ping`. New event
types may be added; unknown events must be handled gracefully [V]. Mid-stream errors do **not**
follow the HTTP error mechanisms, and the HTTP status is already 200 [V]. Tool-use deltas are
`input_json_delta`; text is `text_delta` [V].

### 3.5 Context window / refusal [R/V]

- Over-long prompt is `invalid_request_error` (400) with text `"prompt is too long"` /
  `"prompt: length"`; litellm maps those to `ContextWindowExceededError` [R
  `exception_mapping_utils.py:511-520`].
- Stop/refusal is a normal 200: `stop_reason` (e.g. `refusal`, `max_tokens`, `stop_sequence`,
  `tool_use`), not an error [V concept; exact stop_reason enum is message-schema].

---

## 4. Google Gemini v1beta (generateContent)

There are **two different error formats** under the Gemini banner. Do not conflate them:

- **`v1beta`/`v1` generateContent** (the surface this proxy speaks): classic **gRPC
  `google.rpc.Status`** envelope, `error.code` is an **integer**. [V]
- **Newer Interactions API**: `{"error":{"code":"snake_case_string","message":...}}`. [V]

### 4.1 generateContent error envelope [V]

`ai.google.dev/gemini-api/docs/generate-content/api-errors`:

```json
{
  "error": {
    "code": 400,
    "message": "API key not valid. Please pass a valid API key.",
    "status": "INVALID_ARGUMENT",
    "details": [
      { "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID", "domain": "googleapis.com",
        "metadata": { "service": "generativelanguage.googleapis.com" } },
      { "@type": "type.googleapis.com/google.rpc.LocalizedMessage",
        "locale": "en-US", "message": "API key not valid. Please pass a valid API key." }
    ]
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `error.code` | **integer** | HTTP/numeric gRPC status |
| `error.message` | string | human-readable |
| `error.status` | string | gRPC status in `SCREAMING_CASE` |
| `error.details` | array | `@type`-packed `google.protobuf.Any` members |

**`error.status` (string) must take priority over HTTP code.** This is the exact G1 "N1
status-shadowing" fix (`error_handler.py:814`).

### 4.2 Status code table [V]

From `generate-content/api-errors`, AIP-193, and `google/rpc/code.proto`:

| HTTP | gRPC status | `code` int | Retrieve/retry meaning |
|---|---|---|---|
| 400 | `INVALID_ARGUMENT` | 3 | malformed request; **do not retry** |
| 400 | `FAILED_PRECONDITION` | 9 | free tier unavailable in country / billing not enabled; fix account, **not retryable** |
| 401 | `UNAUTHENTICATED` | 16 | missing/invalid/expired OAuth or API key |
| 403 | `PERMISSION_DENIED` | 7 | key lacks permission |
| 404 | `NOT_FOUND` | 5 | file/resource/model not found |
| 429 | `RESOURCE_EXHAUSTED` | 8 | RPM/TPM/RPD/spend exceeded — retry w/ backoff |
| 499 | `CANCELLED` | 1 | client closed connection |
| 500 | `INTERNAL` / `UNKNOWN` | 13 / 2 | server/overload; retry |
| 503 | `UNAVAILABLE` | 14 | overloaded/down; retry |
| 504 | `DEADLINE_EXCEEDED` | 4 | prompt/context too large or too-short client deadline; raise timeout |

Full `google.rpc.Code` mapping also includes `OK=0`, `ALREADY_EXISTS=6`, `ABORTED=10`,
`OUT_OF_RANGE=11`, `UNIMPLEMENTED=12`, `DATA_LOSS=15` [V code.proto]. Note the HttpMapping listed
in the proto can differ from what Gemini actually returns (e.g. proto maps 8→429, 9→400); always
trust the returned pair.

Vertex API errors page (`model-reference/api-errors`) confirms the same canonical names and adds
`499 CANCELLED`, `504 DEADLINE_EXCEEDED`, and that 429's cause may be quota **or** shared-capacity
overload **or** daily `logprobs` limit [V].

### 4.3 `details[]` members [V]

`google/rpc/error_details.proto` defines, each packed as `google.protobuf.Any` with
`@type = "type.googleapis.com/google.rpc.<Name>"` and included **at most once** per type
(AIP-193):

| Type | Fields | Use |
|---|---|---|
| `ErrorInfo` | `reason` (UPPER_SNAKE, ≤63), `domain`, `metadata` map | machine-readable cause; **must always be present** per AIP-193 |
| `RetryInfo` | `retry_delay` (`google.protobuf.Duration`) | authoritative retry-after for 429/503 |
| `QuotaFailure` | `violations[]`: `subject`, `description`, `api_service`, `quota_metric`, `quota_id`, `quota_dimensions` map, `quota_value` int64, optional `future_quota_value` | which quota, how much |
| `Help` | `links[]: {description, url}` | troubleshooting docs |
| `LocalizedMessage` | `locale` (BCP-47), `message` | localized copy |
| `BadRequest` | `field_violations[]: {field, description}` | request-shape problems |
| `PreconditionFailure` | `violations[]: {type, subject, description}` | e.g. ToS not accepted |
| `DebugInfo` | `stack_entries[]`, `detail` | server debugging |

`google.protobuf.Duration` serializes to JSON as a string with an `s` suffix — `"3.5s"`, `"5s"`,
`"0.290979975s"`, and for long resets compound-looking values like `"156h14m36.752463453s"` (the
proto duration formatter emits `HhMmS` up to hours). `retryDelay` is camelCase in the JSON wire
form even though the proto field is `retry_delay` [V proto; R observed wire form — the repo already
parses both `retryDelay` and `quotaResetDelay` at `error_handler.py:536-593`].

`ErrorInfo.metadata` for quota errors has been observed to carry `quotaResetDelay` alongside (or
instead of) `RetryInfo` [R — repo parser `error_handler.py:578-588`; treat as a secondary
authoritative reset hint].

### 4.4 Retry semantics [V]

`ai.google.dev/gemini-api/docs/troubleshooting`:
- Official Python SDK auto-retries transient errors (timeouts, network, 429, 5xx) up to **4 times**,
  initial delay ~1s, max delay 60s, exponential backoff.
- Guidance: exponential backoff + jitter; **retry only** `429`, `408`, `5xx`; **never retry**
  `400`/`403`. Max ~2 retries for direct REST; min 1s.
- Vertex guidance: don't retry more than twice; avoid traffic spikes.
- `RESOURCE_EXHAUSTED` unifies RPM, TPM, RPD, spend, and shared-capacity overload — the
  `RetryInfo.retryDelay`/`QuotaFailure` (or the message's "reset after") is the discriminator [V].

### 4.5 Finish reasons / content blocking (NOT errors) [V]

`Candidate.FinishReason` (v1beta `generative_service.proto` + Vertex `GenerateContentResponse`):

`FINISH_REASON_UNSPECIFIED=0`, `STOP=1`, `MAX_TOKENS=2`, `SAFETY=3`, `RECITATION=4`, `OTHER=5`,
`LANGUAGE=6`, `BLOCKLIST=7`, `PROHIBITED_CONTENT=8`, `SPII=9`, `MALFORMED_FUNCTION_CALL=10`,
`IMAGE_SAFETY=11`, plus Vertex-only `MODEL_ARMOR`, `IMAGE_PROHIBITED_CONTENT`,
`IMAGE_RECITATION`, `IMAGE_OTHER`, `UNEXPECTED_TOOL_CALL`, `NO_IMAGE` [V].

`PromptFeedback.BlockReason`: `BLOCK_REASON_UNSPECIFIED=0`, `SAFETY=1`, `OTHER=2`, `BLOCKLIST=3`,
`PROHIBITED_CONTENT=4`, `IMAGE_SAFETY=5` [V]. When the prompt is blocked, **no candidates are
returned** and the HTTP is 200 — surface `prompt_feedback.block_reason`, not an error [V].

Vertex explicitly notes: **when streaming, `candidate.content` is empty if content filters block the
output** — so an empty `parts` with `finishReason: SAFETY` is a refusal, not a transport failure
[V]. `GoModel` maps `SAFETY/RECITATION/LANGUAGE/BLOCKLIST/PROHIBITED_CONTENT/SPII/IMAGE_*` to
OpenAI `finish_reason: "content_filter"` (`internal/providers/gemini/native.go:1087`) [V].

### 4.6 Streaming framing [V/R]

- `streamGenerateContent` **default returns a JSON array** of `GenerateContentResponse` objects;
  add `?alt=sse` to get Server-Sent Events (`data: {...}` frames). SDK streaming requires SSE
  [V — google-gemini/cookbook `Streaming_REST.ipynb`; ai.google.dev `api/generate-content`].
- Each SSE data frame is a full `GenerateContentResponse` chunk (candidates/usageMetadata), not an
  OpenAI-style delta object [V].
- **Mid-stream errors can arrive as HTTP 200 with an `{"error": {...}}` frame inside the
  stream.** litellm detects this explicitly: `_check_streaming_error()` reads `error.code`
  (int-coerced, default 500), `error.message`, `error.status` and raises `VertexAIError` [R —
  `llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:3107-3130`, called at `:3243-3245`].
  This is the single most under-modeled Gemini stream shape.
- Interactions-API streaming (different surface) uses `event_type: "error"` with
  `{code, message}` [V].

### 4.7 `503 MODEL_CAPACITY_EXHAUSTED` [R]

Observed in the repo (`error_handler.py:999-1020`) and `stuff/GoModel` tests as a distinct 503
capacity signal separate from plain `UNAVAILABLE`. Report as a capacity/overload sub-code, treat as
retryable server error with cooldown; not in official docs.

---

## 5. Cross-cutting

### 5.1 Rate-limit & retry comparison

| Aspect | OpenAI (Chat + Responses) | Anthropic Messages | Gemini v1beta |
|---|---|---|---|
| Retry-After header | `retry-after` (seconds; SDK honors, cap 120s) | `retry-after` (seconds) | no standard header; use `RetryInfo.retryDelay` / body "reset after" |
| Rate headers | `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`, project-token variants [V] | `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` + priority [V] | none documented; `RetryInfo` in body [V] |
| Reset unit | duration string (`6m0s`) [V] | RFC 3339 timestamp [V] | protobuf Duration (`3.5s`) [V] |
| Quota vs rate | both 429, split by `error.code` (`insufficient_quota` vs `rate_limit_exceeded`) [V] | 429 rate_limit_error; spend cap 429 with **no** retry-after [V] | both `RESOURCE_EXHAUSTED` (429); split via `QuotaFailure`/`RetryInfo`/message [V] |
| Safe to retry | 408, 409, 429, ≥500; not 400/401/403/404/413 [V SDK] | connection errors, 429, 5xx; SDK 2x default [V] | 429, 408, 5xx only; not 400/403 [V] |
| Billing-specific | 429 `insufficient_quota`; retrying won't restore [V] | 402 `billing_error` [V] | 400 `FAILED_PRECONDITION` (billing not enabled) [V] |
| Timeout status | 408 / `APITimeoutError` (connection-class) [V] | 504 `timeout_error` [V] | 504 `DEADLINE_EXCEEDED` [V] |
| Overload status | 503 "Slow Down" / "engine overloaded" [V] | **529** `overloaded_error` (distinct) [V] | 503 `UNAVAILABLE` (+ `MODEL_CAPACITY_EXHAUSTED`) [V/R] |

### 5.2 Streaming-error shape comparison

| Protocol | Terminal success | Terminal failure | Error frame shape |
|---|---|---|---|
| OpenAI Chat | `data: [DONE]` | 4xx/5xx before stream, else abrupt close or `data:{"error":…}` [I/R] | none standardized |
| OpenAI Responses | `response.completed` | `response.failed` (carries `response.error`) AND/OR bare `error` event; may end with none [V] | `error`: `{type:"error", code?, message, param?, sequence_number}`; `response.failed.response.error`: `{code,message}` [V] |
| Anthropic | `message_stop` | `event: error` after HTTP 200 [V] | `{"type":"error","error":{"type":<error_type>,"message":…}}` [V] |
| Gemini | final chunk with `finishReason` (SSE) / end of JSON array | HTTP 200 with embedded `{"error":{code,message,status}}` SSE frame [R] | gRPC `Status` reused as a frame; int `code` [R litellm] |

Reusable invariant from the Responses evidence: **a stream is not successful unless a success
terminal was observed.** Failure terminals and silent truncation are distinct but both must not be
coerced into success [V spec, R langchain #39039].

### 5.3 Quota-at-400 and envelope/status mismatches (the G1 headline bug) [R]

- **Gemini-compatible gateways** (new-api etc.) wrap an upstream 429 inside an HTTP **500/503**
  envelope: `{"error":{"message":"…high demand…","type":"upstream_error","param":"","code":429}}`.
  litellm PR #30417 added body-`code`-429 promotion, then narrowed it to `500 ≤ status < 600` so
  HTTP 400/401 with body `code:429` stays BadRequest/Auth. [R]
- litellm issue #16189 / reopened #16881: Vertex quota returned **HTTP 500 instead of 429** because
  the mapper did not recognize `"Resource exhausted"`; fixed by matching it to `RateLimitError`
  regardless of status (PR #16363). [R]
- litellm PR #34977: an unanchored `"403"` substring in the Gemini 429 body matched the digits of a
  floating-point retry delay and misclassified a retryable 429 as a hard `BadRequestError`. [R]
- litellm PR #38706 expands rate-limit string patterns (`resource_exhausted`,
  `quota exceeded`, `insufficient_quota`, `provisioned throughput exceeded`, etc.) for non-429 or
  wrapped statuses. [R]

**Decision rule:** classify on the strongest structured signal available in this priority order —
(1) provider `error.status`/`error.type` string, (2) `error.code` body value, (3) HTTP status,
(4) message text — and never let an HTTP 400 hard-stop when an explicit quota/rate candidate exists
in the body. This is exactly G1 deliverable 2 in `fix-pass-plan.md:62-63`.

### 5.4 Transport exception taxonomy

`httpx` [V, local package]:
- `httpx.HTTPStatusError` (has `.response.status_code`, `.response.text`, `.response.headers`).
- `httpx.TimeoutException` (base; subtypes `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`,
  `PoolTimeout`), `httpx.ConnectError`, `httpx.NetworkError`, `httpx.TransportError`.
- The repo's classifier imports `httpx.NetworkError` and `httpx.TimeoutException`
  (`error_handler.py:1029-1034`) — note `httpx.Timeout` is not used.

`openai` SDK [V]:
- `APIConnectionError(APIError)` — no status, connection-class; `APITimeoutError(APIConnectionError)`
  — "Request timed out."; `APIStatusError` for 4xx/5xx with `.status_code`, `.request_id`
  (`x-request-id`); typed subclasses per status.

`anthropic` SDK [V]: `APIConnectionError`, `APITimeoutError(APIConnectionError)` (message mentions
network timeout / dropped connection / cancellation), `APIStatusError` with `.request_id` from
`request-id`; `RetryableError` is a marker for middleware to opt into retries.

`google.genai` SDK [V, `python-genai/google/genai/errors.py`]:
- `APIError` with `.code` (int), `.status` (string), `.message`, `.details` (raw JSON).
- `ClientError` for 4xx, `ServerError` for 5xx (both subclass `APIError`).
- Retry in `_api_client.py`: `MAX_RETRY_COUNT=3`, `INITIAL_RETRY_DELAY=1`, `DELAY_MULTIPLIER=2`;
  transient = `httpx.TimeoutException`/`ConnectError`.

`litellm` [V, `litellm/exceptions.py`]:
- `APIConnectionError` sets `status_code = 500` (line 833) — **connection failures look like 5xx**.
- `Timeout` keeps `status_code` (408 default; 504 when mapped from 504) (lines 330-347, 472-479).
- `RateLimitError` adds `category` (vendor vs litellm), `rate_limit_type`
  (requests/tokens/concurrent_requests/budget/max_iterations), and `headers` (only proxy-supplied —
  vendor response headers deliberately excluded for security) (lines 21-85, 413-500).
- `BudgetExceededError` = 429, category `litellm_rate_limit`, type `budget` (lines 960-986).
- `MidStreamFallbackError(ServiceUnavailableError)` propagates the original status, else 503, and
  carries `generated_content`, `is_pre_first_chunk`, `original_exception` (lines 1086-1160).

### 5.5 litellm exception → status map (the ladder the repo currently duplicates) [V]

`litellm/exceptions.py` (note: **deprecated `InvalidRequestError` is not in
`LITELLM_EXCEPTION_TYPES`**; `BadRequestError` is the live 400):

| Class | Status |
|---|---|
| `AuthenticationError` | 401 |
| `PermissionDeniedError` | 403 |
| `NotFoundError` | 404 |
| `BadRequestError` / `UnsupportedParamsError` / `RejectedRequestError` / `ImageFetchError` | 400 |
| `ContextWindowExceededError(BadRequestError)` | 400 |
| `ContentPolicyViolationError(BadRequestError)` | 400 |
| `UnprocessableEntityError` | 422 |
| `RateLimitError` / `BudgetExceededError` | 429 |
| `Timeout` | 408 (or 504 if `exception_status_code=504`) |
| `InternalServerError` | 500 |
| `BadGatewayError` | 502 |
| `ServiceUnavailableError` | 503 |
| `MidStreamFallbackError(ServiceUnavailableError)` | original status else 503 |
| `APIConnectionError` | 500 |
| `APIResponseValidationError` / `JSONSchemaValidationError` | 500 |
| `APIError` | carried `status_code` |

Per-provider mapper entry points in `litellm_core_utils/exception_mapping_utils.py`:
- OpenAI-like: `_map_openai_exception` (line 257) — string checks first (`rate limit`, 429,
  context-window patterns, `content_policy_violation`, `invalid_encrypted_content`), then status.
- Anthropic: `_map_anthropic_exception` (line 501) — **529 and 500 both map to
  `litellm.InternalServerError`** (lines 584-590); 503→`ServiceUnavailableError`; 429→`RateLimitError`.
- Gemini/Vertex: `_map_vertex_exception` (line 1114) — `"429 Quota exceeded"`, `"Quota exceeded for"`,
  `"Resource exhausted"`, capacity messages → `RateLimitError` **even when HTTP is 400/500**
  (lines 1201-1220); body `error.code == 429` inside a 5xx → `RateLimitError` (lines 1221-1242);
  then the status ladder.
- Context-window string set: `ExceptionCheckers.is_error_str_context_window_exceeded`
  (lines 74-106) — `exceed context limit`, `this model's maximum context length is`,
  `is longer than the model's context length`, `exceeds the maximum number of tokens allowed`
  (Gemini), `current length is … while limit is`, etc.
- Rate-limit string set: `is_error_str_rate_limit` (lines 37-71) — `\b429\b` (gated by status),
  `rate limit`, `service tier capacity exceeded`.

**Gap to correct in G1:** litellm's Anthropic 529 → `InternalServerError` loses the
overloaded distinction; the proxy should preserve/carry 529 as a distinct retryable class
(the repo's `StructuredAPIResponseError` has no 529 concept today — `core/errors.py:88-99`).

### 5.6 Content filter / refusal vs error

- OpenAI: `finish_reason == "content_filter"` on a normal 200; SDK raises local
  `ContentFilterFinishReasonError` only when parsing structured output [V]. 400
  `content_policy_violation` (`error.type: invalid_request_error`) is the hard server-side variant
  [V litellm `exceptions.py:588`].
- Responses: `status: "incomplete"` with `incomplete_details.reason: "content_filter"`; not
  `response.failed` [V/R].
- Anthropic: `stop_reason` refusal on a 200; content filtering on Vertex surfaces as
  `ContentPolicyViolationError` 400 with "The response was blocked." / "Output blocked by content
  filtering policy" [R litellm].
- Gemini: `finishReason` `SAFETY`/`RECITATION`/`PROHIBITED_CONTENT`/`BLOCKLIST`/`SPII`/`IMAGE_*` and
  `promptFeedback.block_reason`; empty content when streaming-blocked [V].
- **Rule:** a refusal is a terminal *successful* HTTP response with a stop signal; never rotate
  credentials for it. Rotating on refusal is a real defect class (litellm PR #34977 review flagged
  content-policy fallback leakage).

### 5.7 Context-window exhaustion spellings (for a single matcher)

| Protocol | Where | Spelling |
|---|---|---|
| OpenAI Chat | 400 `error.code` | `context_length_exceeded`; message "maximum context length is" |
| OpenAI Responses | 400 `error.code` | `context_length_exceeded` (param `input`) — **not** in `ResponseError.code` |
| Anthropic | 400 `error.type` invalid_request_error | message `prompt is too long` / `prompt: length` |
| Gemini | 400 `INVALID_ARGUMENT` / 400 `DEADLINE_EXCEEDED` / 500 | "exceeds the maximum number of tokens allowed", "input token limit"; 504 when prompt too large in time |
| Vertex | 400 / 413-like | "400 Request payload size exceeds", "prompt is too long", "Too many input tokens", "Input is too long" [R litellm] |
| Others litellm covers | — | "exceed context limit", "is longer than the model's context length", "`inputs` tokens + `max_new_tokens` must be", "current length is X while limit is Y" |

The repo's `_CONTEXT_WINDOW_ERROR_PATTERNS` (`error_handler.py:39-47`) is broad (`"too long"`,
`"max_tokens"`, `"too many tokens"`) and will false-positive on unrelated parameter errors; G1
should restrict to structured `code/type/status` fields first (per `fix-pass-plan.md:62`).

### 5.8 Companion gateway taxonomies (what peers do)

**Plexus** (`stuff/plexus/packages/backend`):
- Failover policy is status/token based: `isRetryableStatus` = `402` or configured
  `retryableStatusCodes`; default test config `[400, 402, 500, 502, 503, 504, 429]` [V
  `services/dispatch/failover-policy.ts:1-3`, `services/__tests__/dispatcher-quota-errors.test.ts:50`].
  Note it treats **400 as retryable** at the routing layer and uses `QUOTA_ERROR_PATTERNS` /
  `DEADLINE_EXPIRED_PATTERNS` constants for message matching.
- `isRetryableOAuthError` retries: no status, 5xx, 429, 402, plus timeout/network tokens [V].
- `isRetryableNetworkError`: token match on `.code`/`.message` against configured tokens
  (`ECONNREFUSED`/`ETIMEDOUT`/`ENOTFOUND`) [V].
- Quota gets a **distinct terminal responseStatus** `quota_exceeded` (not generic `error`) and a
  protocol-specific 429 reply, with fire-and-forget usage/error persistence [V
  `routes/inference/_quota-error.ts:22-59`].

**GoModel** (`stuff/GoModel`):
- Single `core.GatewayError` with `ErrorType` enum: `provider_error`, `rate_limit_error`,
  `invalid_request_error`, `authentication_error`, `not_found_error`, `permission_error`,
  `internal_error`; status defaults per type (429/400/401/404/403/502/500) [V
  `internal/core/errors.go:19-122`].
- `ParseProviderError(provider,status,body,err)` classifies primarily by HTTP status, extracts
  `message`/`param`/`code`/`error_type` from OpenAI-style or Azure-nested bodies,
  `auth_*`→401/403, 429→rate_limit, 404→not_found, other 4xx→invalid_request (preserving status),
  5xx→provider_error [V `errors.go:241-301`].
- Detects **embedded errors in 2xx bodies** (OpenRouter "Provider returned …" with `metadata.raw`,
  bare `{"error":…}` with no resource key) and preserves an `error.code` HTTP status else 502
  [V `errors.go:303-368`; `internal/llmclient/embedded_stream_error.go`].
- `model_not_found` code emitted on 404 [V `errors.go:236-238`].
- Anthropic rendering: 400→`invalid_request_error` (413→`request_too_large`),
  401→`authentication_error`/403→`permission_error`, 404→`not_found_error`, 429→`rate_limit_error`,
  503→`overloaded_error`, else `api_error` [V `internal/anthropicapi/errors.go:29-52`].
- Gemini: `finishReasonFromGemini` → OpenAI `finish_reason` with safety family→`content_filter`,
  `MAX_TOKENS`→`length`, tool calls→`tool_calls`; streams always use `:streamGenerateContent?alt=sse`
  [V `internal/providers/gemini/native.go:1087-1106`].

### 5.9 Decision-matrix implications for the single pipeline

Concrete rules the evidence supports (align with `fix-pass-plan.md:64`):

1. **Priority order** for classification: structured `error.status` string → `error.code` body
   value → HTTP status → message regex. Never hard-stop a 400 when an explicit quota/rate/status
   candidate exists (Gemini `RESOURCE_EXHAUSTED` at 400/500).
2. **Rotate** (try next credential): 429 rate, quota (`insufficient_quota`, `RESOURCE_EXHAUSTED`,
   Anthropic spend-cap 429), 401/403, 5xx, connection, 529 overloaded, `MODEL_CAPACITY_EXHAUSTED`.
3. **Failover, do not stop,** on auth (per user directive, `fix-pass-plan.md:64`).
4. **Stop** (no rotation): `configuration_error`, `not_found`, `invalid_request` (unless it carries
   a quota/rate body), `context_window_exceeded` (client must shrink request), `pre_request_callback_error`.
5. **Retry same key** with backoff for transient 5xx / connection and for small `retry_after`
   (< threshold).
6. **Cooldown** scope: quota → credential cooldown to parsed reset time
   (`RetryInfo.retryDelay` / `ErrorInfo.metadata.quotaResetDelay` / `quota will reset after`);
   503/capacity → model/provider cooldown; 429 rate → short credential cooldown.
7. **Refusal is not an error**; `finish_reason`/`stop_reason`/`finishReason` content-filter signals
   terminate successfully and never rotate.
8. **429 with no retry timing** is a transient rate limit (rotate) — the repo's `TransientQuotaError`
   already encodes this (`error_handler.py:215-238`), but `should_retry_same_key` currently routes
   it through `server_error`/503.
9. **Stream failures**: only a success terminal proves success. Recognize
   `response.failed`/`error` (Responses), `event: error` (Anthropic), and embedded
   `{"error":…}` frames at HTTP 200 (Gemini); a missing terminal is a failure.
10. **Normalize headers once**: OpenAI rest values are duration strings, Anthropic reset values are
    RFC 3339, Gemini delays are protobuf Durations, Gemini resets live in body `details[]` — one
    duration parser must handle all four (`error_handler.py:57-115` handles compound and `ms`, but
    not RFC 3339 or bare `1s`/`6m0s` cleanly — verify: `_parse_duration_string("1s")` works, but
    `x-ratelimit-reset-requests: 6m0s` is currently fed to `int()` and dropped).

### 5.10 Local defects this research confirms (for G1 tests)

- `error_handler.py:814` — `status_code` is taken from `payload.get("code")` before the Gemini
  `error.status` string is interpreted; `_classify_structured_error_text` receives `status` as a
  string but the numeric code wins the branch dispatch. This is the "N1 status-shadowing" defect.
- `core/errors.py:173` — `any(token in descriptor for token in ("rate", "quota", ...))` matches
  `"generate"`/`"accurate"` substrings. Restrict to exact structured fields.
- `error_handler.py:953` — quota sniff at 429 is `"quota" in error_body or "resource_exhausted"`;
  the 400 branch (`:978-990`) does **not** apply the same sniff, so gemini quota-at-400 becomes
  `invalid_request` and (per `should_rotate_on_error`) is non-rotatable. G1 fixes this.
- `error_handler.py:690-695` — `retry-after` parsed only as `int()`; HTTP-date and compound
  duration forms are silently dropped. Anthropic reset headers are RFC 3339 and ignored entirely.
- litellm Anthropic 529 → `InternalServerError` means any litellm-sourced Anthropic overload loses
  the 529/overloaded distinction; the proxy should re-derive from the body `error.type`.
- `classify_error` imports `InvalidRequestError` (deprecated, not in `LITELLM_EXCEPTION_TYPES`)
  alongside `BadRequestError` (`error_handler.py:20,1104`) — harmless but dead.

---

## Sources

Official docs (fetched 2026-09-11):
- OpenAI errors — https://platform.openai.com/docs/guides/error-codes · https://developers.openai.com/api/docs/guides/error-codes
- OpenAI rate limits — https://platform.openai.com/docs/guides/rate-limits
- OpenAI streaming — https://platform.openai.com/docs/guides/streaming-responses · https://developers.openai.com/api/reference/resources/responses/streaming-events/
- Anthropic errors — https://docs.anthropic.com/en/api/errors
- Anthropic rate limits — https://docs.anthropic.com/en/api/rate-limits
- Anthropic streaming — https://docs.anthropic.com/en/api/messages-streaming
- Gemini generateContent errors (legacy) — https://ai.google.dev/gemini-api/docs/generate-content/api-errors
- Gemini Interactions API errors — https://ai.google.dev/gemini-api/docs/api-errors
- Gemini troubleshooting — https://ai.google.dev/gemini-api/docs/troubleshooting
- Gemini generateContent reference — https://ai.google.dev/api/generate-content
- Vertex API errors — https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/api-errors
- Vertex GenerateContentResponse / FinishReason — https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse
- AIP-193 — https://cloud.google.com/apis/design/errors
- google.rpc error_details.proto — https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/error_details.proto
- google.rpc code.proto — https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/code.proto
- generativelanguage v1/v1beta generative_service.proto — github.com/googleapis/googleapis

SDK sources:
- openai-python — `_exceptions.py`, `types/shared/error_object.py`, `types/responses/response_error.py`, `types/responses/response_error_event.py`, `types/responses/response_failed_event.py`, `_base_client.py`, `_constants.py`; local install `C:\Python312\Lib\site-packages\openai` (2.54.0)
- anthropic-sdk-python — `src/anthropic/_exceptions.py`, `src/anthropic/types/shared/error_type.py`
- python-genai — `google/genai/errors.py`, `google/genai/_api_client.py`
- google-gemini/cookbook — `quickstarts/rest/Streaming_REST.ipynb` (`alt=sse`)

litellm (local `C:\Python312\Lib\site-packages\litellm`): `exceptions.py`,
`litellm_core_utils/exception_mapping_utils.py`,
`llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`.
litellm PRs/issues: #30417, #16363, #16189/#16881, #34977, #38706.

Issues / real-world: bifrost #4413 (+PR #4418), langchain #39039.

Repo: `src/rotator_library/error_handler.py`, `src/rotator_library/core/errors.py`,
`docs/experimental/fix-pass-plan.md`, `docs/experimental/audit-sweep-findings.md`,
`stuff/plexus/packages/backend/src/services/dispatch/failover-policy.ts`,
`stuff/plexus/packages/backend/src/routes/inference/_quota-error.ts`,
`stuff/GoModel/internal/core/errors.go`, `stuff/GoModel/internal/anthropicapi/errors.go`,
`stuff/GoModel/internal/server/error_support.go`,
`stuff/GoModel/internal/llmclient/embedded_stream_error.go`,
`stuff/GoModel/internal/providers/gemini/native.go`.
