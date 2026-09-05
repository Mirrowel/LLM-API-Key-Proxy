# Transform Pass Logging

Debuggability is a core requirement. Every request, response, and stream payload must be inspectable after each transformation pass.

## Existing Baseline

The current project already has transaction and raw I/O logging, but it does not consistently show every intermediate transformed state. The experimental protocol layer must improve this without making normal operation too noisy.

## Transform Trace Model

Each request should have a trace containing ordered pass records. A pass record should include:

- request ID.
- pass name.
- direction: request, response, stream_in, stream_out, error.
- protocol/provider/model context.
- timestamp.
- payload snapshot, redacted if needed.
- optional notes or warnings.
- exception information if the pass failed.

## Required Request Passes

Suggested pass names:

- `raw_client_request`
- `parsed_unified_request`
- `after_session_inference`
- `after_field_cache_injection`
- `after_request_adapters`
- `after_provider_override`
- `provider_request`
- `litellm_fallback_request` when fallback is used

## Required Response Passes

- `raw_provider_response`
- `parsed_unified_response`
- `after_field_cache_extraction`
- `after_response_adapters`
- `after_client_protocol_format`
- `final_client_response`

## Required Stream Passes

For streams, every event can be large. Logging should support configurable sampling/full capture, but the architecture must be able to record:

- `raw_provider_stream_event`
- `parsed_unified_stream_event`
- `after_stream_field_cache_extraction`
- `after_stream_adapters`
- `formatted_client_stream_event`

The transaction logger should be able to record stream events as JSONL to avoid retaining large streams in memory.

## Redaction

Even though full multi-user security is out of scope, transform logging must avoid accidental credential leakage.

Redaction hooks should cover:

- `Authorization` headers.
- `x-api-key` and related API key headers.
- provider API keys.
- OAuth access and refresh tokens.
- cookies.
- obvious `api_key`, `access_token`, `refresh_token`, `client_secret`, and `Authorization` fields in JSON payloads.

Redaction should happen at the logging boundary, not by mutating live request objects.

## Failure Debugging

When a transform fails, the trace should identify:

- failed pass name.
- protocol class.
- provider/model.
- whether the failure occurred before or after provider execution.
- original error type.
- redacted payload snapshot when possible.

## Future Expansion

Later admin/debug endpoints can read these traces. For now, file-based transaction logging is sufficient.
