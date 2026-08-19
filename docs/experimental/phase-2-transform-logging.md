# Phase 2 Plan: Transform-Pass Logging

## Goal

Make transaction logging capable of recording every meaningful request, response, and stream state after each transformation pass, without yet replacing runtime execution with the native protocol adapters. This phase creates the durable trace format and wires it into existing request/executor/provider logging points so later protocol, adapter, and field-cache phases can add more trace entries without redesigning observability.

## Non-Goals

- Do not route execution through native protocols yet.
- Do not implement field-cache extraction or injection yet.
- Do not add Responses routes yet.
- Do not add full multi-user/security work.
- Do not rewrite provider-specific logging systems.
- Do not change request/response behavior.

## Current Code Context

- `TransactionLogger` creates one directory per transaction and writes legacy files such as `request.json`, `request_transformed.json`, `response.json`, `streaming_chunks.jsonl`, and `metadata.json`.
- `RequestContextBuilder` creates `TransactionLogger` and calls `log_request(kwargs)`.
- `RequestExecutor` calls `log_transformed_request(kwargs, context.kwargs)` after request preparation.
- `RequestExecutor` logs non-streaming responses through `log_response(response_data)`.
- `RequestExecutor._transaction_logging_stream_wrapper()` logs parsed chunks and assembles/logs the final streamed response.
- `ProviderLogger` writes provider-level request payloads, raw response chunks, final responses, and provider errors.
- Phase 1 unified types already provide JSON-safe serialization through `serialize_value()`.

## Files To Add

- `src/rotator_library/transform_trace.py`
- `tests/test_transform_trace.py`
- `tests/test_transaction_logger_transform_trace.py`

## Files Likely To Touch

- `src/rotator_library/transaction_logger.py`
- `src/rotator_library/client/request_builder.py`
- `src/rotator_library/client/executor.py`

## Trace Model

`TransformTraceEntry` fields:

- `sequence`: monotonically increasing integer per writer instance.
- `timestamp_utc`.
- `component`: `client` or `provider`.
- `pass_name`: stable machine-readable name.
- `direction`: `request`, `response`, `stream`, `error`, or `metadata`.
- `stage`: `client`, `protocol`, `adapter`, `provider`, or `final`.
- `protocol`: optional protocol name.
- `provider`: optional provider name.
- `model`: optional model name.
- `credential_id`: optional stable or masked credential identifier only.
- `transport`: `http`, `sse`, `websocket`, or a future string.
- `changed_from_previous`: optional bool.
- `metadata`: JSON-safe dict.
- `data`: JSON-safe sanitized payload.

`TransformTraceWriter` responsibilities:

- Maintain a local sequence counter.
- Write compact ordered entries to `transform_trace.jsonl`.
- Optionally write individual request/response snapshots under `transforms/`.
- Avoid per-chunk snapshot files for stream chunks.
- Never raise logging failures into request execution.

## Log Files

Existing files stay for compatibility:

- `request.json`
- `request_transformed.json`
- `response.json`
- `streaming_chunks.jsonl`
- `metadata.json`

New files:

- `transform_trace.jsonl`
- `transforms/0001_raw_client_request.json`
- `transforms/0002_prepared_provider_request.json`
- `transforms/NNNN_<pass_name>.json` for non-stream snapshots

Stream chunks use JSONL entries rather than one file per chunk.

## Required Pass Names

Request passes:

- `raw_client_request`
- `prepared_provider_request`
- `provider_request_payload`

Response passes:

- `raw_client_response`
- `final_client_response`
- `provider_final_response`

Stream passes:

- `raw_stream_chunk`
- `parsed_stream_chunk`
- `assembled_stream_response`
- `provider_raw_stream_chunk`

Error passes:

- `provider_error`
- `transform_log_error`

## Sanitization

This is pragmatic trace hygiene, not full security work.

Redact recursively by key name:

- `api_key`
- `credential_identifier`
- `authorization`
- `x-api-key`
- `x-goog-api-key`
- `access_token`
- `refresh_token`
- `client_secret`
- `password`
- `secret`
- `token`

Rules:

- Redacted value is `"[REDACTED]"`.
- Redact by key only, not by text value.
- Do not hide ordinary model text containing words such as token.
- Always serialize through Phase 1 JSON-safe helpers.

## Integration Behavior

- Trace logging is additive and backward compatible.
- Logging failures never fail requests.
- `log_request()` records `raw_client_request`.
- `log_transformed_request()` records `prepared_provider_request` even when legacy transformed file is skipped because payloads compare equal.
- `log_response()` records `final_client_response`.
- A dedicated method can record `raw_client_response` before final normalization when available.
- `log_stream_chunk()` records `parsed_stream_chunk`.
- `_transaction_logging_stream_wrapper()` records `raw_stream_chunk` before JSON parsing and `assembled_stream_response` before final `log_response()`.
- `ProviderLogger.log_request()` records `provider_request_payload`.
- `ProviderLogger.log_response_chunk()` records `provider_raw_stream_chunk`.
- `ProviderLogger.log_final_response()` records `provider_final_response`.
- `ProviderLogger.log_error()` records `provider_error`.

## Context Design

- `TransactionLogger` owns a client-side `TransformTraceWriter`.
- `TransactionContext` carries enough trace metadata for providers to create provider-side trace entries without sharing mutable logger objects.
- `ProviderLogger` creates its own writer with `component="provider"`.
- Client and provider sequence numbers are local to their writer instances. Entries include component and timestamps, so Phase 2 does not promise one global order across independent writers.

## Comments And Docstrings

- Public trace classes must explain they are observability-only and must not affect request behavior.
- Redaction helpers must explain why redaction is key-based rather than value-based.
- TransactionLogger comments should distinguish legacy files from the new trace ledger.
- Stream comments should explain why stream chunks use JSONL rather than per-chunk snapshots.

## Tests

`tests/test_transform_trace.py`:

- JSON-safe serialization handles dataclasses and non-primitives.
- Redaction recurses through nested payloads.
- Redaction does not redact normal text values.
- Sequence increments per writer.
- Snapshot filenames are stable and sanitized.

`tests/test_transaction_logger_transform_trace.py`:

- `log_request()` writes legacy `request.json` and trace entry `raw_client_request`.
- `log_transformed_request()` writes trace entry even when legacy transformed file is skipped because payloads are equal.
- `log_response()` writes `final_client_response`.
- `log_stream_chunk()` writes `parsed_stream_chunk`.
- ProviderLogger writes provider request, chunk, final, and error trace entries.
- Logging disabled writes nothing.

Regression tests:

- Phase 1 protocol tests.
- `tests/test_session_tracking.py`.
- `tests/test_selection_engine.py`.

## Commit Checkpoints

1. Add transform trace dataclasses, writer, redaction, and tests.
2. Integrate client-side `TransactionLogger` trace entries and tests.
3. Integrate provider-side `ProviderLogger` trace entries and tests.
4. Run focused and regression tests.
5. Review with `explore` and `explore-heavy`, fix findings, and write the uncommitted Phase 2 report.

## Risks And Mitigations

- Stream logs can grow quickly. Use JSONL only for stream chunks.
- Redaction can hide useful fields if too broad. Redact by sensitive key names only.
- Provider/client entry ordering is not globally sequenced in Phase 2. Include component and timestamps.
- Provider SDK objects may not be JSON-native. Always use `serialize_value()`.
- Tests may import an installed package if local `src` is not first on `sys.path`. Keep `tests/conftest.py`.
