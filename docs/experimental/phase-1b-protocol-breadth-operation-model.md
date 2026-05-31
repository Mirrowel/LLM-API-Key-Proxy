# Phase 1b: Protocol Breadth And Operation Model

## Goal

Correct the main Phase 1 audit failure. Phase 1 created a good chat/message protocol foundation, but the starting requirements asked for protocols as the highest priority and for broad protocol coverage. The current protocol layer is still too chat-oriented: it has OpenAI Chat, Anthropic Messages, Gemini, Responses, and LiteLLM fallback, but no explicit operation model for embeddings, image generation, audio transcription, speech/TTS, Ollama-native chat/generate, MCP-style tool gateway calls, or count-token operations.

## Non-Goals

- Do not wire every new operation into live FastAPI routes in this corrective slice. This is protocol foundation work.
- Do not replace current embeddings or Anthropic count_tokens routes yet.
- Do not replace existing provider execution, `UsageManager`, `SessionTracker`, routing, or LiteLLM fallback.
- Do not add SQLite or persistent admin/config databases.
- Do not implement full MCP transport/proxy behavior yet; define the protocol shape and operation seam so it is not blocked later.
- Do not fake WebSocket runtime support; keep explicit future transport/capability metadata.
- Do not touch uncommitted phase reports or `docs/issues/`.

## Current Code Context

- `src/rotator_library/protocols/types.py` has unified chat/message/request/response/stream dataclasses, but `UnifiedRequest` has no `operation` field and no dedicated multimodal operation carrier fields.
- `ProtocolAdapter` has `supported_transports` but no `supported_operations` or `supports_operation()`.
- Existing concrete protocols are chat/message/generate-content focused.
- `ContentBlock` already has generic `type`, `source`, `raw`, and `extra`, which can support multimodal payloads with careful extension.
- Registry auto-discovery exists and should be reused for new protocol modules.
- Tests exist only for the initial protocol modules.

## Files To Add

- `src/rotator_library/protocols/operation.py`
- `src/rotator_library/protocols/openai_embeddings.py`
- `src/rotator_library/protocols/openai_images.py`
- `src/rotator_library/protocols/openai_audio.py`
- `src/rotator_library/protocols/ollama.py`
- `src/rotator_library/protocols/mcp.py`
- `tests/test_protocol_operation_model.py`
- `tests/test_protocol_openai_embeddings.py`
- `tests/test_protocol_openai_images_audio.py`
- `tests/test_protocol_ollama_mcp.py`

## Files To Touch

- `src/rotator_library/protocols/types.py`
- `src/rotator_library/protocols/base.py`
- `src/rotator_library/protocols/__init__.py`
- Existing protocol modules where they should declare `supported_operations`.
- Existing protocol tests if serialization expectations need operation fields added.

## Data Model Additions

- Add operation constants in a small `operation.py` module, not a rigid enum, so custom/local protocols can introduce operation strings without fighting the core.
- Initial standard operation names: `chat`, `messages`, `responses`, `count_tokens`, `embeddings`, `image_generation`, `image_edit`, `image_variation`, `audio_transcription`, `audio_translation`, `speech`, `ollama_chat`, `ollama_generate`, `mcp`, and `unknown`.
- Add `operation` to `UnifiedRequest`.
- Add flexible operation carrier fields to `UnifiedRequest`: `input`, `modalities`, and `files`.
- Add `operation` to `UnifiedResponse`.
- Add flexible response fields to `UnifiedResponse`: `data` and `content_type`.
- Keep existing fields and defaults chat-compatible so existing adapters do not break.
- Update explicit serialization fields so transform logging can see the new values.

## Protocol Adapter Additions

- Add class attribute `supported_operations`.
- Add `supports_operation(operation_name)`.
- Existing protocols should declare their primary operations.
- The base adapter defaults to `operation="unknown"` when no operation is identified.
- LiteLLM fallback stays broad and explicit, not a native implementation claim.

## New Protocol Modules

### OpenAI Embeddings

- Protocol name: `openai_embeddings`.
- Aliases: `embeddings`.
- Operation: `embeddings`.
- Parse/build `/v1/embeddings` style fields: `model`, `input`, `encoding_format`, `dimensions`, and `user`.
- Parse `data[].embedding` and `usage` from responses.
- Preserve unknown fields.

### OpenAI Images

- Protocol name: `openai_images`.
- Aliases: `images`, `image_generation`.
- Operations: `image_generation`, `image_edit`, and `image_variation`.
- Parse/build standard OpenAI image fields: `prompt`, `model`, `n`, `size`, `quality`, `style`, `response_format`, `image`, `mask`, and `user`.
- Parse response `data` entries with `url`, `b64_json`, and `revised_prompt`.
- Preserve file/source metadata without reading file contents.

### OpenAI Audio

- Protocol name: `openai_audio`.
- Aliases: `audio`, `audio_transcription`, and `speech`.
- Operations: `audio_transcription`, `audio_translation`, and `speech`.
- Parse/build transcription/translation fields: `file`, `model`, `language`, `prompt`, `response_format`, `temperature`, and `timestamp_granularities`.
- Parse/build speech fields: `input`, `model`, `voice`, `response_format`, and `speed`.
- Preserve binary or text provider responses in `raw`; map JSON responses into `data` or `output` when possible.

### Ollama

- Protocol name: `ollama`.
- Operations: `ollama_chat`, `ollama_generate`, and compatible `embeddings`.
- Parse/build `/api/chat`, `/api/generate`, and `/api/embeddings` shapes.
- Map `messages`, `prompt`, `options`, `system`, `template`, `context`, `keep_alive`, `format`, and `stream`.
- Parse final and stream chunks while preserving raw context/performance fields.

### MCP

- Protocol name: `mcp`.
- Operation: `mcp`.
- Define a JSON-RPC carrier for `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, and generic method calls.
- Do not claim a full MCP proxy implementation in this slice.
- Preserve `jsonrpc`, `method`, `params`, `id`, `result`, and `error` fields exactly.

## Count Tokens

- Add count-token operation support where it naturally fits, especially Anthropic Messages.
- Avoid duplicating current count-token routes in this slice.
- Ensure `ProtocolAdapter.supports_operation("count_tokens")` can represent a protocol's support.

## Tests

- Operation model serialization includes `operation`, `input`, `modalities`, `files`, response `data`, and `content_type`.
- `ProtocolAdapter.supports_operation()` works and remains override-friendly.
- Registry discovers all new protocols.
- Embeddings request/response parse/build round-trip and usage extraction.
- Images request/response parse/build, including edit file/mask references.
- Audio transcription and speech request/response parse/build.
- Ollama chat/generate parse/build and stream parsing.
- MCP request/response/error round-trip.
- Existing protocol regression tests remain clean.

## Transaction Logging Implications

- This slice only adds trace-friendly fields and protocol adapters.
- Do not wire live transform logs here.
- Preserve `raw` and `extra` so Phase 2b can log every state.

## Docstrings And Comments Required

- Public classes must explain what API shape they model and what remains intentionally future-facing.
- Operation constants must document that strings are intentionally extensible.
- Lossy conversions, especially binary/audio/image payload preservation, need comments explaining that file contents are not inspected and provider-specific metadata stays in `extra`.

## Commit Checkpoints

1. Plan doc commit: `docs(experimental): plan protocol breadth correction`.
2. Operation model/types/base adapter commit with tests.
3. Embeddings/images/audio protocol commit with tests.
4. Ollama/MCP/count-token capability commit with tests.
5. Focused protocol regression run.
6. `explore` and `explore-heavy` review against this plan and the source requirements.
7. Fix findings and re-review if substantial.
8. Write uncommitted Phase 1b report.

## Risks

- Adding fields to dataclasses can break exact dict assertions. Mitigation: update tests only where serialization now intentionally includes new fields.
- A rigid operation enum could block custom providers. Mitigation: use string constants/helpers, not a closed enum.
- New non-chat adapters could overclaim live support. Mitigation: protocol package supports parse/build only; route/provider wiring comes later.
- MCP can sprawl. Mitigation: implement JSON-RPC carrier shape only, not a full proxy.
