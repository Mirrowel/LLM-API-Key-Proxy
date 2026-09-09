# Manual Test Guide — Pre-Merge Acceptance

The automated suite is green (1321 tests, zero failures) and all six protocol surfaces carry gatekeeper sign-off. This guide is the human acceptance pass over the real proxy process before merging `experimental` → `dev`.

## 0. Setup

```powershell
git switch experimental
python src/proxy_app/main.py            # TUI launcher, or:
python src/proxy_app/main.py --port 8000 --enable-request-logging
```

Confirm startup logs show: OAuth bootstrap (if `oauth_creds/` present), provider registry, native execution defaults (no unexpected `litellm_fallback` warnings on model discovery).

## 1. Chat Completions (OpenAI wire)

```powershell
curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $env:PROXY_API_KEY" -H "Content-Type: application/json" -d '{\"model\":\"openai/gpt-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'
```

- [ ] Non-streaming: legal `choices[0].message`, `usage` with `prompt_tokens_details.cached_tokens` when the provider reports cache
- [ ] Streaming: first chunk has `delta.role`, terminal chunk carries `finish_reason`, usage-only frame after finish (`choices: []`), then `[DONE]`
- [ ] Tool round-trip: send a `tools` definition; follow the tool call back as a `tool` message; second call completes
- [ ] Same model via a `provider:profile/model` address (if the provider is multi-profile)

## 2. Anthropic Messages (Claude clients)

Point Claude Code / any Anthropic SDK at `http://localhost:8000` (base URL) with the proxy key.

- [ ] `POST /v1/messages` non-streaming: `content` blocks, `stop_reason`, `usage` (input tokens cache-INCLUSIVE), `thinking` blocks when extended thinking requested
- [ ] Streaming: `message_start` → `content_block_*` lifecycle → `message_delta` (cumulative usage) → `message_stop`; thinking streams with `signature_delta` on same-protocol clients
- [ ] Tool use round-trip (`tool_use` / `tool_result` pairing)
- [ ] `POST /v1/messages/count_tokens` returns `input_tokens` and the `x-proxy-estimate: local-projection` header
- [ ] Error shape: bad model → Anthropic-formatted error envelope (not FastAPI `{"detail": ...}`)

## 3. Responses API

```powershell
curl http://localhost:8000/v1/responses -H "Authorization: Bearer $env:PROXY_API_KEY" -H "Content-Type: application/json" -d '{\"model\":\"openai/gpt-5.2\",\"input\":\"hi\",\"store\":true}'
```

- [ ] Non-streaming: `output` items (`message`, `reasoning` with `encrypted_content` when the provider emits it), `usage` (`input_tokens` cache-inclusive)
- [ ] Streaming: full lifecycle with monotonic `sequence_number` on every event, terminal `response.completed`
- [ ] Continuation: send `previous_response_id` from the stored response — lineage resolves through the scoped store
- [ ] `store: false` (ZDR) over WebSocket (`/v1/responses/ws`): `response.create` turns chain connection-locally; a failed turn's id MISSES on the next turn unless store-failed policy says otherwise
- [ ] Post-start provider failure mid-stream: stream ends in `response.failed` + `[DONE]` (never a raw transport abort)

## 4. Gemini

Point a Gemini client at the proxy with `gemini/<model>` addressing.

- [ ] `contents`/`parts` round-trip incl. `fileData.fileUri` media and `functionCall`/`functionResponse` pairs
- [ ] `generationConfig`: `responseJsonSchema` structured output, `thinkingConfig` (`thinkingBudget`, `thinkingLevel`), `candidateCount > 1` returns multiple candidates with per-candidate `finishReason`
- [ ] Streaming: SSE candidates, `usageMetadata` on the final chunk; `:countTokens` operation

## 5. Cross-protocol conversion spot checks

- [ ] Anthropic client → OpenAI-provider model (thinking budget ↔ effort mapping; conversion summary warnings visible in logs when approximated)
- [ ] Chat client → Gemini model with images (inline base64 parts; `modalities`/`audio` pairing)
- [ ] Responses client → non-OpenAI model (reasoning summaries degrade with disclosure; `previous_response_id` stays proxy-local)
- [ ] Verify no `Cannot represent required content type` for ordinary traffic (custom-tool conversations replay; genuinely unrepresentable hosted tools reject with a clear message)

## 6. Rotation & fallback (live)

- [ ] Two keys for one provider; force a 429 (or use quota-limited keys): rotation advances, cooldown lands in `quota_viewer` / `/quota` stats
- [ ] `FALLBACK_GROUP_x=provider-a/model,provider-b/model` + `MODEL_ROUTE_alias=group:x`: request the alias, kill provider-a's key, confirm ordered failover with the response still in the CLIENT's protocol
- [ ] LiteLLM fallback: request a model with explicit `@litellm_fallback` — confirm the warning log + `execution_mode` in `logs/transactions/*/metadata.json`

## 7. Transaction logging (L1)

With `--enable-request-logging`:

- [ ] Each request writes one directory under `logs/transactions/`: `request.json`, `response.json`, `provider/request_payload.json`, `provider/final_response.json`, `streaming_chunks.jsonl` (zstd-suffixed when available), `metadata.json` v2
- [ ] `metadata.json` shows `attempts[]` with execution mode, routing decisions, timing, and the reconstruct shortcut
- [ ] Trigger a 400-class error (bad model): `capture/captured_trace.json` archives intermediates; a 429 does NOT
- [ ] `python tools/reconstruct_traces.py logs/transactions/<dir>` regenerates the intermediate trace offline

## 8. Restart persistence

- [ ] `SESSION_PERSISTENCE_ENABLED=true`, chat a few turns, restart, continue the conversation — session continues (log line shows `origin=persisted`)
- [ ] Responses store with `provider_cache` backend survives restart (`previous_response_id` resolves)

## Exit criteria

All boxes checked (or deviations understood) → merge `experimental` → `dev` from your side. Nothing in the branch auto-merges or pushes.
