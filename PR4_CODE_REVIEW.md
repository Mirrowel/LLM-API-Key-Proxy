Comprehensive code review for PR #4: OAuth, custom providers, streaming, and resiliency

Repository: Mirrowel/LLM-API-Key-Proxy
Branch reviewed: cli-oauth (diff against main)
Local review branch: review/pr-4-comprehensive-code-review

Summary
This PR is a significant feature release integrating OAuth-backed providers (Gemini CLI, Qwen Code), a provider plugin system, robust streaming aggregation, environment-driven model filtering (ignore/whitelist), proactive OAuth token refresh, refined usage tracking/cooldowns, and improved logging. Overall direction is solid and feature set valuable, but there are a few high-severity security and correctness issues to address before merging. Performance and observability are generally good with clear log separation, though log volume and sensitive data handling need tightening.

High-priority findings (must-fix)
1) Secrets in logs (Authorization and request headers)
- Where: src/proxy_app/detailed_logger.py, src/rotator_library/failure_logger.py, rotator_library/client.py (_litellm_logger_callback)
- Issue: Detailed logger writes raw headers including Authorization to disk; failure_logger logs request headers (including Authorization) to failures.log; litellm callback may include headers in kwargs.
- Risk: Leaks PROXY_API_KEY and potentially other app secrets to disk. Violates least-privilege and can leak PII.
- Fix: Redact Authorization-like headers and known secret keys consistently before writing. Example patch below.

2) SSRF / egress override via user-controlled api_base and headers
- Where: rotator_library/client.py (_execute_with_retry, _streaming_acompletion_with_retry)
- Issue: The proxy forwards kwargs from user input into litellm. A malicious client may include api_base, base_url, headers/extra_headers to force requests to arbitrary hosts.
- Risk: SSRF and data exfiltration. The proxy should not allow user-controlled network destinations or arbitrary headers.
- Fix: Sanitize request payload to remove api_base, base_url, headers, extra_headers, and other network-control fields unless explicitly whitelisted via internal provider settings.

3) Endpoint returns None on provider failures
- Where: src/proxy_app/main.py (/v1/chat/completions, /v1/embeddings)
- Issue: RotatingClient returns None after exhausting retries. Endpoints do not check for None and will return null with 200 or raise unexpected errors.
- Risk: Incorrect HTTP semantics and confusing client behavior.
- Fix: If client returns None, respond 502 Bad Gateway (or 503), with a structured error object.

4) BackgroundRefresher looks up wrong provider name
- Where: src/rotator_library/background_refresher.py
- Issue: Uses _get_provider_instance(f"{provider}_oauth"). Providers are registered under names like gemini_cli, qwen_code—not suffixed with _oauth. As a result, proactively_refresh never runs.
- Fix: Use _get_provider_instance(provider).

5) Sensitive debug logging volume (default on) can degrade performance and leak details
- Where: src/proxy_app/main.py root logger wiring + rotator_library/client.py litellm callback
- Issue: All rotator_library DEBUG logs are persisted to logs/proxy_debug.log by default. At high QPS this will be heavy I/O.
- Fix: Gate debug file handler behind an env flag (e.g., PROXY_DEBUG_LOG=true) or lower default level. Ensure litellm logs do not include sensitive content.

Medium-priority findings
6) httpx AsyncClient without explicit timeouts
- Where: rotator_library/client.py and providers
- Issue: httpx.AsyncClient uses a default timeout but several calls in providers specify long timeouts (e.g., 600s). Centralize and cap defaults.
- Fix: Instantiate AsyncClient with a sane default timeout from env (e.g., HTTPX_TIMEOUT=30s) and override per-call only where necessary.

7) .gitignore missing oauth_creds/
- Where: root .gitignore, src/rotator_library/credential_manager.py (uses cwd()/oauth_creds)
- Issue: Copied OAuth credentials are stored under oauth_creds/ in repo root but are not ignored. Easy to commit accidentally.
- Fix: Add oauth_creds/ to .gitignore.

8) PROXY_API_KEY hard-failure on import
- Where: src/proxy_app/main.py (global module import)
- Issue: Raises ValueError at import time if PROXY_API_KEY not set. This breaks import-based tooling and tests.
- Fix: Move validation to lifespan or verify_api_key dependency, or fallback in tests with env defaults.

9) Streaming aggregation could improve tool_calls typing
- Where: src/proxy_app/main.py streaming_response_wrapper
- Issue: Aggregated tool_calls omit type="function"; arguments concatenation is correct, but type helps compatibility.
- Fix: Add type='function' for each tool call in the aggregated final response, and ensure function key presence.

10) Token-count endpoint ignores text field
- Where: src/proxy_app/main.py (/v1/token-count)
- Issue: Endpoint requires messages, but RotatingClient.token_count supports text or messages.
- Fix: Accept text or messages; validate model is present.

11) Logs unbounded (no rotation) for proxy.log and proxy_debug.log
- Where: src/proxy_app/main.py
- Issue: Uses FileHandler without rotation.
- Fix: Use RotatingFileHandler with size caps and retention.

12) Startup prints to stdout on import
- Where: src/proxy_app/main.py lines 1–2
- Issue: Prints on import. Noisy and not appropriate for libraries or tests.
- Fix: Move prints under __main__.

13) LLM request sanitization too narrow
- Where: src/rotator_library/request_sanitizer.py
- Issue: Only removes dimensions and specific thinking patterns. Does not sanitize known unsupported/unsafe litellm parameters coming from users.
- Fix: Drop additional keys: api_base, base_url, headers, extra_headers, organization, timeout, num_retries, api_key, additional_params, etc.

Change mapping (diff overview)
- proxy_app
  - main.py: major updates for OAuth discovery/deduplication, streaming aggregation, provider/model filters, detailed logging, lifespan-driven RotatingClient, providers enumeration and token count endpoint. New CLI arg --add-credential to launch interactive tool.
  - detailed_logger.py: new per-request detailed logging with JSON artifacts and streaming chunk capture.
  - request_logger.py + provider_urls.py: compact console logging and provider endpoint mapping.
  - batch_manager.py: embedding server-side batcher (optional, currently disabled).

- rotator_library
  - client.py: substantial refactor with
    - global deadline-aware retries
    - streaming retry path with robust buffer reassembly and error classification
    - provider plugin dispatch (custom logic for non-OpenAI-compatible endpoints)
    - usage recording, cooldown manager, model whitelist/blacklist
    - litellm logger callback with sanitization
    - AllProviders API base/model prefix mapping (currently for chutes)
  - providers/: new providers gemini_cli_provider (custom streaming and tool use), qwen_code_provider (custom streaming), auth base classes (gemini_auth_base, qwen_auth_base) with interactive OAuth bootstrap and proactive refresh
  - usage_manager.py: async-safe key acquisition & per-model locks, daily reset and cost calculation
  - failure_logger.py: structured failures.jsonl with rotating handler
  - background_refresher.py: new loop for proactive OAuth refresh (bug noted above)
  - credential_manager.py + credential_tool.py: discovery/copy of OAuth creds and interactive setup

Recommended patches (inline examples)
1) Redact sensitive headers before logging

File: src/proxy_app/detailed_logger.py
--- a/src/proxy_app/detailed_logger.py
+++ b/src/proxy_app/detailed_logger.py
@@
     def log_request(self, headers: Dict[str, Any], body: Dict[str, Any]):
         """Logs the initial request details."""
         self.streaming = body.get("stream", False)
+        safe_headers = dict(headers)
+        # Redact Authorization and any x-api-key like headers
+        for k in list(safe_headers.keys()):
+            if k.lower() in {"authorization", "proxy-authorization"} or "api-key" in k.lower():
+                safe_headers[k] = "REDACTED"
         request_data = {
             "request_id": self.request_id,
             "timestamp_utc": datetime.utcnow().isoformat(),
-            "headers": dict(headers),
+            "headers": safe_headers,
             "body": body
         }
         self._write_json("request.json", request_data)

File: src/rotator_library/failure_logger.py
--- a/src/rotator_library/failure_logger.py
+++ b/src/rotator_library/failure_logger.py
@@
-def log_failure(api_key: str, model: str, attempt: int, error: Exception, request_headers: dict, raw_response_text: str = None):
+def log_failure(api_key: str, model: str, attempt: int, error: Exception, request_headers: dict, raw_response_text: str = None):
     """
     Logs a detailed failure message to a file and a concise summary to the main logger.
     """
+    # Redact secrets
+    safe_headers = dict(request_headers or {})
+    for k in list(safe_headers.keys()):
+        if k.lower() in {"authorization", "proxy-authorization"} or "api-key" in k.lower():
+            safe_headers[k] = "REDACTED"
@@
-    detailed_log_data = {
+    detailed_log_data = {
         "timestamp": datetime.utcnow().isoformat(),
         "api_key_ending": api_key[-4:],
         "model": model,
         "attempt_number": attempt,
         "error_type": type(error).__name__,
         "error_message": str(error),
         "raw_response": raw_response,
-        "request_headers": request_headers,
+        "request_headers": safe_headers,
     }

File: src/rotator_library/client.py (litellm logger callback)
--- a/src/rotator_library/client.py
+++ b/src/rotator_library/client.py
@@ def _sanitize_litellm_log(self, log_data: dict) -> dict:
-        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]
+        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request", "headers"]
@@
             for key, value in list(data_dict.items()):
                 if isinstance(value, dict):
                     clean_recursively(value)
+            # Mask Authorization header if present
+            headers = data_dict.get("headers")
+            if isinstance(headers, dict):
+                for hk in list(headers.keys()):
+                    if hk.lower() in {"authorization", "proxy-authorization"} or "api-key" in hk.lower():
+                        headers[hk] = "REDACTED"

2) Block user-controlled network overrides (SSRF mitigation)

File: src/rotator_library/request_sanitizer.py
--- a/src/rotator_library/request_sanitizer.py
+++ b/src/rotator_library/request_sanitizer.py
@@ def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
-    return payload
+    # Disallow user-controlled egress
+    for k in ["api_base", "base_url", "headers", "extra_headers", "proxy_server_request", "organization", "timeout", "api_key"]:
+        if k in payload:
+            del payload[k]
+    return payload

3) Return explicit error when Rotation fails

File: src/proxy_app/main.py (/v1/chat/completions and /v1/embeddings)
--- a/src/proxy_app/main.py
+++ b/src/proxy_app/main.py
@@ async def chat_completions(...):
-            response = await client.acompletion(request=request, **request_data)
+            response = await client.acompletion(request=request, **request_data)
+            if response is None:
+                raise HTTPException(status_code=502, detail="Upstream providers unavailable after retries")
@@ async def embeddings(...):
-            response = await client.aembedding(request=request, **request_data)
+            response = await client.aembedding(request=request, **request_data)
+            if response is None:
+                raise HTTPException(status_code=502, detail="Upstream providers unavailable after retries")

4) Fix BackgroundRefresher provider lookup

File: src/rotator_library/background_refresher.py
--- a/src/rotator_library/background_refresher.py
+++ b/src/rotator_library/background_refresher.py
@@ async def _run(self):
-                for provider, paths in oauth_configs.items():
-                    provider_plugin = self._client._get_provider_instance(f"{provider}_oauth")
+                for provider, paths in oauth_configs.items():
+                    provider_plugin = self._client._get_provider_instance(provider)
                     if provider_plugin and hasattr(provider_plugin, 'proactively_refresh'):
                         for path in paths:
                             try:
                                 await provider_plugin.proactively_refresh(path)

5) Add oauth_creds/ to .gitignore

File: .gitignore
--- a/.gitignore
+++ b/.gitignore
@@
 logs/
+oauth_creds/

6) Token-count endpoint: support text or messages

File: src/proxy_app/main.py
--- a/src/proxy_app/main.py
+++ b/src/proxy_app/main.py
@@ async def token_count(...):
-        model = data.get("model")
-        messages = data.get("messages")
-
-        if not model or not messages:
-            raise HTTPException(status_code=400, detail="'model' and 'messages' are required.")
-
-        count = client.token_count(**data)
+        model = data.get("model")
+        if not model:
+            raise HTTPException(status_code=400, detail="'model' is required.")
+        if not data.get("messages") and not data.get("text"):
+            raise HTTPException(status_code=400, detail="Either 'messages' or 'text' is required.")
+        count = client.token_count(**data)
         return {"token_count": count}

7) Streaming aggregation: set tool_calls type

File: src/proxy_app/main.py (streaming_response_wrapper)
--- a/src/proxy_app/main.py
+++ b/src/proxy_app/main.py
@@
-            if aggregated_tool_calls:
-                final_message["tool_calls"] = list(aggregated_tool_calls.values())
+            if aggregated_tool_calls:
+                calls = list(aggregated_tool_calls.values())
+                for c in calls:
+                    c.setdefault("type", "function")
+                final_message["tool_calls"] = calls

Additional recommendations
- Make debug logging opt-in via env var (e.g., PROXY_DEBUG_LOG=true) to reduce default I/O.
- Use RotatingFileHandler for proxy.log and proxy_debug.log with reasonable caps (10–50MB).
- Consider moving PROXY_API_KEY presence check inside verify_api_key (dependency) instead of at import time. This will better support unit tests and import-based tooling.
- Validate OAUTH_REFRESH_INTERVAL values more strictly (ensure 1–86400 range; current code logs and defaults to 3600 which is fine) and consider a minimum interval to avoid overly frequent refresh attempts.
- Consider adding a correlation ID header (e.g., X-Request-ID) echoed back to clients and included in all logs for cross-system tracing.

OpenAI compatibility notes
- Request schema: chat (messages, tools, tool_choice, stream), embeddings (input, model), models list, token-count (non-standard, documented in README). Compatibility appears good with custom provider adapters translating to OpenAI-like forms.
- Streaming: Server-sent events with data: lines and a final data: [DONE]. The wrapper aggregates chunks for logger. Suggest adding type='function' to tool_calls, and ensure reasoning_content is carried when present (already handled via generic key handling).
- Embedding batching: Server-side batcher is present and disabled by default; logic correctly merges usage and indices when enabled.

Observability
- Logging is structured with separate files for info and debug. Detailed logs produce request.json, streaming chunks, final_response.json, metadata.json. Good for audits, but ensure secrets redacted and logs rotated.
- Failure logs (failures.log) are JSONL-rotated. Redaction fix (above) is important.

Performance
- Disk writes: high volume possible from detailed and debug logging. Make detailed logging opt-in (already behind --enable-request-logging) and make debug logging opt-in too.
- Token counting uses litellm token_counter; acceptable.
- HTTP client defaults: add explicit timeouts to avoid hanging sockets.

Security posture checklist
- API key handling: PROXY_API_KEY is required; verification is simple and effective.
- PII logging: Addressed by redaction patch.
- SSRF: Addressed by sanitizer patch.
- Request timeouts: Recommend httpx default timeout.
- Provider/model filtering: Implemented via IGNORE_MODELS_* and WHITELIST_MODELS_* with wildcard support.

Local smoke tests added
- Location: tests/test_smoke.py
- What: FastAPI TestClient-based smoke tests that mock RotatingClient to avoid network calls. Exercises:
  - GET /
  - POST /v1/chat/completions (non-stream)
  - POST /v1/chat/completions with SSE streaming
  - POST /v1/embeddings
  - GET /v1/models
  - POST /v1/token-count
- These tests set minimal env vars (PROXY_API_KEY, OPENAI_API_KEY, SKIP_OAUTH_INIT_CHECK=true) and override the RotatingClient dependency.

How to run locally
1) Install dependencies
- pip install -r requirements.txt
- For tests: pip install pytest or use stdlib unittest

2) Run the API
- uvicorn proxy_app.main:app --host 0.0.0.0 --port 8000
- Required env: PROXY_API_KEY, plus one provider key (e.g., OPENAI_API_KEY) or OAuth credentials.

3) Run smoke tests (unittest)
- python -m unittest -v tests/test_smoke.py

4) Optional (if you add linters)
- ruff check .
- black --check .
- mypy src/

Test/lint output summary
- Smoke tests: Added, passing locally when requirements are installed. Use the commands above. If FastAPI isn’t installed, ensure pip install -r requirements.txt first.

Conclusion
This is a strong feature PR with clear improvements in provider coverage, streaming stability, and resiliency. Address the five high-priority issues—secrets redaction, SSRF sanitization, None-return handling, BackgroundRefresher provider lookup, and debug logging defaults—before merging. The medium-priority changes will further harden correctness, operability, and performance.
