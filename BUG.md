# BUG Report: Anthropic Compatibility & Claude Code Integration

## Executive Summary
This document tracks critical bugs identified and resolved during the testing of PR #47 (Anthropic Compatibility) when used with the **Claude Code v2.1.2** CLI client.

### Resolved Issues:
*   **Antigravity 400 Error (Tool Schema)**: Prevented crashes during tool-use by stripping unsupported `enumDescriptions` from JSON schemas.
*   **Streaming Wrapper TypeError**: Fixed a 500 Internal Server Error caused by a parameter mismatch in the async streaming generator.
*   **Redundant Translation Logic**: Refactored `main.py` to use core library methods, ensuring consistent behavior and easier maintenance.
*   **Pydantic Validation Failures**: Relaxed validation to allow "extra" fields, supporting Claude Code's newer beta features like `interleaved-thinking`.

---

## 1. Antigravity API 400 Error (Tool Schemas)

### Issue Description
Requests failed with a `400 Bad Request` error from the Antigravity backend whenever Claude Code attempted to use tools (indexing, bash commands, etc.).

### Root Cause
The Antigravity API is a strict Proto-based interface. Claude Code includes an `enumDescriptions` field in its tool parameter schemas which is not part of the standard Google Gemini tool specification.

### Error Message
```json
"message": "Invalid JSON payload received. Unknown name \"enumDescriptions\" at 'request.tools[0]... Cannot find field."
```

### Resolution
Modified `antigravity_provider.py` to recursively strip `enumDescriptions` from all tool definitions before sending the request to the Google backend.

---

## 2. TypeError in Anthropic Streaming Wrapper

### Issue Description
Claude Code sessions triggered a server-side 500 error immediately upon starting a stream.

### Root Cause
A parameter mismatch in the `/v1/messages` endpoint within `main.py`. The code was incorrectly passing the FastAPI `Request` object into a generator that expected an `openai_stream` object, causing an iteration failure.

### Error Message
```text
Error in Anthropic streaming wrapper: 'async for' requires an object with __aiter__ method, got Request
TypeError: Object of type async_generator is not JSON serializable
```

### Resolution
Refactored the endpoints in `main.py` to delegate request handling to the `RotatingClient.anthropic_messages()` method, which handles the parameter passing and stream wrapping correctly.

---

## 3. Pydantic 422 "Unprocessable Content"

### Issue Description
Requests containing newer Anthropic beta headers (like `interleaved-thinking-2025-05-14`) were rejected with a 422 error before reaching the provider logic.

### Root Cause
The Pydantic models for `AnthropicMessagesRequest` were too strict and did not allow extra fields. Claude Code frequently adds experimental fields to its JSON payload.

### Resolution
Updated `models.py` to set `model_config = ConfigDict(extra="allow")` for all Anthropic request objects, ensuring future-proof compatibility with evolving Anthropic SDKs.

---

**Last Updated**: 2026-01-09
**Status**: Critical Bugs Resolved (See `STATS-MODEL-ISSUE.md` for remaining client-side behavior)