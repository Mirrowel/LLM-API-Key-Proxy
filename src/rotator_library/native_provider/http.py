# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Small HTTP transport wrapper for native provider calls."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..core.errors import StructuredAPIResponseError, structured_api_response_error


class NativeHTTPTransport:
    """Execute provider-native JSON HTTP requests through an injected client."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def post_json(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        """POST JSON and return a decoded response body.

        The wrapper keeps HTTP behavior easy to mock. It does not own retries or
        credential rotation; those remain in the existing executor/usage layer.
        """

        response = await self.client.post(endpoint, headers=headers, json=payload)
        await _raise_for_http_error(response)
        if hasattr(response, "json"):
            return response.json()
        return response

    async def stream_json_lines(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any]) -> AsyncIterator[Any]:
        """Yield provider stream chunks from an injected streaming-capable client.

        Provider-specific test clients can still expose `stream_json_lines()`.
        When a normal `httpx.AsyncClient`-style object is injected, this method
        now uses `client.stream()` directly so native streaming has a real HTTP
        seam without enabling any provider that has not opted in safely.
        """

        if hasattr(self.client, "stream_json_lines"):
            async for chunk in self.client.stream_json_lines(endpoint, headers=headers, json=payload):
                yield chunk
            return
        if hasattr(self.client, "stream"):
            async with self.client.stream("POST", endpoint, headers=headers, json=payload) as response:
                await _raise_for_http_error(response, read_stream=True)
                if hasattr(response, "aiter_lines"):
                    decoder = _SSELineDecoder()
                    async for line in response.aiter_lines():
                        for parsed in decoder.feed(line):
                            yield parsed
                    for parsed in decoder.flush():
                        yield parsed
                    return
                if hasattr(response, "aiter_bytes"):
                    buffer = ""
                    decoder = _SSELineDecoder()
                    async for chunk in response.aiter_bytes():
                        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                        buffer += text
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            for parsed in decoder.feed(line):
                                yield parsed
                    if buffer:
                        for parsed in decoder.feed(buffer):
                            yield parsed
                    for parsed in decoder.flush():
                        yield parsed
                    return
        raise NotImplementedError("Injected native HTTP client does not expose streaming support")


async def _raise_for_http_error(response: Any, *, read_stream: bool = False) -> None:
    """Preserve non-2xx provider bodies and status for retry/error formatting."""

    status = getattr(response, "status_code", None)
    if status is None:
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        return
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        status_code = None
    if status_code is None or 200 <= status_code < 300:
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        return
    if read_stream:
        reader = getattr(response, "aread", None)
        if callable(reader):
            await reader()
    payload: Any = None
    if hasattr(response, "json"):
        try:
            payload = response.json()
        except Exception:
            payload = None
    if not isinstance(payload, dict):
        text = getattr(response, "text", None)
        payload = {
            "error": {
                "message": str(text or f"Provider returned HTTP {status_code}"),
                "code": status_code,
            }
        }
    payload.setdefault("status_code", status_code)
    if isinstance(payload.get("error"), dict):
        payload["error"].setdefault("status_code", status_code)
    headers = dict(getattr(response, "headers", {}) or {})
    error = structured_api_response_error(payload, headers=headers)
    if error:
        raise error
    raise StructuredAPIResponseError(
        f"Provider returned HTTP {status_code}",
        error_type="server_error" if status_code >= 500 else "invalid_request",
        status_code=status_code,
        response=payload,
        headers=headers,
    )


def _parse_stream_line(line: Any) -> Any:
    """Parse one HTTP streaming line while preserving provider sentinels."""

    if line is None:
        return None
    text = line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
    text = text.strip()
    if not text:
        return None
    if text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if text == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _SSELineDecoder:
    """Assemble SSE fields until the blank-line event delimiter."""

    def __init__(self) -> None:
        self.event_name: str | None = None
        self.data_lines: list[str] = []

    def feed(self, line: Any) -> list[Any]:
        text = line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
        text = text.rstrip("\r")
        if not text:
            return self.flush()
        if text.startswith(":"):
            return []
        if text.startswith("event:"):
            self.event_name = text[len("event:") :].strip()
            return []
        if text.startswith("data:"):
            self.data_lines.append(text[len("data:") :].lstrip())
            return []
        output = self.flush()
        parsed = _parse_stream_line(text)
        if parsed is not None:
            output.append(parsed)
        return output

    def flush(self) -> list[Any]:
        if not self.data_lines:
            self.event_name = None
            return []
        text = "\n".join(self.data_lines).strip()
        event_name = self.event_name
        self.event_name = None
        self.data_lines = []
        if text == "[DONE]":
            return ["[DONE]"]
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        if isinstance(parsed, dict) and event_name and not (parsed.get("type") or parsed.get("event")):
            parsed["type"] = event_name
        return [parsed]
