# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Small HTTP transport wrapper for native provider calls."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator


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
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
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
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                if hasattr(response, "aiter_lines"):
                    async for line in response.aiter_lines():
                        parsed = _parse_stream_line(line)
                        if parsed is not None:
                            yield parsed
                    return
                if hasattr(response, "aiter_bytes"):
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                        buffer += text
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            parsed = _parse_stream_line(line)
                            if parsed is not None:
                                yield parsed
                    parsed = _parse_stream_line(buffer)
                    if parsed is not None:
                        yield parsed
                    return
        raise NotImplementedError("Injected native HTTP client does not expose streaming support")


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
