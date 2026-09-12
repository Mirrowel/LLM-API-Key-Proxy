# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Small HTTP transport wrapper for native provider calls.

G4 conditional re-serialization: :meth:`stream_raw_frames` yields one
:class:`RawStreamFrame` per provider event carrying BOTH the original wire
text (relay-able, byte-identical) and the parsed payload (observation). The
relay decision belongs to the executor, not the transport: same protocol
alone never implies pass-through — hooks, overlays, or repair needs
disengage the relay and switch to formatter output (operator ruling:
intercept-and-edit is always possible, streams and non-streams alike).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..core.errors import StructuredAPIResponseError, structured_api_response_error


@dataclass(slots=True)
class RawStreamFrame:
    """One provider stream event with its original wire representation.

    ``raw`` is the exact SSE event text (``data:`` payload plus ``event:``
    name where present, comments included verbatim); ``parsed`` is the
    decoded payload, ``"[DONE]"`` for the chat sentinel, or the raw text
    when undecodable. ``raw is None`` means the injected client never
    exposed bytes (custom test transports) — relay is impossible and the
    executor must fall back to formatter output.
    """

    raw: str | None
    parsed: Any
    event_name: str | None = None
    is_comment: bool = False


class NativeHTTPTransport:
    """Execute provider-native JSON HTTP requests through an injected client."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def post_json(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float | None = None) -> Any:
        """POST JSON and return a decoded response body.

        The wrapper keeps HTTP behavior easy to mock. It does not own retries or
        credential rotation; those remain in the existing executor/usage layer.
        ``timeout_seconds`` carries hook-mandated per-request timeouts
        (G2 transport slot) through to capable clients.
        """

        request_kwargs: dict[str, Any] = {}
        if timeout_seconds is not None:
            if hasattr(self.client, "post"):
                try:
                    import httpx

                    request_kwargs["timeout"] = httpx.Timeout(timeout_seconds)
                except ImportError:
                    request_kwargs["timeout"] = timeout_seconds
        response = await self.client.post(endpoint, headers=headers, json=payload, **request_kwargs)
        await _raise_for_http_error(response)
        if hasattr(response, "json"):
            return response.json()
        return response

    async def stream_json_lines(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float | None = None) -> AsyncIterator[Any]:
        """Yield provider stream chunks from an injected streaming-capable client.

        Provider-specific test clients can still expose `stream_json_lines()`.
        When a normal `httpx.AsyncClient`-style object is injected, this method
        now uses `client.stream()` directly so native streaming has a real HTTP
        seam without enabling any provider that has not opted in safely.
        """

        async for frame in self.stream_raw_frames(endpoint, headers=headers, payload=payload, timeout_seconds=timeout_seconds):
            if frame.is_comment:
                # Parsed-dict consumers skip comment frames (pre-G4
                # behavior); only stream_raw_frames exposes them.
                continue
            yield frame.parsed

    async def stream_raw_frames(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float | None = None) -> AsyncIterator[RawStreamFrame]:
        """Yield raw provider stream events (wire text + parsed payload).

        The G4 relay seam: the executor can relay ``frame.raw`` bytes to the
        client untouched while feeding ``frame.parsed`` to the observing
        pipeline (usage/anchors/metrics/repair). Callers that cannot provide
        bytes yield frames with ``raw=None``.
        """

        if hasattr(self.client, "stream_raw_frames"):
            async for frame in self.client.stream_raw_frames(endpoint, headers=headers, json=payload):
                yield frame
            return
        if hasattr(self.client, "stream_json_lines"):
            async for chunk in self.client.stream_json_lines(endpoint, headers=headers, json=payload):
                yield RawStreamFrame(raw=None, parsed=chunk)
            return
        if hasattr(self.client, "stream"):
            async with self.client.stream("POST", endpoint, headers=headers, json=payload) as response:
                await _raise_for_http_error(response, read_stream=True)
                if hasattr(response, "aiter_lines"):
                    decoder = _SSEFrameDecoder()
                    async for line in response.aiter_lines():
                        for frame in decoder.feed(line):
                            yield frame
                    for frame in decoder.flush():
                        yield frame
                    return
                if hasattr(response, "aiter_bytes"):
                    buffer = ""
                    decoder = _SSEFrameDecoder()
                    async for chunk in response.aiter_bytes():
                        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                        buffer += text
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            for frame in decoder.feed(line):
                                yield frame
                    if buffer:
                        for frame in decoder.feed(buffer):
                            yield frame
                    for frame in decoder.flush():
                        yield frame
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


class _SSEFrameDecoder:
    """Assemble SSE fields until the blank-line event delimiter.

    Produces :class:`RawStreamFrame` objects preserving the original wire
    text so the G4 relay can forward provider bytes untouched. Comment
    frames survive as observable heartbeat evidence instead of being
    dropped silently.
    """

    def __init__(self) -> None:
        self.event_name: str | None = None
        self.data_lines: list[str] = []
        self.pending_comment: str | None = None

    def feed(self, line: Any) -> list[RawStreamFrame]:
        text = line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
        text = text.rstrip("\r")
        if not text:
            return self.flush()
        if text.startswith(":"):
            # Comments relay verbatim and double as provider heartbeat
            # evidence for the stall detector.
            self.pending_comment = text
            return []
        if text.startswith("event:"):
            self.event_name = text[len("event:") :].strip()
            return []
        if text.startswith("data:"):
            # Strip exactly ONE leading space per the SSE spec — never
            # lstrip(), which would corrupt payloads that intentionally
            # begin with whitespace.
            value = text[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            self.data_lines.append(value)
            return []
        output = self.flush()
        parsed = _parse_stream_line(text)
        if parsed is not None:
            output.append(RawStreamFrame(raw=text, parsed=parsed))
        return output

    def flush(self) -> list[RawStreamFrame]:
        frames: list[RawStreamFrame] = []
        if self.pending_comment is not None:
            frames.append(RawStreamFrame(raw=self.pending_comment, parsed=None, is_comment=True))
            self.pending_comment = None
        if not self.data_lines:
            self.event_name = None
            return frames
        text = "\n".join(self.data_lines)
        event_name = self.event_name
        raw = "data: " + text if text != "[DONE]" else "data: [DONE]"
        if event_name:
            raw = f"event: {event_name}\n{raw}"
        self.event_name = None
        self.data_lines = []
        if text.strip() == "[DONE]":
            frames.append(RawStreamFrame(raw=raw, parsed="[DONE]", event_name=event_name))
            return frames
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        if isinstance(parsed, dict) and event_name and not (parsed.get("type") or parsed.get("event")):
            parsed["type"] = event_name
        frames.append(RawStreamFrame(raw=raw, parsed=parsed, event_name=event_name))
        return frames
