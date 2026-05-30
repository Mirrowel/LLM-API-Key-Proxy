# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Small HTTP transport wrapper for native provider calls."""

from __future__ import annotations

from typing import Any


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

    async def stream_json_lines(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any]):
        """Yield provider stream chunks from an injected streaming-capable client.

        Tests and provider-specific clients can expose `stream_json_lines()` to
        avoid binding this foundation to one HTTP client's streaming API. A later
        streaming phase can add richer `httpx.stream()` support without changing
        native provider executor semantics.
        """

        if hasattr(self.client, "stream_json_lines"):
            async for chunk in self.client.stream_json_lines(endpoint, headers=headers, json=payload):
                yield chunk
            return
        raise NotImplementedError("Injected native HTTP client does not expose stream_json_lines")
