# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Data types for the Responses API compatibility layer."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..protocols import serialize_value


def generate_response_id() -> str:
    """Return a local Responses-compatible identifier.

    Upstream IDs are preserved when providers return them. This helper is only
    used by the bridge path when the current chat-completions backend has no
    native Responses ID to expose.
    """

    return f"resp_{secrets.token_urlsafe(18).replace('-', '').replace('_', '')[:24]}"


@dataclass
class StoredResponse:
    """Persisted response object used for retrieval and continuation.

    The shape stores both client-facing response data and enough request/session
    metadata for `previous_response_id` debugging. It deliberately avoids storing
    credential secrets; callers should pass only stable identifiers if they need
    credential correlation later.
    """

    id: str
    model: str
    status: str
    response: dict[str, Any]
    request: dict[str, Any] = field(default_factory=dict)
    input_items: list[Any] = field(default_factory=list)
    output_items: list[Any] = field(default_factory=list)
    usage: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    scope_key: Optional[str] = None
    classifier: Optional[str] = None
    expires_at: Optional[float] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return whether the response should be treated as unavailable."""

        return self.expires_at is not None and (now if now is not None else time.time()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize with JSON-safe values for disk/cache persistence."""

        return serialize_value(
            {
                "id": self.id,
                "created_at": self.created_at,
                "model": self.model,
                "status": self.status,
                "request": self.request,
                "response": self.response,
                "input_items": self.input_items,
                "output_items": self.output_items,
                "usage": self.usage,
                "metadata": self.metadata,
                "session_id": self.session_id,
                "scope_key": self.scope_key,
                "classifier": self.classifier,
                "expires_at": self.expires_at,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredResponse":
        """Rehydrate a stored response from a JSON-compatible dict."""

        return cls(
            id=str(data["id"]),
            created_at=float(data.get("created_at") or time.time()),
            model=str(data.get("model") or ""),
            status=str(data.get("status") or "completed"),
            request=dict(data.get("request") or {}),
            response=dict(data.get("response") or {}),
            input_items=list(data.get("input_items") or []),
            output_items=list(data.get("output_items") or []),
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
            metadata=dict(data.get("metadata") or {}),
            session_id=data.get("session_id"),
            scope_key=data.get("scope_key"),
            classifier=data.get("classifier"),
            expires_at=data.get("expires_at"),
        )
