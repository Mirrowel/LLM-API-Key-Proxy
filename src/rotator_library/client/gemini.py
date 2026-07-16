"""Gemini client-surface handler backed by the shared protocol runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from ..protocols import OPERATION_COUNT_TOKENS, ProtocolContext, get_protocol
from ..routing import load_routing_config_from_env

if TYPE_CHECKING:
    from .rotating_client import RotatingClient


class GeminiHandler:
    """Expose Gemini wire requests without leaking that format into providers."""

    def __init__(self, client: "RotatingClient") -> None:
        self._client = client

    async def generate(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        raw_request: Optional[Any] = None,
    ) -> Any:
        """Execute one Gemini generateContent request through canonical routing."""

        request_payload = dict(payload)
        if request_payload.get("stream"):
            raise ValueError(
                "Gemini generateContent does not accept stream=true; use streamGenerateContent"
            )
        request_payload["model"] = self._routable_model(model)
        return await self._client.agenerate(
            request_payload,
            input_protocol="gemini",
            request=raw_request,
        )

    async def stream_generate(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        raw_request: Optional[Any] = None,
    ) -> Any:
        """Execute one Gemini streamGenerateContent request canonically."""

        request_payload = dict(payload)
        request_payload["model"] = self._routable_model(model)
        request_payload["stream"] = True
        return await self._client.agenerate(
            request_payload,
            input_protocol="gemini",
            request=raw_request,
        )

    def count_tokens(self, payload: dict[str, Any], *, model: str) -> dict[str, int]:
        """Count a Gemini request locally using its canonical Chat projection."""

        request_payload = dict(payload)
        request_payload["model"] = self._routable_model(model)
        gemini = get_protocol("gemini")
        unified = gemini.parse_request(
            request_payload,
            ProtocolContext(
                source_protocol="gemini",
                target_protocol="gemini",
                model=request_payload["model"],
                metadata={"operation": OPERATION_COUNT_TOKENS},
            ),
        )
        chat_request = get_protocol("openai_chat").build_request(
            unified,
            ProtocolContext(
                source_protocol="gemini",
                target_protocol="openai_chat",
                input_protocol="gemini",
                output_protocol="gemini",
                model=request_payload["model"],
            ),
        )
        total = self._client.token_count(
            model=request_payload["model"],
            messages=chat_request.get("messages") or [],
        )
        tools = chat_request.get("tools") or []
        if tools:
            total += self._client.token_count(
                model=request_payload["model"],
                text=json.dumps(tools, separators=(",", ":")),
            )
        return {"totalTokens": total}

    @staticmethod
    def _routable_model(model: str) -> str:
        """Keep configured aliases, otherwise default Gemini-style IDs to Gemini."""

        normalized = str(model or "").removeprefix("models/")
        if "/" in normalized:
            return normalized
        routes = load_routing_config_from_env().model_routes
        return normalized if normalized.lower() in routes else f"gemini/{normalized}"
