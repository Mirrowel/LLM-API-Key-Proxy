"""Gemini client-surface handler backed by the shared protocol runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from ..protocols import OPERATION_COUNT_TOKENS, ProtocolContext, get_protocol
from ..routing import load_routing_config_from_env  # noqa: F401  (kept for test import compatibility)

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

    async def embeddings(self, payload: dict[str, Any], *, model: str, operation: str = "embeddings") -> dict[str, Any]:
        """Embed natively (provider embedContent / batchEmbedContents).

        G9 ingress parity: the client's chosen endpoint IS the batching
        choice — the single route stamps operation ``embeddings``, the
        batch route ``embeddings_batch``, and the runtime honors both
        through the same protocol machinery cross-protocol traffic uses.
        """

        request_payload = dict(payload)
        request_payload["model"] = self._routable_model(model)
        captured: dict[str, Any] = {}

        def _capture_context(ctx: Any) -> None:
            captured["logger"] = getattr(ctx, "transaction_logger", None)

        result = await self._client.agenerate(
            request_payload,
            input_protocol="gemini",
            _requested_operation=operation,
            _request_context_callback=_capture_context,
        )
        _drain_proxy_warnings(result, captured.get("logger"))
        return result

    async def count_tokens(self, payload: dict[str, Any], *, model: str) -> dict[str, Any]:
        """Count tokens natively (provider countTokens) with an opt-in local estimate.

        G14: the official operation passes through to whichever provider the
        request routes to. When that provider cannot serve it, the honest
        error surfaces unless the operator opted into the local projection
        estimate via COUNT_TOKENS_LOCAL_ESTIMATE=1.
        """

        from ..client.executor import RoutingExecutionError

        request_payload = dict(payload)
        request_payload["model"] = self._routable_model(model)
        captured: dict[str, Any] = {}

        def _capture_context(ctx: Any) -> None:
            captured["logger"] = getattr(ctx, "transaction_logger", None)

        try:
            result = await self._client.agenerate(
                request_payload,
                input_protocol="gemini",
                _requested_operation=OPERATION_COUNT_TOKENS,
                _request_context_callback=_capture_context,
            )
        except RoutingExecutionError as error:
            if getattr(error, "error_type", "") != "operation_unsupported" or not _local_estimate_enabled():
                raise
            return self._local_count_estimate(request_payload)
        _drain_proxy_warnings(result, captured.get("logger"))
        return result

    def _local_count_estimate(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy local estimate: canonical Chat projection token counting."""

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
                client_protocol="gemini",
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
        return {"totalTokens": total, "x-proxy-estimate": "local-projection"}

    @staticmethod
    def _routable_model(model: str) -> str:
        """Keep configured aliases, otherwise default Gemini-style IDs to Gemini."""

        normalized = str(model or "").removeprefix("models/")
        if "/" in normalized:
            return normalized
        from ..routing.config import load_routing_config_from_env

        routes = load_routing_config_from_env().model_routes
        return normalized if normalized.lower() in routes else f"gemini/{normalized}"


def _local_estimate_enabled() -> bool:
    """COUNT_TOKENS_LOCAL_ESTIMATE=1 opts into the local projection fallback."""

    import os

    return str(os.environ.get("COUNT_TOKENS_LOCAL_ESTIMATE", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _drain_proxy_warnings(payload: Any, logger: Any) -> None:
    """Pop the private count-token warning channel and sink it to the record.

    G10 Phase B: the count-tokens adapters return ``_proxy_warnings`` because
    they have no logger; the facade owns the logger handle, pops the key so
    it never reaches the client, and records the warnings in the change log.
    """

    if not isinstance(payload, dict):
        return
    warnings = payload.pop("_proxy_warnings", None)
    if warnings and logger is not None:
        logger.log_conversion_warnings(warnings, stage="count_tokens")
