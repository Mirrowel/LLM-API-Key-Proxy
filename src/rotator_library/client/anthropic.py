# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Anthropic Messages facade over the canonical protocol runtime.

The handler owns nothing but facade concerns: the original /v1/messages
body is transported verbatim through the runtime (D4 raw fast path keeps
unknown fields, explicit nulls, and Anthropic extensions byte-identical),
the response id is stamped for non-streaming calls, and transaction
logging traces the boundary. All translation lives in the
anthropic_messages protocol adapter.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..protocols import get_protocol
from ..protocols.operation import OPERATION_COUNT_TOKENS
from ..protocols.types import ProtocolContext
from ..transaction_logger import TransactionLogger

if TYPE_CHECKING:
    from .rotating_client import RotatingClient


def _trace_anthropic(
    logger: Optional[TransactionLogger],
    pass_name: str,
    payload: Any,
    *,
    direction: str,
    stage: str,
) -> None:
    """Emit an Anthropic transform trace when logging is enabled."""

    if not logger:
        return
    logger.log_transform_pass(
        pass_name,
        payload,
        direction=direction,
        stage=stage,
        protocol="anthropic_messages",
    )


class AnthropicHandler:
    """Handle Anthropic Messages API traffic through the protocol runtime."""

    def __init__(self, client: "RotatingClient"):
        self._client = client

    async def messages(
        self,
        request: Dict[str, Any],
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[Any] = None,
    ) -> Any:
        """Execute one Anthropic Messages request canonically.

        Accepts the raw /v1/messages payload (dict) — validation and any
        conversion are owned by the anthropic_messages adapter; unknown
        fields and explicit nulls transport verbatim (D4).
        """

        payload = dict(request)
        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        original_model = str(payload.get("model") or "")
        provider = original_model.split("/")[0] if "/" in original_model else "unknown"

        anthropic_logger = None
        if self._client.enable_request_logging:
            anthropic_logger = TransactionLogger(
                provider,
                original_model,
                enabled=True,
                api_format="ant",
            )
            anthropic_logger.log_request(payload, filename="anthropic_request.json")
            _trace_anthropic(
                anthropic_logger,
                "anthropic_raw_request",
                payload,
                direction="request",
                stage="client",
            )

        response = await self._client.agenerate(
            payload,
            input_protocol="anthropic_messages",
            request=raw_request,
            pre_request_callback=pre_request_callback,
            _parent_log_dir=anthropic_logger.log_dir if anthropic_logger and anthropic_logger.log_dir else None,
        )
        if payload.get("stream"):
            return response
        anthropic_response = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        # Same-protocol fidelity: the upstream msg_* id survives (thinking
        # signature binding and client idempotency depend on it); the local
        # request_id only fills a missing one.
        anthropic_response.setdefault("id", request_id)
        _trace_anthropic(
            anthropic_logger,
            "anthropic_native_protocol_response",
            anthropic_response,
            direction="response",
            stage="final",
        )
        if anthropic_logger:
            anthropic_logger.log_response(anthropic_response, filename="anthropic_response.json")
        return anthropic_response

    async def count_tokens(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Count tokens natively (upstream count_tokens) with an opt-in local estimate.

        G14: the official operation passes through to the routed provider;
        the local projection estimate only serves when the operator opted in
        via COUNT_TOKENS_LOCAL_ESTIMATE=1.
        """

        payload = dict(request)
        payload.setdefault("model", "")
        model = str(payload.get("model") or "")
        captured: Dict[str, Any] = {}

        def _capture_context(ctx: Any) -> None:
            captured["logger"] = getattr(ctx, "transaction_logger", None)

        try:
            result = await self._client.agenerate(
                payload,
                input_protocol="anthropic_messages",
                _requested_operation=OPERATION_COUNT_TOKENS,
                _request_context_callback=_capture_context,
            )
        except Exception as error:
            from ..client.executor import RoutingExecutionError

            if not (
                isinstance(error, RoutingExecutionError)
                and getattr(error, "error_type", "") == "operation_unsupported"
                and _local_estimate_enabled()
            ):
                raise
            return self._local_count_estimate(payload, model)
        _drain_proxy_warnings(result, captured.get("logger"))
        return result

    def _local_count_estimate(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Legacy local estimate via the canonical Chat projection."""

        anthropic = get_protocol("anthropic_messages")
        unified = anthropic.parse_request(
            payload,
            ProtocolContext(
                source_protocol="anthropic_messages",
                target_protocol="anthropic_messages",
                model=model,
                metadata={"operation": OPERATION_COUNT_TOKENS},
            ),
        )
        chat_request = get_protocol("openai_chat").build_request(
            unified,
            ProtocolContext(
                source_protocol="anthropic_messages",
                target_protocol="openai_chat",
                input_protocol="anthropic_messages",
                client_protocol="anthropic_messages",
                model=model,
            ),
        )
        projected_messages = chat_request.get("messages") or []
        # Prior-turn thinking is ignored by the upstream counter (only the
        # current turn's thinking bills as input): strip reasoning content
        # from history turns so the estimate does not overcount.
        for projected in projected_messages[:-1]:
            if isinstance(projected.get("content"), str):
                continue
            blocks = projected.get("content")
            if isinstance(blocks, list):
                filtered = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "reasoning")]
                if len(filtered) != len(blocks):
                    projected["content"] = filtered or None
        total = self._client.token_count(
            model=model,
            messages=projected_messages,
        )
        if chat_request.get("tools"):
            tools_text = json.dumps(chat_request["tools"])
            total += self._client.token_count(model=model, text=tools_text)
        # Local estimate disclosure (spec: exact counts come from the
        # upstream /v1/messages/count_tokens endpoint; the projection here
        # approximates images/PDFs and may include prior-turn thinking the
        # upstream counter ignores). Anthropic clients ignore unknown keys.
        return {"input_tokens": total, "x-proxy-estimate": "local-projection"}


def _local_estimate_enabled() -> bool:
    """COUNT_TOKENS_LOCAL_ESTIMATE=1 opts into the local projection fallback."""

    import os

    return str(os.environ.get("COUNT_TOKENS_LOCAL_ESTIMATE", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _drain_proxy_warnings(payload: Any, logger: Optional[TransactionLogger]) -> None:
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
