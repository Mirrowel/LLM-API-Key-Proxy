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
        anthropic_response["id"] = request_id
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

    async def count_tokens(self, request: Dict[str, Any]) -> dict[str, int]:
        """Count an Anthropic request locally using its canonical Chat projection."""

        payload = dict(request)
        model = str(payload.get("model") or "")
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
        total = self._client.token_count(
            model=model,
            messages=chat_request.get("messages") or [],
        )
        if chat_request.get("tools"):
            tools_text = json.dumps(chat_request["tools"])
            total += self._client.token_count(model=model, text=tools_text)
        return {"input_tokens": total}
