# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
First-party provider for Player2 (https://player2.game).

Player2 exposes an OpenAI-compatible `/chat/completions` endpoint intended
for AI-driven game NPCs. Authentication uses a single bearer token ("p2Key")
obtained either from the Player2 desktop app (if running locally) or via the
OAuth 2.0 Device Authorization Flow - see `utilities/player2_auth.py` and
`credential_tool.py`'s Player2 login wizard for how that key is acquired.

Notable differences from a typical OpenAI-compatible provider:

- The `/chat/completions` request schema does NOT include a `model` field
  (confirmed against https://player2.game/api/api.yaml). The model actually
  used is controlled by the user's Player2 account/app configuration, not by
  the caller. Because of this, `model` is deliberately stripped from the
  outgoing payload rather than passed through - this proxy still requires a
  `player2/<anything>` model string for routing purposes, but the suffix is
  cosmetic.
- There is no `/models` discovery endpoint, so `get_models()` returns a
  single static placeholder model name.
- Error responses include a `402 Insufficient credits` case in addition to
  the usual `401`/`429`, which this provider surfaces as a rate-limit style
  error so it's retried/rotated the same way quota exhaustion is elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.exceptions import RateLimitError

from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


DEFAULT_API_BASE = "https://api.player2.game/v1"

# Player2 doesn't expose model selection or discovery via the API - the
# model is whatever the user's Player2 account/app is configured to use.
PLACEHOLDER_MODEL = "default"

# Fields Player2's /chat/completions schema actually documents.
# Deliberately excludes "model" - see module docstring.
SUPPORTED_PARAMS = {
    "messages",
    "max_tokens",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "response_format",
}

# Player2 doesn't document a reset timestamp for "insufficient credits" (402)
# responses, so a fixed, conservative cooldown is used instead of retrying
# an out-of-credits key like a transient rate limit would be.
INSUFFICIENT_CREDITS_COOLDOWN_SECONDS = 86400  # 24 hours

class Player2Provider(ProviderInterface):
    """First-party Player2 provider using its native OpenAI-compatible API."""

    skip_cost_calculation = True
    default_rotation_mode: str = "sequential"

    def __init__(self):
        pass

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Player2 has no model discovery endpoint and doesn't accept a `model`
        field in completion requests - the model is decided by the user's
        Player2 account/app, not by the caller. A single placeholder is
        returned so the proxy has something to route on.
        """
        return [f"player2/{PLACEHOLDER_MODEL}"]

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Player2 keys (p2Key) are used as a plain bearer token."""
        return {"Authorization": f"Bearer {credential_identifier}"}

    def _api_base(self, override: Optional[str] = None) -> str:
        return (override or os.getenv("PLAYER2_API_BASE") or DEFAULT_API_BASE).rstrip("/")

    def _chat_url(self, api_base: Optional[str] = None) -> str:
        return f"{self._api_base(api_base)}/chat/completions"

    def _build_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        # Fold max_completion_tokens into max_tokens if that's what was sent.
        if "max_completion_tokens" in kwargs and "max_tokens" not in payload:
            payload["max_tokens"] = kwargs["max_completion_tokens"]

        # Unlike some sibling providers, extra_body is filtered (not merged
        # wholesale) because Player2's schema is strict about which fields
        # it accepts - passing through an unrecognized field risks a 4xx
        # from Player2 itself. Any new Player2 param needs to be added to
        # SUPPORTED_PARAMS above to be forwarded via extra_body.
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update({k: v for k, v in extra_body.items() if k in SUPPORTED_PARAMS})

        return payload

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[
        litellm.ModelResponse,
        AsyncGenerator[litellm.ModelResponseStream, None],
    ]:
        api_key = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        file_logger = ProviderLogger(transaction_context)

        api_base = kwargs.pop("api_base", None)
        model = kwargs.get("model", f"player2/{PLACEHOLDER_MODEL}")

        payload = self._build_payload(kwargs)
        url = self._chat_url(api_base)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }

        file_logger.log_request(payload)

        if payload.get("stream"):
            return self._stream_completion(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                model=model,
                file_logger=file_logger,
            )

        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )
        await self._raise_for_status(response, model)
        response_data = response.json()
        response_data["model"] = model
        file_logger.log_final_response(response_data)
        return litellm.ModelResponse(**response_data)

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: ProviderLogger,
    ) -> AsyncGenerator[litellm.ModelResponseStream, None]:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            await self._raise_for_status(response, model)

            async for line in response.aiter_lines():
                file_logger.log_response_chunk(line)
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    lib_logger.warning(f"Could not decode JSON from Player2: {line}")
                    continue

                chunk["model"] = model
                yield litellm.ModelResponseStream(**chunk)

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Distinguishes a genuine "insufficient credits" (402) condition from
        an ordinary rate limit, so it gets a long cooldown instead of being
        retried again within seconds. See the comment in _raise_for_status
        for why a custom attribute (rather than the exception's status_code)
        is what we check here.

        Player2 doesn't document a reset timestamp for insufficient-credits
        responses, so a conservative fixed 24h cooldown is used instead of
        retrying an out-of-credits key like a transient rate limit.
        """
        if getattr(error, "player2_error_reason", None) != "insufficient_credits":
            return None

        return {
            "retry_after": INSUFFICIENT_CREDITS_COOLDOWN_SECONDS,
            "reason": "INSUFFICIENT_CREDITS",
            "reset_timestamp": None,
            "quota_reset_timestamp": None,
        }

    async def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.status_code < 400:
            return

        content = await response.aread()
        error_text = content.decode("utf-8", errors="replace") if content else ""

        if response.status_code == 429:
            raise RateLimitError(
                f"Player2 rate limit exceeded: {error_text}",
                llm_provider="player2",
                model=model,
                response=response,
            )

        if response.status_code == 402:
            # Insufficient credits - this is a persistent condition (it won't
            # clear after a short rate-limit-style cooldown, unlike a real
            # 429), so it's surfaced as a RateLimitError too (for credential
            # rotation) but tagged with a custom attribute. litellm's
            # RateLimitError normalizes response.status_code to 429
            # internally, so this tag is the only reliable way for
            # parse_quota_error() above to tell "out of credits" apart from
            # an actual rate limit and apply a much longer cooldown.
            exc = RateLimitError(
                f"Player2 insufficient credits: {error_text}",
                llm_provider="player2",
                model=model,
                response=response,
            )
            exc.player2_error_reason = "insufficient_credits"
            raise exc

        raise httpx.HTTPStatusError(
            f"Player2 HTTP {response.status_code}: {error_text}",
            request=response.request,
            response=response,
        )