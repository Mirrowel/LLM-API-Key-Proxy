# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpenAIProvider(ProviderInterface):
    """OpenAI — the reference provider, Responses-first.

    Responses is the primary surface (reasoning, encrypted-content
    replay, native token counting via ``/responses/input_tokens``); the
    chat face stays first-class for multi-candidate traffic and clients
    that speak chat. Bare ``openai/model`` keeps matching the client's
    own protocol through profile resolution; the declared default steers
    conversion cases to Responses, matching OpenAI's own primary API.
    """

    protocol_name = "responses"
    native_streaming_supported = True
    default_api_base = "https://api.openai.com/v1"
    default_profile = "responses"
    transport_profiles = {
        "responses": {
            "protocol": "responses",
            "endpoint_paths": {
                "responses": "/responses",
                "count_tokens": "/responses/input_tokens",
                "models": "/models",
            },
        },
        "chat": {
            "protocol": "openai_chat",
            "endpoint_paths": {
                "chat": "/chat/completions",
                "models": "/models",
            },
        },
    }

    def _models_url(self) -> str:
        return f"{self.get_provider_api_base()}/models"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(
                self._models_url(),
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"openai/{model['id']}"
                for model in response.json().get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch OpenAI models: {e}")
            return []
