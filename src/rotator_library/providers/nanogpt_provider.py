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


class NanoGPTProvider(ProviderInterface):
    """NanoGPT — OpenAI-compatible aggregator with a subscription face.

    Pay-as-you-go is the default wire (``nano-gpt.com/api/v1``); the
    ``subscription`` profile swaps to the subscription-included pool
    (``/api/subscription/v1``) and a ``responses`` profile exposes their
    Responses surface. Deposit exhaustion is a 402 (rotate, never retry);
    subscription exhaustion is a 429 until reset. The ``nanogpt`` wire
    adapter owns the length-parameter mapping and the reasoning-field
    spelling.
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://nano-gpt.com/api/v1"
    default_profile = "chat"
    adapter_names = ("nanogpt",)
    transport_profiles = {
        "chat": {"protocol": "openai_chat"},
        "responses": {"protocol": "responses"},
        "subscription": {
            "protocol": "openai_chat",
            "endpoint_paths": {
                "chat": "/chat/completions",
                "models": "/models",
            },
            "base_url": "https://nano-gpt.com/api/subscription/v1",
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
                f"nanogpt/{model['id']}"
                for model in response.json().get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch NanoGPT models: {e}")
            return []
