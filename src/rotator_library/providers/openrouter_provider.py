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


class OpenRouterProvider(ProviderInterface):
    """OpenRouter — three native faces over one credential pool.

    Chat is the default wire; the Responses and Anthropic-compat faces are
    addressed as ``openrouter:responses/<model>`` and
    ``openrouter:anthropic/<vendor/model>``. Model ids keep their colons
    (``:free``/``:nitro`` variants route through the grammar untouched) and
    the unified ``reasoning`` parameter rides same-protocol passthrough.
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://openrouter.ai/api/v1"
    default_profile = "chat"
    transport_profiles = {
        "chat": {"protocol": "openai_chat"},
        "responses": {"protocol": "responses"},
        "anthropic": {"protocol": "anthropic_messages"},
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
                f"openrouter/{model['id']}"
                for model in response.json().get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch OpenRouter models: {e}")
            return []
