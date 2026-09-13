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


class GroqProvider(ProviderInterface):
    """Groq — OpenAI-compatible chat on GroqLPUs.

    Reasoning arrives as ``reasoning`` (not ``reasoning_content``) and the
    ``groq`` wire adapters own the rename plus parameter hygiene
    (temperature clamping, unsupported-knob stripping, ``reasoning_format``
    selection when tools or JSON force parsed output).
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://api.groq.com/openai/v1"
    adapter_names = ("groq",)

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
                f"groq/{model['id']}"
                for model in response.json().get("data", [])
                if isinstance(model, dict) and model.get("id") and model.get("active", True)
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Groq models: {e}")
            return []
