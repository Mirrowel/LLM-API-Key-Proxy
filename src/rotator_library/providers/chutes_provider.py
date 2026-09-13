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


class ChutesProvider(ProviderInterface):
    """Chutes — OpenAI-compatible gateway over TEE-hosted open models.

    Quota is dollar-based at the credential level: a 402 means the account
    balance or plan allowance is gone (never retried), 429 is throttling.
    Model ids keep their vendor prefix and -TEE suffix; routing syntax
    (``default``, comma lists, ``:latency``/``:throughput``) never appears
    in listings. The ``chutes`` wire adapter owns parameter hygiene and
    the dual reasoning-field spellings (vLLM ``reasoning`` vs SGLang
    ``reasoning_content``).
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://llm.chutes.ai/v1"
    adapter_names = ("chutes",)

    def _models_url(self) -> str:
        return f"{self.get_provider_api_base()}/models"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(self._models_url())
            response.raise_for_status()
            models = []
            for model in response.json().get("data", []):
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("id") or "")
                if not model_id or model_id.startswith("default") or "," in model_id:
                    continue
                models.append(f"chutes/{model_id}")
            return models
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Chutes models: {e}")
            return []
