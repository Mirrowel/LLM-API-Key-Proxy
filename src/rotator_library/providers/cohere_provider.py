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


class CohereProvider(ProviderInterface):
    """Cohere — the OpenAI-compatible face is the declared wire.

    Cohere's native v2 API is not one of the proxy's protocols; the
    compatibility surface (``api.cohere.ai/compatibility/v1``) carries
    tools, strict json_schema, and ``reasoning_effort`` (none|high), which
    the canonical effort vocabulary narrows with a recorded warning. Model
    listing on this face is OpenAI-shaped (``data[].id``), unlike the
    native ``/v1/models`` (``models[].name``).
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://api.cohere.ai/compatibility/v1"

    def _models_url(self) -> str:
        return f"{self.get_provider_api_base()}/models"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(
                self._models_url(),
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("data") if isinstance(payload, dict) else None
            if entries is None and isinstance(payload, dict):
                entries = [
                    {"id": model.get("name")}
                    for model in payload.get("models", [])
                    if isinstance(model, dict)
                    if "chat" in (model.get("endpoints") or [])
                ]
            return [
                f"cohere/{model['id']}"
                for model in entries or []
                if isinstance(model, dict) and model.get("id")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Cohere models: {e}")
            return []
