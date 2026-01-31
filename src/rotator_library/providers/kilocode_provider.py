# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class KilocodeProvider(ProviderInterface):
    """
    Provider implementation for the Kilocode API.
    
    Kilocode routes requests to various providers through model prefixes:
    - minimax/minimax-m2.1:free
    - moonshotai/kimi-k2.5:free
    - z-ai/glm-4.7:free
    - And other provider/model combinations
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Kilocode API.
        """
        try:
            response = await client.get(
                "https://kilocode.ai/api/openrouter/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"kilocode/{model['id']}" for model in response.json().get("data", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Kilocode models: {e}")
            return []
