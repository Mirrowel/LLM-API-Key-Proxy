# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any
import litellm
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class NvidiaProvider(ProviderInterface):
    skip_cost_calculation = True
    """
    Provider implementation for the NVIDIA API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the NVIDIA API.
        """
        try:
            response = await client.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            models = [f"nvidia_nim/{model['id']}" for model in response.json().get("data", [])]
            return models
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch NVIDIA models: {e}")
            return []

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Adds the 'thinking' parameter for specific DeepSeek models on the NVIDIA provider,
        only if reasoning_effort is set to low, medium, or high.
        """
        deepseek_models = [
            "deepseek-ai/deepseek-v3.1",
            "deepseek-ai/deepseek-v3.1-terminus",
            "deepseek-ai/deepseek-v3.2"
        ]

        # The model name in the payload is prefixed with 'nvidia_nim/'
        model_name = model.split('/', 1)[1] if '/' in model else model
        reasoning_effort = payload.get("reasoning_effort")

        if model_name in deepseek_models and reasoning_effort in ["low", "medium", "high"]:
            if "extra_body" not in payload:
                payload["extra_body"] = {}
            if "chat_template_kwargs" not in payload["extra_body"]:
                payload["extra_body"]["chat_template_kwargs"] = {}
            
            payload["extra_body"]["chat_template_kwargs"]["thinking"] = True
            lib_logger.info(f"Enabled 'thinking' parameter for model: {model_name} due to reasoning_effort: '{reasoning_effort}'")
