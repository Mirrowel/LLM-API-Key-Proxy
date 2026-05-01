# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any
import litellm
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False  # Ensure this logger doesn't propagate to root
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
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            models = [
                f"nvidia_nim/{model['id']}" for model in response.json().get("data", [])
            ]
            return models
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch NVIDIA models: {e}")
            return []

    V3_MODEL_PREFIXES = [
        "deepseek-ai/deepseek-v3.1",
    ]
    V3_MODEL_EXACT = [
        "deepseek-ai/deepseek-v3.2",
    ]
    V4_MODEL_EXACT = [
        "deepseek-ai/deepseek-v4-pro",
        "deepseek-ai/deepseek-v4-flash",
    ]

    V4_EFFORT_MAP = {
        "low": "high",
        "medium": "high",
        "high": "max",
        "max": "max",
    }
    DISABLE_VALUES = {"none", "disable", "off"}

    def _is_v3_deepseek(self, model_name: str) -> bool:
        if model_name in self.V3_MODEL_EXACT:
            return True
        return any(model_name.startswith(p) for p in self.V3_MODEL_PREFIXES)

    def _is_v4_deepseek(self, model_name: str) -> bool:
        return model_name in self.V4_MODEL_EXACT

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Configures thinking and reasoning_effort for DeepSeek models on NVIDIA.

        V3.x models: only thinking=True/False in chat_template_kwargs.
        V4 models: thinking + mapped reasoning_effort (high/max).
        Incoming reasoning_effort of none/disable/off disables thinking.
        """
        model_name = model.split("/", 1)[1] if "/" in model else model

        is_v3 = self._is_v3_deepseek(model_name)
        is_v4 = self._is_v4_deepseek(model_name)

        if not is_v3 and not is_v4:
            return

        reasoning_effort = payload.get("reasoning_effort")

        if "extra_body" not in payload:
            payload["extra_body"] = {}
        if "chat_template_kwargs" not in payload["extra_body"]:
            payload["extra_body"]["chat_template_kwargs"] = {}

        kwargs = payload["extra_body"]["chat_template_kwargs"]
        is_disabled = (
            isinstance(reasoning_effort, str)
            and reasoning_effort.lower() in self.DISABLE_VALUES
        )

        if is_disabled:
            kwargs["thinking"] = False
            lib_logger.info(
                f"NVIDIA: DeepSeek '{model_name}' — thinking DISABLED "
                f"(reasoning_effort='{reasoning_effort}')"
            )
            return

        kwargs["thinking"] = True

        if is_v3:
            lib_logger.info(f"NVIDIA: DeepSeek V3 '{model_name}' — thinking=True")
        else:
            effort_key = (
                reasoning_effort.lower() if isinstance(reasoning_effort, str) else None
            )
            mapped = self.V4_EFFORT_MAP.get(effort_key, "max")
            kwargs["reasoning_effort"] = mapped
            lib_logger.info(
                f"NVIDIA: DeepSeek V4 '{model_name}' — thinking=True, "
                f"reasoning_effort='{mapped}' (input: '{reasoning_effort}')"
            )
