# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import litellm

from ..minimax_config import (
    ANTHROPIC_PROTOCOL,
    MINIMAX_DEFAULT_MODELS,
    get_minimax_endpoint,
    get_minimax_protocol,
    get_minimax_region,
)
from ..model_definitions import ModelDefinitions
from .provider_interface import ProviderInterface


class MinimaxProvider(ProviderInterface):
    """MiniMax provider using the configured OpenAI or Anthropic protocol."""

    provider_env_name = "minimax"

    def __init__(self):
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """List current text models first, followed by explicit user definitions."""
        models = [f"minimax/{model_id}" for model_id in MINIMAX_DEFAULT_MODELS]
        for model in self.model_definitions.get_all_provider_models("minimax"):
            if model not in models:
                models.append(model)
        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """Return user-defined options for a MiniMax model."""
        model_name = model_name.rsplit("/", 1)[-1]
        options = dict(self.model_definitions.get_model_options("minimax", model_name))
        self._adapt_flash_effort(options, model_name, get_minimax_protocol())
        return options

    @staticmethod
    def _adapt_flash_effort(options: Dict[str, Any], model: str, protocol: str) -> None:
        if model == "MiniMax-M3.1-Flash" and protocol == ANTHROPIC_PROTOCOL:
            effort = options.pop("reasoning_effort", None)
            if effort is not None:
                litellm.register_model(
                    {
                        "anthropic/MiniMax-M3.1-Flash": {
                            **litellm.model_cost.get(
                                "anthropic/MiniMax-M3.1-Flash", {}
                            ),
                            "litellm_provider": "anthropic",
                            "mode": "chat",
                            "supports_reasoning": True,
                            "supports_output_config": True,
                            "supports_xhigh_reasoning_effort": True,
                            "supports_max_reasoning_effort": True,
                        }
                    }
                )
                options["output_config"] = {
                    **options.get("output_config", {}),
                    "effort": effort,
                }

    async def transform_request(
        self,
        kwargs: Dict[str, Any],
        model: str,
        credential: str,
    ) -> List[str]:
        """Route the normalized request through the selected compatibility API."""
        protocol = get_minimax_protocol()
        model_name = model.rsplit("/", 1)[-1]
        self._adapt_flash_effort(kwargs, model_name, protocol)
        kwargs["model"] = f"{protocol}/{model_name}"
        kwargs["api_base"] = get_minimax_endpoint(protocol=protocol)
        kwargs["custom_llm_provider"] = protocol
        return [
            f"minimax: selected {get_minimax_region()} {protocol}-compatible endpoint"
        ]

    def has_custom_logic(self) -> bool:
        """Use LiteLLM after selecting the configured compatibility adapter."""
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Return the standard API key authorization header."""
        return {"Authorization": f"Bearer {credential_identifier}"}
