# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from ..minimax_config import (
    MINIMAX_DEFAULT_MODELS,
    OPENAI_PROTOCOL,
    get_minimax_endpoint,
    get_minimax_protocol,
    get_minimax_region,
)
from ..model_definitions import ModelDefinitions
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class MinimaxProvider(ProviderInterface):
    """MiniMax provider using the configured OpenAI or Anthropic protocol."""

    provider_env_name = "minimax"

    def __init__(self):
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return configured fallback models and any models exposed by the API."""
        models: List[str] = []
        seen: set[str] = set()

        configured_models = self.model_definitions.get_all_provider_models("minimax")
        for model in configured_models:
            if model not in seen:
                models.append(model)
                seen.add(model)

        for model_id in MINIMAX_DEFAULT_MODELS:
            model = f"minimax/{model_id}"
            if model not in seen:
                models.append(model)
                seen.add(model)

        try:
            models_url = f"{get_minimax_endpoint(protocol=OPENAI_PROTOCOL)}/models"
            response = await client.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            for model_data in response.json().get("data", []):
                model_id = model_data.get("id")
                model = f"minimax/{model_id}" if model_id else ""
                if model and model not in seen:
                    models.append(model)
                    seen.add(model)
        except Exception as exc:
            lib_logger.debug("MiniMax model discovery failed: %s", exc)

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """Return user-defined options for a MiniMax model."""
        model_name = model_name.rsplit("/", 1)[-1]
        return self.model_definitions.get_model_options("minimax", model_name)

    async def transform_request(
        self,
        kwargs: Dict[str, Any],
        model: str,
        credential: str,
    ) -> List[str]:
        """Route the normalized request through the selected compatibility API."""
        protocol = get_minimax_protocol()
        model_name = model.rsplit("/", 1)[-1]
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
