# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class MistralProvider(ProviderInterface):
    """
    Provider implementation for the Mistral API.
    """

    MISTRAL_MODEL_PATTERNS = [
        "mistral-medium",
        "mistral-small",
    ]

    DISABLE_VALUES = {"none", "disable", "off"}

    def _is_mistral_reasoning(self, model_name: str) -> bool:
        return any(p in model_name for p in self.MISTRAL_MODEL_PATTERNS)

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Mistral API.
        """
        try:
            response = await client.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"mistral/{model['id']}"
                for model in response.json().get("data", [])
            ]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Mistral models: {e}")
            return []

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Configures reasoning_effort for Mistral reasoning models.

        Mistral's reasoning models (medium, small) support a
        ``reasoning_effort`` parameter, but LiteLLM does not recognise it as a
        top-level kwarg and silently drops it before forwarding the request.
        To work around this we remove the top-level key (if present) and inject
        it through ``extra_body``, which LiteLLM passes through verbatim to the
        underlying Mistral API.

        Incoming ``reasoning_effort`` of none/disable/off disables reasoning
        entirely — the key is removed and no ``extra_body`` override is set.
        """
        model_name = model.split("/", 1)[1] if "/" in model else model

        if not self._is_mistral_reasoning(model_name):
            return

        reasoning_effort = payload.get("reasoning_effort")

        is_disabled = (
            isinstance(reasoning_effort, str)
            and reasoning_effort.lower() in self.DISABLE_VALUES
        )

        if is_disabled:
            payload.pop("reasoning_effort", None)
            lib_logger.info(
                f"Mistral '{model_name}' — reasoning effort DISABLED "
                f"(reasoning_effort='{reasoning_effort}')"
            )
            return

        payload.pop("reasoning_effort", None)
        if "extra_body" not in payload:
            payload["extra_body"] = {}
        payload["extra_body"]["reasoning_effort"] = "high"
        lib_logger.info(
            f"Mistral '{model_name}' — reasoning_effort='high' (via extra_body)"
        )
