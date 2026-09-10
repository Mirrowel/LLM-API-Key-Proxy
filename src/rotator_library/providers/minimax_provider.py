# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any, AsyncGenerator, Union
import litellm

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# Static fallback models used when the MiniMax /v1/models endpoint is unavailable.
# MiniMax-M2.7 is the latest generation; all models share 204K context.
_MINIMAX_STATIC_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]

# MiniMax requires temperature strictly greater than 0.
# Values at or below 0 are clamped to this minimum before forwarding the request.
_MINIMAX_TEMPERATURE_MIN = 0.01


class MiniMaxProvider(ProviderInterface):
    """
    Provider implementation for the MiniMax API.

    MiniMax exposes an OpenAI-compatible chat/completions endpoint at
    https://api.minimax.io/v1.  LiteLLM routes requests using the
    ``minimax/`` prefix, so no custom acompletion logic is required for the
    happy path.

    The only provider-specific behaviour handled here is:

    * **Model discovery** – fetches the live model list from the MiniMax
      ``/v1/models`` endpoint and falls back to a static list of known
      M2.7 / M2.5 models when the endpoint is unreachable.

    * **Temperature clamping** – MiniMax rejects ``temperature=0`` (or any
      value ≤ 0).  Requests that carry such a temperature are silently
      adjusted to ``_MINIMAX_TEMPERATURE_MIN`` (0.01) before being forwarded
      to LiteLLM.

    Configuration
    -------------
    Set one or more API keys using the numbered ``_API_KEY`` suffix pattern::

        MINIMAX_API_KEY_1=your_key_here
        MINIMAX_API_KEY_2=another_key_here

    Optionally override the API base URL (defaults to
    ``https://api.minimax.io/v1``)::

        MINIMAX_API_BASE=https://api.minimax.io/v1

    To avoid sending ``temperature=0`` to MiniMax from any client, you may
    also set the global proxy option::

        OVERRIDE_TEMPERATURE_ZERO=set
    """

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns available MiniMax model names (with ``minimax/`` prefix).

        Tries to discover models dynamically from the ``/v1/models`` endpoint.
        Falls back to the static list when the endpoint is unavailable or
        returns an unexpected payload.
        """
        try:
            response = await client.get(
                "https://api.minimax.io/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            if data:
                models = [f"minimax/{model['id']}" for model in data]
                lib_logger.debug(
                    f"MiniMaxProvider: discovered {len(models)} models from API"
                )
                return models
        except httpx.RequestError as e:
            lib_logger.warning(
                f"MiniMaxProvider: failed to fetch models from API, using static list: {e}"
            )
        except Exception as e:
            lib_logger.warning(
                f"MiniMaxProvider: unexpected error fetching models, using static list: {e}"
            )

        # Return the static fallback list
        static = [f"minimax/{m}" for m in _MINIMAX_STATIC_MODELS]
        lib_logger.info(
            f"MiniMaxProvider: returning {len(static)} static fallback models"
        )
        return static

    # ------------------------------------------------------------------
    # Temperature clamping
    # ------------------------------------------------------------------

    def has_custom_logic(self) -> bool:
        """
        Enable custom acompletion so we can clamp temperature before the
        request reaches the MiniMax API.
        """
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs: Any,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Forward the completion request to LiteLLM after clamping temperature.

        MiniMax rejects any ``temperature`` value that is not strictly
        positive.  When the caller sends ``temperature=0`` (common for
        deterministic / tool-use requests) this method silently raises the
        value to ``_MINIMAX_TEMPERATURE_MIN``.
        """
        temperature = kwargs.get("temperature")
        if temperature is not None and temperature <= 0:
            lib_logger.debug(
                f"MiniMaxProvider: clamping temperature {temperature!r} → "
                f"{_MINIMAX_TEMPERATURE_MIN} (MiniMax requires temperature > 0)"
            )
            kwargs = dict(kwargs)
            kwargs["temperature"] = _MINIMAX_TEMPERATURE_MIN

        # Remove internal proxy keys that must not reach LiteLLM
        kwargs.pop("credential_identifier", None)
        kwargs.pop("transaction_context", None)

        return await litellm.acompletion(**kwargs)
