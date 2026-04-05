# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import os
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import httpx
import litellm

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpencodeZenProvider(ProviderInterface):
    """
    Provider for OpenCode Zen gateway - OpenAI-compatible API.

    Accesses free tier models through OpenCode's Zen gateway.
    Uses a public API key for free models.

    Free models have the "-free" suffix in their model IDs.

    Environment Variables:
        OPENCODE_ZEN_API_BASE - The API base URL (default: https://opencode.ai/zen/v1)

    Custom Headers Required:
        HTTP-Referer: https://opencode.ai/
        X-Title: opencode
    """

    provider_env_name = "opencode_zen"
    skip_cost_calculation: bool = True

    def __init__(self):
        super().__init__()
        self.api_base = os.getenv("OPENCODE_ZEN_API_BASE", "https://opencode.ai/zen/v1")

    def _get_headers(self) -> Dict[str, str]:
        """Return the custom headers required by OpenCode Zen."""
        return {
            "HTTP-Referer": "https://opencode.ai/",
            "X-Title": "opencode",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from OpenCode Zen.

        The models endpoint is public and doesn't require authentication.
        """
        models = []
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url,
                headers=self._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()

            data = response.json()
            for model in data.get("data", []):
                model_id = model.get("id")
                if model_id:
                    models.append(f"opencode_zen/{model_id}")

            lib_logger.info(f"Discovered {len(models)} models from OpenCode Zen")

        except Exception as e:
            lib_logger.warning(f"Failed to fetch models from OpenCode Zen: {e}")

        return models

    def has_custom_logic(self) -> bool:
        """
        Returns True because we need to handle API calls with custom headers.
        """
        return True

    @staticmethod
    def _strip_unsupported_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strip non-text content parts from messages.

        ZenMux free-tier models (DeepSeek, etc.) don't support multimodal
        inputs. If content is a list, keep only text parts; if only non-text
        parts remain, flatten to an empty string to avoid sending an empty array.
        """
        new_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part)
                    elif isinstance(part, str):
                        text_parts.append({"type": "text", "text": part})
                if not text_parts:
                    new_messages.append({**msg, "content": ""})
                elif len(text_parts) == 1:
                    new_messages.append({**msg, "content": text_parts[0].get("text", "")})
                else:
                    new_messages.append({**msg, "content": text_parts})
            else:
                new_messages.append(msg)
        return new_messages

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,  # client unused - LiteLLM manages its own
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion calls with ZenMux custom headers.

        We use LiteLLM but override the headers to include ZenMux's required
        identification headers.
        """
        # Clean up kwargs not needed by LiteLLM
        kwargs.pop("credential_identifier", None)
        kwargs.pop("transaction_context", None)

        # Strip unsupported multimodal content (image_url etc.)
        messages = kwargs.get("messages")
        if messages:
            kwargs["messages"] = self._strip_unsupported_content(messages)

        # Transform model name for LiteLLM's OpenAI provider
        # "opencode_zen/deepseek-v4-flash-free" -> "openai/deepseek-v4-flash-free"
        model = kwargs.get("model", "")
        if "/" in model:
            kwargs["model"] = "openai/" + model.split("/", 1)[1]

        # Add custom headers to the kwargs (without mutating caller's dict)
        extra_headers = self._get_headers()
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}

        # Ensure api_base is set
        kwargs["api_base"] = self.api_base

        # Use the public API key for the OpenCode Zen gateway
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "public"

        # Disable LiteLLM internal retries; the executor handles retry logic
        kwargs["max_retries"] = 0
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            # Return an async generator for streaming
            async def stream_wrapper():
                async for chunk in await litellm.acompletion(**kwargs):
                    yield chunk

            return stream_wrapper()
        else:
            return await litellm.acompletion(**kwargs)

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,  # client unused - LiteLLM manages its own
    ) -> litellm.EmbeddingResponse:
        """
        Handle embedding calls with ZenMux custom headers.
        """
        # Clean up kwargs not needed by LiteLLM
        kwargs.pop("credential_identifier", None)
        kwargs.pop("transaction_context", None)

        # Transform model name for LiteLLM's OpenAI provider
        model = kwargs.get("model", "")
        if "/" in model:
            kwargs["model"] = "openai/" + model.split("/", 1)[1]

        # Add custom headers (without mutating caller's dict)
        extra_headers = self._get_headers()
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}

        kwargs["api_base"] = self.api_base

        if not kwargs.get("api_key"):
            kwargs["api_key"] = "public"

        kwargs["max_retries"] = 0

        return await litellm.aembedding(**kwargs)

    def convert_safety_settings(
        self, settings: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
        """OpenCode Zen doesn't have specific safety settings to convert."""
        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """All OpenCode Zen models are free tier."""
        return "free-tier"

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """All models available through this provider are free tier."""
        return None
