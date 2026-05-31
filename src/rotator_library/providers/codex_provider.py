# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Codex provider integration for native Responses execution."""

from __future__ import annotations

import os
from typing import Any, List

import httpx

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

DEFAULT_API_BASE = "https://api.openai.com"
FALLBACK_MODELS = ["codex/codex-mini-latest", "codex/gpt-5.1-codex"]


class CodexProvider(ProviderInterface):
    """Provider declaration for Codex-style native Responses requests."""

    provider_env_name = "codex"
    protocol_name = "responses"
    adapter_names: tuple[str, ...] = ()
    native_streaming_supported = False
    field_cache_rules = (
        FieldCacheRule(
            name="codex_previous_response_id",
            source="response",
            path="id",
            scope=("provider", "model", "credential", "session"),
            inject=FieldCacheInjection(target="request", path="previous_response_id", when_missing_only=True),
            metadata={"purpose": "preserve Responses continuation IDs for Codex sessions"},
        ),
    )
    default_rotation_mode = "sequential"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch Codex-visible models with fallback names for offline tests."""

        try:
            response = await client.get(self.get_native_endpoint(operation="models"), headers=self.get_native_headers(api_key), timeout=30)
            response.raise_for_status()
            models = [item.get("id") for item in response.json().get("data", []) if isinstance(item, dict) and item.get("id")]
            codex_models = [model for model in models if "codex" in model.lower()]
            if codex_models:
                return [self._with_prefix(model) for model in codex_models]
        except Exception:
            return list(FALLBACK_MODELS)
        return list(FALLBACK_MODELS)

    def get_api_base(self) -> str:
        """Return the configured Codex API base URL."""

        return os.getenv("CODEX_API_BASE", DEFAULT_API_BASE)

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "responses") -> dict[str, str]:
        """Return headers for Codex native HTTP calls."""

        return {"Authorization": f"Bearer {credential_identifier}", "content-type": "application/json"}

    def get_native_operation(self, model: str = "", request: dict | None = None, stream: bool = False) -> str:
        """Codex native calls use the Responses operation."""

        return "responses"

    def normalize_native_model(self, model: str) -> str:
        """Strip the proxy provider prefix before sending upstream."""

        return model.split("/", 1)[1] if model.startswith("codex/") else model

    def supports_native_streaming(self, model: str = "", operation: str = "responses") -> bool:
        """Return false until the generic native stream wrapper is compatible."""

        return False

    def prepare_native_request(self, request: dict[str, Any], model: str = "", operation: str = "") -> dict[str, Any]:
        """Convert chat-style proxy input into Responses API input.

        Direct `/v1/responses` calls already provide `input`; routed chat calls
        usually provide `messages`. Converting here keeps the reusable Responses
        protocol focused on its native shape while still making Codex mock-live
        through the generic native executor.
        """

        prepared = dict(request)
        if model:
            prepared["model"] = model
        if "input" not in prepared and isinstance(prepared.get("messages"), list):
            prepared["input"] = [_message_to_responses_input(message) for message in prepared.pop("messages")]
        return prepared

    def get_native_endpoint(self, model: str = "", operation: str = "responses") -> str:
        """Return the native Codex endpoint for an operation."""

        suffix = "/v1/models" if operation == "models" else "/v1/responses"
        return f"{self.get_api_base().rstrip('/')}{suffix}"

    @staticmethod
    def _with_prefix(model: str) -> str:
        if model.startswith("codex/"):
            return model
        return f"codex/{model}"


def _message_to_responses_input(message: Any) -> dict[str, Any]:
    """Return a minimal Responses input item from an OpenAI-style message."""

    if not isinstance(message, dict):
        return {"role": "user", "content": str(message)}
    return {"role": message.get("role", "user"), "content": message.get("content", "")}
