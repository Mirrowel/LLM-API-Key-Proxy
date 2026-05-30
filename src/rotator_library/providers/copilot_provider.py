# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Copilot provider integration skeleton for native OpenAI Chat execution."""

from __future__ import annotations

import os
from typing import List

import httpx

from .provider_interface import ProviderInterface

DEFAULT_API_BASE = "https://api.githubcopilot.com"
FALLBACK_MODELS = ["copilot/gpt-4.1", "copilot/claude-sonnet-4-5"]


class CopilotProvider(ProviderInterface):
    """Provider declaration for Copilot-style OpenAI-compatible chat calls.

    The skeleton intentionally avoids inventing field-cache rules until a stable
    provider session/conversation field is identified. This keeps Copilot native
    support explicit and testable without guessing hidden behavior.
    """

    provider_env_name = "copilot"
    protocol_name = "openai_chat"
    adapter_names = ("suppress_developer_role",)
    field_cache_rules: tuple = ()
    default_rotation_mode = "sequential"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch Copilot-visible models with a safe fallback list."""

        try:
            response = await client.get(self.get_native_endpoint(operation="models"), headers=self.get_native_headers(api_key), timeout=30)
            response.raise_for_status()
            models = [item.get("id") for item in response.json().get("data", []) if isinstance(item, dict) and item.get("id")]
            if models:
                return [self._with_prefix(model) for model in models]
        except Exception:
            return list(FALLBACK_MODELS)
        return list(FALLBACK_MODELS)

    def get_api_base(self) -> str:
        """Return the configured Copilot API base URL."""

        return os.getenv("COPILOT_API_BASE", DEFAULT_API_BASE)

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "chat") -> dict[str, str]:
        """Return headers for Copilot native HTTP calls."""

        return {
            "Authorization": f"Bearer {credential_identifier}",
            "content-type": "application/json",
            "Copilot-Integration-Id": os.getenv("COPILOT_INTEGRATION_ID", "llm-api-key-proxy"),
        }

    def get_native_endpoint(self, model: str = "", operation: str = "chat") -> str:
        """Return the Copilot endpoint for a native operation."""

        suffix = "/models" if operation == "models" else "/chat/completions"
        return f"{self.get_api_base().rstrip('/')}{suffix}"

    def get_adapter_config(self, model: str = "") -> dict[str, dict[str, str]]:
        """Configure role suppression declaratively for OpenAI-compatible chat."""

        return {"suppress_developer_role": {"mode": "system"}}

    @staticmethod
    def _with_prefix(model: str) -> str:
        if model.startswith("copilot/"):
            return model
        return f"copilot/{model}"
