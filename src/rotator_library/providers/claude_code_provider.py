# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Claude Code provider integration skeleton for native protocol execution."""

from __future__ import annotations

import os
from typing import Any, List

import httpx

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

DEFAULT_API_BASE = "https://api.anthropic.com"
FALLBACK_MODELS = ["claude_code/claude-sonnet-4-5", "claude_code/claude-opus-4-5"]


class ClaudeCodeProvider(ProviderInterface):
    """Provider declaration for Claude Code style native requests.

    The provider starts as an explicit integration path rather than a guessed live
    implementation. It declares protocol/adapters/cache rules and exposes mocked
    auth/model helpers so later native wiring can use it without reintroducing a
    monolithic provider transform.
    """

    provider_env_name = "claude_code"
    protocol_name = "anthropic_messages"
    adapter_names = ("suppress_developer_role",)
    field_cache_rules = (
        FieldCacheRule(
            name="claude_code_thinking_signature",
            source="response",
            path="content.*.signature",
            mode="all",
            scope=("provider", "model", "credential", "session"),
            inject=FieldCacheInjection(target="request", path="metadata.thinking_signatures", as_list=True),
            metadata={"purpose": "preserve Claude thinking signatures for follow-up requests"},
        ),
    )
    default_rotation_mode = "sequential"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch provider models with a conservative fallback list.

        Model discovery is intentionally mock-friendly. If the configured service
        does not expose a standard `/v1/models` response, provider work can later
        override this without changing protocol declarations.
        """

        try:
            response = await client.get(f"{self.get_api_base().rstrip('/')}/v1/models", headers=self.get_native_headers(api_key), timeout=30)
            response.raise_for_status()
            models = [item.get("id") for item in response.json().get("data", []) if isinstance(item, dict) and item.get("id")]
            if models:
                return [self._with_prefix(model) for model in models]
        except Exception:
            return list(FALLBACK_MODELS)
        return list(FALLBACK_MODELS)

    def get_api_base(self) -> str:
        """Return the configured Claude Code API base URL."""

        return os.getenv("CLAUDE_CODE_API_BASE", DEFAULT_API_BASE)

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "messages") -> dict[str, str]:
        """Return headers for native mocked HTTP requests."""

        return {
            "Authorization": f"Bearer {credential_identifier}",
            "anthropic-version": os.getenv("CLAUDE_CODE_ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
        }

    def get_native_endpoint(self, model: str = "", operation: str = "messages") -> str:
        """Return the provider endpoint for a native operation."""

        if operation == "models":
            return f"{self.get_api_base().rstrip('/')}/v1/models"
        return f"{self.get_api_base().rstrip('/')}/v1/messages"

    def get_adapter_config(self, model: str = "") -> dict[str, dict[str, Any]]:
        """Configure adapters without hardcoding provider transforms."""

        return {"suppress_developer_role": {"mode": "user"}}

    @staticmethod
    def _with_prefix(model: str) -> str:
        if model.startswith("claude_code/"):
            return model
        return f"claude_code/{model}"
