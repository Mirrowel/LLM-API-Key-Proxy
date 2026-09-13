# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Native protocol provider for a local (or remote) Ollama server.

Ollama speaks newline-delimited JSON natively on ``/api/chat``,
``/api/generate``, and ``/api/embed``. Auth is optional: a bare local server
needs no credential, while a reverse-proxied deployment may present an
``Authorization: Bearer`` token. Zero-credential routing is declared through
``default_auth_mode = "none"`` so the client mints the internal no-auth slot.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from .provider_interface import ProviderInterface, render_endpoint_path

lib_logger = logging.getLogger("rotator_library")


class OllamaProvider(ProviderInterface):
    """Provider plugin exposing Ollama's native HTTP protocol."""

    provider_env_name = "ollama"
    protocol_name = "ollama"
    default_api_base = "http://localhost:11434"
    native_streaming_supported = True
    # Local Ollama needs no credential; the header is added only when a real
    # secret is configured (optional bearer).
    default_auth_mode = "none"
    skip_cost_calculation = True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return ``/api/tags`` model names, provider-prefixed for routing."""

        try:
            response = await client.get(f"{self.get_provider_api_base()}/api/tags")
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("models") if isinstance(payload, dict) else []
            models: List[str] = []
            for entry in entries or []:
                name = entry.get("name") or entry.get("model") if isinstance(entry, dict) else entry
                if name:
                    models.append(f"ollama/{str(name).removeprefix('ollama/')}")
            return models
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Ollama models: {e}")
            return []

    def get_native_headers(
        self,
        credential_identifier: str,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> Dict[str, str]:
        """Optional bearer auth: no header when the no-auth slot is in play."""

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._is_real_credential(credential_identifier):
            headers["Authorization"] = f"Bearer {credential_identifier}"
        return headers

    def get_native_endpoint(
        self,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> str:
        """Map operations onto the native ``/api/*`` endpoints."""

        path = self._default_endpoint_path("ollama", operation)
        rendered = render_endpoint_path(
            path,
            model=self.normalize_native_model(model),
            operation=operation,
            provider=self.provider_env_name,
        )
        return f"{self.get_provider_api_base()}{rendered}"

    @staticmethod
    def _is_real_credential(credential_identifier: str) -> bool:
        if not credential_identifier:
            return False
        try:
            from ..client.scopes import NO_AUTH_CREDENTIAL

            return credential_identifier != NO_AUTH_CREDENTIAL
        except Exception:  # pragma: no cover - import guard
            return not str(credential_identifier).startswith("__proxy_")


def get_provider() -> OllamaProvider:
    """Convenience accessor used by tests and tooling."""

    return OllamaProvider()


__all__ = ["OllamaProvider", "get_provider"]
