# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""The universal dynamic provider (G8).

One declaration carrier for every provider created from config or env —
no protocol-specific behavior lives here. Whatever protocol a dynamic
provider declares (openai_chat, responses, anthropic_messages, gemini,
ollama, or their siblings via transport profiles), the generic native
runtime executes it; this class only resolves and exposes the declared
identity:

- ``<NAME>_API_BASE`` + ``<NAME>_API_KEY`` (+ optional ``<NAME>_PROTOCOL``)
  is the complete minimal env form; the JSON providers section carries the
  full surface (endpoint paths per operation, per-profile auth, cache
  replay rules, adapters, models).
- ``bind_runtime_config`` is honored: the startup snapshot this instance
  resolves against updates when the client rebinds it — transport
  identity is immutable per run but never silently stale.
- Provider-name case is normalized at the runtime-config lookup, so a
  mixed-case declaration address finds its config.
- Model discovery runs through the resolved protocol's listing endpoint
  (``/models`` for chat-family faces) with the declared auth headers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Type

from .provider_interface import (
    ProviderInterface,
    auth_header_pair,
    declared_endpoint_path,
    render_endpoint_path,
)

# Per-protocol default endpoint paths for dynamic declarations. Kept in
# one table so the interface default and the dynamic base can never drift.
DYNAMIC_DEFAULT_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "openai_chat": {"chat": "/chat/completions", "models": "/models"},
    "responses": {"responses": "/responses"},
    "anthropic_messages": {
        "messages": "/v1/messages",
        "count_tokens": "/v1/messages/count_tokens",
    },
    "gemini": {
        "generate": "/models/{model}:generateContent",
        "stream_generate": "/models/{model}:streamGenerateContent?alt=sse",
        "count_tokens": "/models/{model}:countTokens",
        "models": "/models",
    },
    "ollama": {
        "ollama_chat": "/api/chat",
        "ollama_generate": "/api/generate",
        "embeddings": "/api/embed",
        "models": "/api/tags",
    },
}


class DynamicProvider:
    """Declaration-driven provider for config/env-created entries."""

    skip_cost_calculation: bool = True

    def __init__(self, provider_name: str, *, config_snapshot: Any = None):
        self.provider_name = str(provider_name).lower()
        self.provider_env_name = provider_name
        from ..config.experimental import get_provider_runtime_config, load_experimental_config

        self._config_snapshot = (
            config_snapshot
            if config_snapshot is not None
            else load_experimental_config()
        )
        runtime = self._resolve_runtime()
        self.api_base = runtime.api_base or os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base or not str(self.api_base).strip():
            raise ValueError(
                f"API base URL is required for dynamic provider {provider_name!r}"
            )

        self.transport_profiles = (
            dict(runtime.transport_profiles) if runtime.transport_profiles else None
        )
        self.default_profile = runtime.default_profile
        self.protocol_name = runtime.protocol_name or "openai_chat"

        from ..model_definitions import ModelDefinitions

        self.model_definitions = ModelDefinitions()

    # -- runtime resolution ------------------------------------------------

    def _resolve_runtime(self, model: str = ""):
        from ..config.experimental import get_provider_runtime_config

        return get_provider_runtime_config(
            self.provider_name.lower(),
            model,
            config=self._config_snapshot,
        )

    def _runtime_config(self, model: str = ""):
        return self._resolve_runtime(model)

    def _get_runtime_config(self, model: str = ""):
        """Custom provider transport identity is immutable after startup."""

        return self._resolve_runtime(model)

    def bind_runtime_config(self, config: Any) -> None:
        """Honor rebinds: the resolved snapshot updates with the client."""

        ProviderInterface.bind_runtime_config(self, config)
        self._config_snapshot = config

    # -- identity -----------------------------------------------------------

    def get_api_base(self) -> str:
        return str(self._resolve_runtime().api_base or self.api_base).rstrip("/")

    def get_provider_api_base(self, model: str = "") -> str:
        return self.get_api_base()

    def normalize_native_model(self, model: str = "") -> str:
        prefix = f"{self.provider_name}/"
        return model[len(prefix):] if model.startswith(prefix) else model

    def has_custom_logic(self) -> bool:
        return False

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        return self.get_native_headers(credential_identifier)

    def get_native_operation(
        self,
        model: str = "",
        request: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        profile: Optional[str] = None,
    ) -> str:
        return ProviderInterface.get_native_operation(
            self, model, request, stream=stream, profile=profile
        )

    # -- models --------------------------------------------------------------

    async def get_models(self, api_key: str, client):
        """Configured models first; otherwise the protocol's listing endpoint."""

        runtime = self._resolve_runtime()
        configured = runtime.models
        if configured:
            return [
                model if model.startswith(f"{self.provider_name}/") else f"{self.provider_name}/{model}"
                for model in configured
            ]
        response = await client.get(
            f"{self.get_api_base()}{self._listing_path()}",
            headers=self.get_native_headers(api_key, operation="models"),
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("data") or payload.get("models") or [] if isinstance(payload, dict) else []
        models: list[str] = []
        for entry in entries:
            raw_id = entry.get("id") or entry.get("name") if isinstance(entry, dict) else entry
            model_id = str(raw_id or "").removeprefix("models/")
            if model_id:
                models.append(f"{self.provider_name}/{model_id}")
        return models

    def _listing_path(self) -> str:
        from ..config.experimental import get_provider_runtime_config

        runtime = self._resolve_runtime()
        declared = runtime.endpoint_paths.get("models")
        if declared:
            return "/" + declared.lstrip("/")
        defaults = DYNAMIC_DEFAULT_ENDPOINTS.get(self.protocol_name or "openai_chat", {})
        return defaults.get("models", "/models")

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        return self.model_definitions.get_model_options(self.provider_name, model_name)

    # -- endpoints + headers ---------------------------------------------------

    def get_native_endpoint(
        self,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> str:
        runtime = self._resolve_runtime(model)
        protocol = self.get_protocol_name(model, profile=profile) if profile else self.get_protocol_name(model)
        protocol = protocol or "openai_chat"
        entry: Optional[Any] = None
        if profile and self.transport_profiles:
            entry = self.transport_profiles.get(profile)
        path = declared_endpoint_path(entry, operation) or runtime.endpoint_paths.get(operation)
        if not path:
            defaults = DYNAMIC_DEFAULT_ENDPOINTS.get(
                protocol, DYNAMIC_DEFAULT_ENDPOINTS.get("openai_chat", {})
            )
            path = defaults.get(operation)
        if not path:
            raise NotImplementedError(
                f"Dynamic provider {self.provider_name} has no endpoint for {protocol}/{operation}"
            )
        rendered = render_endpoint_path(
            path,
            model=self.normalize_native_model(model),
            operation=operation,
            provider=self.provider_name,
        )
        if rendered.startswith(("http://", "https://")):
            return rendered
        return f"{self.get_api_base()}/{rendered.lstrip('/')}"

    def get_native_headers(
        self,
        credential_identifier: str,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> Dict[str, str]:
        runtime = self._resolve_runtime(model)
        auth_mode = runtime.auth_mode
        auth_header_name = runtime.auth_header_name
        if profile and self.transport_profiles:
            entry = self.transport_profiles.get(profile)
            if isinstance(entry, Mapping):
                auth_mode = entry.get("auth_mode") or auth_mode
                auth_header_name = entry.get("auth_header_name") or auth_header_name
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if operation == "stream_generate":
            headers["Accept"] = "text/event-stream"
        headers.update(
            auth_header_pair(
                credential_identifier,
                auth_mode,
                auth_header_name,
                provider=self.provider_name,
            )
        )
        return headers
