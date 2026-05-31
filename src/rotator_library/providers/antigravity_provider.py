# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Antigravity provider integration restored from safe retired pieces."""

from __future__ import annotations

import os
from typing import Any, List, Optional

import httpx

from ..field_cache import FieldCacheInjection, FieldCacheRule
from .provider_interface import ProviderInterface

BASE_URLS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://daily-cloudcode-pa.googleapis.com/v1internal",
    "https://cloudcode-pa.googleapis.com/v1internal",
]
ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.15.8 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-flash",
    "claude-sonnet-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
]
MODEL_ALIAS_MAP = {
    "rev19-uic3-1p": "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-pro-image": "gemini-3-pro-image-preview",
    "gemini-3-pro-low": "gemini-3-pro-preview",
    "gemini-3-pro-high": "gemini-3-pro-preview",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-opus-4-5": "claude-opus-4.5",
    "claude-opus-4-6": "claude-opus-4.6",
}
MODEL_ALIAS_REVERSE = {public: internal for internal, public in MODEL_ALIAS_MAP.items()}
EXCLUDED_MODELS = {"chat_20706", "chat_23310", "gemini-2.5-flash-thinking", "gemini-2.5-pro"}


class AntigravityProvider(ProviderInterface):
    """Safe restored Antigravity integration path.

    The retired provider contained valuable model/header/endpoint knowledge mixed
    with fragile device-profile and monolithic transform logic. This active
    skeleton restores only stable declarations and helpers so native provider
    work can proceed behind tests before any live routing is enabled.
    """

    provider_env_name = "antigravity"
    protocol_name = "gemini"
    adapter_names: tuple[str, ...] = ("antigravity_envelope",)
    field_cache_rules = (
        FieldCacheRule(
            name="antigravity_thought_signature",
            source="response",
            path="candidates.*.content.parts.*.thoughtSignature",
            mode="all",
            scope=("provider", "model", "credential", "session"),
            inject=FieldCacheInjection(target="request", path="request.metadata.thoughtSignatures", as_list=True),
            metadata={"purpose": "preserve Gemini thought signatures across Antigravity turns"},
        ),
    )
    native_streaming_supported = False
    model_quota_groups = {
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview", "gemini-3-flash"],
        "claude": ["claude-sonnet-4.5", "claude-opus-4.5", "claude-opus-4.6"],
    }
    default_rotation_mode = "sequential"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available Antigravity models or return the restored safe list."""

        try:
            response = await client.post(self.get_native_endpoint(operation="models"), headers=self.get_native_headers(api_key), json={}, timeout=30)
            response.raise_for_status()
            models = self._models_from_response(response.json())
            if models:
                return [self._with_prefix(model) for model in models]
        except Exception:
            return [self._with_prefix(model) for model in AVAILABLE_MODELS]
        return [self._with_prefix(model) for model in AVAILABLE_MODELS]

    def get_api_base(self) -> str:
        """Return the first configured Antigravity base URL."""

        configured = os.getenv("ANTIGRAVITY_API_BASE")
        return configured.rstrip("/") if configured else BASE_URLS[0]

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "generate") -> dict[str, str]:
        """Return static Antigravity headers plus bearer auth.

        Retired per-device fingerprint headers are intentionally not restored;
        they are brittle and should only return if a current service requirement
        is verified with tests.
        """

        headers = {
            "Authorization": f"Bearer {credential_identifier}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_HEADERS,
        }
        if operation == "stream_generate":
            headers["Accept"] = "text/event-stream"
        return headers

    def get_native_operation(self, model: str = "", request: dict[str, Any] | None = None, stream: bool = False) -> str:
        """Return the Gemini generate operation used by Antigravity endpoints."""

        return "stream_generate" if stream else "generate"

    def normalize_native_model(self, model: str) -> str:
        """Strip the proxy prefix and map public aliases to upstream names."""

        clean = model.split("/", 1)[1] if model.startswith("antigravity/") else model
        return self._alias_to_internal(clean)

    def prepare_native_request(self, request: dict[str, Any], model: str = "", operation: str = "") -> dict[str, Any]:
        """Return a request with the upstream model and Gemini contents shape.

        The provider intentionally keeps this to model alias and message-shape
        handling only. Device profile and fingerprint behavior stays out of the
        active integration until it is verified against current service behavior.
        """

        prepared = dict(request)
        public_model = str(request.get("_proxy_model") or request.get("model") or "")
        thinking_level = _thinking_level_from_model(public_model)
        if model:
            prepared["model"] = _model_with_thinking_variant(model, thinking_level)
        prepared.pop("_proxy_model", None)
        if thinking_level:
            generation_config = prepared.setdefault("generationConfig", {})
            generation_config.setdefault("thinkingConfig", {})["thinkingLevel"] = thinking_level
            prepared.setdefault("metadata", {})["thinking_level"] = thinking_level
        if "contents" not in prepared and isinstance(prepared.get("messages"), list):
            prepared["contents"] = [_message_to_gemini_content(message) for message in prepared.pop("messages")]
        return prepared

    def supports_native_streaming(self, model: str = "", operation: str = "generate") -> bool:
        """Return false until native stream wrapping is provider-safe."""

        return False

    def get_native_endpoint(self, model: str = "", operation: str = "generate") -> str:
        """Return Antigravity internal operation endpoints."""

        if operation == "models":
            return f"{self.get_api_base()}:fetchAvailableModels"
        if operation == "stream_generate":
            return f"{self.get_api_base()}:streamGenerateContent?alt=sse"
        return f"{self.get_api_base()}:generateContent"

    def get_adapter_config(self, model: str = "") -> dict[str, dict[str, Any]]:
        """Configure the safe Antigravity internal request envelope."""

        return {
            "antigravity_envelope": {
                "project": os.getenv("ANTIGRAVITY_PROJECT", ""),
                "user_agent": ANTIGRAVITY_HEADERS["User-Agent"],
                "request_type": os.getenv("ANTIGRAVITY_REQUEST_TYPE", "CHAT_COMPLETION"),
            }
        }

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """Antigravity exposes no restored model-tier restriction."""

        return None

    def normalize_model_for_tracking(self, model: str) -> str:
        """Normalize internal names to public aliases while preserving prefix."""

        if "/" in model:
            provider, clean_model = model.split("/", 1)
            return f"{provider}/{self._api_to_user_model(clean_model)}"
        return self._api_to_user_model(model)

    @staticmethod
    def _with_prefix(model: str) -> str:
        if model.startswith("antigravity/"):
            return model
        return f"antigravity/{model}"

    @staticmethod
    def _alias_to_internal(alias: str) -> str:
        if alias in {"rev19-uic3-1p", "gemini-3-pro-image", "gemini-3-pro-low", "gemini-3-pro-high"}:
            return MODEL_ALIAS_MAP.get(alias, alias)
        return MODEL_ALIAS_REVERSE.get(alias, MODEL_ALIAS_MAP.get(alias, alias))

    @staticmethod
    def _api_to_user_model(internal: str) -> str:
        return MODEL_ALIAS_MAP.get(internal, internal)

    def _models_from_response(self, payload: dict[str, Any]) -> list[str]:
        raw_models: list[str] = []
        if isinstance(payload.get("models"), dict):
            raw_models.extend(payload["models"].keys())
        if isinstance(payload.get("data"), list):
            raw_models.extend(item.get("id") for item in payload["data"] if isinstance(item, dict) and item.get("id"))
        result = []
        for model in raw_models:
            public = self._api_to_user_model(str(model))
            if public not in EXCLUDED_MODELS and public in AVAILABLE_MODELS and public not in result:
                result.append(public)
        return result


def _message_to_gemini_content(message: Any) -> dict[str, Any]:
    """Return a minimal Gemini content item from an OpenAI-style message."""

    if not isinstance(message, dict):
        return {"role": "user", "parts": [{"text": str(message)}]}
    role = "model" if message.get("role") == "assistant" else "user"
    content = message.get("content", "")
    parts = content if isinstance(content, list) else [{"text": str(content)}]
    return {"role": role, "parts": parts}


def _thinking_level_from_model(model: str) -> Optional[str]:
    clean = model.split("/", 1)[1] if model.startswith("antigravity/") else model
    if clean.endswith("-low"):
        return "low"
    if clean.endswith("-high"):
        return "high"
    return None


def _model_with_thinking_variant(model: str, thinking_level: Optional[str]) -> str:
    if model == "gemini-3-pro-preview" and thinking_level in {"low", "high"}:
        return f"gemini-3-pro-{thinking_level}"
    return model
