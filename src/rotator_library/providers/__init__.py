# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import importlib
import pkgutil
import os
from typing import Any, Dict, Optional, Type
from .provider_interface import ProviderInterface

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}


def _create_dynamic_plugin_class(
    name: str,
    config_snapshot: Any = None,
) -> Type[ProviderInterface]:
    """Create one ProviderInterface implementation bound to a config name."""

    class DynamicPlugin(DynamicOpenAICompatibleProvider, ProviderInterface):
        provider_env_name = name

        def __init__(self):
            DynamicOpenAICompatibleProvider.__init__(
                self,
                name,
                config_snapshot=config_snapshot,
            )

    DynamicPlugin.__name__ = f"{''.join(part.title() for part in name.split('_'))}DynamicProvider"
    return DynamicPlugin


class DynamicOpenAICompatibleProvider:
    """
    Dynamic provider for safe config or ``*_API_BASE`` declarations.

    Environment-only declarations retain the existing OpenAI-compatible
    LiteLLM path. Structured config may additionally opt the provider into any
    registered native protocol without storing credentials in the config file.
    """

    # Class attribute - no need to instantiate
    skip_cost_calculation: bool = True

    def __init__(self, provider_name: str, *, config_snapshot: Any = None):
        self.provider_name = provider_name
        self.provider_env_name = provider_name
        from ..config.experimental import get_provider_runtime_config, load_experimental_config

        self._config_snapshot = (
            config_snapshot
            if config_snapshot is not None
            else load_experimental_config()
        )
        runtime = get_provider_runtime_config(
            provider_name,
            config=self._config_snapshot,
        )
        self.api_base = runtime.api_base or os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"API base URL is required for dynamic provider {provider_name!r}"
            )

        # Import model definitions
        from ..model_definitions import ModelDefinitions

        self.model_definitions = ModelDefinitions()

    def _runtime_config(self, model: str = ""):
        from ..config.experimental import get_provider_runtime_config

        return get_provider_runtime_config(
            self.provider_name,
            model,
            config=self._config_snapshot,
        )

    def _get_runtime_config(self, model: str = ""):
        """Keep custom provider transport identity immutable after startup."""

        return self._runtime_config(model)

    def get_api_base(self) -> str:
        return str(self._runtime_config().api_base or self.api_base).rstrip("/")

    async def get_models(self, api_key: str, client):
        """Return configured models or discover common ``/models`` shapes."""
        configured = self._runtime_config().models
        if configured:
            return [
                model if model.startswith(f"{self.provider_name}/") else f"{self.provider_name}/{model}"
                for model in configured
            ]
        response = await client.get(
            f"{self.get_api_base()}/models",
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

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """Get model options from static definitions."""
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """Returns False since we want to use the standard litellm flow."""
        return False

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Return the configured credential header."""
        return self.get_native_headers(credential_identifier)

    def get_native_operation(
        self,
        model: str = "",
        request: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        protocol = self.get_protocol_name(model)
        if protocol == "responses":
            return "responses"
        if protocol == "anthropic_messages":
            return "messages"
        if protocol == "gemini":
            return "stream_generate" if stream else "generate"
        return "chat"

    def normalize_native_model(self, model: str = "") -> str:
        prefix = f"{self.provider_name}/"
        return model[len(prefix):] if model.startswith(prefix) else model

    def get_native_endpoint(self, model: str = "", operation: str = "chat") -> str:
        runtime = self._runtime_config(model)
        protocol = self.get_protocol_name(model) or "openai_chat"
        defaults = {
            "openai_chat": {"chat": "/chat/completions"},
            "responses": {"responses": "/responses"},
            "anthropic_messages": {"messages": "/messages"},
            "gemini": {
                "generate": "/models/{model}:generateContent",
                "stream_generate": "/models/{model}:streamGenerateContent?alt=sse",
                "count_tokens": "/models/{model}:countTokens",
            },
        }
        path = runtime.endpoint_paths.get(operation) or defaults.get(protocol, {}).get(operation)
        if not path:
            raise NotImplementedError(
                f"Dynamic provider {self.provider_name} has no endpoint for {protocol}/{operation}"
            )
        rendered = path.format(
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
    ) -> Dict[str, str]:
        runtime = self._runtime_config(model)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if operation == "stream_generate":
            headers["Accept"] = "text/event-stream"
        if runtime.auth_mode == "none":
            return headers
        if runtime.auth_mode == "x-api-key":
            headers["x-api-key"] = credential_identifier
        elif runtime.auth_mode == "x-goog-api-key":
            headers["x-goog-api-key"] = credential_identifier
        elif runtime.auth_mode == "custom":
            if not runtime.auth_header_name:
                raise ValueError(
                    f"Dynamic provider {self.provider_name} requires auth_header_name for custom auth"
                )
            headers[runtime.auth_header_name] = credential_identifier
        else:
            headers["Authorization"] = f"Bearer {credential_identifier}"
        return headers


def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    Also creates dynamic plugins for custom OpenAI-compatible providers.
    """
    package_path = __path__
    package_name = __name__
    from ..config.experimental import load_experimental_config

    config_snapshot = load_experimental_config()

    # First, register file-based providers. Archive/private modules are skipped
    # so retired providers can remain in-tree without becoming accessible.
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.startswith("_"):
            continue

        # Construct the full module path
        full_module_path = f"{package_name}.{module_name}"

        # Import the module
        module = importlib.import_module(full_module_path)

        # Look for a class that inherits from ProviderInterface
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, ProviderInterface)
                and attribute is not ProviderInterface
            ):
                # Derives 'openrouter' from 'openrouter_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                import logging

                logging.getLogger("rotator_library").debug(
                    f"Registered provider: {provider_name}"
                )

    # Then, create dynamic plugins for custom OpenAI-compatible providers
    # These use the pattern: <NAME>_API_BASE where NAME is not a known LiteLLM provider
    # Known providers just get their api_base overridden via ProviderConfig

    # Import KNOWN_PROVIDERS to check against
    from ..provider_config import KNOWN_PROVIDERS

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip if this is a known LiteLLM provider (not a custom provider)
            if provider_name in KNOWN_PROVIDERS:
                continue

            # Skip if this provider name already exists (file-based plugin)
            if provider_name in PROVIDER_PLUGINS:
                continue

            plugin_class = _create_dynamic_plugin_class(
                provider_name,
                config_snapshot=config_snapshot,
            )
            PROVIDER_PLUGINS[provider_name] = plugin_class
            import logging

            logging.getLogger("rotator_library").debug(
                f"Registered dynamic provider: {provider_name}"
            )

    # Structured config can define custom providers without a parallel API_BASE
    # environment variable. Credentials remain in the existing secret stores.
    configured = config_snapshot.providers
    for raw_name, raw in configured.items():
        provider_name = str(raw_name).lower()
        if provider_name in PROVIDER_PLUGINS:
            transport_keys = {
                "api_base",
                "endpoint_paths",
                "auth_mode",
                "auth_header_name",
                "models",
            } & set(raw if isinstance(raw, dict) else {})
            if transport_keys:
                raise ValueError(
                    f"Provider {provider_name!r} is implemented in code; custom transport keys are not applied: "
                    f"{', '.join(sorted(transport_keys))}"
                )
            continue
        if not isinstance(raw, dict):
            continue
        configured_protocol = raw.get("protocol_name")
        if configured_protocol:
            from ..protocols import get_protocol

            protocol_name = get_protocol(str(configured_protocol).strip().lower()).name
            if protocol_name not in {
                "openai_chat",
                "responses",
                "anthropic_messages",
                "gemini",
            }:
                raise ValueError(
                    f"Configured custom provider {provider_name!r} requires a supported generative protocol, got {protocol_name!r}"
                )
        api_base = str(raw.get("api_base") or os.getenv(f"{provider_name.upper()}_API_BASE") or "").strip()
        if not api_base:
            raise ValueError(
                f"Configured custom provider {provider_name!r} requires providers.{provider_name}.api_base"
            )
        PROVIDER_PLUGINS[provider_name] = _create_dynamic_plugin_class(
            provider_name,
            config_snapshot=config_snapshot,
        )
        import logging

        logging.getLogger("rotator_library").debug(
            f"Registered config-defined provider: {provider_name}"
        )


# Discover and register providers when the package is imported
_register_providers()
