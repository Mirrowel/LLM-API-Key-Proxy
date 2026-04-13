# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import importlib
import pkgutil
import os
from typing import Dict, Type
from .provider_interface import ProviderInterface

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}


from .openai_compatible_provider import OpenAICompatibleProvider

class DynamicOpenAICompatibleProvider(OpenAICompatibleProvider):
    """
    Dynamic provider class for custom OpenAI-compatible providers.
    Created at runtime for providers with _API_BASE environment variables
    that are NOT known LiteLLM providers.

    Environment variable pattern:
        <NAME>_API_BASE - The API base URL (required)
        <NAME>_API_KEY  - The API key

    Example:
        MYSERVER_API_BASE=http://localhost:8000/v1
        MYSERVER_API_KEY=sk-xxx

    Note: For known providers (openai, anthropic, etc.), setting _API_BASE
    will override their default endpoint without creating a custom provider.
    """
    pass


def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    Also creates dynamic plugins for custom OpenAI-compatible providers.
    """
    package_path = __path__
    package_name = __name__

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
        _skip_bases = (ProviderInterface, OpenAICompatibleProvider, DynamicOpenAICompatibleProvider)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, ProviderInterface)
                and attribute not in _skip_bases
            ):
                # Derives 'gemini_cli' from 'gemini_cli_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                if provider_name == "gemini":
                    PROVIDER_PLUGINS["google"] = attribute
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

            # Create a dynamic plugin class
            def create_plugin_class(name):
                class DynamicPlugin(DynamicOpenAICompatibleProvider):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            PROVIDER_PLUGINS[provider_name] = plugin_class
            import logging

            logging.getLogger("rotator_library").debug(
                f"Registered dynamic provider: {provider_name}"
            )


# Discover and register providers when the package is imported
_register_providers()
