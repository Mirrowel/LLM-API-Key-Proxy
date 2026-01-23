# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Model name resolution and filtering.

Extracts model-related logic from client.py including:
- _resolve_model_id (lines 867-902)
- _is_model_ignored (lines 587-619)
- _is_model_whitelisted (lines 621-651)
"""

import fnmatch
import logging
from typing import Any, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class ModelResolver:
    """
    Resolve model names and apply filtering rules.

    Handles:
    - Model ID resolution (display name -> actual ID)
    - Whitelist/blacklist filtering
    - Provider prefix handling
    """

    def __init__(
        self,
        provider_plugins: Dict[str, Any],
        model_definitions: Optional[Any] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        provider_instances: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ModelResolver.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes
            model_definitions: ModelDefinitions instance for ID mapping
            ignore_models: Models to ignore/blacklist per provider
            whitelist_models: Models to explicitly whitelist per provider
            provider_instances: Shared dict for caching provider instances.
                If None, creates a new dict (not recommended - leads to duplicate instances).
        """
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = (
            provider_instances if provider_instances is not None else {}
        )
        self._definitions = model_definitions
        self._ignore = ignore_models or {}
        self._whitelist = whitelist_models or {}

    def _get_plugin_instance(self, provider: str) -> Optional[Any]:
        """
        Get or create a plugin instance for a provider.
        """
        if provider not in self._plugin_instances:
            plugin_class = self._plugins.get(provider)
            if plugin_class:
                if isinstance(plugin_class, type):
                    self._plugin_instances[provider] = plugin_class()
                else:
                    self._plugin_instances[provider] = plugin_class
            else:
                return None
        return self._plugin_instances[provider]

    def resolve_model_id(self, model: str, provider: str) -> str:
        """
        Resolve display name to actual model ID.

        For custom models with name/ID mappings, returns the ID.
        Otherwise, returns the model name unchanged.

        Args:
            model: Full model string with provider (e.g., "iflow/DS-v3.2")
            provider: Provider name (e.g., "iflow")

        Returns:
            Full model string with ID (e.g., "iflow/deepseek-v3.2")
        """
        model_name = model.split("/")[-1] if "/" in model else model

        # Check provider plugin first
        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "model_definitions"):
            resolved = plugin.model_definitions.get_model_id(provider, model_name)
            if resolved and resolved != model_name:
                return f"{provider}/{resolved}"

        # Fallback to client-level definitions
        if self._definitions:
            resolved = self._definitions.get_model_id(provider, model_name)
            if resolved and resolved != model_name:
                return f"{provider}/{resolved}"

        return model

    def is_model_allowed(self, model: str, provider: str) -> bool:
        """
        Check if model passes whitelist/blacklist filters.

        Whitelist takes precedence over blacklist.

        Args:
            model: Model string (with or without provider prefix)
            provider: Provider name

        Returns:
            True if model is allowed, False if blocked
        """
        # Whitelist takes precedence
        if self._is_whitelisted(model, provider):
            return True

        # Then check blacklist
        if self._is_blacklisted(model, provider):
            return False

        return True

    def _is_blacklisted(self, model: str, provider: str) -> bool:
        """
        Check if model is blacklisted.

        Supports glob patterns:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard
        - "*-preview" - suffix wildcard
        - "*" - match all

        Args:
            model: Model string
            provider: Provider name (used to get ignore list)

        Returns:
            True if model is blacklisted
        """
        model_provider = model.split("/")[0] if "/" in model else provider

        if model_provider not in self._ignore:
            return False

        ignore_list = self._ignore[model_provider]
        if ignore_list == ["*"]:
            return True

        # Extract model name without provider prefix
        model_name = model.split("/", 1)[1] if "/" in model else model

        for pattern in ignore_list:
            # Use fnmatch for glob pattern support
            if fnmatch.fnmatch(model_name, pattern):
                return True
            if fnmatch.fnmatch(model, pattern):
                return True

        return False

    def _is_whitelisted(self, model: str, provider: str) -> bool:
        """
        Check if model is whitelisted.

        Same pattern support as blacklist.

        Args:
            model: Model string
            provider: Provider name

        Returns:
            True if model is whitelisted
        """
        model_provider = model.split("/")[0] if "/" in model else provider

        if model_provider not in self._whitelist:
            return False

        whitelist = self._whitelist[model_provider]
        model_name = model.split("/", 1)[1] if "/" in model else model

        for pattern in whitelist:
            if fnmatch.fnmatch(model_name, pattern):
                return True
            if fnmatch.fnmatch(model, pattern):
                return True

        return False

    @staticmethod
    def extract_provider(model: str) -> str:
        """
        Extract provider name from model string.

        Args:
            model: Model string (e.g., "openai/gpt-4")

        Returns:
            Provider name (e.g., "openai") or empty string if no prefix
        """
        return model.split("/")[0] if "/" in model else ""

    @staticmethod
    def strip_provider(model: str) -> str:
        """
        Strip provider prefix from model string.

        Args:
            model: Model string (e.g., "openai/gpt-4")

        Returns:
            Model name without prefix (e.g., "gpt-4")
        """
        return model.split("/", 1)[1] if "/" in model else model

    @staticmethod
    def ensure_provider_prefix(model: str, provider: str) -> str:
        """
        Ensure model string has provider prefix.

        Args:
            model: Model string
            provider: Provider name to add if missing

        Returns:
            Model string with provider prefix
        """
        if "/" in model:
            return model
        return f"{provider}/{model}"
