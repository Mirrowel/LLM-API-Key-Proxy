# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Provider-specific request transformations.

This module isolates all provider-specific request mutations that were
scattered throughout client.py, including:
- gemma-3 system message conversion
- qwen_code provider remapping
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- iflow stream_options removal

Transforms are applied in a defined order with logging of modifications.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class ProviderTransforms:
    """
    Centralized provider-specific request transformations.

    Transforms are applied in order:
    1. Built-in transforms (gemma-3, qwen_code, etc.)
    2. Provider hook transforms (from provider plugins)
    3. Safety settings conversions
    """

    def __init__(
        self,
        provider_plugins: Dict[str, Any],
        provider_config: Optional[Any] = None,
        provider_instances: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ProviderTransforms.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes
            provider_config: ProviderConfig instance for LiteLLM conversions
            provider_instances: Shared dict for caching provider instances.
                If None, creates a new dict (not recommended - leads to duplicate instances).
        """
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = (
            provider_instances if provider_instances is not None else {}
        )
        self._config = provider_config

        # Registry of built-in transforms
        # Each provider can have multiple transform functions
        self._transforms: Dict[str, List[Callable]] = {
            "gemma": [self._transform_gemma_system_messages],
            "qwen_code": [self._transform_qwen_code_provider],
            "gemini": [self._transform_gemini_safety, self._transform_gemini_thinking],
            "nvidia_nim": [self._transform_nvidia_thinking],
            "iflow": [self._transform_iflow_stream_options],
        }

    def _get_plugin_instance(self, provider: str) -> Optional[Any]:
        """Get or create a plugin instance for a provider."""
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

    async def apply(
        self,
        provider: str,
        model: str,
        credential: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply all applicable transforms to request kwargs.

        Args:
            provider: Provider name
            model: Model being requested
            credential: Selected credential
            kwargs: Request kwargs (will be mutated)

        Returns:
            Modified kwargs
        """
        modifications: List[str] = []

        # 1. Apply built-in transforms
        for transform_provider, transforms in self._transforms.items():
            # Check if transform applies (provider match or model contains pattern)
            if transform_provider == provider or transform_provider in model.lower():
                for transform in transforms:
                    result = transform(kwargs, model, provider)
                    if result:
                        modifications.append(result)

        # 2. Apply provider hook transforms (async)
        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "transform_request"):
            try:
                hook_result = await plugin.transform_request(kwargs, model, credential)
                if hook_result:
                    modifications.extend(hook_result)
            except Exception as e:
                lib_logger.debug(f"Provider transform_request hook failed: {e}")

        # 3. Apply model-specific options from provider
        if plugin and hasattr(plugin, "get_model_options"):
            model_options = plugin.get_model_options(model)
            if model_options:
                for key, value in model_options.items():
                    if key == "reasoning_effort":
                        kwargs["reasoning_effort"] = value
                    elif key not in kwargs:
                        kwargs[key] = value
                modifications.append(f"applied model options for {model}")

        # 4. Apply LiteLLM conversion if config available
        if self._config and hasattr(self._config, "convert_for_litellm"):
            kwargs = self._config.convert_for_litellm(**kwargs)

        if modifications:
            lib_logger.debug(
                f"Applied transforms for {provider}/{model}: {modifications}"
            )

        return kwargs

    def apply_sync(
        self,
        provider: str,
        model: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply built-in transforms synchronously (no provider hooks).

        Useful when async is not available.

        Args:
            provider: Provider name
            model: Model being requested
            kwargs: Request kwargs

        Returns:
            Modified kwargs
        """
        modifications: List[str] = []

        for transform_provider, transforms in self._transforms.items():
            if transform_provider == provider or transform_provider in model.lower():
                for transform in transforms:
                    result = transform(kwargs, model, provider)
                    if result:
                        modifications.append(result)

        if modifications:
            lib_logger.debug(
                f"Applied sync transforms for {provider}/{model}: {modifications}"
            )

        return kwargs

    # =========================================================================
    # BUILT-IN TRANSFORMS
    # =========================================================================

    def _transform_gemma_system_messages(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Convert system messages to user messages for Gemma-3.

        Gemma-3 models don't support system messages, so we convert them
        to user messages to maintain functionality.
        """
        if "gemma-3" not in model.lower():
            return None

        messages = kwargs.get("messages", [])
        if not messages:
            return None

        converted = False
        new_messages = []
        for m in messages:
            if m.get("role") == "system":
                new_messages.append({"role": "user", "content": m["content"]})
                converted = True
            else:
                new_messages.append(m)

        if converted:
            kwargs["messages"] = new_messages
            return "gemma-3: converted system->user messages"
        return None

    def _transform_qwen_code_provider(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remap qwen_code to qwen provider for LiteLLM.

        The qwen_code provider is a custom wrapper that needs to be
        translated to the qwen provider for LiteLLM compatibility.
        """
        if provider != "qwen_code":
            return None

        kwargs["custom_llm_provider"] = "qwen"
        if "/" in model:
            kwargs["model"] = model.split("/", 1)[1]
        return "qwen_code: remapped to qwen provider"

    def _transform_gemini_safety(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Apply default Gemini safety settings.

        Ensures safety settings are present without overriding explicit settings.
        """
        if provider != "gemini":
            return None

        # Default safety settings (generic form)
        default_generic = {
            "harassment": "OFF",
            "hate_speech": "OFF",
            "sexually_explicit": "OFF",
            "dangerous_content": "OFF",
            "civic_integrity": "BLOCK_NONE",
        }

        # Default Gemini-native settings
        default_gemini = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]

        # If generic form present, fill in missing keys
        if "safety_settings" in kwargs and isinstance(kwargs["safety_settings"], dict):
            for k, v in default_generic.items():
                if k not in kwargs["safety_settings"]:
                    kwargs["safety_settings"][k] = v
            return "gemini: filled missing safety settings"

        # If Gemini form present, fill in missing categories
        if "safetySettings" in kwargs and isinstance(kwargs["safetySettings"], list):
            present = {
                item.get("category")
                for item in kwargs["safetySettings"]
                if isinstance(item, dict)
            }
            added = 0
            for d in default_gemini:
                if d["category"] not in present:
                    kwargs["safetySettings"].append(d)
                    added += 1
            if added > 0:
                return f"gemini: added {added} missing safety categories"
            return None

        # Neither present: set generic defaults
        if "safety_settings" not in kwargs and "safetySettings" not in kwargs:
            kwargs["safety_settings"] = default_generic.copy()
            return "gemini: applied default safety settings"

        return None

    def _transform_gemini_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter for Gemini.

        Delegates to provider plugin's handle_thinking_parameter method.
        """
        if provider != "gemini":
            return None

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            return "gemini: handled thinking parameter"
        return None

    def _transform_nvidia_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter for NVIDIA NIM.

        Delegates to provider plugin's handle_thinking_parameter method.
        """
        if provider != "nvidia_nim":
            return None

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            return "nvidia_nim: handled thinking parameter"
        return None

    def _transform_iflow_stream_options(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remove stream_options for iflow provider.

        The iflow provider returns HTTP 406 if stream_options is present.
        """
        if provider != "iflow":
            return None

        if "stream_options" in kwargs:
            del kwargs["stream_options"]
            return "iflow: removed stream_options"
        return None

    # =========================================================================
    # SAFETY SETTINGS CONVERSION
    # =========================================================================

    def convert_safety_settings(
        self,
        provider: str,
        settings: Dict[str, str],
    ) -> Optional[Any]:
        """
        Convert generic safety settings to provider-specific format.

        Args:
            provider: Provider name
            settings: Generic safety settings dict

        Returns:
            Provider-specific settings or None
        """
        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "convert_safety_settings"):
            return plugin.convert_safety_settings(settings)
        return None
