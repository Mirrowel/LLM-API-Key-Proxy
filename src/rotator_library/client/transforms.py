# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Provider-specific request transformations.

This module isolates all provider-specific request mutations that were
scattered throughout client.py, including:
- gemma-3 system message conversion
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- dedaluslabs tool_choice=auto removal

Transforms are applied in a defined order with logging of modifications.
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class ProviderTransforms:
    """
    Centralized provider-specific request transformations.

    Transforms are applied in order:
    1. Built-in transforms (gemma-3, etc.)
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
            "gemini": [self._transform_gemini_safety, self._transform_gemini_thinking],
            "nvidia_nim": [self._transform_nvidia_thinking],
            "dedaluslabs": [self._transform_dedaluslabs_tool_choice],
            "mistral": [self._transform_mistral_thinking],
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
        provider_config_override: Optional[Dict[str, Any]] = None,
        *,
        transaction_logger: Optional[Any] = None,
        credential_id: Optional[str] = None,
        transport: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
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
        trace_metadata = dict(trace_metadata or {})
        _trace_transform_pass(
            transaction_logger,
            "pre_provider_transform_request",
            kwargs,
            provider=provider,
            model=model,
            credential_id=credential_id,
            transport=transport,
            metadata={"phase": "start", **trace_metadata},
        )

        # 1. Apply built-in transforms
        for transform_provider, transforms in self._transforms.items():
            # Check if transform applies (provider match or model contains pattern)
            if transform_provider == provider or transform_provider in model.lower():
                for transform in transforms:
                    before = deepcopy(kwargs) if transaction_logger else None
                    try:
                        result = transform(kwargs, model, provider)
                    except Exception as exc:
                        if transaction_logger:
                            transaction_logger.log_transform_error(
                                "builtin_provider_transform",
                                exc,
                                payload=before if before is not None else kwargs,
                                stage="client",
                                transport=transport,
                                metadata={
                                    "provider": provider,
                                    "model": model,
                                    "credential_id": credential_id,
                                    "transform_provider": transform_provider,
                                    "transform_name": getattr(transform, "__name__", repr(transform)),
                                    **trace_metadata,
                                },
                            )
                        raise
                    if result:
                        modifications.append(result)
                        _trace_transform_pass(
                            transaction_logger,
                            "after_builtin_provider_transform",
                            kwargs,
                            provider=provider,
                            model=model,
                            credential_id=credential_id,
                            transport=transport,
                            changed_from_previous=(before != kwargs) if before is not None else None,
                            metadata={
                                "transform_provider": transform_provider,
                                "transform_name": getattr(transform, "__name__", repr(transform)),
                                "modification": result,
                                **trace_metadata,
                            },
                        )

        # 2. Apply provider hook transforms (async)
        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "transform_request"):
            try:
                before = deepcopy(kwargs) if transaction_logger else None
                hook_result = await plugin.transform_request(kwargs, model, credential)
                if hook_result:
                    modifications.extend(hook_result)
                if hook_result or (before is not None and before != kwargs):
                    _trace_transform_pass(
                        transaction_logger,
                        "after_provider_hook_transform",
                        kwargs,
                        provider=provider,
                        model=model,
                        credential_id=credential_id,
                        transport=transport,
                        changed_from_previous=(before != kwargs) if before is not None else None,
                        metadata={"modifications": hook_result or [], **trace_metadata},
                    )
            except Exception as e:
                lib_logger.debug(f"Provider transform_request hook failed: {e}")
                if transaction_logger:
                    transaction_logger.log_transform_error(
                        "provider_hook_transform",
                        e,
                        payload=kwargs,
                        stage="client",
                        transport=transport,
                        metadata={"provider": provider, "model": model, "credential_id": credential_id, **trace_metadata},
                    )

        # 3. Apply model-specific options from provider
        if plugin and hasattr(plugin, "get_model_options"):
            model_options = plugin.get_model_options(model)
            if model_options:
                before = deepcopy(kwargs) if transaction_logger else None
                for key, value in model_options.items():
                    if key == "reasoning_effort":
                        kwargs["reasoning_effort"] = value
                    elif key not in kwargs:
                        kwargs[key] = value
                modifications.append(f"applied model options for {model}")
                _trace_transform_pass(
                    transaction_logger,
                    "after_provider_model_options",
                    kwargs,
                    provider=provider,
                    model=model,
                    credential_id=credential_id,
                    transport=transport,
                    changed_from_previous=(before != kwargs) if before is not None else None,
                    metadata={"model_options": deepcopy(model_options) if transaction_logger else None, **trace_metadata},
                )

        # 4. Apply LiteLLM conversion if config available
        if self._config and hasattr(self._config, "convert_for_litellm"):
            before = deepcopy(kwargs) if transaction_logger else None
            _trace_transform_pass(
                transaction_logger,
                "before_litellm_conversion",
                kwargs,
                provider=provider,
                model=model,
                credential_id=credential_id,
                transport=transport,
                metadata={"provider_config_override": bool(provider_config_override), **trace_metadata},
            )
            kwargs = self._config.convert_for_litellm(
                provider_override=provider_config_override,
                **kwargs,
            )
            _trace_transform_pass(
                transaction_logger,
                "after_litellm_conversion",
                kwargs,
                provider=provider,
                model=model,
                credential_id=credential_id,
                transport=transport,
                changed_from_previous=(before != kwargs) if before is not None else None,
                metadata={"provider_config_override": bool(provider_config_override), **trace_metadata},
            )

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

    def _transform_gemini_safety(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        # Safety settings are passed through unchanged. No defaults are injected
        # because some Gemini-family models (e.g. Gemma) reject unknown safety
        # categories with a 400 error.
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

    def _transform_mistral_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter for Mistral.

        Delegates to provider plugin's handle_thinking_parameter method.
        """
        if provider != "mistral":
            return None

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            return "mistral: handled thinking parameter"
        return None

    def _transform_dedaluslabs_tool_choice(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remove tool_choice=auto for dedaluslabs provider.

        Dedaluslabs API returns HTTP 422 if tool_choice is passed as a string
        ("auto") instead of an object. Since "auto" is the default behavior,
        removing it fixes the issue without changing functionality.
        """
        if provider != "dedaluslabs":
            return None

        if kwargs.get("tool_choice") == "auto":
            del kwargs["tool_choice"]
            return "dedaluslabs: removed tool_choice=auto"
        return None

    # =========================================================================
    # SAFETY SETTINGS CONVERSION (REMOVED)
    # =========================================================================
    # Previously had convert_safety_settings() wrapper that delegated to
    # provider plugins. Removed because auto-injecting/merging safety defaults
    # caused 400 errors on models that don't support those categories (e.g. Gemma).
    # See gemini_provider.py for the full removal comment with previous defaults.
    # =========================================================================


def _trace_transform_pass(
    transaction_logger: Optional[Any],
    pass_name: str,
    payload: Dict[str, Any],
    *,
    provider: str,
    model: str,
    credential_id: Optional[str],
    transport: Optional[str],
    metadata: Dict[str, Any],
    changed_from_previous: Optional[bool] = None,
) -> None:
    """Record provider-transform states without changing transform behavior.

    Transform tracing is observability-only. This helper centralizes pass
    metadata so individual transforms can stay focused on payload mutation while
    the transaction trace still shows each live request state.
    """

    if not transaction_logger:
        return
    transaction_logger.log_transform_pass(
        pass_name,
        payload,
        direction="request",
        stage="client",
        credential_id=credential_id,
        transport=transport,
        changed_from_previous=changed_from_previous,
        metadata={"provider": provider, "model": model, **metadata},
    )
