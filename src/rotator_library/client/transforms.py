# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Request transformations — global guards and provider-specific mutations.

This module centralises all request mutations applied before litellm /
provider plugin ``acompletion``:

Global (run for every request):
- thinking-mode tool-call guard (disable thinking when reasoning_content
  was dropped from a tool-call turn — prevents 400 errors across all
  thinking-capable APIs)

Provider-specific (keyed by provider name or model substring):
- gemma-3 system message conversion
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- dedaluslabs tool_choice=auto removal
- Mistral reasoning_content / thinking_signature stripping
- chutes allowed_openai_params injection
- kimi-k2.5 mandatory top_p
- GLM-5 / GLM-4 max_tokens floor for thinking models
- Gitlawb / Xiaomi-Mimo compression workaround

Transforms are applied in a defined order with logging of modifications.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class ProviderTransforms:
    """
    Centralized request transformations.

    Transforms are applied in order:
    0. Global pre-transforms (thinking-mode guard, etc.)
    1. Built-in keyed transforms (gemma-3, mistral, etc.)
    2. Provider hook transforms (from provider plugins)
    3. Model-specific options from provider plugins
    4. LiteLLM conversion
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
            "chutes": [self._transform_chutes_allowed_params],
            "kimi-k2.5": [self._transform_kimi_parameters],
            "glm-5": [self._transform_glm5_max_tokens],
            "glm-4": [self._transform_glm5_max_tokens],
            "xiaomi_mimo": [self._transform_opengateway_compression],
            "gitlawb": [self._transform_opengateway_compression],
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
        request_type: str = "chat",
    ) -> Dict[str, Any]:
        """
        Apply all applicable transforms to request kwargs.

        Args:
            provider: Provider name
            model: Model being requested
            credential: Selected credential
            kwargs: Request kwargs (will be mutated)
            request_type: Type of request ("chat", "embedding", etc.)

        Returns:
            Modified kwargs
        """
        modifications: List[str] = []

        # 0. Global pre-transforms (run for every provider)
        guard_result = self._guard_thinking_tool_calls(kwargs, model, provider)
        if guard_result:
            modifications.append(guard_result)

        # 1. Apply built-in transforms (keyed by provider / model substring)
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
            kwargs["request_type"] = request_type
            kwargs = self._config.convert_for_litellm(
                provider_override=provider_config_override,
                **kwargs,
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

        # 0. Global pre-transforms
        guard_result = self._guard_thinking_tool_calls(kwargs, model, provider)
        if guard_result:
            modifications.append(guard_result)

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
    # GLOBAL PRE-TRANSFORMS (run for every provider before keyed transforms)
    # =========================================================================

    def _guard_thinking_tool_calls(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Prevent 400 errors from thinking-mode APIs when the client dropped
        ``reasoning_content`` from an assistant turn that had tool_calls.

        Many reasoning APIs (DeepSeek, Moonshot/Kimi, etc.) require the
        original ``reasoning_content`` to be passed back on assistant messages
        that contain ``tool_calls``.  Clients often strip this field because
        the standard OpenAI SDK doesn't persist it.

        When we detect an assistant message with tool_calls but no
        reasoning_content, the proxy cannot reconstruct the missing text.
        Rather than send a placeholder that the API will reject, we
        proactively disable thinking mode so the request can still succeed
        (without chain-of-thought on this turn).

        Providers that check ``if "thinking" not in extra_body`` before
        enabling thinking will automatically defer to the guard's decision.

        Gemini/Vertex models are exempt: they don't require
        ``reasoning_content`` on replay and think by default — disabling
        thinking would suppress useful output with no benefit.
        """
        if provider in ("vertex", "gemini", "google"):
            return None

        messages = kwargs.get("messages")
        if not messages:
            return None

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            if msg.get("tool_calls") and not msg.get("reasoning_content"):
                extra_body = kwargs.get("extra_body")
                if not isinstance(extra_body, dict):
                    extra_body = {}
                extra_body = {**extra_body, "thinking": {"type": "disabled"}}
                kwargs["extra_body"] = extra_body
                return (
                    "disabled thinking — assistant tool-call turn "
                    "missing reasoning_content"
                )

        return None

    # =========================================================================
    # BUILT-IN TRANSFORMS (keyed by provider / model substring)
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

    # Fields set on assistant messages by the proxy's response processing
    # (streaming.py / executor.py) that upstream APIs do not accept on input.
    _RESPONSE_ONLY_MESSAGE_FIELDS = ("reasoning_content", "thinking_signature")

    def _strip_response_only_fields(
        self,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Strip proxy-added response fields from request messages.

        Returns True if any fields were removed.
        """
        messages = kwargs.get("messages")
        if not messages:
            return False

        stripped = False
        for msg in messages:
            for field in self._RESPONSE_ONLY_MESSAGE_FIELDS:
                if field in msg:
                    del msg[field]
                    stripped = True
        return stripped

    def _transform_mistral_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter and message sanitization for Mistral.

        Strips reasoning_content / thinking_signature from messages (these are
        set by the proxy on responses but cause 422 errors when sent back to
        Mistral's strict input validation) and delegates reasoning_effort
        configuration to the provider plugin.

        Also strips ``extra_body["thinking"]`` for models that do not support
        thinking (e.g. ministral).  The global guard ``_guard_thinking_tool_calls``
        unconditionally injects ``extra_body: {"thinking": {"type": "disabled"}}``
        for any provider when a tool-call turn is missing reasoning_content, but
        Mistral's strict input validation rejects the ``thinking`` field on models
        that don't support it (HTTP 422).
        """
        if provider != "mistral":
            return None

        modifications = []

        if self._strip_response_only_fields(kwargs):
            modifications.append("stripped response-only message fields")

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            modifications.append("handled thinking parameter")

        # Strip extra_body["thinking"] for models that don't support it.
        # The global _guard_thinking_tool_calls injects this for every provider
        # when a tool-call turn lacks reasoning_content, but Mistral's strict
        # input validation rejects the field on non-reasoning models (422).
        model_name = model.split("/", 1)[1] if "/" in model else model
        if (
            kwargs.get("extra_body")
            and isinstance(kwargs["extra_body"], dict)
            and "thinking" in kwargs["extra_body"]
        ):
            # Only keep thinking for Mistral reasoning models
            if not self._is_mistral_reasoning_model(model_name, plugin):
                del kwargs["extra_body"]["thinking"]
                modifications.append("stripped extra_body thinking (non-reasoning model)")

        if modifications:
            return "mistral: " + ", ".join(modifications)
        return None

    def _is_mistral_reasoning_model(
        self, model_name: str, plugin: Optional[Any]
    ) -> bool:
        """Check if a Mistral model supports thinking/reasoning."""
        if plugin and hasattr(plugin, "_is_mistral_reasoning"):
            return plugin._is_mistral_reasoning(model_name)
        # Fallback: match known reasoning model patterns
        return any(
            p in model_name
            for p in ["mistral-medium", "mistral-small"]
        )

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

    # OpenAI-compatible params that LiteLLM's Chutes provider config
    # doesn't declare support for.  Without this list, drop_params=True
    # causes LiteLLM to silently strip tools / tool_choice / etc.
    _CHUTES_ALLOWED_OPENAI_PARAMS = [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    ]

    def _transform_chutes_allowed_params(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Inject allowed_openai_params for Chutes provider.

        LiteLLM's built-in Chutes provider config doesn't advertise support
        for tool calling parameters (tools, tool_choice, etc.), so with
        litellm.drop_params=True they get silently removed.  This transform
        tells LiteLLM these standard OpenAI params are safe to pass through
        to the Chutes API, which is fully OpenAI-compatible.
        """
        if provider != "chutes":
            return None

        # Only inject if the request actually uses any of these params
        has_tool_params = any(k in kwargs for k in self._CHUTES_ALLOWED_OPENAI_PARAMS)
        if not has_tool_params:
            return None

        existing = kwargs.get("allowed_openai_params", [])
        merged = list(set(existing) | set(self._CHUTES_ALLOWED_OPENAI_PARAMS))
        kwargs["allowed_openai_params"] = merged
        return "chutes: injected allowed_openai_params for tool calling"

    def _transform_kimi_parameters(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Set top_p=0.95 for Kimi K2.5 models.

        The Kimi K2.5 API (via various providers) strictly requires top_p to be 0.95.
        Other values or missing top_p results in a 400 error.
        """
        if "kimi-k2.5" not in model.lower():
            return None

        if kwargs.get("top_p") != 0.95:
            kwargs["top_p"] = 0.95
            return "kimi-k2.5: set top_p=0.95 (mandatory)"
        return None

    # GLM-5 / GLM-4 thinking model minimum token floor
    GLM_MIN_MAX_TOKENS = 4096

    def _transform_glm5_max_tokens(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Enforce a minimum max_tokens floor for GLM-5/GLM-4 thinking models.

        GLM-5 (and GLM-4.x) thinking variants share a single max_tokens budget
        between reasoning tokens and content tokens. When max_tokens is too low,
        the model exhausts the entire budget on chain-of-thought reasoning and
        returns content: null/"". This affects all providers hosting these models
        (Modal, NanoGPT, Kilo, Zenmux, etc.).

        This transform enforces a minimum floor so the model always has enough
        headroom to produce actual response content after reasoning.
        """
        model_lower = model.lower()
        # Only apply to GLM thinking/reasoning model variants
        if not any(prefix in model_lower for prefix in ("glm-5", "glm-4")):
            return None

        current = kwargs.get("max_tokens")
        if current is None or current < self.GLM_MIN_MAX_TOKENS:
            kwargs["max_tokens"] = self.GLM_MIN_MAX_TOKENS
            if current is not None:
                return (
                    f"glm: raised max_tokens from {current} to "
                    f"{self.GLM_MIN_MAX_TOKENS} (thinking budget floor)"
                )
            return (
                f"glm: set max_tokens to {self.GLM_MIN_MAX_TOKENS} "
                f"(thinking budget floor)"
            )
        return None

    def _transform_opengateway_compression(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Disable compression for Gitlawb Opengateway / Xiaomi-Mimo.
        The server sends 'content-encoding: gzip' but the body is actually plain
        text, which causes 'incorrect header check' decoding errors in httpx.
        """
        # Apply to xiaomi_mimo known provider or any provider name containing gitlawb
        is_gitlawb = "gitlawb" in provider.lower() or provider == "xiaomi_mimo"
        if not is_gitlawb:
            return None

        headers = kwargs.get("headers", {})
        # Force identity encoding to prevent httpx from attempting decompression
        headers["Accept-Encoding"] = "identity"
        kwargs["headers"] = headers
        return f"{provider}: disabled compression (identity)"

    # =========================================================================
    # SAFETY SETTINGS CONVERSION (REMOVED)
    # =========================================================================
    # Previously had convert_safety_settings() wrapper that delegated to
    # provider plugins. Removed because auto-injecting/merging safety defaults
    # caused 400 errors on models that don't support those categories (e.g. Gemma).
    # See gemini_provider.py for the full removal comment with previous defaults.
    # =========================================================================
