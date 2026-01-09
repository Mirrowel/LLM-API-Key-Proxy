import asyncio
import fnmatch
import json
import re
import codecs
import time
import os
import random
import httpx
import litellm
from litellm.exceptions import APIConnectionError
from litellm.litellm_core_utils.token_counter import token_counter
import logging
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

lib_logger = logging.getLogger("rotator_library")
# Ensure the logger is configured to propagate to the root logger
# which is set up in main.py. This allows the main app to control
# log levels and handlers centrally.
lib_logger.propagate = False

from .usage_manager import UsageManager
from .failure_logger import log_failure, configure_failure_logger
from .error_handler import (
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    classify_error,
    AllProviders,
    NoAvailableKeysError,
    should_rotate_on_error,
    should_retry_same_key,
    RequestErrorAccumulator,
    mask_credential,
)
from .providers import PROVIDER_PLUGINS
from .providers.openai_compatible_provider import OpenAICompatibleProvider
from .request_sanitizer import sanitize_request_payload
from .cooldown_manager import CooldownManager
from .credential_manager import CredentialManager
from .background_refresher import BackgroundRefresher
from .model_definitions import ModelDefinitions
from .transaction_logger import TransactionLogger
from .utils.paths import get_default_root, get_logs_dir, get_oauth_dir, get_data_file


class StreamedAPIError(Exception):
    """Custom exception to signal an API error received over a stream."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data


class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, List[str]]] = None,
        oauth_credentials: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 2,
        usage_file_path: Optional[Union[str, Path]] = None,
        configure_logging: bool = True,
        global_timeout: int = 30,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
        rotation_tolerance: float = 3.0,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the RotatingClient with intelligent credential rotation.

        Args:
            api_keys: Dictionary mapping provider names to lists of API keys
            oauth_credentials: Dictionary mapping provider names to OAuth credential paths
            max_retries: Maximum number of retry attempts per credential
            usage_file_path: Path to store usage statistics. If None, uses data_dir/key_usage.json
            configure_logging: Whether to configure library logging
            global_timeout: Global timeout for requests in seconds
            abort_on_callback_error: Whether to abort on pre-request callback errors
            litellm_provider_params: Provider-specific parameters for LiteLLM
            ignore_models: Models to ignore/blacklist per provider
            whitelist_models: Models to explicitly whitelist per provider
            enable_request_logging: Whether to enable detailed request logging
            max_concurrent_requests_per_key: Max concurrent requests per key by provider
            rotation_tolerance: Tolerance for weighted random credential rotation.
                - 0.0: Deterministic, least-used credential always selected
                - 2.0 - 4.0 (default, recommended): Balanced randomness, can pick credentials within 2 uses of max
                - 5.0+: High randomness, more unpredictable selection patterns
            data_dir: Root directory for all data files (logs, cache, oauth_creds, key_usage.json).
                      If None, auto-detects: EXE directory if frozen, else current working directory.
        """
        # Resolve data_dir early - this becomes the root for all file operations
        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = get_default_root()

        # Configure failure logger to use correct logs directory
        configure_failure_logger(get_logs_dir(self.data_dir))

        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        litellm.drop_params = True
        if configure_logging:
            # When True, this allows logs from this library to be handled
            # by the parent application's logging configuration.
            lib_logger.propagate = True
            # Remove any default handlers to prevent duplicate logging
            if lib_logger.hasHandlers():
                lib_logger.handlers.clear()
                lib_logger.addHandler(logging.NullHandler())
        else:
            lib_logger.propagate = False

        api_keys = api_keys or {}
        oauth_credentials = oauth_credentials or {}

        # Filter out providers with empty lists of credentials to ensure validity
        api_keys = {provider: keys for provider, keys in api_keys.items() if keys}
        oauth_credentials = {
            provider: paths for provider, paths in oauth_credentials.items() if paths
        }

        if not api_keys and not oauth_credentials:
            lib_logger.warning(
                "No provider credentials configured. The client will be unable to make any API requests."
            )

        self.api_keys = api_keys
        # Use provided oauth_credentials directly if available (already discovered by main.py)
        # Only call discover_and_prepare() if no credentials were passed
        if oauth_credentials:
            self.oauth_credentials = oauth_credentials
        else:
            self.credential_manager = CredentialManager(
                os.environ, oauth_dir=get_oauth_dir(self.data_dir)
            )
            self.oauth_credentials = self.credential_manager.discover_and_prepare()
        self.background_refresher = BackgroundRefresher(self)
        self.oauth_providers = set(self.oauth_credentials.keys())

        all_credentials = {}
        for provider, keys in api_keys.items():
            all_credentials.setdefault(provider, []).extend(keys)
        for provider, paths in self.oauth_credentials.items():
            all_credentials.setdefault(provider, []).extend(paths)
        self.all_credentials = all_credentials

        self.max_retries = max_retries
        self.global_timeout = global_timeout
        self.abort_on_callback_error = abort_on_callback_error

        # Initialize provider plugins early so they can be used for rotation mode detection
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = {}

        # Build provider rotation modes map
        # Each provider can specify its preferred rotation mode ("balanced" or "sequential")
        provider_rotation_modes = {}
        for provider in self.all_credentials.keys():
            provider_class = self._provider_plugins.get(provider)
            if provider_class and hasattr(provider_class, "get_rotation_mode"):
                # Use class method to get rotation mode (checks env var + class default)
                mode = provider_class.get_rotation_mode(provider)
            else:
                # Fallback: check environment variable directly
                env_key = f"ROTATION_MODE_{provider.upper()}"
                mode = os.getenv(env_key, "balanced")

            provider_rotation_modes[provider] = mode
            if mode != "balanced":
                lib_logger.info(f"Provider '{provider}' using rotation mode: {mode}")

        # Build priority-based concurrency multiplier maps
        # These are universal multipliers based on credential tier/priority
        priority_multipliers: Dict[str, Dict[int, int]] = {}
        priority_multipliers_by_mode: Dict[str, Dict[str, Dict[int, int]]] = {}
        sequential_fallback_multipliers: Dict[str, int] = {}

        for provider in self.all_credentials.keys():
            provider_class = self._provider_plugins.get(provider)

            # Start with provider class defaults
            if provider_class:
                # Get default priority multipliers from provider class
                if hasattr(provider_class, "default_priority_multipliers"):
                    default_multipliers = provider_class.default_priority_multipliers
                    if default_multipliers:
                        priority_multipliers[provider] = dict(default_multipliers)

                # Get sequential fallback from provider class
                if hasattr(provider_class, "default_sequential_fallback_multiplier"):
                    fallback = provider_class.default_sequential_fallback_multiplier
                    if fallback != 1:  # Only store if different from global default
                        sequential_fallback_multipliers[provider] = fallback

            # Override with environment variables
            # Format: CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>=<multiplier>
            # Format: CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>_<MODE>=<multiplier>
            for key, value in os.environ.items():
                prefix = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_"
                if key.startswith(prefix):
                    remainder = key[len(prefix) :]
                    try:
                        multiplier = int(value)
                        if multiplier < 1:
                            lib_logger.warning(f"Invalid {key}: {value}. Must be >= 1.")
                            continue

                        # Check if mode-specific (e.g., _PRIORITY_1_SEQUENTIAL)
                        if "_" in remainder:
                            parts = remainder.rsplit("_", 1)
                            priority = int(parts[0])
                            mode = parts[1].lower()
                            if mode in ("sequential", "balanced"):
                                # Mode-specific override
                                if provider not in priority_multipliers_by_mode:
                                    priority_multipliers_by_mode[provider] = {}
                                if mode not in priority_multipliers_by_mode[provider]:
                                    priority_multipliers_by_mode[provider][mode] = {}
                                priority_multipliers_by_mode[provider][mode][
                                    priority
                                ] = multiplier
                                lib_logger.info(
                                    f"Provider '{provider}' priority {priority} ({mode} mode) multiplier: {multiplier}x"
                                )
                            else:
                                # Assume it's part of the priority number (unlikely but handle gracefully)
                                lib_logger.warning(f"Unknown mode in {key}: {mode}")
                        else:
                            # Universal priority multiplier
                            priority = int(remainder)
                            if provider not in priority_multipliers:
                                priority_multipliers[provider] = {}
                            priority_multipliers[provider][priority] = multiplier
                            lib_logger.info(
                                f"Provider '{provider}' priority {priority} multiplier: {multiplier}x"
                            )
                    except ValueError:
                        lib_logger.warning(
                            f"Invalid {key}: {value}. Could not parse priority or multiplier."
                        )

        # Log configured multipliers
        for provider, multipliers in priority_multipliers.items():
            if multipliers:
                lib_logger.info(
                    f"Provider '{provider}' priority multipliers: {multipliers}"
                )
        for provider, fallback in sequential_fallback_multipliers.items():
            lib_logger.info(
                f"Provider '{provider}' sequential fallback multiplier: {fallback}x"
            )

        # Resolve usage file path - use provided path or default to data_dir
        if usage_file_path is not None:
            resolved_usage_path = Path(usage_file_path)
        else:
            resolved_usage_path = self.data_dir / "key_usage.json"

        self.usage_manager = UsageManager(
            file_path=resolved_usage_path,
            rotation_tolerance=rotation_tolerance,
            provider_rotation_modes=provider_rotation_modes,
            provider_plugins=PROVIDER_PLUGINS,
            priority_multipliers=priority_multipliers,
            priority_multipliers_by_mode=priority_multipliers_by_mode,
            sequential_fallback_multipliers=sequential_fallback_multipliers,
        )
        self._model_list_cache = {}
        self.http_client = httpx.AsyncClient()
        self.all_providers = AllProviders()
        self.cooldown_manager = CooldownManager()
        self.litellm_provider_params = litellm_provider_params or {}
        self.ignore_models = ignore_models or {}
        self.whitelist_models = whitelist_models or {}
        self.enable_request_logging = enable_request_logging
        self.model_definitions = ModelDefinitions()

        # Store and validate max concurrent requests per key
        self.max_concurrent_requests_per_key = max_concurrent_requests_per_key or {}
        # Validate all values are >= 1
        for provider, max_val in self.max_concurrent_requests_per_key.items():
            if max_val < 1:
                lib_logger.warning(
                    f"Invalid max_concurrent for '{provider}': {max_val}. Setting to 1."
                )
                self.max_concurrent_requests_per_key[provider] = 1

    def _is_model_ignored(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model should be ignored based on the ignore list.
        Supports full glob/fnmatch patterns for both full model IDs and model names.

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard (matches gpt-4-preview, o1-preview, etc.)
        - "*-preview*" - contains wildcard (matches anything with -preview)
        - "*" - match all
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in self.ignore_models:
            return False

        ignore_list = self.ignore_models[model_provider]
        if ignore_list == ["*"]:
            return True

        try:
            # This is the model name as the provider sees it (e.g., "gpt-4" or "google/gemma-7b")
            provider_model_name = model_id.split("/", 1)[1]
        except IndexError:
            provider_model_name = model_id

        for ignored_pattern in ignore_list:
            # Use fnmatch for full glob pattern support
            if fnmatch.fnmatch(provider_model_name, ignored_pattern) or fnmatch.fnmatch(
                model_id, ignored_pattern
            ):
                return True
        return False

    def _is_model_whitelisted(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model is explicitly whitelisted.
        Supports full glob/fnmatch patterns for both full model IDs and model names.

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard (matches gpt-4-preview, o1-preview, etc.)
        - "*-preview*" - contains wildcard (matches anything with -preview)
        - "*" - match all
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in self.whitelist_models:
            return False

        whitelist = self.whitelist_models[model_provider]

        try:
            # This is the model name as the provider sees it (e.g., "gpt-4" or "google/gemma-7b")
            provider_model_name = model_id.split("/", 1)[1]
        except IndexError:
            provider_model_name = model_id

        for whitelisted_pattern in whitelist:
            # Use fnmatch for full glob pattern support
            if fnmatch.fnmatch(
                provider_model_name, whitelisted_pattern
            ) or fnmatch.fnmatch(model_id, whitelisted_pattern):
                return True
        return False

    def _sanitize_litellm_log(self, log_data: dict) -> dict:
        """
        Recursively removes large data fields and sensitive information from litellm log
        dictionaries to keep debug logs clean and secure.
        """
        if not isinstance(log_data, dict):
            return log_data

        # Keys to remove at any level of the dictionary
        keys_to_pop = [
            "messages",
            "input",
            "response",
            "data",
            "api_key",
            "api_base",
            "original_response",
            "additional_args",
        ]

        # Keys that might contain nested dictionaries to clean
        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]

        # Create a deep copy to avoid modifying the original log object in memory
        clean_data = json.loads(json.dumps(log_data, default=str))

        def clean_recursively(data_dict):
            if not isinstance(data_dict, dict):
                return

            # Remove sensitive/large keys
            for key in keys_to_pop:
                data_dict.pop(key, None)

            # Recursively clean nested dictionaries
            for key in nested_keys:
                if key in data_dict and isinstance(data_dict[key], dict):
                    clean_recursively(data_dict[key])

            # Also iterate through all values to find any other nested dicts
            for key, value in list(data_dict.items()):
                if isinstance(value, dict):
                    clean_recursively(value)

        clean_recursively(clean_data)
        return clean_data

    def _litellm_logger_callback(self, log_data: dict):
        """
        Callback function to redirect litellm's logs to the library's logger.
        This allows us to control the log level and destination of litellm's output.
        It also cleans up error logs for better readability in debug files.
        """
        # Filter out verbose pre_api_call and post_api_call logs
        log_event_type = log_data.get("log_event_type")
        if log_event_type in ["pre_api_call", "post_api_call"]:
            return  # Skip these verbose logs entirely

        # For successful calls or pre-call logs, a simple debug message is enough.
        if not log_data.get("exception"):
            sanitized_log = self._sanitize_litellm_log(log_data)
            # We log it at the DEBUG level to ensure it goes to the debug file
            # and not the console, based on the main.py configuration.
            lib_logger.debug(f"LiteLLM Log: {sanitized_log}")
            return

        # For failures, extract key info to make debug logs more readable.
        model = log_data.get("model", "N/A")
        call_id = log_data.get("litellm_call_id", "N/A")
        error_info = log_data.get("standard_logging_object", {}).get(
            "error_information", {}
        )
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get(
            "error_message", str(log_data.get("exception", ""))
        )
        error_message = " ".join(error_message.split())  # Sanitize

        lib_logger.debug(
            f"LiteLLM Callback Handled Error: Model={model} | "
            f"Type={error_class} | Message='{error_message}'"
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client to prevent resource leaks."""
        if hasattr(self, "http_client") and self.http_client:
            await self.http_client.aclose()

    def _convert_model_params(self, **kwargs) -> Dict[str, Any]:
        """
        Converts model parameters for specific providers.
        For example, the 'chutes' provider requires the model name to be prepended
        with 'openai/' and a specific 'api_base'.
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        provider = model.split("/")[0]
        if provider == "chutes":
            kwargs["model"] = f"openai/{model.split('/', 1)[1]}"
            kwargs["api_base"] = "https://llm.chutes.ai/v1"

        return kwargs

    def _convert_model_params_for_litellm(self, **kwargs) -> Dict[str, Any]:
        """
        Converts model parameters specifically for LiteLLM calls.
        This is called right before calling LiteLLM to handle custom providers.
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        provider = model.split("/")[0]

        # Handle custom OpenAI-compatible providers
        # Check if this is a custom provider by looking for API_BASE environment variable
        import os

        api_base_env = f"{provider.upper()}_API_BASE"
        if os.getenv(api_base_env):
            # For custom providers, tell LiteLLM to use openai provider with custom model name
            # This preserves original model name in logs but converts for LiteLLM
            kwargs = kwargs.copy()  # Don't modify original
            kwargs["model"] = f"openai/{model.split('/', 1)[1]}"
            kwargs["api_base"] = os.getenv(api_base_env).rstrip("/")
            kwargs["custom_llm_provider"] = "openai"

        return kwargs

    def _apply_default_safety_settings(
        self, litellm_kwargs: Dict[str, Any], provider: str
    ):
        """
        Ensure default Gemini safety settings are present when calling the Gemini provider.
        This will not override any explicit settings provided by the request. It accepts
        either OpenAI-compatible generic `safety_settings` (dict) or direct Gemini-style
        `safetySettings` (list of dicts). Missing categories will be added with safe defaults.
        """
        if provider != "gemini":
            return

        # Generic defaults (openai-compatible style)
        default_generic = {
            "harassment": "OFF",
            "hate_speech": "OFF",
            "sexually_explicit": "OFF",
            "dangerous_content": "OFF",
            "civic_integrity": "BLOCK_NONE",
        }

        # Gemini defaults (direct Gemini format)
        default_gemini = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]

        # If generic form is present, ensure missing generic keys are filled in
        if "safety_settings" in litellm_kwargs and isinstance(
            litellm_kwargs["safety_settings"], dict
        ):
            for k, v in default_generic.items():
                if k not in litellm_kwargs["safety_settings"]:
                    litellm_kwargs["safety_settings"][k] = v
            return

        # If Gemini form is present, ensure missing gemini categories are appended
        if "safetySettings" in litellm_kwargs and isinstance(
            litellm_kwargs["safetySettings"], list
        ):
            present = {
                item.get("category")
                for item in litellm_kwargs["safetySettings"]
                if isinstance(item, dict)
            }
            for d in default_gemini:
                if d["category"] not in present:
                    litellm_kwargs["safetySettings"].append(d)
            return

        # Neither present: set generic defaults so provider conversion will translate them
        if (
            "safety_settings" not in litellm_kwargs
            and "safetySettings" not in litellm_kwargs
        ):
            litellm_kwargs["safety_settings"] = default_generic.copy()

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        return self.oauth_credentials

    def _is_custom_openai_compatible_provider(self, provider_name: str) -> bool:
        """Checks if a provider is a custom OpenAI-compatible provider."""
        import os

        # Check if the provider has an API_BASE environment variable
        api_base_env = f"{provider_name.upper()}_API_BASE"
        return os.getenv(api_base_env) is not None

    def _get_provider_instance(self, provider_name: str):
        """
        Lazily initializes and returns a provider instance.
        Only initializes providers that have configured credentials.

        Args:
            provider_name: The name of the provider to get an instance for.
                          For OAuth providers, this may include "_oauth" suffix
                          (e.g., "antigravity_oauth"), but credentials are stored
                          under the base name (e.g., "antigravity").

        Returns:
            Provider instance if credentials exist, None otherwise.
        """
        # For OAuth providers, credentials are stored under base name (without _oauth suffix)
        # e.g., "antigravity_oauth" plugin → credentials under "antigravity"
        credential_key = provider_name
        if provider_name.endswith("_oauth"):
            base_name = provider_name[:-6]  # Remove "_oauth"
            if base_name in self.oauth_providers:
                credential_key = base_name

        # Only initialize providers for which we have credentials
        if credential_key not in self.all_credentials:
            lib_logger.debug(
                f"Skipping provider '{provider_name}' initialization: no credentials configured"
            )
            return None

        if provider_name not in self._provider_instances:
            if provider_name in self._provider_plugins:
                self._provider_instances[provider_name] = self._provider_plugins[
                    provider_name
                ]()
            elif self._is_custom_openai_compatible_provider(provider_name):
                # Create a generic OpenAI-compatible provider for custom providers
                try:
                    self._provider_instances[provider_name] = OpenAICompatibleProvider(
                        provider_name
                    )
                except ValueError:
                    # If the provider doesn't have the required environment variables, treat it as a standard provider
                    return None
            else:
                return None
        return self._provider_instances[provider_name]

    def _resolve_model_id(self, model: str, provider: str) -> str:
        """
        Resolves the actual model ID to send to the provider.

        For custom models with name/ID mappings, returns the ID.
        Otherwise, returns the model name unchanged.

        Args:
            model: Full model string with provider (e.g., "iflow/DS-v3.2")
            provider: Provider name (e.g., "iflow")

        Returns:
            Full model string with ID (e.g., "iflow/deepseek-v3.2")
        """
        # Extract model name from "provider/model_name" format
        model_name = model.split("/")[-1] if "/" in model else model

        # Try to get provider instance to check for model definitions
        provider_plugin = self._get_provider_instance(provider)

        # Check if provider has model definitions
        if provider_plugin and hasattr(provider_plugin, "model_definitions"):
            model_id = provider_plugin.model_definitions.get_model_id(
                provider, model_name
            )
            if model_id and model_id != model_name:
                # Return with provider prefix
                return f"{provider}/{model_id}"

        # Fallback: use client's own model definitions
        model_id = self.model_definitions.get_model_id(provider, model_name)
        if model_id and model_id != model_name:
            return f"{provider}/{model_id}"

        # No conversion needed, return original
        return model

    async def _safe_streaming_wrapper(
        self,
        stream: Any,
        key: str,
        model: str,
        request: Optional[Any] = None,
        provider_plugin: Optional[Any] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        A hybrid wrapper for streaming that buffers fragmented JSON, handles client disconnections gracefully,
        and distinguishes between content and streamed errors.

        FINISH_REASON HANDLING:
        Providers just translate chunks - this wrapper handles ALL finish_reason logic:
        1. Strip finish_reason from intermediate chunks (litellm defaults to "stop")
        2. Track accumulated_finish_reason with priority: tool_calls > length/content_filter > stop
        3. Only emit finish_reason on final chunk (detected by usage.completion_tokens > 0)
        """
        last_usage = None
        stream_completed = False
        stream_iterator = stream.__aiter__()
        json_buffer = ""
        accumulated_finish_reason = None  # Track strongest finish_reason across chunks
        has_tool_calls = False  # Track if ANY tool calls were seen in stream

        try:
            while True:
                if request and await request.is_disconnected():
                    lib_logger.info(
                        f"Client disconnected. Aborting stream for credential {mask_credential(key)}."
                    )
                    break

                try:
                    chunk = await stream_iterator.__anext__()
                    if json_buffer:
                        lib_logger.warning(
                            f"Discarding incomplete JSON buffer from previous chunk: {json_buffer}"
                        )
                        json_buffer = ""

                    # Convert chunk to dict, handling both litellm.ModelResponse and raw dicts
                    if hasattr(chunk, "dict"):
                        chunk_dict = chunk.dict()
                    elif hasattr(chunk, "model_dump"):
                        chunk_dict = chunk.model_dump()
                    else:
                        chunk_dict = chunk

                    # === FINISH_REASON LOGIC ===
                    # Providers send raw chunks without finish_reason logic.
                    # This wrapper determines finish_reason based on accumulated state.
                    if "choices" in chunk_dict and chunk_dict["choices"]:
                        choice = chunk_dict["choices"][0]
                        delta = choice.get("delta", {})
                        usage = chunk_dict.get("usage", {})

                        # Track tool_calls across ALL chunks - if we ever see one, finish_reason must be tool_calls
                        if delta.get("tool_calls"):
                            has_tool_calls = True
                            accumulated_finish_reason = "tool_calls"

                        # Detect final chunk: has usage with completion_tokens > 0
                        has_completion_tokens = (
                            usage
                            and isinstance(usage, dict)
                            and usage.get("completion_tokens", 0) > 0
                        )

                        if has_completion_tokens:
                            # FINAL CHUNK: Determine correct finish_reason
                            if has_tool_calls:
                                # Tool calls always win
                                choice["finish_reason"] = "tool_calls"
                            elif accumulated_finish_reason:
                                # Use accumulated reason (length, content_filter, etc.)
                                choice["finish_reason"] = accumulated_finish_reason
                            else:
                                # Default to stop
                                choice["finish_reason"] = "stop"
                        else:
                            # INTERMEDIATE CHUNK: Never emit finish_reason
                            # (litellm.ModelResponse defaults to "stop" which is wrong)
                            choice["finish_reason"] = None

                    yield f"data: {json.dumps(chunk_dict)}\n\n"

                    if hasattr(chunk, "usage") and chunk.usage:
                        last_usage = chunk.usage

                except StopAsyncIteration:
                    stream_completed = True
                    if json_buffer:
                        lib_logger.info(
                            f"Stream ended with incomplete data in buffer: {json_buffer}"
                        )
                    if last_usage:
                        # Create a dummy ModelResponse for recording (only usage matters)
                        dummy_response = litellm.ModelResponse(usage=last_usage)
                        await self.usage_manager.record_success(
                            key, model, dummy_response
                        )
                    else:
                        # If no usage seen (rare), record success without tokens/cost
                        await self.usage_manager.record_success(key, model)

                    break

                except CredentialNeedsReauthError as e:
                    # This credential needs re-authentication but re-auth is already queued.
                    # Wrap it so the outer retry loop can rotate to the next credential.
                    # No scary traceback needed - this is an expected recovery scenario.
                    raise StreamedAPIError("Credential needs re-authentication", data=e)

                except (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                    litellm.InternalServerError,
                    APIConnectionError,
                    httpx.HTTPStatusError,
                ) as e:
                    # This is a critical, typed error from litellm or httpx that signals a key failure.
                    # We do not try to parse it here. We wrap it and raise it immediately
                    # for the outer retry loop to handle.
                    lib_logger.warning(
                        f"Caught a critical API error mid-stream: {type(e).__name__}. Signaling for credential rotation."
                    )
                    raise StreamedAPIError("Provider error received in stream", data=e)

                except Exception as e:
                    try:
                        raw_chunk = ""
                        # Google streams errors inside a bytes representation (b'{...}').
                        # We use regex to extract the content, which is more reliable than splitting.
                        match = re.search(r"b'(\{.*\})'", str(e), re.DOTALL)
                        if match:
                            # The extracted string is unicode-escaped (e.g., '\\n'). We must decode it.
                            raw_chunk = codecs.decode(match.group(1), "unicode_escape")
                        else:
                            # Fallback for other potential error formats that use "Received chunk:".
                            chunk_from_split = (
                                str(e).split("Received chunk:")[-1].strip()
                            )
                            if chunk_from_split != str(
                                e
                            ):  # Ensure the split actually did something
                                raw_chunk = chunk_from_split

                        if not raw_chunk:
                            # If we could not extract a valid chunk, we cannot proceed with reassembly.
                            # This indicates a different, unexpected error type. Re-raise it.
                            raise e

                        # Append the clean chunk to the buffer and try to parse.
                        json_buffer += raw_chunk
                        parsed_data = json.loads(json_buffer)

                        # If parsing succeeds, we have the complete object.
                        lib_logger.info(
                            f"Successfully reassembled JSON from stream: {json_buffer}"
                        )

                        # Wrap the complete error object and raise it. The outer function will decide how to handle it.
                        raise StreamedAPIError(
                            "Provider error received in stream", data=parsed_data
                        )

                    except json.JSONDecodeError:
                        # This is the expected outcome if the JSON in the buffer is not yet complete.
                        lib_logger.info(
                            f"Buffer still incomplete. Waiting for more chunks: {json_buffer}"
                        )
                        continue  # Continue to the next loop to get the next chunk.
                    except StreamedAPIError:
                        # Re-raise to be caught by the outer retry handler.
                        raise
                    except Exception as buffer_exc:
                        # If the error was not a JSONDecodeError, it's an unexpected internal error.
                        lib_logger.error(
                            f"Error during stream buffering logic: {buffer_exc}. Discarding buffer."
                        )
                        json_buffer = (
                            ""  # Clear the corrupted buffer to prevent further issues.
                        )
                        raise buffer_exc

        except StreamedAPIError:
            # This is caught by the acompletion retry logic.
            # We re-raise it to ensure it's not caught by the generic 'except Exception'.
            raise

        except Exception as e:
            # Catch any other unexpected errors during streaming.
            lib_logger.error(f"Caught unexpected exception of type: {type(e).__name__}")
            lib_logger.error(
                f"An unexpected error occurred during the stream for credential {mask_credential(key)}: {e}"
            )
            # We still need to raise it so the client knows something went wrong.
            raise

        finally:
            # This block now runs regardless of how the stream terminates (completion, client disconnect, etc.).
            # The primary goal is to ensure usage is always logged internally.
            await self.usage_manager.release_key(key, model)
            lib_logger.info(
                f"STREAM FINISHED and lock released for credential {mask_credential(key)}."
            )

            # Only send [DONE] if the stream completed naturally and the client is still there.
            # This prevents sending [DONE] to a disconnected client or after an error.
            if stream_completed and (
                not request or not await request.is_disconnected()
            ):
                yield "data: [DONE]\n\n"

    async def _transaction_logging_stream_wrapper(
        self,
        stream: Any,
        transaction_logger: Optional[TransactionLogger],
        request_data: Dict[str, Any],
    ) -> Any:
        """
        Wrap a stream to log chunks and final response to TransactionLogger.

        This wrapper:
        1. Yields chunks unchanged (passthrough)
        2. Parses SSE chunks and logs them via transaction_logger.log_stream_chunk()
        3. Collects chunks for final response assembly
        4. After stream ends, assembles and logs final response

        Args:
            stream: The streaming generator (yields SSE strings like "data: {...}")
            transaction_logger: Optional TransactionLogger instance
            request_data: Original request data for context
        """
        chunks = []
        try:
            async for chunk_str in stream:
                yield chunk_str

                # Log chunk if logging enabled
                if (
                    transaction_logger
                    and isinstance(chunk_str, str)
                    and chunk_str.strip()
                    and chunk_str.startswith("data:")
                ):
                    content = chunk_str[len("data:") :].strip()
                    if content and content != "[DONE]":
                        try:
                            chunk_data = json.loads(content)
                            chunks.append(chunk_data)
                            transaction_logger.log_stream_chunk(chunk_data)
                        except json.JSONDecodeError:
                            lib_logger.warning(
                                f"TransactionLogger: Failed to parse chunk: {content[:100]}"
                            )
        finally:
            # Assemble and log final response after stream ends
            if transaction_logger and chunks:
                try:
                    final_response = TransactionLogger.assemble_streaming_response(
                        chunks, request_data
                    )
                    transaction_logger.log_response(final_response)
                except Exception as e:
                    lib_logger.warning(
                        f"TransactionLogger: Failed to assemble/log final response: {e}"
                    )

    async def _execute_with_retry(
        self,
        api_call: callable,
        request: Optional[Any],
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """A generic retry mechanism for non-streaming API calls."""
        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' is a required parameter.")

        provider = model.split("/")[0]
        if provider not in self.all_credentials:
            raise ValueError(
                f"No API keys or OAuth credentials configured for provider: {provider}"
            )

        # Establish a global deadline for the entire request lifecycle.
        deadline = time.time() + self.global_timeout

        # Create transaction logger if request logging is enabled
        transaction_logger = None
        if self.enable_request_logging:
            transaction_logger = TransactionLogger(provider, model, enabled=True)
            transaction_logger.log_request(kwargs)

        # Create a mutable copy of the keys and shuffle it to ensure
        # that the key selection is randomized, which is crucial when
        # multiple keys have the same usage stats.
        credentials_for_provider = list(self.all_credentials[provider])
        random.shuffle(credentials_for_provider)

        # Filter out credentials that are unavailable (queued for re-auth)
        provider_plugin = self._get_provider_instance(provider)
        if provider_plugin and hasattr(provider_plugin, "is_credential_available"):
            available_creds = [
                cred
                for cred in credentials_for_provider
                if provider_plugin.is_credential_available(cred)
            ]
            if available_creds:
                credentials_for_provider = available_creds
            # If all credentials are unavailable, keep the original list
            # (better to try unavailable creds than fail immediately)

        tried_creds = set()
        last_exception = None
        kwargs = self._convert_model_params(**kwargs)

        # The main rotation loop. It continues as long as there are untried credentials and the global deadline has not been exceeded.

        # Resolve model ID early, before any credential operations
        # This ensures consistent model ID usage for acquisition, release, and tracking
        resolved_model = self._resolve_model_id(model, provider)
        if resolved_model != model:
            lib_logger.info(f"Resolved model '{model}' to '{resolved_model}'")
            model = resolved_model
            kwargs["model"] = model  # Ensure kwargs has the resolved model for litellm

        # [NEW] Filter by model tier requirement and build priority map
        credential_priorities = None
        if provider_plugin and hasattr(provider_plugin, "get_model_tier_requirement"):
            required_tier = provider_plugin.get_model_tier_requirement(model)
            if required_tier is not None:
                # Filter OUT only credentials we KNOW are too low priority
                # Keep credentials with unknown priority (None) - they might be high priority
                incompatible_creds = []
                compatible_creds = []
                unknown_creds = []

                for cred in credentials_for_provider:
                    if hasattr(provider_plugin, "get_credential_priority"):
                        priority = provider_plugin.get_credential_priority(cred)
                        if priority is None:
                            # Unknown priority - keep it, will be discovered on first use
                            unknown_creds.append(cred)
                        elif priority <= required_tier:
                            # Known compatible priority
                            compatible_creds.append(cred)
                        else:
                            # Known incompatible priority (too low)
                            incompatible_creds.append(cred)
                    else:
                        # Provider doesn't support priorities - keep all
                        unknown_creds.append(cred)

                # If we have any known-compatible or unknown credentials, use them
                tier_compatible_creds = compatible_creds + unknown_creds
                if tier_compatible_creds:
                    credentials_for_provider = tier_compatible_creds
                    if compatible_creds and unknown_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible + {len(unknown_creds)} unknown-tier credentials."
                        )
                    elif compatible_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible credentials."
                        )
                    else:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(unknown_creds)} unknown-tier credentials (will discover on use)."
                        )
                elif incompatible_creds:
                    # Only known-incompatible credentials remain
                    lib_logger.warning(
                        f"Model {model} requires priority <= {required_tier} credentials, "
                        f"but all {len(incompatible_creds)} known credentials have priority > {required_tier}. "
                        f"Request will likely fail."
                    )

        # Build priority map and tier names map for usage_manager
        credential_tier_names = None
        if provider_plugin and hasattr(provider_plugin, "get_credential_priority"):
            credential_priorities = {}
            credential_tier_names = {}
            for cred in credentials_for_provider:
                priority = provider_plugin.get_credential_priority(cred)
                if priority is not None:
                    credential_priorities[cred] = priority
                # Also get tier name for logging
                if hasattr(provider_plugin, "get_credential_tier_name"):
                    tier_name = provider_plugin.get_credential_tier_name(cred)
                    if tier_name:
                        credential_tier_names[cred] = tier_name

            if credential_priorities:
                lib_logger.debug(
                    f"Credential priorities for {provider}: {', '.join(f'P{p}={len([c for c in credentials_for_provider if credential_priorities.get(c) == p])}' for p in sorted(set(credential_priorities.values())))}"
                )

        # Initialize error accumulator for tracking errors across credential rotation
        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        while (
            len(tried_creds) < len(credentials_for_provider) and time.time() < deadline
        ):
            current_cred = None
            key_acquired = False
            try:
                # Check for a provider-wide cooldown first.
                if await self.cooldown_manager.is_cooling_down(provider):
                    remaining_cooldown = (
                        await self.cooldown_manager.get_cooldown_remaining(provider)
                    )
                    remaining_budget = deadline - time.time()

                    # If the cooldown is longer than the remaining time budget, fail fast.
                    if remaining_cooldown > remaining_budget:
                        lib_logger.warning(
                            f"Provider {provider} cooldown ({remaining_cooldown:.2f}s) exceeds remaining request budget ({remaining_budget:.2f}s). Failing early."
                        )
                        break

                    lib_logger.warning(
                        f"Provider {provider} is in cooldown. Waiting for {remaining_cooldown:.2f} seconds."
                    )
                    await asyncio.sleep(remaining_cooldown)

                creds_to_try = [
                    c for c in credentials_for_provider if c not in tried_creds
                ]
                if not creds_to_try:
                    break

                # Get count of credentials not on cooldown for this model
                available_creds = (
                    await self.usage_manager.get_available_credentials_for_model(
                        creds_to_try, model
                    )
                )
                available_count = len(available_creds)
                total_count = len(credentials_for_provider)

                lib_logger.info(
                    f"Acquiring key for model {model}. Tried keys: {len(tried_creds)}/{available_count}({total_count})"
                )
                max_concurrent = self.max_concurrent_requests_per_key.get(provider, 1)
                current_cred = await self.usage_manager.acquire_key(
                    available_keys=creds_to_try,
                    model=model,
                    deadline=deadline,
                    max_concurrent=max_concurrent,
                    credential_priorities=credential_priorities,
                    credential_tier_names=credential_tier_names,
                )
                key_acquired = True
                tried_creds.add(current_cred)

                litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())

                # [NEW] Merge provider-specific params
                if provider in self.litellm_provider_params:
                    litellm_kwargs["litellm_params"] = {
                        **self.litellm_provider_params[provider],
                        **litellm_kwargs.get("litellm_params", {}),
                    }

                provider_plugin = self._get_provider_instance(provider)

                # Model ID is already resolved before the loop, and kwargs['model'] is updated.
                # No further resolution needed here.

                # Apply model-specific options for custom providers
                if provider_plugin and hasattr(provider_plugin, "get_model_options"):
                    model_options = provider_plugin.get_model_options(model)
                    if model_options:
                        # Merge model options into litellm_kwargs
                        for key, value in model_options.items():
                            if key == "reasoning_effort":
                                litellm_kwargs["reasoning_effort"] = value
                            elif key not in litellm_kwargs:
                                litellm_kwargs[key] = value

                if provider_plugin and provider_plugin.has_custom_logic():
                    lib_logger.debug(
                        f"Provider '{provider}' has custom logic. Delegating call."
                    )
                    litellm_kwargs["credential_identifier"] = current_cred
                    litellm_kwargs["transaction_context"] = (
                        transaction_logger.get_context() if transaction_logger else None
                    )

                    # Retry loop for custom providers - mirrors streaming path error handling
                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting call with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            response = await provider_plugin.acompletion(
                                self.http_client, **litellm_kwargs
                            )

                            # For non-streaming, success is immediate
                            await self.usage_manager.record_success(
                                current_cred, model, response
                            )

                            await self.usage_manager.release_key(current_cred, model)
                            key_acquired = False

                            # Log response to transaction logger
                            if transaction_logger:
                                response_data = (
                                    response.model_dump()
                                    if hasattr(response, "model_dump")
                                    else response
                                )
                                transaction_logger.log_response(response_data)

                            return response

                        except (
                            litellm.RateLimitError,
                            httpx.HTTPStatusError,
                        ) as e:
                            last_exception = e
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}) during custom provider call. Failing."
                                )
                                raise last_exception

                            # Handle rate limits with cooldown (exclude quota_exceeded)
                            if classified_error.error_type == "rate_limit":
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.warning(
                                f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code}). Rotating."
                            )
                            break  # Rotate to next credential

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                        ) as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} failed after max retries. Rotating."
                                )
                                break

                            wait_time = classified_error.retry_after or (
                                2**attempt
                            ) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()
                            if wait_time > remaining_budget:
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Retry wait ({wait_time:.2f}s) exceeds budget. Rotating."
                                )
                                break

                            lib_logger.warning(
                                f"Cred {mask_credential(current_cred)} server error. Retrying in {wait_time:.2f}s."
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        except Exception as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Record in accumulator
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            lib_logger.warning(
                                f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code})."
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing."
                                )
                                raise last_exception

                            # Handle rate limits with cooldown (exclude quota_exceeded)
                            if (
                                classified_error.status_code == 429
                                and classified_error.error_type != "quota_exceeded"
                            ) or classified_error.error_type == "rate_limit":
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Rotate to next credential

                    # If the inner loop breaks, it means the key failed and we need to rotate.
                    # Continue to the next iteration of the outer while loop to pick a new key.
                    continue

                else:  # This is the standard API Key / litellm-handled provider logic
                    is_oauth = provider in self.oauth_providers
                    if is_oauth:  # Standard OAuth provider (not custom)
                        # ... (logic to set headers) ...
                        pass
                    else:  # API Key
                        litellm_kwargs["api_key"] = current_cred

                    provider_instance = self._get_provider_instance(provider)
                    if provider_instance:
                        # Ensure default Gemini safety settings are present (without overriding request)
                        try:
                            self._apply_default_safety_settings(
                                litellm_kwargs, provider
                            )
                        except Exception:
                            # If anything goes wrong here, avoid breaking the request flow.
                            lib_logger.debug(
                                "Could not apply default safety settings; continuing."
                            )

                        if "safety_settings" in litellm_kwargs:
                            converted_settings = (
                                provider_instance.convert_safety_settings(
                                    litellm_kwargs["safety_settings"]
                                )
                            )
                            if converted_settings is not None:
                                litellm_kwargs["safety_settings"] = converted_settings
                            else:
                                del litellm_kwargs["safety_settings"]

                    if provider == "gemini" and provider_instance:
                        provider_instance.handle_thinking_parameter(
                            litellm_kwargs, model
                        )
                    if provider == "nvidia_nim" and provider_instance:
                        provider_instance.handle_thinking_parameter(
                            litellm_kwargs, model
                        )

                    if "gemma-3" in model and "messages" in litellm_kwargs:
                        litellm_kwargs["messages"] = [
                            {"role": "user", "content": m["content"]}
                            if m.get("role") == "system"
                            else m
                            for m in litellm_kwargs["messages"]
                        ]

                    litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting call with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            # Convert model parameters for custom providers right before LiteLLM call
                            final_kwargs = self._convert_model_params_for_litellm(
                                **litellm_kwargs
                            )

                            response = await api_call(
                                **final_kwargs,
                                logger_fn=self._litellm_logger_callback,
                            )

                            await self.usage_manager.record_success(
                                current_cred, model, response
                            )

                            await self.usage_manager.release_key(current_cred, model)
                            key_acquired = False

                            # Log response to transaction logger
                            if transaction_logger:
                                response_data = (
                                    response.model_dump()
                                    if hasattr(response, "model_dump")
                                    else response
                                )
                                transaction_logger.log_response(response_data)

                            return response

                        except litellm.RateLimitError as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)

                            # Extract a clean error message for the user-facing log
                            error_message = str(e).split("\n")[0]

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            lib_logger.info(
                                f"Key {mask_credential(current_cred)} hit rate limit for {model}. Rotating key."
                            )

                            # Only trigger provider-wide cooldown for rate limits, not quota issues
                            if (
                                classified_error.status_code == 429
                                and classified_error.error_type != "quota_exceeded"
                            ):
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Move to the next key

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                        ) as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                # Record in accumulator only on final failure for this key
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Key {mask_credential(current_cred)} failed after max retries due to server error. Rotating."
                                )
                                break  # Move to the next key

                            # For temporary errors, wait before retrying with the same key.
                            wait_time = classified_error.retry_after or (
                                2**attempt
                            ) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()

                            # If the required wait time exceeds the budget, don't wait; rotate to the next key immediately.
                            if wait_time > remaining_budget:
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Retry wait ({wait_time:.2f}s) exceeds budget ({remaining_budget:.2f}s). Rotating key."
                                )
                                break

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} server error. Retrying in {wait_time:.2f}s."
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry with the same key

                        except httpx.HTTPStatusError as e:
                            # Handle HTTP errors from httpx (e.g., from custom providers like Antigravity)
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )

                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} HTTP {e.response.status_code} ({classified_error.error_type})."
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record in accumulator after confirming it's a rotatable error
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            # Handle rate limits with cooldown (exclude quota_exceeded from provider-wide cooldown)
                            if classified_error.error_type == "rate_limit":
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )

                            # Check if we should retry same key (server errors with retries left)
                            if (
                                should_retry_same_key(classified_error)
                                and attempt < self.max_retries - 1
                            ):
                                wait_time = classified_error.retry_after or (
                                    2**attempt
                                ) + random.uniform(0, 1)
                                remaining_budget = deadline - time.time()
                                if wait_time <= remaining_budget:
                                    lib_logger.warning(
                                        f"Server error, retrying same key in {wait_time:.2f}s."
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue

                            # Record failure and rotate to next key
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.info(
                                f"Rotating to next key after {classified_error.error_type} error."
                            )
                            break

                        except Exception as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )

                            if request and await request.is_disconnected():
                                lib_logger.warning(
                                    f"Client disconnected. Aborting retries for {mask_credential(current_cred)}."
                                )
                                raise last_exception

                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code})."
                            )

                            # Handle rate limits with cooldown (exclude quota_exceeded from provider-wide cooldown)
                            if (
                                classified_error.status_code == 429
                                and classified_error.error_type != "quota_exceeded"
                            ) or classified_error.error_type == "rate_limit":
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record in accumulator after confirming it's a rotatable error
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Try next key for other errors
            finally:
                if key_acquired and current_cred:
                    await self.usage_manager.release_key(current_cred, model)

        # Check if we exhausted all credentials or timed out
        if time.time() >= deadline:
            error_accumulator.timeout_occurred = True

        if error_accumulator.has_errors():
            # Log concise summary for server logs
            lib_logger.error(error_accumulator.build_log_message())

            # Return the structured error response for the client
            return error_accumulator.build_client_error_response()

        # Return None to indicate failure without error details (shouldn't normally happen)
        lib_logger.warning(
            "Unexpected state: request failed with no recorded errors. "
            "This may indicate a logic error in error tracking."
        )
        return None

    async def _streaming_acompletion_with_retry(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """A dedicated generator for retrying streaming completions with full request preparation and per-key retries."""
        model = kwargs.get("model")
        provider = model.split("/")[0]

        # Create a mutable copy of the keys and shuffle it.
        credentials_for_provider = list(self.all_credentials[provider])
        random.shuffle(credentials_for_provider)

        # Filter out credentials that are unavailable (queued for re-auth)
        provider_plugin = self._get_provider_instance(provider)
        if provider_plugin and hasattr(provider_plugin, "is_credential_available"):
            available_creds = [
                cred
                for cred in credentials_for_provider
                if provider_plugin.is_credential_available(cred)
            ]
            if available_creds:
                credentials_for_provider = available_creds
            # If all credentials are unavailable, keep the original list
            # (better to try unavailable creds than fail immediately)

        deadline = time.time() + self.global_timeout

        # Create transaction logger if request logging is enabled
        transaction_logger = None
        if self.enable_request_logging:
            transaction_logger = TransactionLogger(provider, model, enabled=True)
            transaction_logger.log_request(kwargs)

        tried_creds = set()
        last_exception = None
        kwargs = self._convert_model_params(**kwargs)

        consecutive_quota_failures = 0

        # Resolve model ID early, before any credential operations
        # This ensures consistent model ID usage for acquisition, release, and tracking
        resolved_model = self._resolve_model_id(model, provider)
        if resolved_model != model:
            lib_logger.info(f"Resolved model '{model}' to '{resolved_model}'")
            model = resolved_model
            kwargs["model"] = model  # Ensure kwargs has the resolved model for litellm

        # [NEW] Filter by model tier requirement and build priority map
        credential_priorities = None
        if provider_plugin and hasattr(provider_plugin, "get_model_tier_requirement"):
            required_tier = provider_plugin.get_model_tier_requirement(model)
            if required_tier is not None:
                # Filter OUT only credentials we KNOW are too low priority
                # Keep credentials with unknown priority (None) - they might be high priority
                incompatible_creds = []
                compatible_creds = []
                unknown_creds = []

                for cred in credentials_for_provider:
                    if hasattr(provider_plugin, "get_credential_priority"):
                        priority = provider_plugin.get_credential_priority(cred)
                        if priority is None:
                            # Unknown priority - keep it, will be discovered on first use
                            unknown_creds.append(cred)
                        elif priority <= required_tier:
                            # Known compatible priority
                            compatible_creds.append(cred)
                        else:
                            # Known incompatible priority (too low)
                            incompatible_creds.append(cred)
                    else:
                        # Provider doesn't support priorities - keep all
                        unknown_creds.append(cred)

                # If we have any known-compatible or unknown credentials, use them
                tier_compatible_creds = compatible_creds + unknown_creds
                if tier_compatible_creds:
                    credentials_for_provider = tier_compatible_creds
                    if compatible_creds and unknown_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible + {len(unknown_creds)} unknown-tier credentials."
                        )
                    elif compatible_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible credentials."
                        )
                    else:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(unknown_creds)} unknown-tier credentials (will discover on use)."
                        )
                elif incompatible_creds:
                    # Only known-incompatible credentials remain
                    lib_logger.warning(
                        f"Model {model} requires priority <= {required_tier} credentials, "
                        f"but all {len(incompatible_creds)} known credentials have priority > {required_tier}. "
                        f"Request will likely fail."
                    )

        # Build priority map and tier names map for usage_manager
        credential_tier_names = None
        if provider_plugin and hasattr(provider_plugin, "get_credential_priority"):
            credential_priorities = {}
            credential_tier_names = {}
            for cred in credentials_for_provider:
                priority = provider_plugin.get_credential_priority(cred)
                if priority is not None:
                    credential_priorities[cred] = priority
                # Also get tier name for logging
                if hasattr(provider_plugin, "get_credential_tier_name"):
                    tier_name = provider_plugin.get_credential_tier_name(cred)
                    if tier_name:
                        credential_tier_names[cred] = tier_name

            if credential_priorities:
                lib_logger.debug(
                    f"Credential priorities for {provider}: {', '.join(f'P{p}={len([c for c in credentials_for_provider if credential_priorities.get(c) == p])}' for p in sorted(set(credential_priorities.values())))}"
                )

        # Initialize error accumulator for tracking errors across credential rotation
        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        try:
            while (
                len(tried_creds) < len(credentials_for_provider)
                and time.time() < deadline
            ):
                current_cred = None
                key_acquired = False
                try:
                    if await self.cooldown_manager.is_cooling_down(provider):
                        remaining_cooldown = (
                            await self.cooldown_manager.get_cooldown_remaining(provider)
                        )
                        remaining_budget = deadline - time.time()
                        if remaining_cooldown > remaining_budget:
                            lib_logger.warning(
                                f"Provider {provider} cooldown ({remaining_cooldown:.2f}s) exceeds remaining request budget ({remaining_budget:.2f}s). Failing early."
                            )
                            break
                        lib_logger.warning(
                            f"Provider {provider} is in a global cooldown. All requests to this provider will be paused for {remaining_cooldown:.2f} seconds."
                        )
                        await asyncio.sleep(remaining_cooldown)

                    creds_to_try = [
                        c for c in credentials_for_provider if c not in tried_creds
                    ]
                    if not creds_to_try:
                        lib_logger.warning(
                            f"All credentials for provider {provider} have been tried. No more credentials to rotate to."
                        )
                        break

                    # Get count of credentials not on cooldown for this model
                    available_creds = (
                        await self.usage_manager.get_available_credentials_for_model(
                            creds_to_try, model
                        )
                    )
                    available_count = len(available_creds)
                    total_count = len(credentials_for_provider)

                    lib_logger.info(
                        f"Acquiring credential for model {model}. Tried credentials: {len(tried_creds)}/{available_count}({total_count})"
                    )
                    max_concurrent = self.max_concurrent_requests_per_key.get(
                        provider, 1
                    )
                    current_cred = await self.usage_manager.acquire_key(
                        available_keys=creds_to_try,
                        model=model,
                        deadline=deadline,
                        max_concurrent=max_concurrent,
                        credential_priorities=credential_priorities,
                        credential_tier_names=credential_tier_names,
                    )
                    key_acquired = True
                    tried_creds.add(current_cred)

                    litellm_kwargs = self.all_providers.get_provider_kwargs(
                        **kwargs.copy()
                    )
                    if "reasoning_effort" in kwargs:
                        litellm_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]

                    # [NEW] Merge provider-specific params
                    if provider in self.litellm_provider_params:
                        litellm_kwargs["litellm_params"] = {
                            **self.litellm_provider_params[provider],
                            **litellm_kwargs.get("litellm_params", {}),
                        }

                    provider_plugin = self._get_provider_instance(provider)

                    # Model ID is already resolved before the loop, and kwargs['model'] is updated.
                    # No further resolution needed here.

                    # Apply model-specific options for custom providers
                    if provider_plugin and hasattr(
                        provider_plugin, "get_model_options"
                    ):
                        model_options = provider_plugin.get_model_options(model)
                        if model_options:
                            # Merge model options into litellm_kwargs
                            for key, value in model_options.items():
                                if key == "reasoning_effort":
                                    litellm_kwargs["reasoning_effort"] = value
                                elif key not in litellm_kwargs:
                                    litellm_kwargs[key] = value
                    if provider_plugin and provider_plugin.has_custom_logic():
                        lib_logger.debug(
                            f"Provider '{provider}' has custom logic. Delegating call."
                        )
                        litellm_kwargs["credential_identifier"] = current_cred
                        litellm_kwargs["transaction_context"] = (
                            transaction_logger.get_context()
                            if transaction_logger
                            else None
                        )

                        for attempt in range(self.max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting stream with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                                )

                                if pre_request_callback:
                                    try:
                                        await pre_request_callback(
                                            request, litellm_kwargs
                                        )
                                    except Exception as e:
                                        if self.abort_on_callback_error:
                                            raise PreRequestCallbackError(
                                                f"Pre-request callback failed: {e}"
                                            ) from e
                                        else:
                                            lib_logger.warning(
                                                f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                            )

                                response = await provider_plugin.acompletion(
                                    self.http_client, **litellm_kwargs
                                )

                                lib_logger.info(
                                    f"Stream connection established for credential {mask_credential(current_cred)}. Processing response."
                                )

                                key_acquired = False
                                stream_generator = self._safe_streaming_wrapper(
                                    response,
                                    current_cred,
                                    model,
                                    request,
                                    provider_plugin,
                                )

                                # Wrap with transaction logging
                                logged_stream = (
                                    self._transaction_logging_stream_wrapper(
                                        stream_generator, transaction_logger, kwargs
                                    )
                                )

                                async for chunk in logged_stream:
                                    yield chunk
                                return

                            except (
                                StreamedAPIError,
                                litellm.RateLimitError,
                                httpx.HTTPStatusError,
                            ) as e:
                                last_exception = e
                                # If the exception is our custom wrapper, unwrap the original error
                                original_exc = getattr(e, "data", e)
                                classified_error = classify_error(
                                    original_exc, provider=provider
                                )
                                error_message = str(original_exc).split("\n")[0]

                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=dict(request.headers)
                                    if request
                                    else {},
                                )

                                # Record in accumulator for client reporting
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )

                                # Check if this error should trigger rotation
                                if not should_rotate_on_error(classified_error):
                                    lib_logger.error(
                                        f"Non-recoverable error ({classified_error.error_type}) during custom stream. Failing."
                                    )
                                    raise last_exception

                                # Handle rate limits with cooldown (exclude quota_exceeded)
                                if classified_error.error_type == "rate_limit":
                                    cooldown_duration = (
                                        classified_error.retry_after or 60
                                    )
                                    await self.cooldown_manager.start_cooldown(
                                        provider, cooldown_duration
                                    )

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code}). Rotating."
                                )
                                break

                            except (
                                APIConnectionError,
                                litellm.InternalServerError,
                                litellm.ServiceUnavailableError,
                            ) as e:
                                last_exception = e
                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=dict(request.headers)
                                    if request
                                    else {},
                                )
                                classified_error = classify_error(e, provider=provider)
                                error_message = str(e).split("\n")[0]

                                # Provider-level error: don't increment consecutive failures
                                await self.usage_manager.record_failure(
                                    current_cred,
                                    model,
                                    classified_error,
                                    increment_consecutive_failures=False,
                                )

                                if attempt >= self.max_retries - 1:
                                    error_accumulator.record_error(
                                        current_cred, classified_error, error_message
                                    )
                                    lib_logger.warning(
                                        f"Cred {mask_credential(current_cred)} failed after max retries. Rotating."
                                    )
                                    break

                                wait_time = classified_error.retry_after or (
                                    2**attempt
                                ) + random.uniform(0, 1)
                                remaining_budget = deadline - time.time()
                                if wait_time > remaining_budget:
                                    error_accumulator.record_error(
                                        current_cred, classified_error, error_message
                                    )
                                    lib_logger.warning(
                                        f"Retry wait ({wait_time:.2f}s) exceeds budget. Rotating."
                                    )
                                    break

                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} server error. Retrying in {wait_time:.2f}s."
                                )
                                await asyncio.sleep(wait_time)
                                continue

                            except Exception as e:
                                last_exception = e
                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=dict(request.headers)
                                    if request
                                    else {},
                                )
                                classified_error = classify_error(e, provider=provider)
                                error_message = str(e).split("\n")[0]

                                # Record in accumulator
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )

                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code})."
                                )

                                # Check if this error should trigger rotation
                                if not should_rotate_on_error(classified_error):
                                    lib_logger.error(
                                        f"Non-recoverable error ({classified_error.error_type}). Failing."
                                    )
                                    raise last_exception

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                break

                        # If the inner loop breaks, it means the key failed and we need to rotate.
                        # Continue to the next iteration of the outer while loop to pick a new key.
                        continue

                    else:  # This is the standard API Key / litellm-handled provider logic
                        is_oauth = provider in self.oauth_providers
                        if is_oauth:  # Standard OAuth provider (not custom)
                            # ... (logic to set headers) ...
                            pass
                        else:  # API Key
                            litellm_kwargs["api_key"] = current_cred

                    provider_instance = self._get_provider_instance(provider)
                    if provider_instance:
                        # Ensure default Gemini safety settings are present (without overriding request)
                        try:
                            self._apply_default_safety_settings(
                                litellm_kwargs, provider
                            )
                        except Exception:
                            lib_logger.debug(
                                "Could not apply default safety settings for streaming path; continuing."
                            )

                        if "safety_settings" in litellm_kwargs:
                            converted_settings = (
                                provider_instance.convert_safety_settings(
                                    litellm_kwargs["safety_settings"]
                                )
                            )
                            if converted_settings is not None:
                                litellm_kwargs["safety_settings"] = converted_settings
                            else:
                                del litellm_kwargs["safety_settings"]

                    if provider == "gemini" and provider_instance:
                        provider_instance.handle_thinking_parameter(
                            litellm_kwargs, model
                        )
                    if provider == "nvidia_nim" and provider_instance:
                        provider_instance.handle_thinking_parameter(
                            litellm_kwargs, model
                        )

                    if "gemma-3" in model and "messages" in litellm_kwargs:
                        litellm_kwargs["messages"] = [
                            {"role": "user", "content": m["content"]}
                            if m.get("role") == "system"
                            else m
                            for m in litellm_kwargs["messages"]
                        ]

                    litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                    # If the provider is 'qwen_code', set the custom provider to 'qwen'
                    # and strip the prefix from the model name for LiteLLM.
                    if provider == "qwen_code":
                        litellm_kwargs["custom_llm_provider"] = "qwen"
                        litellm_kwargs["model"] = model.split("/", 1)[1]

                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting stream with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            # lib_logger.info(f"DEBUG: litellm.acompletion kwargs: {litellm_kwargs}")
                            # Convert model parameters for custom providers right before LiteLLM call
                            final_kwargs = self._convert_model_params_for_litellm(
                                **litellm_kwargs
                            )

                            response = await litellm.acompletion(
                                **final_kwargs,
                                logger_fn=self._litellm_logger_callback,
                            )

                            lib_logger.info(
                                f"Stream connection established for credential {mask_credential(current_cred)}. Processing response."
                            )

                            key_acquired = False
                            stream_generator = self._safe_streaming_wrapper(
                                response,
                                current_cred,
                                model,
                                request,
                                provider_instance,
                            )

                            # Wrap with transaction logging
                            logged_stream = self._transaction_logging_stream_wrapper(
                                stream_generator, transaction_logger, kwargs
                            )

                            async for chunk in logged_stream:
                                yield chunk
                            return

                        except (
                            StreamedAPIError,
                            litellm.RateLimitError,
                            httpx.HTTPStatusError,
                        ) as e:
                            last_exception = e

                            # This is the final, robust handler for streamed errors.
                            error_payload = {}
                            cleaned_str = None
                            # The actual exception might be wrapped in our StreamedAPIError.
                            original_exc = getattr(e, "data", e)
                            classified_error = classify_error(
                                original_exc, provider=provider
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}) during litellm stream. Failing."
                                )
                                raise last_exception

                            try:
                                # The full error JSON is in the string representation of the exception.
                                json_str_match = re.search(
                                    r"(\{.*\})", str(original_exc), re.DOTALL
                                )
                                if json_str_match:
                                    cleaned_str = codecs.decode(
                                        json_str_match.group(1), "unicode_escape"
                                    )
                                    error_payload = json.loads(cleaned_str)
                            except (json.JSONDecodeError, TypeError):
                                error_payload = {}

                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                                raw_response_text=cleaned_str,
                            )

                            error_details = error_payload.get("error", {})
                            error_status = error_details.get("status", "")
                            error_message_text = error_details.get(
                                "message", str(original_exc).split("\n")[0]
                            )

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            if (
                                "quota" in error_message_text.lower()
                                or "resource_exhausted" in error_status.lower()
                            ):
                                consecutive_quota_failures += 1

                                quota_value = "N/A"
                                quota_id = "N/A"
                                if "details" in error_details and isinstance(
                                    error_details.get("details"), list
                                ):
                                    for detail in error_details["details"]:
                                        if isinstance(detail.get("violations"), list):
                                            for violation in detail["violations"]:
                                                if "quotaValue" in violation:
                                                    quota_value = violation[
                                                        "quotaValue"
                                                    ]
                                                if "quotaId" in violation:
                                                    quota_id = violation["quotaId"]
                                                if (
                                                    quota_value != "N/A"
                                                    and quota_id != "N/A"
                                                ):
                                                    break

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )

                                if consecutive_quota_failures >= 3:
                                    # Fatal: likely input data too large
                                    client_error_message = (
                                        f"Request failed after 3 consecutive quota errors (input may be too large). "
                                        f"Limit: {quota_value} (Quota ID: {quota_id})"
                                    )
                                    lib_logger.error(
                                        f"Fatal quota error for {mask_credential(current_cred)}. ID: {quota_id}, Limit: {quota_value}"
                                    )
                                    yield f"data: {json.dumps({'error': {'message': client_error_message, 'type': 'proxy_fatal_quota_error'}})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                else:
                                    lib_logger.warning(
                                        f"Cred {mask_credential(current_cred)} quota error ({consecutive_quota_failures}/3). Rotating."
                                    )
                                    break

                            else:
                                consecutive_quota_failures = 0
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type}. Rotating."
                                )

                                if classified_error.error_type == "rate_limit":
                                    cooldown_duration = (
                                        classified_error.retry_after or 60
                                    )
                                    await self.cooldown_manager.start_cooldown(
                                        provider, cooldown_duration
                                    )

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                break

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                        ) as e:
                            consecutive_quota_failures = 0
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message_text = str(e).split("\n")[0]

                            # Record error in accumulator (server errors are transient, not abnormal)
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                lib_logger.warning(
                                    f"Credential {mask_credential(current_cred)} failed after max retries for model {model} due to a server error. Rotating key silently."
                                )
                                # [MODIFIED] Do not yield to the client here.
                                break

                            wait_time = classified_error.retry_after or (
                                2**attempt
                            ) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()
                            if wait_time > remaining_budget:
                                lib_logger.warning(
                                    f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early."
                                )
                                break

                            lib_logger.warning(
                                f"Credential {mask_credential(current_cred)} encountered a server error for model {model}. Reason: '{error_message_text}'. Retrying in {wait_time:.2f}s."
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        except Exception as e:
                            consecutive_quota_failures = 0
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=dict(request.headers)
                                if request
                                else {},
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message_text = str(e).split("\n")[0]

                            # Record error in accumulator
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            lib_logger.warning(
                                f"Credential {mask_credential(current_cred)} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {error_message_text}."
                            )

                            # Handle rate limits with cooldown (exclude quota_exceeded)
                            if (
                                classified_error.status_code == 429
                                and classified_error.error_type != "quota_exceeded"
                            ) or classified_error.error_type == "rate_limit":
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )
                                lib_logger.warning(
                                    f"Rate limit detected for {provider}. Starting {cooldown_duration}s cooldown."
                                )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                # Non-rotatable errors - fail immediately
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record failure and rotate to next key
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.info(
                                f"Rotating to next key after {classified_error.error_type} error."
                            )
                            break

                finally:
                    if key_acquired and current_cred:
                        await self.usage_manager.release_key(current_cred, model)

            # Build detailed error response using error accumulator
            error_accumulator.timeout_occurred = time.time() >= deadline

            if error_accumulator.has_errors():
                # Log concise summary for server logs
                lib_logger.error(error_accumulator.build_log_message())

                # Build structured error response for client
                error_response = error_accumulator.build_client_error_response()
                error_data = error_response
            else:
                # Fallback if no errors were recorded (shouldn't happen)
                final_error_message = (
                    "Request failed: No available API keys after rotation or timeout."
                )
                if last_exception:
                    final_error_message = (
                        f"Request failed. Last error: {str(last_exception)}"
                    )
                error_data = {
                    "error": {"message": final_error_message, "type": "proxy_error"}
                }
                lib_logger.error(final_error_message)

            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except NoAvailableKeysError as e:
            lib_logger.error(
                f"A streaming request failed because no keys were available within the time budget: {e}"
            )
            error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # This will now only catch fatal errors that should be raised, like invalid requests.
            lib_logger.error(
                f"An unhandled exception occurred in streaming retry logic: {e}",
                exc_info=True,
            )
            error_data = {
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "proxy_internal_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Dispatcher for completion requests.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the completion request will be aborted and the exception will propagate.

        Returns:
            The completion response object, or an async generator for streaming responses, or None if all retries fail.
        """
        # Handle iflow provider: remove stream_options to avoid HTTP 406
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""

        if provider == "iflow" and "stream_options" in kwargs:
            lib_logger.debug(
                "Removing stream_options for iflow provider to avoid HTTP 406"
            )
            kwargs.pop("stream_options", None)

        if kwargs.get("stream"):
            # Only add stream_options for providers that support it (excluding iflow)
            if provider != "iflow":
                if "stream_options" not in kwargs:
                    kwargs["stream_options"] = {}
                if "include_usage" not in kwargs["stream_options"]:
                    kwargs["stream_options"]["include_usage"] = True

            return self._streaming_acompletion_with_retry(
                request=request, pre_request_callback=pre_request_callback, **kwargs
            )
        else:
            return self._execute_with_retry(
                litellm.acompletion,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )

    def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """
        Executes an embedding request with retry logic.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the embedding request will be aborted and the exception will propagate.

        Returns:
            The embedding response object, or None if all retries fail.
        """
        return self._execute_with_retry(
            litellm.aembedding,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def token_count(self, **kwargs) -> int:
        """Calculates the number of tokens for a given text or list of messages.

        For Antigravity provider models, this also includes the preprompt tokens
        that get injected during actual API calls (agent instruction + identity override).
        This ensures token counts match actual usage.
        """
        kwargs = self._convert_model_params(**kwargs)
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is a required parameter.")

        # Calculate base token count
        if messages:
            base_count = token_counter(model=model, messages=messages)
        elif text:
            base_count = token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided.")

        # Add preprompt tokens for Antigravity provider
        # The Antigravity provider injects system instructions during actual API calls,
        # so we need to account for those tokens in the count
        provider = model.split("/")[0] if "/" in model else ""
        if provider == "antigravity":
            try:
                from .providers.antigravity_provider import get_antigravity_preprompt_text

                preprompt_text = get_antigravity_preprompt_text()
                if preprompt_text:
                    preprompt_tokens = token_counter(model=model, text=preprompt_text)
                    base_count += preprompt_tokens
            except ImportError:
                # Provider not available, skip preprompt token counting
                pass

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Returns a list of available models for a specific provider, with caching."""
        lib_logger.info(f"Getting available models for provider: {provider}")
        if provider in self._model_list_cache:
            lib_logger.debug(f"Returning cached models for provider: {provider}")
            return self._model_list_cache[provider]

        credentials_for_provider = self.all_credentials.get(provider)
        if not credentials_for_provider:
            lib_logger.warning(f"No credentials for provider: {provider}")
            return []

        # Create a copy and shuffle it to randomize the starting credential
        shuffled_credentials = list(credentials_for_provider)
        random.shuffle(shuffled_credentials)

        provider_instance = self._get_provider_instance(provider)
        if provider_instance:
            # For providers with hardcoded models (like gemini_cli), we only need to call once.
            # For others, we might need to try multiple keys if one is invalid.
            # The current logic of iterating works for both, as the credential is not
            # always used in get_models.
            for credential in shuffled_credentials:
                try:
                    # Display last 6 chars for API keys, or the filename for OAuth paths
                    cred_display = mask_credential(credential)
                    lib_logger.debug(
                        f"Attempting to get models for {provider} with credential {cred_display}"
                    )
                    models = await provider_instance.get_models(
                        credential, self.http_client
                    )
                    lib_logger.info(
                        f"Got {len(models)} models for provider: {provider}"
                    )

                    # Whitelist and blacklist logic
                    final_models = []
                    for m in models:
                        is_whitelisted = self._is_model_whitelisted(provider, m)
                        is_blacklisted = self._is_model_ignored(provider, m)

                        if is_whitelisted:
                            final_models.append(m)
                            continue

                        if not is_blacklisted:
                            final_models.append(m)

                    if len(final_models) != len(models):
                        lib_logger.info(
                            f"Filtered out {len(models) - len(final_models)} models for provider {provider}."
                        )

                    self._model_list_cache[provider] = final_models
                    return final_models
                except Exception as e:
                    classified_error = classify_error(e, provider=provider)
                    cred_display = mask_credential(credential)
                    lib_logger.debug(
                        f"Failed to get models for provider {provider} with credential {cred_display}: {classified_error.error_type}. Trying next credential."
                    )
                    continue  # Try the next credential

        lib_logger.error(
            f"Failed to get models for provider {provider} after trying all credentials."
        )
        return []

    async def get_all_available_models(
        self, grouped: bool = True
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Returns a list of all available models, either grouped by provider or as a flat list."""
        lib_logger.info("Getting all available models...")

        all_providers = list(self.all_credentials.keys())
        tasks = [self.get_available_models(provider) for provider in all_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_provider_models = {}
        for provider, result in zip(all_providers, results):
            if isinstance(result, Exception):
                lib_logger.error(
                    f"Failed to get models for provider {provider}: {result}"
                )
                all_provider_models[provider] = []
            else:
                all_provider_models[provider] = result

        lib_logger.info("Finished getting all available models.")
        if grouped:
            return all_provider_models
        else:
            flat_models = []
            for models in all_provider_models.values():
                flat_models.extend(models)
            return flat_models

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get quota and usage stats for all credentials.

        This returns cached/disk data aggregated by provider.
        For provider-specific quota info (e.g., Antigravity quota groups),
        it enriches the data from provider plugins.

        Args:
            provider_filter: If provided, only return stats for this provider

        Returns:
            Complete stats dict ready for the /v1/quota-stats endpoint
        """
        # Get base stats from usage manager
        stats = await self.usage_manager.get_stats_for_endpoint(provider_filter)

        # Enrich with provider-specific quota data
        for provider, prov_stats in stats.get("providers", {}).items():
            provider_class = self._provider_plugins.get(provider)
            if not provider_class:
                continue

            # Get or create provider instance
            if provider not in self._provider_instances:
                self._provider_instances[provider] = provider_class()
            provider_instance = self._provider_instances[provider]

            # Check if provider has quota tracking (like Antigravity)
            if hasattr(provider_instance, "_get_effective_quota_groups"):
                # Add quota group summary
                quota_groups = provider_instance._get_effective_quota_groups()
                prov_stats["quota_groups"] = {}

                for group_name, group_models in quota_groups.items():
                    group_stats = {
                        "models": group_models,
                        "credentials_total": 0,
                        "credentials_exhausted": 0,
                        "avg_remaining_pct": 0,
                        "total_remaining_pcts": [],
                        # Total requests tracking across all credentials
                        "total_requests_used": 0,
                        "total_requests_max": 0,
                        # Tier breakdown: tier_name -> {"total": N, "active": M}
                        "tiers": {},
                    }

                    # Calculate per-credential quota for this group
                    for cred in prov_stats.get("credentials", []):
                        models_data = cred.get("models", {})
                        group_stats["credentials_total"] += 1

                        # Track tier - get directly from provider cache since cred["tier"] not set yet
                        tier = cred.get("tier")
                        if not tier and hasattr(
                            provider_instance, "project_tier_cache"
                        ):
                            cred_path = cred.get("full_path", "")
                            tier = provider_instance.project_tier_cache.get(cred_path)
                        tier = tier or "unknown"

                        # Initialize tier entry if needed with priority for sorting
                        if tier not in group_stats["tiers"]:
                            priority = 10  # default
                            if hasattr(provider_instance, "_resolve_tier_priority"):
                                priority = provider_instance._resolve_tier_priority(
                                    tier
                                )
                            group_stats["tiers"][tier] = {
                                "total": 0,
                                "active": 0,
                                "priority": priority,
                            }
                        group_stats["tiers"][tier]["total"] += 1

                        # Find model with VALID baseline (not just any model with stats)
                        model_stats = None
                        for model in group_models:
                            candidate = self._find_model_stats_in_data(
                                models_data, model, provider, provider_instance
                            )
                            if candidate:
                                baseline = candidate.get("baseline_remaining_fraction")
                                if baseline is not None:
                                    model_stats = candidate
                                    break
                                # Keep first found as fallback (for request counts)
                                if model_stats is None:
                                    model_stats = candidate

                        if model_stats:
                            baseline = model_stats.get("baseline_remaining_fraction")
                            req_count = model_stats.get("request_count", 0)
                            max_req = model_stats.get("quota_max_requests") or 0

                            # Accumulate totals (one model per group per credential)
                            group_stats["total_requests_used"] += req_count
                            group_stats["total_requests_max"] += max_req

                            if baseline is not None:
                                remaining_pct = int(baseline * 100)
                                group_stats["total_remaining_pcts"].append(
                                    remaining_pct
                                )
                                if baseline <= 0:
                                    group_stats["credentials_exhausted"] += 1
                                else:
                                    # Credential is active (has quota remaining)
                                    group_stats["tiers"][tier]["active"] += 1

                    # Calculate average remaining percentage (per-credential average)
                    if group_stats["total_remaining_pcts"]:
                        group_stats["avg_remaining_pct"] = int(
                            sum(group_stats["total_remaining_pcts"])
                            / len(group_stats["total_remaining_pcts"])
                        )
                    del group_stats["total_remaining_pcts"]

                    # Calculate total remaining percentage (global)
                    if group_stats["total_requests_max"] > 0:
                        used = group_stats["total_requests_used"]
                        max_r = group_stats["total_requests_max"]
                        group_stats["total_requests_remaining"] = max_r - used
                        group_stats["total_remaining_pct"] = max(
                            0, int((1 - used / max_r) * 100)
                        )
                    else:
                        group_stats["total_requests_remaining"] = 0
                        group_stats["total_remaining_pct"] = None

                    prov_stats["quota_groups"][group_name] = group_stats

                # Also enrich each credential with formatted quota group info
                for cred in prov_stats.get("credentials", []):
                    cred["model_groups"] = {}
                    models_data = cred.get("models", {})

                    for group_name, group_models in quota_groups.items():
                        # Find model with VALID baseline (prefer over any model with stats)
                        # Also track the best reset_ts across all models in the group
                        model_stats = None
                        best_reset_ts = None

                        for model in group_models:
                            candidate = self._find_model_stats_in_data(
                                models_data, model, provider, provider_instance
                            )
                            if candidate:
                                # Track the best (latest) reset_ts from any model in group
                                candidate_reset_ts = candidate.get("quota_reset_ts")
                                if candidate_reset_ts:
                                    if (
                                        best_reset_ts is None
                                        or candidate_reset_ts > best_reset_ts
                                    ):
                                        best_reset_ts = candidate_reset_ts

                                baseline = candidate.get("baseline_remaining_fraction")
                                if baseline is not None:
                                    model_stats = candidate
                                    # Don't break - continue to find best reset_ts
                                # Keep first found as fallback
                                if model_stats is None:
                                    model_stats = candidate

                        if model_stats:
                            baseline = model_stats.get("baseline_remaining_fraction")
                            max_req = model_stats.get("quota_max_requests")
                            req_count = model_stats.get("request_count", 0)
                            # Use best_reset_ts from any model in the group
                            reset_ts = best_reset_ts or model_stats.get(
                                "quota_reset_ts"
                            )

                            remaining_pct = (
                                int(baseline * 100) if baseline is not None else None
                            )
                            is_exhausted = baseline is not None and baseline <= 0

                            # Format reset time
                            reset_iso = None
                            if reset_ts:
                                try:
                                    from datetime import datetime, timezone

                                    reset_iso = datetime.fromtimestamp(
                                        reset_ts, tz=timezone.utc
                                    ).isoformat()
                                except (ValueError, OSError):
                                    pass

                            requests_remaining = (
                                max(0, max_req - req_count) if max_req else 0
                            )
                            cred["model_groups"][group_name] = {
                                "remaining_pct": remaining_pct,
                                "requests_used": req_count,
                                "requests_remaining": requests_remaining,
                                "requests_max": max_req,
                                "display": f"{requests_remaining}/{max_req}"
                                if max_req
                                else f"?/?",
                                "is_exhausted": is_exhausted,
                                "reset_time_iso": reset_iso,
                                "models": group_models,
                                "confidence": self._get_baseline_confidence(
                                    model_stats
                                ),
                            }

                    # Recalculate credential's requests from model_groups
                    # This fixes double-counting when models share quota groups
                    if cred.get("model_groups"):
                        group_requests = sum(
                            g.get("requests_used", 0)
                            for g in cred["model_groups"].values()
                        )
                        cred["requests"] = group_requests

                        # HACK: Fix global requests if present
                        # This is a simplified fix that sets global.requests = current group_requests.
                        # TODO: Properly track archived requests per quota group in usage_manager.py
                        # so that global stats correctly sum: current_period + archived_periods
                        # without double-counting models that share quota groups.
                        # See: usage_manager.py lines 2388-2404 where global stats are built
                        # by iterating all models (causing double-counting for grouped models).
                        if cred.get("global"):
                            cred["global"]["requests"] = group_requests

                    # Try to get email from provider's cache
                    cred_path = cred.get("full_path", "")
                    if hasattr(provider_instance, "project_tier_cache"):
                        tier = provider_instance.project_tier_cache.get(cred_path)
                        if tier:
                            cred["tier"] = tier

        return stats

    def _find_model_stats_in_data(
        self,
        models_data: Dict[str, Any],
        model: str,
        provider: str,
        provider_instance: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Find model stats in models_data, trying various name variants.

        Handles aliased model names (e.g., gemini-3-pro-preview -> gemini-3-pro-high)
        by using the provider's _user_to_api_model() mapping.

        Args:
            models_data: Dict of model_name -> stats from credential
            model: Model name to look up (user-facing name)
            provider: Provider name for prefixing
            provider_instance: Provider instance for alias methods

        Returns:
            Model stats dict if found, None otherwise
        """
        # Try direct match with and without provider prefix
        prefixed_model = f"{provider}/{model}"
        model_stats = models_data.get(prefixed_model) or models_data.get(model)

        if model_stats:
            return model_stats

        # Try with API model name (e.g., gemini-3-pro-preview -> gemini-3-pro-high)
        if hasattr(provider_instance, "_user_to_api_model"):
            api_model = provider_instance._user_to_api_model(model)
            if api_model != model:
                prefixed_api = f"{provider}/{api_model}"
                model_stats = models_data.get(prefixed_api) or models_data.get(
                    api_model
                )

        return model_stats

    def _get_baseline_confidence(self, model_stats: Dict) -> str:
        """
        Determine confidence level based on baseline age.

        Args:
            model_stats: Model statistics dict with baseline_fetched_at

        Returns:
            "high" | "medium" | "low"
        """
        baseline_fetched_at = model_stats.get("baseline_fetched_at")
        if not baseline_fetched_at:
            return "low"

        age_seconds = time.time() - baseline_fetched_at
        if age_seconds < 300:  # 5 minutes
            return "high"
        elif age_seconds < 1800:  # 30 minutes
            return "medium"
        return "low"

    async def reload_usage_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        """
        await self.usage_manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force refresh quota from external API.

        For Antigravity, this fetches live quota data from the API.
        For other providers, this is a no-op (just reloads from disk).

        Args:
            provider: If specified, only refresh this provider
            credential: If specified, only refresh this specific credential

        Returns:
            Refresh result dict with success/failure info
        """
        result = {
            "action": "force_refresh",
            "scope": "credential"
            if credential
            else ("provider" if provider else "all"),
            "provider": provider,
            "credential": credential,
            "credentials_refreshed": 0,
            "success_count": 0,
            "failed_count": 0,
            "duration_ms": 0,
            "errors": [],
        }

        start_time = time.time()

        # Determine which providers to refresh
        if provider:
            providers_to_refresh = (
                [provider] if provider in self.all_credentials else []
            )
        else:
            providers_to_refresh = list(self.all_credentials.keys())

        for prov in providers_to_refresh:
            provider_class = self._provider_plugins.get(prov)
            if not provider_class:
                continue

            # Get or create provider instance
            if prov not in self._provider_instances:
                self._provider_instances[prov] = provider_class()
            provider_instance = self._provider_instances[prov]

            # Check if provider supports quota refresh (like Antigravity)
            if hasattr(provider_instance, "fetch_initial_baselines"):
                # Get credentials to refresh
                if credential:
                    # Find full path for this credential
                    creds_to_refresh = []
                    for cred_path in self.all_credentials.get(prov, []):
                        if cred_path.endswith(credential) or cred_path == credential:
                            creds_to_refresh.append(cred_path)
                            break
                else:
                    creds_to_refresh = self.all_credentials.get(prov, [])

                if not creds_to_refresh:
                    continue

                try:
                    # Fetch live quota from API for ALL specified credentials
                    quota_results = await provider_instance.fetch_initial_baselines(
                        creds_to_refresh
                    )

                    # Store baselines in usage manager
                    if hasattr(provider_instance, "_store_baselines_to_usage_manager"):
                        stored = (
                            await provider_instance._store_baselines_to_usage_manager(
                                quota_results, self.usage_manager
                            )
                        )
                        result["success_count"] += stored

                    result["credentials_refreshed"] += len(creds_to_refresh)

                    # Count failures
                    for cred_path, data in quota_results.items():
                        if data.get("status") != "success":
                            result["failed_count"] += 1
                            result["errors"].append(
                                f"{Path(cred_path).name}: {data.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    lib_logger.error(f"Failed to refresh quota for {prov}: {e}")
                    result["errors"].append(f"{prov}: {str(e)}")
                    result["failed_count"] += len(creds_to_refresh)

        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result

    # --- Anthropic API Compatibility Methods ---

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        from .anthropic_compat import (
            translate_anthropic_request,
            openai_to_anthropic_response,
            anthropic_streaming_wrapper,
        )
        import uuid

        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        original_model = request.model

        # Translate Anthropic request to OpenAI format
        openai_request = translate_anthropic_request(request)

        if request.stream:
            # Streaming response
            response_generator = self.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Create disconnect checker if raw_request provided
            is_disconnected = None
            if raw_request is not None and hasattr(raw_request, "is_disconnected"):
                is_disconnected = raw_request.is_disconnected

            # Return the streaming wrapper
            return anthropic_streaming_wrapper(
                openai_stream=response_generator,
                original_model=original_model,
                request_id=request_id,
                is_disconnected=is_disconnected,
            )
        else:
            # Non-streaming response
            response = await self.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Convert OpenAI response to Anthropic format
            openai_response = (
                response.model_dump() if hasattr(response, "model_dump") else dict(response)
            )
            anthropic_response = openai_to_anthropic_response(openai_response, original_model)

            # Override the ID with our request ID
            anthropic_response["id"] = request_id

            return anthropic_response

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        from .anthropic_compat import (
            anthropic_to_openai_messages,
            anthropic_to_openai_tools,
        )
        import json

        anthropic_request = request.model_dump(exclude_none=True)

        openai_messages = anthropic_to_openai_messages(
            anthropic_request.get("messages", []), anthropic_request.get("system")
        )

        # Count tokens for messages
        message_tokens = self.token_count(
            model=request.model,
            messages=openai_messages,
        )

        # Count tokens for tools if present
        tool_tokens = 0
        if request.tools:
            # Tools add tokens based on their definitions
            # Convert to JSON string and count tokens for tool definitions
            openai_tools = anthropic_to_openai_tools(
                [tool.model_dump() for tool in request.tools]
            )
            if openai_tools:
                # Serialize tools to count their token contribution
                tools_text = json.dumps(openai_tools)
                tool_tokens = self.token_count(
                    model=request.model,
                    text=tools_text,
                )

        total_tokens = message_tokens + tool_tokens

        return {"input_tokens": total_tokens}
