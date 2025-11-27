import asyncio
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
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

lib_logger = logging.getLogger("rotator_library")
# Ensure the logger is configured to propagate to the root logger
# which is set up in main.py. This allows the main app to control
# log levels and handlers centrally.
lib_logger.propagate = False

from .usage_manager import UsageManager
from .failure_logger import log_failure
from .error_handler import (
    PreRequestCallbackError,
    classify_error,
    AllProviders,
    NoAvailableKeysError,
)
from .providers import PROVIDER_PLUGINS
from .providers.openai_compatible_provider import OpenAICompatibleProvider
from .request_sanitizer import sanitize_request_payload
from .cooldown_manager import CooldownManager
from .credential_manager import CredentialManager
from .background_refresher import BackgroundRefresher
from .model_definitions import ModelDefinitions


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
        usage_file_path: str = "key_usage.json",
        configure_logging: bool = True,
        global_timeout: int = 30,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
    ):
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
        self.credential_manager = CredentialManager(oauth_credentials)
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
        self.usage_manager = UsageManager(file_path=usage_file_path)
        self._model_list_cache = {}
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = {}
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
                lib_logger.warning(f"Invalid max_concurrent for '{provider}': {max_val}. Setting to 1.")
                self.max_concurrent_requests_per_key[provider] = 1

    def _is_model_ignored(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model should be ignored based on the ignore list.
        Supports exact and partial matching for both full model IDs and model names.
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
            if ignored_pattern.endswith("*"):
                match_pattern = ignored_pattern[:-1]
                # Match wildcard against the provider's model name
                if provider_model_name.startswith(match_pattern):
                    return True
            else:
                # Exact match against the full proxy ID OR the provider's model name
                if (
                    model_id == ignored_pattern
                    or provider_model_name == ignored_pattern
                ):
                    return True
        return False

    def _is_model_whitelisted(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model is explicitly whitelisted.
        Supports exact and partial matching for both full model IDs and model names.
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in self.whitelist_models:
            return False

        whitelist = self.whitelist_models[model_provider]
        for whitelisted_pattern in whitelist:
            if whitelisted_pattern == "*":
                return True

            try:
                # This is the model name as the provider sees it (e.g., "gpt-4" or "google/gemma-7b")
                provider_model_name = model_id.split("/", 1)[1]
            except IndexError:
                provider_model_name = model_id

            if whitelisted_pattern.endswith("*"):
                match_pattern = whitelisted_pattern[:-1]
                # Match wildcard against the provider's model name
                if provider_model_name.startswith(match_pattern):
                    return True
            else:
                # Exact match against the full proxy ID OR the provider's model name
                if (
                    model_id == whitelisted_pattern
                    or provider_model_name == whitelisted_pattern
                ):
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

    def _apply_default_safety_settings(self, litellm_kwargs: Dict[str, Any], provider: str):
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
        if "safety_settings" in litellm_kwargs and isinstance(litellm_kwargs["safety_settings"], dict):
            for k, v in default_generic.items():
                if k not in litellm_kwargs["safety_settings"]:
                    litellm_kwargs["safety_settings"][k] = v
            return

        # If Gemini form is present, ensure missing gemini categories are appended
        if "safetySettings" in litellm_kwargs and isinstance(litellm_kwargs["safetySettings"], list):
            present = {item.get("category") for item in litellm_kwargs["safetySettings"] if isinstance(item, dict)}
            for d in default_gemini:
                if d["category"] not in present:
                    litellm_kwargs["safetySettings"].append(d)
            return

        # Neither present: set generic defaults so provider conversion will translate them
        if "safety_settings" not in litellm_kwargs and "safetySettings" not in litellm_kwargs:
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
        
        Returns:
            Provider instance if credentials exist, None otherwise.
        """
        # Only initialize providers for which we have credentials
        if provider_name not in self.all_credentials:
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
        model_name = model.split('/')[-1] if '/' in model else model
        
        # Try to get provider instance to check for model definitions
        provider_plugin = self._get_provider_instance(provider)
        
        # Check if provider has model definitions
        if provider_plugin and hasattr(provider_plugin, 'model_definitions'):
            model_id = provider_plugin.model_definitions.get_model_id(provider, model_name)
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
        self, stream: Any, key: str, model: str, request: Optional[Any] = None
    ) -> AsyncGenerator[Any, None]:
        """
        A hybrid wrapper for streaming that buffers fragmented JSON, handles client disconnections gracefully,
        and distinguishes between content and streamed errors.
        """
        last_usage = None
        stream_completed = False
        stream_iterator = stream.__aiter__()
        json_buffer = ""

        try:
            while True:
                if request and await request.is_disconnected():
                    lib_logger.info(
                        f"Client disconnected. Aborting stream for credential ...{key[-6:]}."
                    )
                    # Do not yield [DONE] because the client is gone.
                    # The 'finally' block will handle key release.
                    break

                try:
                    chunk = await stream_iterator.__anext__()
                    if json_buffer:
                        # If we are about to discard a buffer, it means data was likely lost.
                        # Log this as a warning to make it visible.
                        lib_logger.warning(
                            f"Discarding incomplete JSON buffer from previous chunk: {json_buffer}"
                        )
                        json_buffer = ""

                    yield f"data: {json.dumps(chunk.dict())}\n\n"

                    if hasattr(chunk, "usage") and chunk.usage:
                        last_usage = (
                            chunk.usage
                        )  # Overwrite with the latest (cumulative)

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

                except (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                    litellm.InternalServerError,
                    APIConnectionError,
                ) as e:
                    # This is a critical, typed error from litellm that signals a key failure.
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
                f"An unexpected error occurred during the stream for credential ...{key[-6:]}: {e}"
            )
            # We still need to raise it so the client knows something went wrong.
            raise

        finally:
            # This block now runs regardless of how the stream terminates (completion, client disconnect, etc.).
            # The primary goal is to ensure usage is always logged internally.
            await self.usage_manager.release_key(key, model)
            lib_logger.info(
                f"STREAM FINISHED and lock released for credential ...{key[-6:]}."
            )

            # Only send [DONE] if the stream completed naturally and the client is still there.
            # This prevents sending [DONE] to a disconnected client or after an error.
            if stream_completed and (
                not request or not await request.is_disconnected()
            ):
                yield "data: [DONE]\n\n"

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

        # Create a mutable copy of the keys and shuffle it to ensure
        # that the key selection is randomized, which is crucial when
        # multiple keys have the same usage stats.
        credentials_for_provider = list(self.all_credentials[provider])
        random.shuffle(credentials_for_provider)
        
        # Filter out credentials that are unavailable (queued for re-auth)
        provider_plugin = self._get_provider_instance(provider)
        if provider_plugin and hasattr(provider_plugin, 'is_credential_available'):
            available_creds = [
                cred for cred in credentials_for_provider
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

                lib_logger.info(
                    f"Acquiring key for model {model}. Tried keys: {len(tried_creds)}/{len(credentials_for_provider)}"
                )
                max_concurrent = self.max_concurrent_requests_per_key.get(provider, 1)
                current_cred = await self.usage_manager.acquire_key(
                    available_keys=creds_to_try, model=model, deadline=deadline,
                    max_concurrent=max_concurrent
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
                    litellm_kwargs["enable_request_logging"] = (
                        self.enable_request_logging
                    )

                    # Check body first for custom_reasoning_budget
                    if "custom_reasoning_budget" in kwargs:
                        litellm_kwargs["custom_reasoning_budget"] = kwargs[
                            "custom_reasoning_budget"
                        ]
                    else:
                        custom_budget_header = None
                        if request and hasattr(request, "headers"):
                            custom_budget_header = request.headers.get(
                                "custom_reasoning_budget"
                            )

                        if custom_budget_header is not None:
                            is_budget_enabled = custom_budget_header.lower() == "true"
                            litellm_kwargs["custom_reasoning_budget"] = (
                                is_budget_enabled
                            )

                    # The plugin handles the entire call, including retries on 401, etc.
                    # The main retry loop here is for key rotation on other errors.
                    response = await provider_plugin.acompletion(
                        self.http_client, **litellm_kwargs
                    )

                    # For non-streaming, success is immediate, and this function only handles non-streaming.
                    await self.usage_manager.record_success(
                        current_cred, model, response
                    )
                    await self.usage_manager.release_key(current_cred, model)
                    key_acquired = False
                    return response

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
                            self._apply_default_safety_settings(litellm_kwargs, provider)
                        except Exception:
                            # If anything goes wrong here, avoid breaking the request flow.
                            lib_logger.debug("Could not apply default safety settings; continuing.")

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
                                f"Attempting call with credential ...{current_cred[-6:]} (Attempt {attempt + 1}/{self.max_retries})"
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
                            classified_error = classify_error(e)

                            # Extract a clean error message for the user-facing log
                            error_message = str(e).split("\n")[0]
                            lib_logger.info(
                                f"Key ...{current_cred[-6:]} hit rate limit for model {model}. Reason: '{error_message}'. Rotating key."
                            )

                            if classified_error.status_code == 429:
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )
                                lib_logger.warning(
                                    f"IP-based rate limit detected for {provider}. Starting a {cooldown_duration}-second global cooldown."
                                )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.warning(
                                f"Key ...{current_cred[-6:]} encountered a rate limit. Trying next key."
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
                            classified_error = classify_error(e)
                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error,
                                increment_consecutive_failures=False
                            )

                            if attempt >= self.max_retries - 1:
                                error_message = str(e).split("\n")[0]
                                lib_logger.warning(
                                    f"Key ...{current_cred[-6:]} failed after max retries for model {model} due to a server error. Reason: '{error_message}'. Rotating key."
                                )
                                break  # Move to the next key

                            # For temporary errors, wait before retrying with the same key.
                            wait_time = classified_error.retry_after or (
                                1 * (2**attempt)
                            ) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()

                            # If the required wait time exceeds the budget, don't wait; rotate to the next key immediately.
                            if wait_time > remaining_budget:
                                lib_logger.warning(
                                    f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early."
                                )
                                break

                            error_message = str(e).split("\n")[0]
                            lib_logger.warning(
                                f"Key ...{current_cred[-6:]} encountered a server error for model {model}. Reason: '{error_message}'. Retrying in {wait_time:.2f}s."
                            )
                            await asyncio.sleep(wait_time)
                            continue  # Retry with the same key

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
                                    f"Client disconnected. Aborting retries for credential ...{current_cred[-6:]}."
                                )
                                raise last_exception

                            classified_error = classify_error(e)
                            error_message = str(e).split("\n")[0]
                            lib_logger.warning(
                                f"Key ...{current_cred[-6:]} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {error_message}. Rotating key."
                            )
                            if classified_error.status_code == 429:
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )
                                lib_logger.warning(
                                    f"IP-based rate limit detected for {provider} from generic exception. Starting a {cooldown_duration}-second global cooldown."
                                )

                            if classified_error.error_type in [
                                "invalid_request",
                                "context_window_exceeded",
                                "authentication",
                            ]:
                                # For these errors, we should not retry with other keys.
                                raise last_exception

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Try next key for other errors
            finally:
                if key_acquired and current_cred:
                    await self.usage_manager.release_key(current_cred, model)

        if last_exception:
            # Log the final error but do not raise it, as per the new requirement.
            # The client should not see intermittent failures.
            lib_logger.error(
                f"Request failed after trying all keys or exceeding global timeout. Last error: {last_exception}"
            )

        # Return None to indicate failure without propagating a disruptive exception.
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
        if provider_plugin and hasattr(provider_plugin, 'is_credential_available'):
            available_creds = [
                cred for cred in credentials_for_provider
                if provider_plugin.is_credential_available(cred)
            ]
            if available_creds:
                credentials_for_provider = available_creds
            # If all credentials are unavailable, keep the original list
            # (better to try unavailable creds than fail immediately)

        deadline = time.time() + self.global_timeout
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

                    lib_logger.info(
                        f"Acquiring credential for model {model}. Tried credentials: {len(tried_creds)}/{len(credentials_for_provider)}"
                    )
                    max_concurrent = self.max_concurrent_requests_per_key.get(provider, 1)
                    current_cred = await self.usage_manager.acquire_key(
                        available_keys=creds_to_try, model=model, deadline=deadline,
                        max_concurrent=max_concurrent
                    )
                    key_acquired = True
                    tried_creds.add(current_cred)

                    litellm_kwargs = self.all_providers.get_provider_kwargs(
                        **kwargs.copy()
                    )
                    if "reasoning_effort" in kwargs:
                        litellm_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
                    # Check body first for custom_reasoning_budget
                    if "custom_reasoning_budget" in kwargs:
                        litellm_kwargs["custom_reasoning_budget"] = kwargs[
                            "custom_reasoning_budget"
                        ]
                    else:
                        custom_budget_header = None
                        if request and hasattr(request, "headers"):
                            custom_budget_header = request.headers.get(
                                "custom_reasoning_budget"
                            )

                        if custom_budget_header is not None:
                            is_budget_enabled = custom_budget_header.lower() == "true"
                            litellm_kwargs["custom_reasoning_budget"] = (
                                is_budget_enabled
                            )

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
                        litellm_kwargs["enable_request_logging"] = (
                            self.enable_request_logging
                        )

                        for attempt in range(self.max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting stream with credential ...{current_cred[-6:]} (Attempt {attempt + 1}/{self.max_retries})"
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
                                    f"Stream connection established for credential ...{current_cred[-6:]}. Processing response."
                                )

                                key_acquired = False
                                stream_generator = self._safe_streaming_wrapper(
                                    response, current_cred, model, request
                                )

                                async for chunk in stream_generator:
                                    yield chunk
                                return

                            except (
                                StreamedAPIError,
                                litellm.RateLimitError,
                                httpx.HTTPStatusError,
                            ) as e:
                                if (
                                    isinstance(e, httpx.HTTPStatusError)
                                    and e.response.status_code != 429
                                ):
                                    raise e

                                last_exception = e
                                # If the exception is our custom wrapper, unwrap the original error
                                original_exc = getattr(e, "data", e)
                                classified_error = classify_error(original_exc)
                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} encountered a recoverable error ({classified_error.error_type}) during custom provider stream. Rotating key."
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
                                classified_error = classify_error(e)
                                # Provider-level error: don't increment consecutive failures
                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error,
                                    increment_consecutive_failures=False
                                )

                                if attempt >= self.max_retries - 1:
                                    lib_logger.warning(
                                        f"Credential ...{current_cred[-6:]} failed after max retries for model {model} due to a server error. Rotating key."
                                    )
                                    break

                                wait_time = classified_error.retry_after or (
                                    1 * (2**attempt)
                                ) + random.uniform(0, 1)
                                remaining_budget = deadline - time.time()
                                if wait_time > remaining_budget:
                                    lib_logger.warning(
                                        f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early."
                                    )
                                    break

                                error_message = str(e).split("\n")[0]
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} encountered a server error for model {model}. Reason: '{error_message}'. Retrying in {wait_time:.2f}s."
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
                                classified_error = classify_error(e)
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {str(e)}. Rotating key."
                                )
                                if classified_error.error_type in [
                                    "invalid_request",
                                    "context_window_exceeded",
                                    "authentication",
                                ]:
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
                            self._apply_default_safety_settings(litellm_kwargs, provider)
                        except Exception:
                            lib_logger.debug("Could not apply default safety settings for streaming path; continuing.")

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
                                f"Attempting stream with credential ...{current_cred[-6:]} (Attempt {attempt + 1}/{self.max_retries})"
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
                                f"Stream connection established for credential ...{current_cred[-6:]}. Processing response."
                            )

                            key_acquired = False
                            stream_generator = self._safe_streaming_wrapper(
                                response, current_cred, model, request
                            )

                            async for chunk in stream_generator:
                                yield chunk
                            return

                        except (StreamedAPIError, litellm.RateLimitError) as e:
                            last_exception = e

                            # This is the final, robust handler for streamed errors.
                            error_payload = {}
                            cleaned_str = None
                            # The actual exception might be wrapped in our StreamedAPIError.
                            original_exc = getattr(e, "data", e)
                            classified_error = classify_error(original_exc)

                            try:
                                # The full error JSON is in the string representation of the exception.
                                json_str_match = re.search(
                                    r"(\{.*\})", str(original_exc), re.DOTALL
                                )
                                if json_str_match:
                                    # The string may contain byte-escaped characters (e.g., \\n).
                                    cleaned_str = codecs.decode(
                                        json_str_match.group(1), "unicode_escape"
                                    )
                                    error_payload = json.loads(cleaned_str)
                            except (json.JSONDecodeError, TypeError):
                                lib_logger.warning(
                                    "Could not parse JSON details from streamed error exception."
                                )
                                error_payload = {}

                            # Now, log the failure with the extracted raw response.
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
                            # Fallback to the full string if parsing fails.
                            error_message_text = error_details.get(
                                "message", str(original_exc)
                            )

                            if (
                                "quota" in error_message_text.lower()
                                or "resource_exhausted" in error_status.lower()
                            ):
                                consecutive_quota_failures += 1
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} hit a quota limit. This is consecutive failure #{consecutive_quota_failures} for this request."
                                )

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
                                    console_log_message = (
                                        f"Terminating stream for credential ...{current_cred[-6:]} due to 3rd consecutive quota error. "
                                        f"This is now considered a fatal input data error. ID: {quota_id}, Limit: {quota_value}."
                                    )
                                    client_error_message = (
                                        "FATAL: Request failed after 3 consecutive quota errors, "
                                        "indicating the input data is too large for the model's per-request limit. "
                                        f"Last Error Message: '{error_message_text}'. Limit: {quota_value} (Quota ID: {quota_id})."
                                    )
                                    lib_logger.error(console_log_message)

                                    yield f"data: {json.dumps({'error': {'message': client_error_message, 'type': 'proxy_fatal_quota_error'}})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                else:
                                    # [MODIFIED] Do not yield to the client. Just log and break to rotate the key.
                                    lib_logger.warning(
                                        f"Quota error on credential ...{current_cred[-6:]} (failure {consecutive_quota_failures}/3). Rotating key silently."
                                    )
                                    break

                            else:
                                consecutive_quota_failures = 0
                                # [MODIFIED] Do not yield to the client. Just log and break to rotate the key.
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} encountered a recoverable error ({classified_error.error_type}) during stream. Rotating key silently."
                                )

                                if (
                                    classified_error.error_type == "rate_limit"
                                    and classified_error.status_code == 429
                                ):
                                    cooldown_duration = (
                                        classified_error.retry_after or 60
                                    )
                                    await self.cooldown_manager.start_cooldown(
                                        provider, cooldown_duration
                                    )
                                    lib_logger.warning(
                                        f"IP-based rate limit detected for {provider}. Starting a {cooldown_duration}-second global cooldown."
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
                            classified_error = classify_error(e)
                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error,
                                increment_consecutive_failures=False
                            )

                            if attempt >= self.max_retries - 1:
                                lib_logger.warning(
                                    f"Credential ...{current_cred[-6:]} failed after max retries for model {model} due to a server error. Rotating key silently."
                                )
                                # [MODIFIED] Do not yield to the client here.
                                break

                            wait_time = classified_error.retry_after or (
                                1 * (2**attempt)
                            ) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()
                            if wait_time > remaining_budget:
                                lib_logger.warning(
                                    f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early."
                                )
                                break

                            error_message = str(e).split("\n")[0]
                            lib_logger.warning(
                                f"Credential ...{current_cred[-6:]} encountered a server error for model {model}. Reason: '{error_message}'. Retrying in {wait_time:.2f}s."
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
                            classified_error = classify_error(e)

                            lib_logger.warning(
                                f"Credential ...{current_cred[-6:]} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {str(e)}. Rotating key."
                            )

                            if classified_error.status_code == 429:
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(
                                    provider, cooldown_duration
                                )
                                lib_logger.warning(
                                    f"IP-based rate limit detected for {provider} from generic stream exception. Starting a {cooldown_duration}-second global cooldown."
                                )

                            if classified_error.error_type in [
                                "invalid_request",
                                "context_window_exceeded",
                                "authentication",
                            ]:
                                raise last_exception

                            # [MODIFIED] Do not yield to the client here.
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break

                finally:
                    if key_acquired and current_cred:
                        await self.usage_manager.release_key(current_cred, model)

            final_error_message = "Failed to complete the streaming request: No available API keys after rotation or global timeout exceeded."
            if last_exception:
                final_error_message = f"Failed to complete the streaming request. Last error: {str(last_exception)}"
                lib_logger.error(
                    f"Streaming request failed after trying all keys. Last error: {last_exception}"
                )
            else:
                lib_logger.error(final_error_message)

            error_data = {
                "error": {"message": final_error_message, "type": "proxy_error"}
            }
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
            lib_logger.debug("Removing stream_options for iflow provider to avoid HTTP 406")
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
        """Calculates the number of tokens for a given text or list of messages."""
        kwargs = self._convert_model_params(**kwargs)
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is a required parameter.")
        if messages:
            return token_counter(model=model, messages=messages)
        elif text:
            return token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided.")

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
                    cred_display = (
                        credential[-6:]
                        if not os.path.isfile(credential)
                        else os.path.basename(credential)
                    )
                    lib_logger.debug(
                        f"Attempting to get models for {provider} with credential ...{cred_display}"
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
                    classified_error = classify_error(e)
                    cred_display = (
                        credential[-6:]
                        if not os.path.isfile(credential)
                        else os.path.basename(credential)
                    )
                    lib_logger.debug(
                        f"Failed to get models for provider {provider} with credential ...{cred_display}: {classified_error.error_type}. Trying next credential."
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
