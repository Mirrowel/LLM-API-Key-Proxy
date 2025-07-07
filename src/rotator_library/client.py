import asyncio
import json
import os
import random
import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False

if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

from .usage_manager import UsageManager
from .failure_logger import log_failure
from .error_handler import classify_error, AllProviders
from .providers import PROVIDER_PLUGINS
from .request_sanitizer import sanitize_request_payload

class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """
    def __init__(self, api_keys: Dict[str, List[str]], max_retries: int = 2, usage_file_path: str = "key_usage.json"):
        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        litellm.drop_params = True
        if not api_keys:
            raise ValueError("API keys dictionary cannot be empty.")
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.usage_manager = UsageManager(file_path=usage_file_path)
        self._model_list_cache = {}
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = {}
        self.http_client = httpx.AsyncClient()
        self.all_providers = AllProviders()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client to prevent resource leaks."""
        if hasattr(self, 'http_client') and self.http_client:
            await self.http_client.aclose()

    def _get_provider_instance(self, provider_name: str):
        """Lazily initializes and returns a provider instance."""
        if provider_name not in self._provider_instances:
            if provider_name in self._provider_plugins:
                self._provider_instances[provider_name] = self._provider_plugins[provider_name]()
            else:
                return None
        return self._provider_instances[provider_name]

    async def _safe_streaming_wrapper(self, stream: Any, key: str, model: str) -> AsyncGenerator[Any, None]:
        """
        A definitive hybrid wrapper for streaming responses that ensures usage is recorded
        and the key lock is released only after the stream is fully consumed.
        It exhaustively checks for usage data in all possible locations and gracefully
        handles JSON decoding errors from the stream.
        """
        usage_recorded = False
        stream_completed = False
        stream_iterator = stream.__aiter__()
        
        try:
            while True:
                try:
                    chunk = await stream_iterator.__anext__()
                    yield f"data: {json.dumps(chunk.dict())}\n\n"
                    
                    # 1. First, try to find usage in a chunk (for providers that send it mid-stream)
                    if not usage_recorded and hasattr(chunk, 'usage') and chunk.usage:
                        await self.usage_manager.record_success(key, model, chunk)
                        usage_recorded = True
                        lib_logger.info(f"Recorded usage from stream chunk for key ...{key[-4:]}")
                
                except StopAsyncIteration:
                    # The stream finished successfully.
                    stream_completed = True
                    break
                
                except Exception as e:
                    # This will catch JSONDecodeError and other potential issues with a chunk
                    lib_logger.warning(f"Skipping a malformed chunk for key ...{key[-4:]}: {e}")
                    continue # Skip to the next chunk

        finally:
            # 2. If not found in a chunk, try the final stream object itself (for other providers)
            if not usage_recorded:
                # This call is now safe because record_success is robust
                await self.usage_manager.record_success(key, model, stream)
                lib_logger.info(f"Recorded usage from final stream object for key ...{key[-4:]}")

            # 3. Release the key only after all attempts to record usage are complete
            await self.usage_manager.release_key(key, model)
            lib_logger.info(f"STREAM FINISHED and lock released for key ...{key[-4:]}.")
            
            # Only yield [DONE] if the stream completed successfully
            if stream_completed:
                yield "data: [DONE]\n\n"


    async def acompletion(self, pre_request_callback: Optional[callable] = None, **kwargs) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Performs a completion call with smart key rotation and retry logic.
        It will try each available key in sequence if the previous one fails.
        """
        model = kwargs.get("model")
        is_streaming = kwargs.get("stream", False)

        if not model:
            raise ValueError("'model' is a required parameter.")

        provider = model.split('/')[0]
        if provider not in self.api_keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        keys_for_provider = self.api_keys[provider]
        tried_keys = set()
        last_exception = None
        
        while len(tried_keys) < len(keys_for_provider):
            current_key = None
            key_acquired = False
            try:
                keys_to_try = [k for k in keys_for_provider if k not in tried_keys]
                if not keys_to_try:
                    break 

                current_key = await self.usage_manager.acquire_key(
                    available_keys=keys_to_try,
                    model=model
                )
                key_acquired = True
                tried_keys.add(current_key)

                # Prepare litellm_kwargs once per key, not on every retry
                litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())

                provider_instance = self._get_provider_instance(provider)
                if provider_instance:
                    
                    # Ensure safety_settings are present, defaulting to lowest if not provided
                    if "safety_settings" not in litellm_kwargs:
                        litellm_kwargs["safety_settings"] = {
                            "harassment": "BLOCK_NONE",
                            "hate_speech": "BLOCK_NONE",
                            "sexually_explicit": "BLOCK_NONE",
                            "dangerous_content": "BLOCK_NONE",
                        }

                    converted_settings = provider_instance.convert_safety_settings(litellm_kwargs["safety_settings"])
                    
                    if converted_settings is not None:
                        litellm_kwargs["safety_settings"] = converted_settings
                    else:
                        # If conversion returns None, remove it to avoid sending empty settings
                        del litellm_kwargs["safety_settings"]
                
                if provider == "gemini":
                    if provider_instance:
                        provider_instance.handle_thinking_parameter(litellm_kwargs, model)

                if "gemma-3" in model and "messages" in litellm_kwargs:
                    new_messages = [
                        {"role": "user", "content": m["content"]} if m.get("role") == "system" else m
                        for m in litellm_kwargs["messages"]
                    ]
                    litellm_kwargs["messages"] = new_messages
                
                if provider == "chutes":
                    litellm_kwargs["model"] = f"openai/{model.split('/', 1)[1]}"
                    litellm_kwargs["api_base"] = "https://llm.chutes.ai/v1"

                litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        
                        if pre_request_callback:
                            await pre_request_callback()

                        response = await litellm.acompletion(api_key=current_key, **litellm_kwargs)

                        if is_streaming:
                            # The wrapper is now responsible for releasing the key.
                            key_acquired = False  # Transfer responsibility to wrapper
                            return self._safe_streaming_wrapper(response, current_key, model)
                        else:
                            # For non-streaming, record and release here.
                            await self.usage_manager.record_success(current_key, model, response)
                            await self.usage_manager.release_key(current_key, model)
                            key_acquired = False # Key has been released
                            return response

                    except Exception as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)
                        
                        classified_error = classify_error(e)

                        if classified_error.error_type in ['invalid_request', 'context_window_exceeded']:
                            # These errors are not recoverable by rotating keys, so fail fast.
                            lib_logger.error(f"Unrecoverable error '{classified_error.error_type}' with key ...{current_key[-4:]}. Failing request.")
                            raise last_exception

                        if classified_error.error_type in ['server_error', 'api_connection']:
                            # These are temporary, so record the failure and retry with backoff.
                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            
                            if attempt >= self.max_retries - 1:
                                lib_logger.warning(f"Key ...{current_key[-4:]} failed on final retry for {classified_error.error_type}. Trying next key.")
                                break

                            # Use a longer cooldown for API connection errors
                            base_wait = 5 if classified_error.error_type == 'api_connection' else 1
                            wait_time = classified_error.retry_after or (base_wait * (2 ** attempt)) + random.uniform(0, 1)
                            
                            lib_logger.warning(f"Key ...{current_key[-4:]} encountered a {classified_error.error_type}. Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                            continue

                        # For other errors (rate_limit, authentication, unknown), record failure and try the next key.
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        lib_logger.warning(f"Key ...{current_key[-4:]} encountered '{classified_error.error_type}'. Trying next key.")
                        break
            finally:
                # This block ensures the key is always released if it was acquired but not passed to the wrapper.
                if key_acquired and current_key:
                    await self.usage_manager.release_key(current_key, model)

        if last_exception:
            raise last_exception
        
        raise Exception("Failed to complete the request: No available API keys for the provider or all keys failed.")

    def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:
        """Calculates the number of tokens for a given text or list of messages."""
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
            lib_logger.info(f"Returning cached models for provider: {provider}")
            return self._model_list_cache[provider]

        keys_for_provider = self.api_keys.get(provider)
        if not keys_for_provider:
            lib_logger.warning(f"No API key for provider: {provider}")
            return []

        # Create a copy and shuffle it to randomize the starting key
        shuffled_keys = list(keys_for_provider)
        random.shuffle(shuffled_keys)

        provider_instance = self._get_provider_instance(provider)
        if provider_instance:
            for api_key in shuffled_keys:
                try:
                    lib_logger.info(f"Attempting to get models for {provider} with key ...{api_key[-4:]}")
                    models = await provider_instance.get_models(api_key, self.http_client)
                    lib_logger.info(f"Got {len(models)} models for provider: {provider}")
                    self._model_list_cache[provider] = models
                    return models
                except Exception as e:
                    classified_error = classify_error(e)
                    lib_logger.warning(f"Failed to get models for provider {provider} with key ...{api_key[-4:]}: {classified_error.error_type}. Trying next key.")
                    continue # Try the next key
        
        lib_logger.error(f"Failed to get models for provider {provider} after trying all keys.")
        return []

    async def get_all_available_models(self, grouped: bool = True) -> Union[Dict[str, List[str]], List[str]]:
        """Returns a list of all available models, either grouped by provider or as a flat list."""
        lib_logger.info("Getting all available models...")
        tasks = [self.get_available_models(provider) for provider in self.api_keys.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_provider_models = {}
        for provider, result in zip(self.api_keys.keys(), results):
            if isinstance(result, Exception):
                lib_logger.error(f"Failed to get models for provider {provider}: {result}")
                all_provider_models[provider] = []
            else:
                all_provider_models[provider] = result
        
        lib_logger.info("Finished getting all available models.")
        if grouped:
            return all_provider_models
        else:
            flat_models = []
            for provider, models in all_provider_models.items():
                for model in models:
                    flat_models.append(f"{provider}/{model}")
            return flat_models
