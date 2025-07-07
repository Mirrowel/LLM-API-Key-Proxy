import asyncio
import json
import os
import random
import httpx
import litellm
from litellm.exceptions import APIConnectionError
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
from .cooldown_manager import CooldownManager

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
        self.cooldown_manager = CooldownManager()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client to prevent resource leaks."""
        if hasattr(self, 'http_client') and self.http_client:
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

        provider = model.split('/')[0]
        if provider == "chutes":
            kwargs["model"] = f"openai/{model.split('/', 1)[1]}"
            kwargs["api_base"] = "https://llm.chutes.ai/v1"
        
        return kwargs

    def _get_provider_instance(self, provider_name: str):
        """Lazily initializes and returns a provider instance."""
        if provider_name not in self._provider_instances:
            if provider_name in self._provider_plugins:
                self._provider_instances[provider_name] = self._provider_plugins[provider_name]()
            else:
                return None
        return self._provider_instances[provider_name]

    async def _safe_streaming_wrapper(self, stream: Any, key: str, model: str, request: Optional[Any] = None) -> AsyncGenerator[Any, None]:
        """
        A hybrid wrapper for streaming that buffers fragmented JSON, handles client disconnections gracefully,
        and distinguishes between content and streamed errors.
        """
        usage_recorded = False
        stream_completed = False
        stream_iterator = stream.__aiter__()
        json_buffer = ""

        try:
            while True:
                if request and await request.is_disconnected():
                    lib_logger.warning(f"Client disconnected. Aborting stream for key ...{key[-4:]}.")
                    # Do not yield [DONE] because the client is gone.
                    # The 'finally' block will handle key release.
                    break

                try:
                    chunk = await stream_iterator.__anext__()
                    if json_buffer:
                        lib_logger.warning(f"Discarding incomplete JSON buffer: {json_buffer}")
                        json_buffer = ""
                    
                    yield f"data: {json.dumps(chunk.dict())}\n\n"

                    if not usage_recorded and hasattr(chunk, 'usage') and chunk.usage:
                        await self.usage_manager.record_success(key, model, chunk)
                        usage_recorded = True

                except StopAsyncIteration:
                    stream_completed = True
                    if json_buffer:
                        lib_logger.warning(f"Stream ended with incomplete data in buffer: {json_buffer}")
                    break

                except Exception as e:
                    try:
                        raw_chunk = str(e).split("Received chunk:")[-1].strip()
                        json_buffer += raw_chunk
                        parsed_data = json.loads(json_buffer)
                        
                        lib_logger.info(f"Successfully reassembled JSON from buffer: {json_buffer}")
                        
                        if "error" in parsed_data:
                            lib_logger.warning(f"Reassembled object is an API error. Passing it to the client and raising internally.")
                            yield f"data: {json.dumps(parsed_data)}\n\n"
                            # Signal the error to the outer retry loop so it can try the next key.
                            raise StreamedAPIError("Provider error received in stream", data=parsed_data)
                        else:
                            yield f"data: {json.dumps(parsed_data)}\n\n"
                        
                        json_buffer = ""
                    except json.JSONDecodeError:
                        lib_logger.info(f"Buffer still incomplete. Waiting for more chunks: {json_buffer}")
                        continue
                    except StreamedAPIError:
                        # Re-raise to be caught by the outer handler
                        raise
                    except Exception as buffer_exc:
                        lib_logger.error(f"Error during stream buffering logic: {buffer_exc}. Discarding buffer.")
                        json_buffer = ""
                        continue
        
        except StreamedAPIError:
            # This is caught by the acompletion retry logic.
            # We re-raise it to ensure it's not caught by the generic 'except Exception'.
            raise

        except Exception as e:
            # Catch any other unexpected errors during streaming.
            lib_logger.error(f"An unexpected error occurred during the stream for key ...{key[-4:]}: {e}")
            # We still need to raise it so the client knows something went wrong.
            raise

        finally:
            # Only record usage if the stream completed successfully and usage wasn't already recorded.
            if stream_completed and not usage_recorded:
                await self.usage_manager.record_success(key, model, stream)
            
            await self.usage_manager.release_key(key, model)
            lib_logger.info(f"STREAM FINISHED and lock released for key ...{key[-4:]}.")
            
            if stream_completed:
                yield "data: [DONE]\n\n"

    async def acompletion(self, request: Optional[Any] = None, pre_request_callback: Optional[callable] = None, **kwargs) -> Union[Any, AsyncGenerator[str, None]]:
        kwargs = self._convert_model_params(**kwargs)
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
                if await self.cooldown_manager.is_cooling_down(provider):
                    remaining_time = await self.cooldown_manager.get_cooldown_remaining(provider)
                    lib_logger.warning(f"Provider {provider} is in cooldown. Waiting for {remaining_time:.2f} seconds.")
                    await asyncio.sleep(remaining_time)

                keys_to_try = [k for k in keys_for_provider if k not in tried_keys]
                if not keys_to_try:
                    break

                current_key = await self.usage_manager.acquire_key(available_keys=keys_to_try, model=model)
                key_acquired = True
                tried_keys.add(current_key)

                litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())
                provider_instance = self._get_provider_instance(provider)
                if provider_instance:
                    if "safety_settings" not in litellm_kwargs:
                        litellm_kwargs["safety_settings"] = {"harassment": "BLOCK_NONE", "hate_speech": "BLOCK_NONE", "sexually_explicit": "BLOCK_NONE", "dangerous_content": "BLOCK_NONE"}
                    converted_settings = provider_instance.convert_safety_settings(litellm_kwargs["safety_settings"])
                    if converted_settings is not None: litellm_kwargs["safety_settings"] = converted_settings
                    else: del litellm_kwargs["safety_settings"]
                
                if provider == "gemini":
                    if provider_instance: provider_instance.handle_thinking_parameter(litellm_kwargs, model)

                if "gemma-3" in model and "messages" in litellm_kwargs:
                    litellm_kwargs["messages"] = [{"role": "user", "content": m["content"]} if m.get("role") == "system" else m for m in litellm_kwargs["messages"]]
                
                litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        
                        if pre_request_callback:
                            await pre_request_callback()

                        response = await litellm.acompletion(api_key=current_key, **litellm_kwargs)

                        if is_streaming:
                            key_acquired = False # The wrapper will handle the key release
                            stream_generator = self._safe_streaming_wrapper(response, current_key, model, request)
                            async for chunk in stream_generator:
                                yield chunk
                            return
                        else:
                            await self.usage_manager.record_success(current_key, model, response)
                            await self.usage_manager.release_key(current_key, model)
                            key_acquired = False
                            return response

                    except (StreamedAPIError, APIConnectionError, litellm.RateLimitError) as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)
                        classified_error = classify_error(e)

                        if classified_error.error_type == 'rate_limit' and classified_error.status_code == 429:
                            cooldown_duration = classified_error.retry_after or 60
                            await self.cooldown_manager.start_cooldown(provider, cooldown_duration)
                            lib_logger.error(f"IP-based rate limit detected for {provider}. Starting a {cooldown_duration}-second global cooldown.")
                        
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        lib_logger.warning(f"Key ...{current_key[-4:]} encountered '{classified_error.error_type}'. Trying next key.")
                        break

                    except Exception as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)
                        
                        if request and await request.is_disconnected():
                            lib_logger.warning(f"Client disconnected. Aborting retries for key ...{current_key[-4:]}.")
                            raise last_exception

                        classified_error = classify_error(e)
                        if classified_error.error_type in ['invalid_request', 'context_window_exceeded']:
                            raise last_exception

                        if classified_error.error_type in ['server_error', 'api_connection']:
                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            if attempt >= self.max_retries - 1:
                                break
                            wait_time = classified_error.retry_after or (1 * (2 ** attempt)) + random.uniform(0, 1)
                            await asyncio.sleep(wait_time)
                            continue

                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        break
            finally:
                if key_acquired and current_key:
                    await self.usage_manager.release_key(current_key, model)

        if last_exception:
            raise last_exception
        
        raise Exception("Failed to complete the request: No available API keys for the provider or all keys failed.")

    async def aembedding(self, request: Optional[Any] = None, **kwargs) -> Any:
        kwargs = self._convert_model_params(**kwargs)
        model = kwargs.get("model")
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

                current_key = await self.usage_manager.acquire_key(available_keys=keys_to_try, model=model)
                key_acquired = True
                tried_keys.add(current_key)

                litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())
                litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting embedding call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        response = await litellm.aembedding(api_key=current_key, **litellm_kwargs)
                        
                        await self.usage_manager.record_success(current_key, model, response)
                        return response

                    except Exception as e:
                        last_exception = e
                        if isinstance(e, asyncio.CancelledError): raise e

                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)
                        classified_error = classify_error(e)

                        if classified_error.error_type in ['invalid_request', 'context_window_exceeded']:
                            raise last_exception
                        
                        if request and await request.is_disconnected():
                            lib_logger.warning(f"Client disconnected during embedding. Aborting retries for key ...{current_key[-4:]}.")
                            raise last_exception

                        if classified_error.error_type in ['server_error', 'api_connection']:
                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            if attempt >= self.max_retries - 1: break
                            wait_time = classified_error.retry_after or (1 * (2 ** attempt)) + random.uniform(0, 1)
                            await asyncio.sleep(wait_time)
                            continue
                        
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        break
            
            except (litellm.InvalidRequestError, litellm.ContextWindowExceededError, asyncio.CancelledError) as e:
                raise e
            except Exception as e:
                last_exception = e
                lib_logger.error(f"An unexpected error occurred with key ...{current_key[-4:] if current_key else 'N/A'}: {e}")
                continue
            finally:
                if key_acquired and current_key:
                    await self.usage_manager.release_key(current_key, model)

        if last_exception:
            raise last_exception
        
        raise Exception("Failed to complete the request: No available API keys or all keys failed.")

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
