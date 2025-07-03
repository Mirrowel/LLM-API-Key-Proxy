import asyncio
import json
import os
import random
import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
import logging
from typing import List, Dict, Any, AsyncGenerator

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False

if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

from .usage_manager import UsageManager
from .failure_logger import log_failure
from .error_handler import classify_error, AllProviders
from .providers import PROVIDER_PLUGINS

class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """
    def __init__(self, api_keys: Dict[str, List[str]], max_retries: int = 2, usage_file_path: str = "key_usage.json"):
        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        if not api_keys:
            raise ValueError("API keys dictionary cannot be empty.")
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.usage_manager = UsageManager(file_path=usage_file_path)
        self._model_list_cache = {}
        self._provider_instances = {
            name: plugin() for name, plugin in PROVIDER_PLUGINS.items()
        }
        self.http_client = httpx.AsyncClient()
        self.all_providers = AllProviders()

    async def _gemini_stream_wrapper(self, stream: Any, key: str, model: str) -> AsyncGenerator[Any, None]:
        """
        A wrapper specifically for Gemini streams to handle potential JSON chunking issues.
        It buffers and reassembles chunks before yielding them.
        """
        lib_logger.warning("Using Gemini stream wrapper to reassemble JSON chunks.")
        buffer = ""
        usage_recorded = False
        try:
            async for chunk in stream:
                buffer += chunk.choices[0].delta.content or ""
                try:
                    # Try to parse the buffer as a complete JSON object
                    parsed_chunk = json.loads(buffer)
                    # If successful, yield it and clear the buffer
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': json.dumps(parsed_chunk)}}]})}"
                    buffer = ""
                    
                    # Record usage if present in the parsed chunk
                    if not usage_recorded and isinstance(parsed_chunk, dict) and parsed_chunk.get('usage'):
                        await self.usage_manager.record_success(key, model, parsed_chunk)
                        usage_recorded = True
                        lib_logger.info(f"Recorded usage from reassembled Gemini chunk for key ...{key[-4:]}")

                except json.JSONDecodeError:
                    # If it fails, it's an incomplete chunk, so we continue buffering
                    lib_logger.debug("Incomplete JSON chunk received from Gemini, buffering...")
                    continue
        finally:
            if not usage_recorded:
                await self.usage_manager.record_success(key, model, stream)
                lib_logger.info(f"Recorded usage from final stream object for key ...{key[-4:]}")
            
            await self.usage_manager.release_key(key, model)
            lib_logger.info(f"GEMINI STREAM FINISHED and lock released for key ...{key[-4:]}.")
            yield "data: [DONE]\n\n"

    async def _safe_streaming_wrapper(self, stream: Any, key: str, model: str) -> AsyncGenerator[Any, None]:
        """
        A definitive hybrid wrapper for streaming responses that ensures usage is recorded
        and the key lock is released only after the stream is fully consumed.
        """
        usage_recorded = False
        try:
            async for chunk in stream:
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                if not usage_recorded and hasattr(chunk, 'usage') and chunk.usage:
                    await self.usage_manager.record_success(key, model, chunk)
                    usage_recorded = True
                    lib_logger.info(f"Recorded usage from stream chunk for key ...{key[-4:]}")
        finally:
            if not usage_recorded:
                await self.usage_manager.record_success(key, model, stream)
                lib_logger.info(f"Recorded usage from final stream object for key ...{key[-4:]}")

            await self.usage_manager.release_key(key, model)
            lib_logger.info(f"STREAM FINISHED and lock released for key ...{key[-4:]}.")
            yield "data: [DONE]\n\n"


    async def acompletion(self, pre_request_callback: callable = None, **kwargs) -> Any:
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
            try:
                keys_to_try = [k for k in keys_for_provider if k not in tried_keys]
                if not keys_to_try:
                    break 

                current_key = await self.usage_manager.acquire_key(
                    available_keys=keys_to_try,
                    model=model
                )
                tried_keys.add(current_key)

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        
                        litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())
                        
                        if "gemma-3" in model and "messages" in litellm_kwargs:
                            new_messages = [
                                {"role": "user", "content": m["content"]} if m.get("role") == "system" else m
                                for m in litellm_kwargs["messages"]
                            ]
                            litellm_kwargs["messages"] = new_messages
                        
                        if provider == "chutes":
                            litellm_kwargs["model"] = f"openai/{model.split('/', 1)[1]}"
                            litellm_kwargs["api_base"] = "https://llm.chutes.ai/v1"
                        
                        if pre_request_callback:
                            await pre_request_callback()

                        response = await litellm.acompletion(api_key=current_key, **litellm_kwargs)

                        if is_streaming:
                            # Special handling for Gemini streams due to chunking issues
                            if provider == "gemini":
                                return self._gemini_stream_wrapper(response, current_key, model)
                            return self._safe_streaming_wrapper(response, current_key, model)
                        else:
                            # For non-streaming, record and release here.
                            await self.usage_manager.record_success(current_key, model, response)
                            await self.usage_manager.release_key(current_key, model)
                            return response

                    except Exception as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)
                        
                        classified_error = classify_error(e)
                        
                        if classified_error.error_type in ['invalid_request', 'authentication']:
                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            break 

                        if classified_error.error_type == 'server_error':
                            if attempt < self.max_retries - 1:
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                lib_logger.warning(f"Key ...{current_key[-4:]} encountered a server error. Retrying in {wait_time:.2f} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        break
            finally:
                # This block is now only for handling failures where the key needs to be released
                # without a successful response. The wrapper handles the success case for streams.
                pass

        if last_exception:
            raise last_exception
        
        raise Exception("Failed to complete the request: No available API keys for the provider or all keys failed.")

    def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:
        """Calculates the number of tokens for a given text or list of messages."""
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

        api_key = self.api_keys.get(provider, [None])[0]
        if not api_key:
            lib_logger.warning(f"No API key for provider: {provider}")
            return []

        if provider in self._provider_instances:
            lib_logger.info(f"Calling get_models for provider: {provider}")
            models = await self._provider_instances[provider].get_models(api_key, self.http_client)
            lib_logger.info(f"Got {len(models)} models for provider: {provider}")
            self._model_list_cache[provider] = models
            return models
        else:
            lib_logger.warning(f"Model list fetching not implemented for provider: {provider}")
            return []

    async def get_all_available_models(self, grouped: bool = True) -> Any:
        """Returns a list of all available models, either grouped by provider or as a flat list."""
        lib_logger.info("Getting all available models...")
        all_provider_models = {}
        for provider in self.api_keys.keys():
            lib_logger.info(f"Getting models for provider: {provider}")
            all_provider_models[provider] = await self.get_available_models(provider)
        
        lib_logger.info("Finished getting all available models.")
        if grouped:
            return all_provider_models
        else:
            flat_models = []
            for provider, models in all_provider_models.items():
                for model in models:
                    flat_models.append(f"{model}")
            return flat_models
