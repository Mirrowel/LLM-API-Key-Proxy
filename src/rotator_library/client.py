import asyncio
import json
import os
import random
import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
import logging
from typing import List, Dict, Any, AsyncGenerator

# Set up a dedicated logger for the library
lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False

# You might want to add a handler if you want to see these logs specifically
# For example, a NullHandler to avoid "No handler found" warnings if the
# main app doesn't configure this logger.
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

from .usage_manager import UsageManager
from .failure_logger import log_failure
from .error_handler import is_rate_limit_error, is_server_error, is_unrecoverable_error
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

    async def _streaming_wrapper(self, stream: Any, key: str, model: str) -> AsyncGenerator[Any, None]:
        """
        A wrapper for streaming responses that formats the output as OpenAI-compatible
        Server-Sent Events (SSE) and records usage.
        """
        try:
            async for chunk in stream:
                #lib_logger.info(f"STREAM CHUNK: {chunk}")
                # Convert the litellm chunk object to a dictionary
                chunk_dict = chunk.dict()
                
                # Format as a Server-Sent Event
                yield f"data: {json.dumps(chunk_dict)}\n\n"

                # Safely check for usage data in the chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    lib_logger.info(f"Usage found in chunk for key ...{key[-4:]}: {chunk.usage}")
                    await self.usage_manager.record_success(key, model, chunk)

        finally:
            # Signal the end of the stream
            yield "data: [DONE]\n\n"
            lib_logger.info("STREAM FINISHED and [DONE] signal sent.")


    async def acompletion(self, pre_request_callback: callable = None, **kwargs) -> Any:
        """
        Performs a completion call with smart key rotation and retry logic.
        Handles both streaming and non-streaming requests with thread-safe key acquisition.
        """
        model = kwargs.get("model")
        is_streaming = kwargs.get("stream", False)

        if not model:
            raise ValueError("'model' is a required parameter.")

        provider = model.split('/')[0]
        if provider not in self.api_keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        current_key = None
        try:
            while True: # Loop to acquire a key and make the call
                current_key = await self.usage_manager.acquire_key(
                    available_keys=self.api_keys[provider],
                    model=model
                )

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        
                        litellm_kwargs = kwargs.copy()
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
                            return self._streaming_wrapper(response, current_key, model)
                        else:
                            await self.usage_manager.record_success(current_key, model, response)
                            return response

                    except Exception as e:
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_data=kwargs)

                        if is_unrecoverable_error(e):
                            lib_logger.error(f"Key ...{current_key[-4:]} failed with unrecoverable error: {e}. Raising exception.")
                            raise e

                        if is_rate_limit_error(e):
                            lib_logger.warning(f"Key ...{current_key[-4:]} hit a rate limit for model {model}. Rotating key and setting cooldown.")
                            await self.usage_manager.record_rotation_error(current_key, model, e)
                            break  # Break from retries to get a new key

                        if is_server_error(e):
                            if attempt < self.max_retries - 1:
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                lib_logger.warning(f"Key ...{current_key[-4:]} encountered a server error. Retrying in {wait_time:.2f} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                lib_logger.error(f"Key ...{current_key[-4:]} failed after max retries on a server error. Rotating key.")
                                await self.usage_manager.record_rotation_error(current_key, model, e)
                                break
                        
                        # Fallback for any other unexpected errors
                        lib_logger.error(f"Key ...{current_key[-4:]} failed with an unexpected error: {e}. Rotating key.")
                        await self.usage_manager.record_rotation_error(current_key, model, e)
                        break
                
                # If we exit the retry loop due to failure, release the key and try to get a new one.
                await self.usage_manager.release_key(current_key)
                current_key = None # Ensure key is not released again in finally

        finally:
            if current_key:
                await self.usage_manager.release_key(current_key)

    def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:
        """
        Calculates the number of tokens for a given text or list of messages.
        """
        if messages:
            return token_counter(model=model, messages=messages)
        elif text:
            return token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided.")

    async def get_available_models(self, provider: str) -> List[str]:
        """
        Returns a list of available models for a specific provider, with caching.
        """
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
        """
        Returns a list of all available models, either grouped by provider or as a flat list.
        """
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
