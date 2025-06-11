import asyncio
import json
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
import logging
from typing import List, Dict, Any, AsyncGenerator

from src.rotator_library.usage_manager import UsageManager
from src.rotator_library.failure_logger import log_failure
from src.rotator_library.error_handler import is_server_error, is_unrecoverable_error
from src.rotator_library.providers import PROVIDER_PLUGINS

class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """
    def __init__(self, api_keys: Dict[str, List[str]], max_retries: int = 2, usage_file_path: str = "key_usage.json"):
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

    async def _streaming_wrapper(self, stream: Any, key: str, model: str) -> AsyncGenerator[Any, None]:
        """
        A wrapper for streaming responses that formats the output as OpenAI-compatible
        Server-Sent Events (SSE) and records usage.
        """
        try:
            async for chunk in stream:
                #logging.info(f"STREAM CHUNK: {chunk}")
                # Convert the litellm chunk object to a dictionary
                chunk_dict = chunk.dict()
                
                # Format as a Server-Sent Event
                yield f"data: {json.dumps(chunk_dict)}\n\n"

                # Safely check for usage data in the chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    logging.info(f"Usage found in chunk for key ...{key[-4:]}: {chunk.usage}")
                    self.usage_manager.record_success(key, model, chunk)

        finally:
            # Signal the end of the stream
            yield "data: [DONE]\n\n"
            logging.info("STREAM FINISHED and [DONE] signal sent.")


    async def acompletion(self, **kwargs) -> Any:
        """
        Performs a completion call with smart key rotation and retry logic.
        Handles both streaming and non-streaming requests.
        """
        model = kwargs.get("model")
        is_streaming = kwargs.get("stream", False)

        if not model:
            raise ValueError("'model' is a required parameter.")

        provider = model.split('/')[0]
        if provider not in self.api_keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        while True: # Loop until a key succeeds or we decide to give up
            current_key = self.usage_manager.get_next_smart_key(
                available_keys=self.api_keys[provider],
                model=model
            )

            if not current_key:
                print("All keys are currently on cooldown. Waiting...")
                await asyncio.sleep(5) # Wait 5 seconds before checking for an available key again
                continue

            for attempt in range(self.max_retries):
                try:
                    print(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                    response = await litellm.acompletion(api_key=current_key, **kwargs)

                    if is_streaming:
                        # For streams, we return a wrapper generator that logs usage on completion.
                        return self._streaming_wrapper(response, current_key, model)
                    else:
                        # For non-streams, we can log usage immediately.
                        self.usage_manager.record_success(current_key, model, response)
                        return response

                except Exception as e:
                    log_failure(
                        api_key=current_key,
                        model=model,
                        attempt=attempt + 1,
                        error=e,
                        request_data=kwargs
                    )
                    
                    # For any retriable server error, we just continue the attempt loop
                    if is_server_error(e) and attempt < self.max_retries - 1:
                        print(f"Key ...{current_key[-4:]} failed with server error. Retrying...")
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    
                    # For unrecoverable errors, fail fast
                    if is_unrecoverable_error(e):
                        raise e

                    # For all other errors (Auth, RateLimit, or final Server error), record it and break to get a new key
                    print(f"Key ...{current_key[-4:]} failed permanently. Rotating...")
                    self.usage_manager.record_rotation_error(current_key, model, e)
                    break

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
        if provider in self._model_list_cache:
            return self._model_list_cache[provider]

        api_key = self.api_keys.get(provider, [None])[0]
        if not api_key:
            return []

        if provider in self._provider_instances:
            models = await self._provider_instances[provider].get_models(api_key)
            self._model_list_cache[provider] = models
            return models
        else:
            logging.warning(f"Model list fetching not implemented for provider: {provider}")
            return []

    async def get_all_available_models(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary of all available models, grouped by provider.
        """
        all_provider_models = {}
        for provider in self.api_keys.keys():
            all_provider_models[provider] = await self.get_available_models(provider)
        return all_provider_models
