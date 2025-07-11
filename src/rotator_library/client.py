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

lib_logger = logging.getLogger('rotator_library')
# Ensure the logger is configured to propagate to the root logger
# which is set up in main.py. This allows the main app to control
# log levels and handlers centrally.
lib_logger.propagate = False

from .usage_manager import UsageManager
from .failure_logger import log_failure
from .error_handler import classify_error, AllProviders, NoAvailableKeysError
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
    def __init__(self, api_keys: Dict[str, List[str]], max_retries: int = 2, usage_file_path: str = "key_usage.json", configure_logging: bool = True, global_timeout: int = 30):
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

        if not api_keys:
            raise ValueError("API keys dictionary cannot be empty.")
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.global_timeout = global_timeout
        self.usage_manager = UsageManager(file_path=usage_file_path)
        self._model_list_cache = {}
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = {}
        self.http_client = httpx.AsyncClient()
        self.all_providers = AllProviders()
        self.cooldown_manager = CooldownManager()

    def _sanitize_litellm_log(self, log_data: dict) -> dict:
        """
        Recursively removes large data fields and sensitive information from litellm log
        dictionaries to keep debug logs clean and secure.
        """
        if not isinstance(log_data, dict):
            return log_data

        # Keys to remove at any level of the dictionary
        keys_to_pop = [
            "messages", "input", "response", "data", "api_key",
            "api_base", "original_response", "additional_args"
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
        error_info = log_data.get("standard_logging_object", {}).get("error_information", {})
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get("error_message", str(log_data.get("exception", "")))
        error_message = ' '.join(error_message.split()) # Sanitize

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
                    lib_logger.info(f"Client disconnected. Aborting stream for key ...{key[-4:]}.")
                    # Do not yield [DONE] because the client is gone.
                    # The 'finally' block will handle key release.
                    break

                try:
                    chunk = await stream_iterator.__anext__()
                    if json_buffer:
                        # If we are about to discard a buffer, it means data was likely lost.
                        # Log this as a warning to make it visible.
                        lib_logger.warning(f"Discarding incomplete JSON buffer from previous chunk: {json_buffer}")
                        json_buffer = ""
                    
                    yield f"data: {json.dumps(chunk.dict())}\n\n"

                    if not usage_recorded and hasattr(chunk, 'usage') and chunk.usage:
                        await self.usage_manager.record_success(key, model, chunk)
                        usage_recorded = True

                except StopAsyncIteration:
                    stream_completed = True
                    if json_buffer:
                        lib_logger.debug(f"Stream ended with incomplete data in buffer: {json_buffer}")
                    break

                except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.InternalServerError, APIConnectionError) as e:
                    # This is a critical, typed error from litellm that signals a key failure.
                    # We do not try to parse it here. We wrap it and raise it immediately
                    # for the outer retry loop to handle.
                    lib_logger.warning(f"Caught a critical API error mid-stream: {type(e).__name__}. Signaling for key rotation.")
                    raise StreamedAPIError("Provider error received in stream", data=e)

                except Exception as e:
                    try:
                        raw_chunk = ""
                        # Google streams errors inside a bytes representation (b'{...}').
                        # We use regex to extract the content, which is more reliable than splitting.
                        match = re.search(r"b'(\{.*\})'", str(e), re.DOTALL)
                        if match:
                            # The extracted string is unicode-escaped (e.g., '\\n'). We must decode it.
                            raw_chunk = codecs.decode(match.group(1), 'unicode_escape')
                        else:
                            # Fallback for other potential error formats that use "Received chunk:".
                            chunk_from_split = str(e).split("Received chunk:")[-1].strip()
                            if chunk_from_split != str(e): # Ensure the split actually did something
                                raw_chunk = chunk_from_split
                        
                        if not raw_chunk:
                            # If we could not extract a valid chunk, we cannot proceed with reassembly.
                            # This indicates a different, unexpected error type. Re-raise it.
                            raise e

                        # Append the clean chunk to the buffer and try to parse.
                        json_buffer += raw_chunk
                        parsed_data = json.loads(json_buffer)
                        
                        # If parsing succeeds, we have the complete object.
                        lib_logger.debug(f"Successfully reassembled JSON from stream: {json_buffer}")
                        
                        # Wrap the complete error object and raise it. The outer function will decide how to handle it.
                        raise StreamedAPIError("Provider error received in stream", data=parsed_data)

                    except json.JSONDecodeError:
                        # This is the expected outcome if the JSON in the buffer is not yet complete.
                        lib_logger.debug(f"Buffer still incomplete. Waiting for more chunks: {json_buffer}")
                        continue # Continue to the next loop to get the next chunk.
                    except StreamedAPIError:
                        # Re-raise to be caught by the outer retry handler.
                        raise
                    except Exception as buffer_exc:
                        # If the error was not a JSONDecodeError, it's an unexpected internal error.
                        lib_logger.error(f"Error during stream buffering logic: {buffer_exc}. Discarding buffer.")
                        json_buffer = "" # Clear the corrupted buffer to prevent further issues.
                        raise buffer_exc
        
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

    async def _execute_with_retry(self, api_call: callable, request: Optional[Any], **kwargs) -> Any:
        """A generic retry mechanism for non-streaming API calls."""
        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' is a required parameter.")

        provider = model.split('/')[0]
        if provider not in self.api_keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        # Establish a global deadline for the entire request lifecycle.
        deadline = time.time() + self.global_timeout
        
        # Create a mutable copy of the keys and shuffle it to ensure
        # that the key selection is randomized, which is crucial when
        # multiple keys have the same usage stats.
        keys_for_provider = list(self.api_keys[provider])
        random.shuffle(keys_for_provider)
        
        tried_keys = set()
        last_exception = None
        kwargs = self._convert_model_params(**kwargs)
        
        # The main rotation loop. It continues as long as there are untried keys and the global deadline has not been exceeded.
        while len(tried_keys) < len(keys_for_provider) and time.time() < deadline:
            current_key = None
            key_acquired = False
            try:
                # Check for a provider-wide cooldown first.
                if await self.cooldown_manager.is_cooling_down(provider):
                    remaining_cooldown = await self.cooldown_manager.get_cooldown_remaining(provider)
                    remaining_budget = deadline - time.time()
                    
                    # If the cooldown is longer than the remaining time budget, fail fast.
                    if remaining_cooldown > remaining_budget:
                        lib_logger.warning(f"Provider {provider} cooldown ({remaining_cooldown:.2f}s) exceeds remaining request budget ({remaining_budget:.2f}s). Failing early.")
                        break

                    lib_logger.warning(f"Provider {provider} is in cooldown. Waiting for {remaining_cooldown:.2f} seconds.")
                    await asyncio.sleep(remaining_cooldown)

                keys_to_try = [k for k in keys_for_provider if k not in tried_keys]
                if not keys_to_try:
                    break

                current_key = await self.usage_manager.acquire_key(
                    available_keys=keys_to_try, 
                    model=model,
                    deadline=deadline
                )
                key_acquired = True
                tried_keys.add(current_key)

                litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())
                provider_instance = self._get_provider_instance(provider)
                if provider_instance:
                    if "safety_settings" in litellm_kwargs:
                        converted_settings = provider_instance.convert_safety_settings(litellm_kwargs["safety_settings"])
                        if converted_settings is not None:
                            litellm_kwargs["safety_settings"] = converted_settings
                        else:
                            del litellm_kwargs["safety_settings"]
                
                if provider == "gemini" and provider_instance:
                    provider_instance.handle_thinking_parameter(litellm_kwargs, model)

                if "gemma-3" in model and "messages" in litellm_kwargs:
                    litellm_kwargs["messages"] = [{"role": "user", "content": m["content"]} if m.get("role") == "system" else m for m in litellm_kwargs["messages"]]
                
                litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                for attempt in range(self.max_retries):
                    try:
                        lib_logger.info(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                        response = await api_call(
                            api_key=current_key,
                            **litellm_kwargs,
                            logger_fn=self._litellm_logger_callback
                        )
                        
                        await self.usage_manager.record_success(current_key, model, response)
                        await self.usage_manager.release_key(current_key, model)
                        key_acquired = False
                        return response

                    except litellm.RateLimitError as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                        classified_error = classify_error(e)
                        
                        # Extract a clean error message for the user-facing log
                        error_message = str(e).split('\n')[0]
                        lib_logger.info(f"Key ...{current_key[-4:]} hit rate limit for model {model}. Reason: '{error_message}'. Rotating key.")

                        if classified_error.status_code == 429:
                            cooldown_duration = classified_error.retry_after or 60
                            await self.cooldown_manager.start_cooldown(provider, cooldown_duration)
                            lib_logger.warning(f"IP-based rate limit detected for {provider}. Starting a {cooldown_duration}-second global cooldown.")
                        
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        lib_logger.warning(f"Key ...{current_key[-4:]} encountered a rate limit. Trying next key.")
                        break # Move to the next key

                    except (APIConnectionError, litellm.InternalServerError, litellm.ServiceUnavailableError) as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                        classified_error = classify_error(e)
                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        
                        if attempt >= self.max_retries - 1:
                            error_message = str(e).split('\n')[0]
                            lib_logger.warning(f"Key ...{current_key[-4:]} failed after max retries for model {model} due to a server error. Reason: '{error_message}'. Rotating key.")
                            break # Move to the next key
                        
                        # For temporary errors, wait before retrying with the same key.
                        wait_time = classified_error.retry_after or (1 * (2 ** attempt)) + random.uniform(0, 1)
                        remaining_budget = deadline - time.time()
                        
                        # If the required wait time exceeds the budget, don't wait; rotate to the next key immediately.
                        if wait_time > remaining_budget:
                            lib_logger.warning(f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early.")
                            break

                        error_message = str(e).split('\n')[0]
                        lib_logger.warning(f"Key ...{current_key[-4:]} encountered a server error for model {model}. Reason: '{error_message}'. Retrying in {wait_time:.2f}s.")
                        await asyncio.sleep(wait_time)
                        continue # Retry with the same key

                    except Exception as e:
                        last_exception = e
                        log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                        
                        if request and await request.is_disconnected():
                            lib_logger.warning(f"Client disconnected. Aborting retries for key ...{current_key[-4:]}.")
                            raise last_exception

                        classified_error = classify_error(e)
                        error_message = str(e).split('\n')[0]
                        lib_logger.warning(f"Key ...{current_key[-4:]} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {error_message}. Rotating key.")
                        if classified_error.status_code == 429:
                            cooldown_duration = classified_error.retry_after or 60
                            await self.cooldown_manager.start_cooldown(provider, cooldown_duration)
                            lib_logger.warning(f"IP-based rate limit detected for {provider} from generic exception. Starting a {cooldown_duration}-second global cooldown.")

                        if classified_error.error_type in ['invalid_request', 'context_window_exceeded', 'authentication']:
                            # For these errors, we should not retry with other keys.
                            raise last_exception

                        await self.usage_manager.record_failure(current_key, model, classified_error)
                        break # Try next key for other errors
            finally:
                if key_acquired and current_key:
                    await self.usage_manager.release_key(current_key, model)

        if last_exception:
            # Log the final error but do not raise it, as per the new requirement.
            # The client should not see intermittent failures.
            lib_logger.error(f"Request failed after trying all keys or exceeding global timeout. Last error: {last_exception}")
        
        # Return None to indicate failure without propagating a disruptive exception.
        return None

    async def _streaming_acompletion_with_retry(self, request: Optional[Any], **kwargs) -> AsyncGenerator[str, None]:
        """A dedicated generator for retrying streaming completions with full request preparation and per-key retries."""
        model = kwargs.get("model")
        provider = model.split('/')[0]
        
        # Create a mutable copy of the keys and shuffle it.
        keys_for_provider = list(self.api_keys[provider])
        random.shuffle(keys_for_provider)
        
        deadline = time.time() + self.global_timeout
        tried_keys = set()
        last_exception = None
        kwargs = self._convert_model_params(**kwargs)
        try:
            while len(tried_keys) < len(keys_for_provider) and time.time() < deadline:
                current_key = None
                key_acquired = False
                try:
                    if await self.cooldown_manager.is_cooling_down(provider):
                        remaining_cooldown = await self.cooldown_manager.get_cooldown_remaining(provider)
                        remaining_budget = deadline - time.time()
                        if remaining_cooldown > remaining_budget:
                            lib_logger.warning(f"Provider {provider} cooldown ({remaining_cooldown:.2f}s) exceeds remaining request budget ({remaining_budget:.2f}s). Failing early.")
                            break
                        lib_logger.warning(f"Provider {provider} is in a global cooldown. All requests to this provider will be paused for {remaining_cooldown:.2f} seconds.")
                        await asyncio.sleep(remaining_cooldown)

                    keys_to_try = [k for k in keys_for_provider if k not in tried_keys]
                    if not keys_to_try:
                        lib_logger.warning(f"All keys for provider {provider} have been tried. No more keys to rotate to.")
                        break

                    lib_logger.info(f"Acquiring key for model {model}. Tried keys: {len(tried_keys)}/{len(keys_for_provider)}")
                    current_key = await self.usage_manager.acquire_key(
                        available_keys=keys_to_try,
                        model=model,
                        deadline=deadline
                    )
                    key_acquired = True
                    tried_keys.add(current_key)

                    litellm_kwargs = self.all_providers.get_provider_kwargs(**kwargs.copy())
                    provider_instance = self._get_provider_instance(provider)
                    if provider_instance:
                        if "safety_settings" in litellm_kwargs:
                            converted_settings = provider_instance.convert_safety_settings(litellm_kwargs["safety_settings"])
                            if converted_settings is not None:
                                litellm_kwargs["safety_settings"] = converted_settings
                            else:
                                del litellm_kwargs["safety_settings"]
                    
                    if provider == "gemini" and provider_instance:
                        provider_instance.handle_thinking_parameter(litellm_kwargs, model)

                    if "gemma-3" in model and "messages" in litellm_kwargs:
                        litellm_kwargs["messages"] = [{"role": "user", "content": m["content"]} if m.get("role") == "system" else m for m in litellm_kwargs["messages"]]
                    
                    litellm_kwargs = sanitize_request_payload(litellm_kwargs, model)

                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(f"Attempting stream with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                            response = await litellm.acompletion(
                                api_key=current_key,
                                **litellm_kwargs,
                                logger_fn=self._litellm_logger_callback
                            )
                            
                            lib_logger.info(f"Stream connection established for key ...{current_key[-4:]}. Processing response.")

                            key_acquired = False
                            stream_generator = self._safe_streaming_wrapper(response, current_key, model, request)
                            
                            async for chunk in stream_generator:
                                yield chunk
                            return

                        except (StreamedAPIError, litellm.RateLimitError) as e:
                            last_exception = e
                            log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                            classified_error = classify_error(e)
                            
                            # This is the final, robust handler for streamed errors.
                            error_payload = {}
                            # The actual exception might be wrapped in our StreamedAPIError.
                            original_exc = getattr(e, 'data', e)

                            try:
                                # The full error JSON is in the string representation of the exception.
                                json_str_match = re.search(r'(\{.*\})', str(original_exc), re.DOTALL)
                                if json_str_match:
                                    # The string may contain byte-escaped characters (e.g., \\n).
                                    cleaned_str = codecs.decode(json_str_match.group(1), 'unicode_escape')
                                    error_payload = json.loads(cleaned_str)
                            except (json.JSONDecodeError, TypeError):
                                lib_logger.warning("Could not parse JSON details from streamed error exception.")
                                error_payload = {}

                            error_details = error_payload.get("error", {})
                            error_status = error_details.get("status", "")
                            # Fallback to the full string if parsing fails.
                            error_message_text = error_details.get("message", str(original_exc))

                            if "quota" in error_message_text.lower() or "resource_exhausted" in error_status.lower():
                                # This is a fatal quota error. Terminate the stream with a clear message.
                                quota_value = "N/A"
                                quota_id = "N/A"
                                if "details" in error_details and isinstance(error_details.get("details"), list):
                                    for detail in error_details["details"]:
                                        if isinstance(detail.get("violations"), list):
                                            for violation in detail["violations"]:
                                                if "quotaValue" in violation:
                                                    quota_value = violation["quotaValue"]
                                                if "quotaId" in violation:
                                                    quota_id = violation["quotaId"]
                                                if quota_value != "N/A" and quota_id != "N/A":
                                                    break
                                
                                # 1. Detailed message for the end client
                                client_error_message = (
                                    f"FATAL: You have exceeded your API quota. "
                                    f"Message: '{error_message_text}'. "
                                    f"Limit: {quota_value} (Quota ID: {quota_id})."
                                )

                                # 2. Concise message for the console log
                                console_log_message = (
                                    f"Terminating stream for key ...{current_key[-4:]} due to fatal quota error. "
                                    f"ID: {quota_id}, Limit: {quota_value}."
                                )
                                lib_logger.warning(console_log_message)

                                # 3. Yield the detailed message to the client and terminate
                                yield f"data: {json.dumps({'error': {'message': client_error_message, 'type': 'proxy_quota_error'}})}\n\n"
                                yield "data: [DONE]\n\n"
                                return # Exit the generator completely.

                            # --- NON-QUOTA ERROR: Fallback to key rotation ---
                            rotation_error_message = f"Provider API key failed with {classified_error.error_type}. Rotating to a new key."
                            yield f"data: {json.dumps({'error': {'message': rotation_error_message, 'type': 'proxy_key_rotation_error', 'code': classified_error.status_code}})}\n\n"
                            
                            lib_logger.warning(f"Key ...{current_key[-4:]} encountered a recoverable error during stream for model {model}. Rotating key.")
                            
                            # Only apply global cooldown for non-quota 429s.
                            if classified_error.error_type == 'rate_limit' and classified_error.status_code == 429:
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(provider, cooldown_duration)
                                lib_logger.warning(f"IP-based rate limit detected for {provider}. Starting a {cooldown_duration}-second global cooldown.")

                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            break # Break to try the next key

                        except (APIConnectionError, litellm.InternalServerError, litellm.ServiceUnavailableError) as e:
                            last_exception = e
                            log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                            classified_error = classify_error(e)
                            await self.usage_manager.record_failure(current_key, model, classified_error)

                            if attempt >= self.max_retries - 1:
                                lib_logger.warning(f"Key ...{current_key[-4:]} failed after max retries for model {model} due to a server error. Rotating key.")
                                # Inform the client about the temporary failure before rotating.
                                error_message = f"Key ...{current_key[-4:]} failed after multiple retries. Rotating to a new key."
                                error_data = {
                                    "error": {
                                        "message": error_message,
                                        "type": "proxy_key_rotation_error",
                                        "code": classified_error.status_code
                                    }
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                break
                            
                            wait_time = classified_error.retry_after or (1 * (2 ** attempt)) + random.uniform(0, 1)
                            remaining_budget = deadline - time.time()
                            if wait_time > remaining_budget:
                                lib_logger.warning(f"Required retry wait time ({wait_time:.2f}s) exceeds remaining budget ({remaining_budget:.2f}s). Rotating key early.")
                                break
                            
                            error_message = str(e).split('\n')[0]
                            lib_logger.warning(f"Key ...{current_key[-4:]} encountered a server error for model {model}. Reason: '{error_message}'. Retrying in {wait_time:.2f}s.")
                            await asyncio.sleep(wait_time)
                            continue

                        except Exception as e:
                            last_exception = e
                            log_failure(api_key=current_key, model=model, attempt=attempt + 1, error=e, request_headers=dict(request.headers) if request else {})
                            classified_error = classify_error(e)

                            # For most exceptions, we notify the client and rotate the key.
                            if classified_error.error_type not in ['invalid_request', 'context_window_exceeded', 'authentication']:
                                error_message = f"An unexpected error occurred with key ...{current_key[-4:]}. Rotating to a new key."
                                error_data = {
                                    "error": {
                                        "message": error_message,
                                        "type": "proxy_key_rotation_error",
                                        "code": classified_error.status_code
                                    }
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"

                            lib_logger.warning(f"Key ...{current_key[-4:]} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {str(e)}. Rotating key.")

                            if classified_error.status_code == 429:
                                cooldown_duration = classified_error.retry_after or 60
                                await self.cooldown_manager.start_cooldown(provider, cooldown_duration)
                                lib_logger.warning(f"IP-based rate limit detected for {provider} from generic stream exception. Starting a {cooldown_duration}-second global cooldown.")

                            if classified_error.error_type in ['invalid_request', 'context_window_exceeded', 'authentication']:
                                raise last_exception
                            
                            await self.usage_manager.record_failure(current_key, model, classified_error)
                            break

                finally:
                    if key_acquired and current_key:
                        await self.usage_manager.release_key(current_key, model)
            
            final_error_message = "Failed to complete the streaming request: No available API keys after rotation or global timeout exceeded."
            if last_exception:
                final_error_message = f"Failed to complete the streaming request. Last error: {str(last_exception)}"
                lib_logger.error(f"Streaming request failed after trying all keys. Last error: {last_exception}")

            error_data = {"error": {"message": final_error_message, "type": "proxy_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except NoAvailableKeysError as e:
            lib_logger.error(f"A streaming request failed because no keys were available within the time budget: {e}")
            error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # This will now only catch fatal errors that should be raised, like invalid requests.
            lib_logger.error(f"An unhandled exception occurred in streaming retry logic: {e}", exc_info=True)
            error_data = {"error": {"message": f"An unexpected error occurred: {str(e)}", "type": "proxy_internal_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    def acompletion(self, request: Optional[Any] = None, **kwargs) -> Union[Any, AsyncGenerator[str, None]]:
        """Dispatcher for completion requests."""
        if kwargs.get("stream"):
            return self._streaming_acompletion_with_retry(request, **kwargs)
        else:
            return self._execute_with_retry(litellm.acompletion, request, **kwargs)

    def aembedding(self, request: Optional[Any] = None, **kwargs) -> Any:
        """Executes an embedding request with retry logic."""
        return self._execute_with_retry(litellm.aembedding, request, **kwargs)

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
                    lib_logger.debug(f"Attempting to get models for {provider} with key ...{api_key[-4:]}")
                    models = await provider_instance.get_models(api_key, self.http_client)
                    lib_logger.info(f"Got {len(models)} models for provider: {provider}")
                    self._model_list_cache[provider] = models
                    return models
                except Exception as e:
                    classified_error = classify_error(e)
                    lib_logger.debug(f"Failed to get models for provider {provider} with key ...{api_key[-4:]}: {classified_error.error_type}. Trying next key.")
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
