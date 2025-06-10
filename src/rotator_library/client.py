import asyncio
import json
import litellm
import logging
from typing import List, Dict, Any, AsyncGenerator

from src.rotator_library.usage_manager import UsageManager
from src.rotator_library.error_handler import (
    is_authentication_error,
    is_rate_limit_error,
    is_server_error,
    is_unrecoverable_error,
)

class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """
    def __init__(self, api_keys: List[str], max_retries: int = 2):
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.usage_manager = UsageManager()

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
                    self.usage_manager.record_success(key, model, chunk.usage)

        finally:
            # Signal the end of the stream
            yield "data: [DONE]\n\n"
            logging.info("STREAM FINISHED and [DONE] signal sent.")


    async def acompletion(self, **kwargs) -> Any:
        """
        Performs a completion call with smart key rotation and retry logic.
        Handles both streaming and non-streaming requests.
        """
        failed_keys_for_this_request = []
        model = kwargs.get("model")
        is_streaming = kwargs.get("stream", False)

        if not model:
            raise ValueError("'model' is a required parameter.")

        while len(failed_keys_for_this_request) < len(self.api_keys):
            current_key = self.usage_manager.get_next_smart_key(
                available_keys=self.api_keys,
                model=model,
                excluded_keys=failed_keys_for_this_request
            )

            if not current_key:
                raise Exception("All available API keys have failed for this request.")

            for attempt in range(self.max_retries):
                try:
                    print(f"Attempting call with key ...{current_key[-4:]} (Attempt {attempt + 1}/{self.max_retries})")
                    response = await litellm.acompletion(api_key=current_key, **kwargs)

                    if is_streaming:
                        # For streams, we return a wrapper generator that logs usage on completion.
                        return self._streaming_wrapper(response, current_key, model)
                    else:
                        # For non-streams, we can log usage immediately.
                        self.usage_manager.record_success(current_key, model, response.usage)
                        return response

                except Exception as e:
                    error_message = str(e)
                    print(f"Key ...{current_key[-4:]} failed with error: {error_message}")

                    if is_authentication_error(e) or is_rate_limit_error(e):
                        self.usage_manager.record_rotation_error(current_key, model, error_message)
                        failed_keys_for_this_request.append(current_key)
                        break

                    elif is_server_error(e):
                        if attempt == self.max_retries - 1:
                            self.usage_manager.record_rotation_error(current_key, model, f"Failed after max retries with error: {error_message}")
                            failed_keys_for_this_request.append(current_key)
                            break
                        else:
                            await asyncio.sleep(1 * (attempt + 1))
                            continue

                    elif is_unrecoverable_error(e):
                        raise e

                    else:
                        raise e

        raise Exception("All API keys failed after multiple retries.")
