# src/rotator_library/providers/iflow_provider.py

import json
import time
import os
import httpx
import logging
from typing import Union, AsyncGenerator, List, Dict, Any
from .provider_interface import ProviderInterface
from .iflow_auth_base import IFlowAuthBase
import litellm
from litellm.exceptions import RateLimitError, AuthenticationError

lib_logger = logging.getLogger('rotator_library')

# Model list can be expanded as iFlow supports more models
HARDCODED_MODELS = [
    "deepseek-v3",
    "deepseek-chat",
    "deepseek-coder"
]

# OpenAI-compatible parameters supported by iFlow API
SUPPORTED_PARAMS = {
    'model', 'messages', 'temperature', 'top_p', 'max_tokens',
    'stream', 'tools', 'tool_choice', 'presence_penalty',
    'frequency_penalty', 'n', 'stop', 'seed', 'response_format'
}


class IFlowProvider(IFlowAuthBase, ProviderInterface):
    """
    iFlow provider using OAuth authentication with local callback server.
    API requests use the derived API key (NOT OAuth access_token).
    Based on the Go example implementation.
    """
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a hardcoded list of known compatible iFlow models.
        Validates OAuth credentials if applicable.
        """
        # If it's an OAuth credential (file path), ensure it's valid
        if os.path.isfile(credential):
            try:
                await self.initialize_token(credential)
            except Exception as e:
                lib_logger.warning(f"Failed to validate iFlow OAuth credential: {e}")
        # else: Direct API key, no validation needed here

        return [f"iflow/{model_id}" for model_id in HARDCODED_MODELS]

    def _clean_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes unsupported properties from tool schemas to prevent API errors.
        Similar to Qwen Code implementation.
        """
        import copy
        cleaned_tools = []

        for tool in tools:
            cleaned_tool = copy.deepcopy(tool)

            if "function" in cleaned_tool:
                func = cleaned_tool["function"]

                # Remove strict mode (may not be supported)
                func.pop("strict", None)

                # Clean parameter schema if present
                if "parameters" in func and isinstance(func["parameters"], dict):
                    params = func["parameters"]

                    # Remove additionalProperties if present
                    params.pop("additionalProperties", None)

                    # Recursively clean nested properties
                    if "properties" in params:
                        self._clean_schema_properties(params["properties"])

            cleaned_tools.append(cleaned_tool)

        return cleaned_tools

    def _clean_schema_properties(self, properties: Dict[str, Any]) -> None:
        """Recursively cleans schema properties."""
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                # Remove unsupported fields
                prop_schema.pop("strict", None)
                prop_schema.pop("additionalProperties", None)

                # Recurse into nested properties
                if "properties" in prop_schema:
                    self._clean_schema_properties(prop_schema["properties"])

                # Recurse into array items
                if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                    self._clean_schema_properties({"item": prop_schema["items"]})

    def _build_request_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Builds a clean request payload with only supported parameters.
        This prevents 400 Bad Request errors from litellm-internal parameters.
        """
        # Extract only supported OpenAI parameters
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        # Always force streaming for internal processing
        payload['stream'] = True

        # Always include usage data in stream
        payload['stream_options'] = {"include_usage": True}

        # Handle tool schema cleaning
        if "tools" in payload and payload["tools"]:
            payload["tools"] = self._clean_tool_schemas(payload["tools"])
            lib_logger.debug(f"Cleaned {len(payload['tools'])} tool schemas")

        return payload

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        """
        Converts a raw iFlow SSE chunk to an OpenAI-compatible chunk.
        Since iFlow is OpenAI-compatible, minimal conversion is needed.
        """
        if not isinstance(chunk, dict):
            return

        # Handle usage data
        if usage_data := chunk.get("usage"):
            yield {
                "choices": [], "model": model_id, "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-iflow-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }
            }
            return

        # Handle content data
        choices = chunk.get("choices", [])
        if not choices:
            return

        # iFlow returns OpenAI-compatible format, so we can mostly pass through
        yield {
            "choices": choices,
            "model": model_id,
            "object": "chat.completion.chunk",
            "id": chunk.get("id", f"chatcmpl-iflow-{time.time()}"),
            "created": chunk.get("created", int(time.time()))
        }

    def _stream_to_completion_response(self, chunks: List[litellm.ModelResponse]) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        # Initialize the final response structure
        final_message = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        finish_reason = None

        # Get the first chunk for basic response metadata
        first_chunk = chunks[0]

        # Process each chunk to aggregate content
        for chunk in chunks:
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.get("delta", {})

            # Aggregate content
            if "content" in delta and delta["content"] is not None:
                if "content" not in final_message:
                    final_message["content"] = ""
                final_message["content"] += delta["content"]

            # Aggregate reasoning content (if supported by iFlow)
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                if "reasoning_content" not in final_message:
                    final_message["reasoning_content"] = ""
                final_message["reasoning_content"] += delta["reasoning_content"]

            # Aggregate tool calls
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_chunk in delta["tool_calls"]:
                    index = tc_chunk["index"]
                    if index not in aggregated_tool_calls:
                        aggregated_tool_calls[index] = {"function": {"name": "", "arguments": ""}}
                    if "id" in tc_chunk:
                        aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                    if "type" in tc_chunk:
                        aggregated_tool_calls[index]["type"] = tc_chunk["type"]
                    if "function" in tc_chunk:
                        if "name" in tc_chunk["function"] and tc_chunk["function"]["name"] is not None:
                            aggregated_tool_calls[index]["function"]["name"] += tc_chunk["function"]["name"]
                        if "arguments" in tc_chunk["function"] and tc_chunk["function"]["arguments"] is not None:
                            aggregated_tool_calls[index]["function"]["arguments"] += tc_chunk["function"]["arguments"]

            # Aggregate function calls (legacy format)
            if "function_call" in delta and delta["function_call"] is not None:
                if "function_call" not in final_message:
                    final_message["function_call"] = {"name": "", "arguments": ""}
                if "name" in delta["function_call"] and delta["function_call"]["name"] is not None:
                    final_message["function_call"]["name"] += delta["function_call"]["name"]
                if "arguments" in delta["function_call"] and delta["function_call"]["arguments"] is not None:
                    final_message["function_call"]["arguments"] += delta["function_call"]["arguments"]

            # Get finish reason from the last chunk that has it
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        # Handle usage data from the last chunk that has it
        for chunk in reversed(chunks):
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
                break

        # Add tool calls to final message if any
        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())

        # Ensure standard fields are present for consistent logging
        for field in ["content", "tool_calls", "function_call"]:
            if field not in final_message:
                final_message[field] = None

        # Construct the final response
        final_choice = {
            "index": 0,
            "message": final_message,
            "finish_reason": finish_reason
        }

        # Create the final ModelResponse
        final_response_data = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": first_chunk.model,
            "choices": [final_choice],
            "usage": usage_data
        }

        return litellm.ModelResponse(**final_response_data)

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        enable_request_logging = kwargs.pop("enable_request_logging", False)
        model = kwargs["model"]

        async def make_request():
            """Prepares and makes the actual API call."""
            # CRITICAL: get_api_details returns api_key, NOT access_token
            api_base, api_key = await self.get_api_details(credential_path)

            # Build clean payload with only supported parameters
            payload = self._build_request_payload(**kwargs)

            headers = {
                "Authorization": f"Bearer {api_key}",  # Uses api_key from user info
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "iFlow-Cli"
            }

            url = f"{api_base.rstrip('/')}/chat/completions"

            if enable_request_logging:
                lib_logger.info(f"iFlow Request URL: {url}")
                lib_logger.info(f"iFlow Request Payload: {json.dumps(payload, indent=2)}")
            else:
                lib_logger.debug(f"iFlow Request URL: {url}")

            return client.stream("POST", url, headers=headers, json=payload, timeout=600)

        async def stream_handler(response_stream, attempt=1):
            """Handles the streaming response and converts chunks."""
            try:
                async with response_stream as response:
                    # Check for HTTP errors before processing stream
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_text = error_text.decode('utf-8') if isinstance(error_text, bytes) else error_text

                        # Handle 401: Force token refresh and retry once
                        if response.status_code == 401 and attempt == 1:
                            lib_logger.warning("iFlow returned 401. Forcing token refresh and retrying once.")
                            await self._refresh_token(credential_path, force=True)
                            retry_stream = await make_request()
                            async for chunk in stream_handler(retry_stream, attempt=2):
                                yield chunk
                            return

                        # Handle 429: Rate limit
                        elif response.status_code == 429 or "slow_down" in error_text.lower():
                            raise RateLimitError(
                                f"iFlow rate limit exceeded: {error_text}",
                                llm_provider="iflow",
                                model=model,
                                response=response
                            )

                        # Handle other errors
                        else:
                            if enable_request_logging:
                                lib_logger.error(f"iFlow HTTP {response.status_code} error: {error_text}")
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response
                            )

                    # Process successful streaming response
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                for openai_chunk in self._convert_chunk_to_openai(chunk, model):
                                    yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Could not decode JSON from iFlow: {line}")

            except httpx.HTTPStatusError:
                raise  # Re-raise HTTP errors we already handled
            except Exception as e:
                if enable_request_logging:
                    lib_logger.error(f"Error during iFlow stream processing: {e}", exc_info=True)
                raise

        http_response_stream = await make_request()
        response_generator = stream_handler(http_response_stream)

        if kwargs.get("stream"):
            return response_generator
        else:
            async def non_stream_wrapper():
                chunks = [chunk async for chunk in response_generator]
                return self._stream_to_completion_response(chunks)
            return await non_stream_wrapper()
