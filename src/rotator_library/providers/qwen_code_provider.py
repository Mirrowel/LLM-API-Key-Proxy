# src/rotator_library/providers/qwen_code_provider.py

import json
import time
import os
import httpx
import logging
from typing import Union, AsyncGenerator, List, Dict, Any
from .provider_interface import ProviderInterface
from .qwen_auth_base import QwenAuthBase
from ..model_definitions import ModelDefinitions
import litellm
from litellm.exceptions import RateLimitError, AuthenticationError
from pathlib import Path
import uuid
from datetime import datetime

lib_logger = logging.getLogger('rotator_library')

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
QWEN_CODE_LOGS_DIR = LOGS_DIR / "qwen_code_logs"

class _QwenCodeFileLogger:
    """A simple file logger for a single Qwen Code transaction."""
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = str(uuid.uuid4())
        # Sanitize model name for directory
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        self.log_dir = QWEN_CODE_LOGS_DIR / f"{timestamp}_{safe_model_name}_{request_id}"
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create Qwen Code log directory: {e}")
            self.enabled = False

    def log_request(self, payload: Dict[str, Any]):
        """Logs the request payload sent to Qwen Code."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "request_payload.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"_QwenCodeFileLogger: Failed to write request: {e}")

    def log_response_chunk(self, chunk: str):
        """Logs a raw chunk from the Qwen Code response stream."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "response_stream.log", "a", encoding="utf-8") as f:
                f.write(chunk + "\n")
        except Exception as e:
            lib_logger.error(f"_QwenCodeFileLogger: Failed to write response chunk: {e}")

    def log_error(self, error_message: str):
        """Logs an error message."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "error.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] {error_message}\n")
        except Exception as e:
            lib_logger.error(f"_QwenCodeFileLogger: Failed to write error: {e}")

    def log_final_response(self, response_data: Dict[str, Any]):
        """Logs the final, reassembled response."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "final_response.json", "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"_QwenCodeFileLogger: Failed to write final response: {e}")

HARDCODED_MODELS = [
    "qwen3-coder-plus",
    "qwen3-coder-flash"
]

# OpenAI-compatible parameters supported by Qwen Code API
SUPPORTED_PARAMS = {
    'model', 'messages', 'temperature', 'top_p', 'max_tokens',
    'stream', 'tools', 'tool_choice', 'presence_penalty',
    'frequency_penalty', 'n', 'stop', 'seed', 'response_format'
}

class QwenCodeProvider(QwenAuthBase, ProviderInterface):
    skip_cost_calculation = True
    REASONING_START_MARKER = 'THINK||'

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of Qwen Code models from three sources:
        1. Environment variable models (via QWEN_CODE_MODELS) - ALWAYS included, take priority
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        3. Dynamic discovery from Qwen API (if supported) - added only if ID not in env vars

        Environment variable models always win and are never deduplicated, even if they
        share the same ID (to support different configs like temperature, etc.)

        Validates OAuth credentials if applicable.
        """
        models = []
        env_var_ids = set()  # Track IDs from env vars to prevent hardcoded/dynamic duplicates

        def extract_model_id(item) -> str:
            """Extract model ID from various formats (dict, string with/without provider prefix)."""
            if isinstance(item, dict):
                # Dict format: extract 'id' or 'name' field
                return item.get("id") or item.get("name", "")
            elif isinstance(item, str):
                # String format: extract ID from "provider/id" or just "id"
                return item.split("/")[-1] if "/" in item else item
            return str(item)

        # Source 1: Load environment variable models (ALWAYS include ALL of them)
        static_models = self.model_definitions.get_all_provider_models("qwen_code")
        if static_models:
            for model in static_models:
                # Extract model name from "qwen_code/ModelName" format
                model_name = model.split("/")[-1] if "/" in model else model
                # Get the actual model ID from definitions (which may differ from the name)
                model_id = self.model_definitions.get_model_id("qwen_code", model_name)

                # ALWAYS add env var models (no deduplication)
                models.append(model)
                # Track the ID to prevent hardcoded/dynamic duplicates
                if model_id:
                    env_var_ids.add(model_id)
            lib_logger.info(f"Loaded {len(static_models)} static models for qwen_code from environment variables")

        # Source 2: Add hardcoded models (only if ID not already in env vars)
        for model_id in HARDCODED_MODELS:
            if model_id not in env_var_ids:
                models.append(f"qwen_code/{model_id}")
                env_var_ids.add(model_id)

        # Source 3: Try dynamic discovery from Qwen Code API (only if ID not already in env vars)
        try:
            # Validate OAuth credentials and get API details
            if os.path.isfile(credential):
                await self.initialize_token(credential)

            api_base, access_token = await self.get_api_details(credential)
            models_url = f"{api_base.rstrip('/')}/v1/models"

            response = await client.get(
                models_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()

            dynamic_data = response.json()
            # Handle both {data: [...]} and direct [...] formats
            model_list = dynamic_data.get("data", dynamic_data) if isinstance(dynamic_data, dict) else dynamic_data

            dynamic_count = 0
            for model in model_list:
                model_id = extract_model_id(model)
                if model_id and model_id not in env_var_ids:
                    models.append(f"qwen_code/{model_id}")
                    env_var_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(f"Discovered {dynamic_count} additional models for qwen_code from API")

        except Exception as e:
            # Silently ignore dynamic discovery errors
            lib_logger.debug(f"Dynamic model discovery failed for qwen_code: {e}")
            pass

        return models

    def _clean_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes unsupported properties from tool schemas to prevent API errors.
        Adapted for Qwen's API requirements.
        """
        import copy
        cleaned_tools = []

        for tool in tools:
            cleaned_tool = copy.deepcopy(tool)

            if "function" in cleaned_tool:
                func = cleaned_tool["function"]

                # Remove strict mode (not supported by Qwen)
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
        elif not payload.get("tools"):
            # Per Qwen Code API bug (see: https://github.com/qianwen-team/flash-dance/issues/2),
            # injecting a dummy tool prevents stream corruption when no tools are provided
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "do_not_call_me",
                    "description": "Do not call this tool.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
            lib_logger.debug("Injected dummy tool to prevent Qwen API stream corruption")

        return payload

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        """Converts a raw Qwen SSE chunk to an OpenAI-compatible chunk."""
        if not isinstance(chunk, dict):
            return

        # Handle usage data
        if usage_data := chunk.get("usage"):
            yield {
                "choices": [], "model": model_id, "object": "chat.completion.chunk",
                "id": f"chatcmpl-qwen-{time.time()}", "created": int(time.time()),
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

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle <think> tags for reasoning content
        content = delta.get("content")
        if content and ("<think>" in content or "</think>" in content):
            parts = content.replace("<think>", f"||{self.REASONING_START_MARKER}").replace("</think>", f"||/{self.REASONING_START_MARKER}").split("||")
            for part in parts:
                if not part: continue
                
                new_delta = {}
                if part.startswith(self.REASONING_START_MARKER):
                    new_delta['reasoning_content'] = part.replace(self.REASONING_START_MARKER, "")
                elif part.startswith(f"/{self.REASONING_START_MARKER}"):
                    continue
                else:
                    new_delta['content'] = part
                
                yield {
                    "choices": [{"index": 0, "delta": new_delta, "finish_reason": None}],
                    "model": model_id, "object": "chat.completion.chunk",
                    "id": f"chatcmpl-qwen-{time.time()}", "created": int(time.time())
                }
        else:
            # Standard content chunk
            yield {
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                "model": model_id, "object": "chat.completion.chunk",
                "id": f"chatcmpl-qwen-{time.time()}", "created": int(time.time())
            }

    def _stream_to_completion_response(self, chunks: List[litellm.ModelResponse]) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.
        This replaces the non-existent litellm.utils.stream_to_completion_response function.
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

            # Aggregate reasoning content
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

        # Create dedicated file logger for this request
        file_logger = _QwenCodeFileLogger(
            model_name=model,
            enabled=enable_request_logging
        )

        async def make_request():
            """Prepares and makes the actual API call."""
            api_base, access_token = await self.get_api_details(credential_path)

            # Strip provider prefix from model name (e.g., "qwen_code/qwen3-coder-plus" -> "qwen3-coder-plus")
            model_name = model.split('/')[-1]
            kwargs_with_stripped_model = {**kwargs, 'model': model_name}

            # Build clean payload with only supported parameters
            payload = self._build_request_payload(**kwargs_with_stripped_model)

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "google-api-nodejs-client/9.15.1",
                "X-Goog-Api-Client": "gl-node/22.17.0",
                "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
            }

            url = f"{api_base.rstrip('/')}/v1/chat/completions"

            # Log request to dedicated file
            file_logger.log_request(payload)
            lib_logger.debug(f"Qwen Code Request URL: {url}")

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
                            lib_logger.warning("Qwen Code returned 401. Forcing token refresh and retrying once.")
                            await self._refresh_token(credential_path, force=True)
                            retry_stream = await make_request()
                            async for chunk in stream_handler(retry_stream, attempt=2):
                                yield chunk
                            return

                        # Handle 429: Rate limit
                        elif response.status_code == 429 or "slow_down" in error_text.lower():
                            raise RateLimitError(
                                f"Qwen Code rate limit exceeded: {error_text}",
                                llm_provider="qwen_code",
                                model=model,
                                response=response
                            )

                        # Handle other errors
                        else:
                            error_msg = f"Qwen Code HTTP {response.status_code} error: {error_text}"
                            file_logger.log_error(error_msg)
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response
                            )

                    # Process successful streaming response
                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                for openai_chunk in self._convert_chunk_to_openai(chunk, model):
                                    yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Could not decode JSON from Qwen Code: {line}")

            except httpx.HTTPStatusError:
                raise  # Re-raise HTTP errors we already handled
            except Exception as e:
                file_logger.log_error(f"Error during Qwen Code stream processing: {e}")
                lib_logger.error(f"Error during Qwen Code stream processing: {e}", exc_info=True)
                raise

        async def logging_stream_wrapper():
            """Wraps the stream to log the final reassembled response."""
            openai_chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(openai_chunks)
                    file_logger.log_final_response(final_response.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:
            async def non_stream_wrapper():
                chunks = [chunk async for chunk in logging_stream_wrapper()]
                return self._stream_to_completion_response(chunks)
            return await non_stream_wrapper()