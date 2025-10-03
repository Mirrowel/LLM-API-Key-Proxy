# src/rotator_library/providers/qwen_code_provider.py

import json
import time
import httpx
import logging
from typing import Union, AsyncGenerator, List, Dict, Any
from .provider_interface import ProviderInterface
from .qwen_auth_base import QwenAuthBase
import litellm
from litellm.exceptions import RateLimitError, AuthenticationError

lib_logger = logging.getLogger('rotator_library')

HARDCODED_MODELS = [
    "qwen3-coder-plus",
    "qwen3-coder-flash"
]

class QwenCodeProvider(QwenAuthBase, ProviderInterface):
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """Returns a hardcoded list of known compatible Qwen models."""
        return [f"qwen_code/{model_id}" for model_id in HARDCODED_MODELS]

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
            parts = content.replace("<think>", "||THINK||").replace("</think>", "||/THINK||").split("||")
            for part in parts:
                if not part: continue
                
                new_delta = {}
                if part.startswith("THINK||"):
                    new_delta['reasoning_content'] = part.replace("THINK||", "")
                elif part.startswith("/THINK||"):
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

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        model = kwargs["model"]

        async def do_call():
            api_base, access_token = await self.get_api_details(credential_path)
            
            # Prepare payload
            payload = kwargs.copy()
            payload.pop("litellm_params", None) # Clean up internal params
            
            # Per Go example, inject dummy tool to prevent stream corruption
            if not payload.get("tools"):
                payload["tools"] = [{"type": "function", "function": {"name": "do_not_call_me", "description": "Do not call this tool under any circumstances.", "parameters": {"type": "object", "properties": {}}}}]

            # Ensure usage is included in stream
            payload["stream_options"] = {"include_usage": True}

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if kwargs.get("stream") else "application/json",
                "User-Agent": "google-api-nodejs-client/9.15.1",
                "X-Goog-Api-Client": "gl-node/22.17.0",
                "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
            }
            
            url = f"{api_base.rstrip('/')}/chat/completions"
            lib_logger.debug(f"Qwen Code Request URL: {url}")
            lib_logger.debug(f"Qwen Code Request Payload: {json.dumps(payload, indent=2)}")

            async def stream_handler():
                async with client.stream("POST", url, headers=headers, json=payload, timeout=600) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == "[DONE]": break
                            try:
                                chunk = json.loads(data_str)
                                for openai_chunk in self._convert_chunk_to_openai(chunk, model):
                                    yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Could not decode JSON from Qwen Code: {line}")
            
            return stream_handler()

        try:
            response_gen = await do_call()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                lib_logger.warning("Qwen Code returned 401. Forcing token refresh and retrying once.")
                await self._refresh_token(credential_path, force=True)
                response_gen = await do_call()
            elif e.response.status_code == 429 or "slow_down" in e.response.text.lower():
                raise RateLimitError(
                    message=f"Qwen Code rate limit exceeded: {e.response.text}",
                    llm_provider="qwen_code",
                    response=e.response
                )
            else:
                raise e

        if kwargs.get("stream"):
            return response_gen
        else:
            chunks = [chunk async for chunk in response_gen]
            return litellm.utils.stream_to_completion_response(chunks)