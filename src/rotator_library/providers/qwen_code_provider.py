# src/rotator_library/providers/qwen_code_provider.py

import litellm.exceptions as litellm_exc
import httpx
import logging
from typing import Union, AsyncGenerator, List
from .provider_interface import ProviderInterface
from .qwen_auth_base import QwenAuthBase
import litellm

lib_logger = logging.getLogger('rotator_library')

# [NEW] Hardcoded model list based on Kilo example
HARDCODED_MODELS = [
    "qwen3-coder-plus",
    "qwen3-coder-flash"
]

class QwenCodeProvider(QwenAuthBase, ProviderInterface):
    def has_custom_logic(self) -> bool:
        return True # We use custom logic to handle 401 retries and stream parsing

    # [NEW] get_models implementation
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Returns a hardcoded list of known compatible Qwen models for the OpenAI-compatible API."""
        return [f"qwen_code/{model_id}" for model_id in HARDCODED_MODELS]

    async def _stream_parser(self, stream: AsyncGenerator, model_id: str) -> AsyncGenerator:
        """Parses the stream from litellm to handle Qwen's <think> tags."""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content and ("<think>" in content or "</think>" in content):
                parts = content.replace("<think>", "||THINK||").replace("</think>", "||/THINK||").split("||")
                for part in parts:
                    # [NEW] Check for provider-specific errors in content
                    if "slow_down" in part.lower():
                        lib_logger.warning("Qwen 'slow_down' detected in response content. Treating as rate limit.")
                        raise litellm_exc.RateLimitError(
                            message="Qwen slow_down error detected.", llm_provider="qwen_code"
                        )
                    if not part: continue
                    new_chunk = chunk.copy()
                    if part.startswith("THINK||"):
                        new_chunk.choices[0].delta.reasoning_content = part.replace("THINK||", "")
                        new_chunk.choices[0].delta.content = None
                    elif part.startswith("/THINK||"):
                        continue # Ignore closing tag
                    else:
                        new_chunk.choices[0].delta.content = part
                        new_chunk.choices[0].delta.reasoning_content = None
                    yield new_chunk
            else:
                yield chunk

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        model = kwargs["model"]
        
        async def do_call():
            api_base, access_token = self.get_api_details(credential_path)
            response = await litellm.acompletion(
                # [NEW] Add timeout and retry params if needed, but since rotation handles retries, this is optional
                **kwargs, api_key=access_token, api_base=api_base
            )
            # [NEW] Post-call check for specific finish reasons or errors
            if not kwargs.get("stream") and response.choices[0].finish_reason == "slow_down":
                lib_logger.warning("Qwen 'slow_down' finish reason detected. Treating as rate limit.")
                raise litellm_exc.RateLimitError(
                    message="Qwen slow_down finish reason.", llm_provider="qwen_code"
                )
            return response
        
        try:
            response = await do_call()
        except litellm.AuthenticationError as e:
            if "401" in str(e):
                lib_logger.warning("Qwen Code returned 401. Forcing token refresh and retrying once.")
                await self._refresh_token(credential_path, force=True)
                response = await do_call()
            # [NEW] Catch provider-specific exceptions
            elif "slow_down" in str(e).lower():
                raise litellm_exc.RateLimitError(
                    message="Qwen slow_down error in exception.", llm_provider="qwen_code"
                )
            else:
                raise e

        if kwargs.get("stream"):
            return self._stream_parser(response, model)
        return response