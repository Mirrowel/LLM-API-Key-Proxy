# src/rotator_library/providers/copilot_provider.py
"""
GitHub Copilot Provider - Custom API integration for Copilot Chat.

This provider implements the full Copilot Chat API integration including:
- Custom OAuth authentication (Device Flow)
- Direct API calls bypassing LiteLLM
- X-Initiator header control (user vs agent mode)
- Vision request support
- Both streaming and non-streaming responses

Based on:
- https://github.com/sst/opencode-copilot-auth
- https://github.com/Tarquinen/dotfiles/tree/main/.config/opencode/plugin/copilot-force-agent-header
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm

from .provider_interface import ProviderInterface
from .copilot_auth_base import CopilotAuthBase

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default Copilot API base URLs
COPILOT_API_BASE = "https://api.githubcopilot.com"

# Available Copilot models (these may vary based on subscription)
DEFAULT_COPILOT_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "claude-3.5-sonnet",
    "claude-sonnet-4",
    "o3-mini",
    "o1",
    "gemini-2.0-flash-001",
]

# Responses API alternate input types for agent detection
RESPONSES_API_ALTERNATE_INPUT_TYPES = [
    "file_search_call",
    "computer_call",
    "computer_call_output",
    "web_search_call",
    "function_call",
    "function_call_output",
    "image_generation_call",
    "code_interpreter_call",
    "local_shell_call",
    "local_shell_call_output",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
    "mcp_call",
    "reasoning",
]


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# MAIN PROVIDER CLASS
# =============================================================================


class CopilotProvider(CopilotAuthBase, ProviderInterface):
    """
    GitHub Copilot provider with custom API integration.

    Features:
    - Device Flow OAuth authentication
    - Direct Copilot Chat API calls
    - Configurable X-Initiator header (user vs agent mode)
    - Configurable agent header percentage for first messages
    - Vision request support
    - Both streaming and non-streaming responses

    Environment Variables:
    - COPILOT_FORCE_AGENT_HEADER: Always use "agent" initiator (default: false)
    - COPILOT_AGENT_PERCENTAGE: For first messages, % chance of "agent" (0-100, default: 100)
    - COPILOT_MODELS: Comma-separated list of available models
    - COPILOT_ENTERPRISE_URL: GitHub Enterprise URL (optional)
    """

    skip_cost_calculation = True  # Copilot uses subscription, not token billing

    def __init__(self):
        super().__init__()

        # X-Initiator header configuration
        # Based on https://github.com/Tarquinen/dotfiles/tree/main/.config/opencode/plugin/copilot-force-agent-header
        self._force_agent_header = _env_bool("COPILOT_FORCE_AGENT_HEADER", False)
        self._agent_percentage = _env_int("COPILOT_AGENT_PERCENTAGE", 100)

        # Model configuration
        models_env = os.getenv("COPILOT_MODELS", "")
        if models_env:
            self._available_models = [
                m.strip() for m in models_env.split(",") if m.strip()
            ]
        else:
            self._available_models = DEFAULT_COPILOT_MODELS

        lib_logger.debug(
            f"CopilotProvider initialized: force_agent={self._force_agent_header}, "
            f"agent_percentage={self._agent_percentage}%, models={len(self._available_models)}"
        )

    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================

    def has_custom_logic(self) -> bool:
        """Returns True - Copilot uses custom API calls, not LiteLLM."""
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return available Copilot models.

        The api_key here is actually the credential path for OAuth providers.
        For Copilot, models are configured via environment or defaults.
        """
        return self._available_models

    def get_credential_priority(self, credential: str) -> Optional[int]:
        """
        Returns priority for credential. Copilot doesn't have tiers.
        All Copilot credentials are treated equally.
        """
        return 1  # All credentials have same priority

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        Returns model tier requirement. Copilot doesn't restrict by tier.
        """
        return None

    # =========================================================================
    # X-INITIATOR HEADER LOGIC
    # =========================================================================

    def _determine_initiator(
        self, messages: List[Dict[str, Any]], is_responses_api: bool = False
    ) -> str:
        """
        Determine the X-Initiator header value based on conversation context.

        Logic (based on opencode-copilot-auth):
        1. If message contains tool/assistant roles → "agent" (ongoing conversation)
        2. If using Responses API with certain types → "agent"
        3. For first messages:
           - If COPILOT_FORCE_AGENT_HEADER=true → "agent"
           - Else: COPILOT_AGENT_PERCENTAGE% chance of "agent", else "user"

        Returns:
            "agent" or "user"
        """
        # Check for ongoing agent conversation (has assistant/tool messages)
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                if role in ["tool", "assistant"]:
                    return "agent"

                # Check for vision content in messages
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            pass  # Vision doesn't affect initiator

        # Check for Responses API alternate input types
        if is_responses_api and messages:
            last_input = messages[-1] if messages else {}
            input_type = last_input.get("type", "")
            if input_type in RESPONSES_API_ALTERNATE_INPUT_TYPES:
                return "agent"
            if last_input.get("role") == "assistant":
                return "agent"

        # First message logic
        if self._force_agent_header:
            return "agent"

        if self._agent_percentage >= 100:
            return "agent"
        elif self._agent_percentage <= 0:
            return "user"
        else:
            # Random based on percentage
            return "agent" if random.random() * 100 < self._agent_percentage else "user"

    def _is_vision_request(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if request contains vision/image content."""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in [
                        "image_url",
                        "input_image",
                    ]:
                        return True
        return False

    # =========================================================================
    # API COMPLETION
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests to Copilot API.

        This method:
        1. Gets fresh Copilot API token
        2. Builds request with proper headers (X-Initiator, etc.)
        3. Makes direct API call to Copilot
        4. Parses response into LiteLLM format
        """
        credential_path = kwargs.get("api_key", "")
        model = kwargs.get("model", "gpt-4o")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/")[-1]

        # Get fresh credentials and token
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_copilot_token(credential_path, creds)

        access_token = creds.get("access_token", "")
        enterprise_url = creds.get("enterprise_url", "")

        # Determine base URL
        if enterprise_url:
            base_url = f"https://copilot-api.{self._normalize_domain(enterprise_url)}"
        else:
            base_url = COPILOT_API_BASE

        # Determine headers
        initiator = self._determine_initiator(messages)
        is_vision = self._is_vision_request(messages)

        headers = {
            **self.COPILOT_HEADERS,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Openai-Intent": "conversation-edits",
            "X-Initiator": initiator,
        }

        if is_vision:
            headers["Copilot-Vision-Request"] = "true"

        # Build request body (OpenAI-compatible format)
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        if kwargs.get("temperature") is not None:
            body["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens") is not None:
            body["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("top_p") is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            body["stop"] = kwargs["stop"]
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice"):
            body["tool_choice"] = kwargs["tool_choice"]

        lib_logger.debug(
            f"Copilot request: model={model}, initiator={initiator}, "
            f"stream={stream}, vision={is_vision}"
        )

        if stream:
            return self._handle_streaming_response(
                client, base_url, headers, body, model
            )
        else:
            return await self._handle_non_streaming_response(
                client, base_url, headers, body, model
            )

    async def _handle_non_streaming_response(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """Handle non-streaming Copilot API response."""
        url = f"{base_url}/chat/completions"

        try:
            response = await client.post(
                url,
                headers=headers,
                json=body,
                timeout=300.0,  # 5 minute timeout for long completions
            )
            response.raise_for_status()
            data = response.json()

            # Convert to LiteLLM format
            return self._convert_to_litellm_response(data, model)

        except httpx.HTTPStatusError as e:
            lib_logger.error(
                f"Copilot API error (HTTP {e.response.status_code}): {e.response.text}"
            )
            raise
        except Exception as e:
            lib_logger.error(f"Copilot request failed: {e}")
            raise

    async def _handle_streaming_response(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming Copilot API response."""
        url = f"{base_url}/chat/completions"

        async def stream_generator():
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                    timeout=300.0,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            yield self._convert_to_litellm_chunk(chunk_data, model)
                        except json.JSONDecodeError:
                            continue

            except httpx.HTTPStatusError as e:
                lib_logger.error(
                    f"Copilot streaming error (HTTP {e.response.status_code}): {e.response.text}"
                )
                raise
            except Exception as e:
                lib_logger.error(f"Copilot streaming failed: {e}")
                raise

        return stream_generator()

    def _convert_to_litellm_response(
        self, data: Dict[str, Any], model: str
    ) -> litellm.ModelResponse:
        """Convert Copilot API response to LiteLLM ModelResponse format."""
        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            litellm_choice = litellm.Choices(
                index=choice.get("index", 0),
                message=litellm.Message(
                    role=message.get("role", "assistant"),
                    content=message.get("content", ""),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )

            # Handle tool calls
            if message.get("tool_calls"):
                litellm_choice.message.tool_calls = message["tool_calls"]

            choices.append(litellm_choice)

        usage = data.get("usage", {})
        return litellm.ModelResponse(
            id=data.get("id", f"copilot-{uuid.uuid4()}"),
            choices=choices,
            created=data.get("created", int(time.time())),
            model=f"copilot/{model}",
            usage=litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    def _convert_to_litellm_chunk(
        self, chunk_data: Dict[str, Any], model: str
    ) -> litellm.ModelResponse:
        """Convert Copilot streaming chunk to LiteLLM format."""
        choices = []
        for choice in chunk_data.get("choices", []):
            delta = choice.get("delta", {})
            litellm_choice = litellm.Choices(
                index=choice.get("index", 0),
                delta=litellm.Delta(
                    role=delta.get("role"),
                    content=delta.get("content"),
                ),
                finish_reason=choice.get("finish_reason"),
            )

            # Handle tool call deltas
            if delta.get("tool_calls"):
                litellm_choice.delta.tool_calls = delta["tool_calls"]

            choices.append(litellm_choice)

        return litellm.ModelResponse(
            id=chunk_data.get("id", f"copilot-{uuid.uuid4()}"),
            choices=choices,
            created=chunk_data.get("created", int(time.time())),
            model=f"copilot/{model}",
        )

    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """
        Copilot doesn't support embeddings API.
        Raises NotImplementedError.
        """
        raise NotImplementedError("Copilot does not support embeddings API")
