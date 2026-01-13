# src/rotator_library/providers/codex_provider.py
"""
OpenAI Codex Provider

Provider for OpenAI Codex models via the Responses API.
Supports GPT-5, GPT-5.1, GPT-5.2, Codex, and Codex Mini models.

Key Features:
- OAuth-based authentication with PKCE
- Responses API for streaming
- Reasoning/thinking output with configurable effort levels
- Tool calling support
- OpenAI Chat Completions format translation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import httpx
import litellm

from .provider_interface import ProviderInterface, UsageResetConfigDef, QuotaGroupMap
from .openai_oauth_base import OpenAIOAuthBase
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..error_handler import EmptyResponseError, TransientQuotaError

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# CONFIGURATION
# =============================================================================

def env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default


def env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.getenv(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


# Codex API endpoint configuration
# Default: ChatGPT Backend API (works with OAuth credentials)
# Alternative: OpenAI API (requires API key, set CODEX_USE_OPENAI_API=true)
USE_OPENAI_API = env_bool("CODEX_USE_OPENAI_API", False)

if USE_OPENAI_API:
    CODEX_API_BASE = os.getenv("CODEX_API_BASE", "https://api.openai.com/v1")
    CODEX_RESPONSES_ENDPOINT = f"{CODEX_API_BASE}/responses"
else:
    # Default: ChatGPT backend API (requires OAuth + account_id)
    CODEX_API_BASE = os.getenv("CODEX_API_BASE", "https://chatgpt.com/backend-api/codex")
    CODEX_RESPONSES_ENDPOINT = f"{CODEX_API_BASE}/responses"

# Available models - base models
BASE_MODELS = [
    # GPT-5 models
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    # Codex models
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "codex-mini",
]

# Reasoning effort levels
REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}

# Models that support reasoning effort variants
# Maps model -> allowed effort levels
REASONING_MODEL_EFFORTS = {
    "gpt-5": {"low", "medium", "high"},
    "gpt-5.1": {"low", "medium", "high"},
    "gpt-5.2": {"low", "medium", "high", "xhigh"},
    "gpt-5-codex": {"low", "medium", "high"},
    "gpt-5.1-codex": {"low", "medium", "high"},
    "gpt-5.2-codex": {"low", "medium", "high", "xhigh"},
    "gpt-5.1-codex-max": {"low", "medium", "high", "xhigh"},
    "gpt-5.1-codex-mini": {"low", "medium", "high"},
    "codex-mini": {"low", "medium", "high"},
}

def _build_available_models() -> list:
    """Build full list of available models including reasoning variants."""
    models = list(BASE_MODELS)

    # Add reasoning effort variants for each model
    for model, efforts in REASONING_MODEL_EFFORTS.items():
        for effort in sorted(efforts):
            models.append(f"{model}:{effort}")

    return models

AVAILABLE_MODELS = _build_available_models()

# Default reasoning configuration
DEFAULT_REASONING_EFFORT = os.getenv("CODEX_REASONING_EFFORT", "medium")
DEFAULT_REASONING_SUMMARY = os.getenv("CODEX_REASONING_SUMMARY", "auto")
DEFAULT_REASONING_COMPAT = os.getenv("CODEX_REASONING_COMPAT", "think-tags")

# Empty response retry configuration
EMPTY_RESPONSE_MAX_ATTEMPTS = max(1, env_int("CODEX_EMPTY_RESPONSE_ATTEMPTS", 3))
EMPTY_RESPONSE_RETRY_DELAY = env_int("CODEX_EMPTY_RESPONSE_RETRY_DELAY", 2)

# System instruction for Codex models - loaded from file to preserve exact bytes
# The ChatGPT backend API validates this instruction matches exactly
def _load_codex_prompt() -> str:
    """Load the Codex system prompt from file to preserve exact Unicode characters."""
    prompt_file = Path(__file__).parent / "codex_prompt.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        lib_logger.warning(f"Codex prompt file not found at {prompt_file}, using fallback")
        return "You are a coding agent."

CODEX_SYSTEM_INSTRUCTION = _load_codex_prompt()

# Whether to inject the Codex system instruction
INJECT_CODEX_INSTRUCTION = env_bool("CODEX_INJECT_INSTRUCTION", True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _allowed_efforts_for_model(model: str) -> set:
    """Get allowed reasoning effort levels for a model."""
    base = (model or "").strip().lower()
    if not base:
        return REASONING_EFFORTS

    normalized = base.split(":")[0]
    if normalized.startswith("gpt-5.2"):
        return {"low", "medium", "high", "xhigh"}
    if normalized.startswith("gpt-5.1-codex-max"):
        return {"low", "medium", "high", "xhigh"}
    if normalized.startswith("gpt-5.1"):
        return {"low", "medium", "high"}

    return REASONING_EFFORTS


def _extract_reasoning_from_model_name(model: str) -> Optional[Dict[str, Any]]:
    """Extract reasoning effort from model name suffix."""
    if not isinstance(model, str) or not model:
        return None

    s = model.strip().lower()
    if not s:
        return None

    # Check for suffix like :high or -high
    if ":" in s:
        maybe = s.rsplit(":", 1)[-1].strip()
        if maybe in REASONING_EFFORTS:
            return {"effort": maybe}

    for sep in ("-", "_"):
        for effort in REASONING_EFFORTS:
            if s.endswith(f"{sep}{effort}"):
                return {"effort": effort}

    return None


def _build_reasoning_param(
    base_effort: str = "medium",
    base_summary: str = "auto",
    overrides: Optional[Dict[str, Any]] = None,
    allowed_efforts: Optional[set] = None,
) -> Dict[str, Any]:
    """Build reasoning parameter for Responses API."""
    effort = (base_effort or "").strip().lower()
    summary = (base_summary or "").strip().lower()

    valid_efforts = allowed_efforts or REASONING_EFFORTS
    valid_summaries = {"auto", "concise", "detailed", "none"}

    if isinstance(overrides, dict):
        o_eff = str(overrides.get("effort", "")).strip().lower()
        o_sum = str(overrides.get("summary", "")).strip().lower()
        if o_eff in valid_efforts and o_eff:
            effort = o_eff
        if o_sum in valid_summaries and o_sum:
            summary = o_sum

    if effort not in valid_efforts:
        effort = "medium"
    if summary not in valid_summaries:
        summary = "auto"

    reasoning: Dict[str, Any] = {"effort": effort}
    if summary != "none":
        reasoning["summary"] = summary

    return reasoning


def _normalize_model_name(name: str) -> str:
    """Normalize model name, stripping reasoning effort suffix."""
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"

    base = name.split(":", 1)[0].strip()

    # Strip effort suffix
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in REASONING_EFFORTS:
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break

    # Model name mapping
    mapping = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "codex": "codex-mini",
        "codex-mini-latest": "codex-mini",
    }

    return mapping.get(base.lower(), base)


def _convert_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI chat messages format to Responses API input format.
    """
    input_items = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            # System messages become user messages (ChatMock pattern)
            # The instructions field is used ONLY for the base opencode instructions
            if isinstance(content, str) and content.strip():
                input_items.insert(0, {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}]
                })
            continue

        if role == "user":
            # User messages with content
            if isinstance(content, str):
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}]
                })
            elif isinstance(content, list):
                # Handle multimodal content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"type": "input_text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                            parts.append({"type": "input_image", "image_url": url})
                if parts:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": parts
                    })
            continue

        if role == "assistant":
            # Assistant messages
            if isinstance(content, str) and content:
                input_items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}]
                })
            elif isinstance(content, list):
                # Handle assistant content as a list
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "text":
                            parts.append({"type": "output_text", "text": part.get("text", "")})
                        elif part_type == "output_text":
                            parts.append({"type": "output_text", "text": part.get("text", "")})
                if parts:
                    input_items.append({
                        "role": "assistant",
                        "content": parts
                    })

            # Handle tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("type") == "function":
                    func = tc.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", str(uuid.uuid4())),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    })
            continue

        if role == "tool":
            # Tool result messages
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content if isinstance(content, str) else json.dumps(content),
            })
            continue

    return input_items


def _convert_tools_to_responses_format(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI tools format to Responses API format.
    """
    if not tools:
        return []

    responses_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "function")

        if tool_type == "function":
            func = tool.get("function", {})
            name = func.get("name", "")
            # Skip tools without a name
            if not name:
                continue
            params = func.get("parameters", {})
            # Ensure parameters is a valid object
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
            responses_tools.append({
                "type": "function",
                "name": name,
                "description": func.get("description") or "",
                "parameters": params,
                "strict": False,
            })
        elif tool_type in ("web_search", "web_search_preview"):
            responses_tools.append({"type": tool_type})

    return responses_tools


def _apply_reasoning_to_message(
    message: Dict[str, Any],
    reasoning_summary_text: str,
    reasoning_full_text: str,
    compat: str,
) -> Dict[str, Any]:
    """Apply reasoning output to message based on compatibility mode."""
    try:
        compat = (compat or "think-tags").strip().lower()
    except Exception:
        compat = "think-tags"

    if compat == "o3":
        # OpenAI o3 format with reasoning object
        rtxt_parts = []
        if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
            rtxt_parts.append(reasoning_summary_text)
        if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
            rtxt_parts.append(reasoning_full_text)
        rtxt = "\n\n".join([p for p in rtxt_parts if p])
        if rtxt:
            message["reasoning"] = {"content": [{"type": "text", "text": rtxt}]}
        return message

    if compat in ("legacy", "current"):
        # Legacy format with separate fields
        if reasoning_summary_text:
            message["reasoning_summary"] = reasoning_summary_text
        if reasoning_full_text:
            message["reasoning"] = reasoning_full_text
        return message

    # Default: think-tags format (prepend to content)
    rtxt_parts = []
    if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
        rtxt_parts.append(reasoning_summary_text)
    if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
        rtxt_parts.append(reasoning_full_text)
    rtxt = "\n\n".join([p for p in rtxt_parts if p])

    if rtxt:
        think_block = f"<think>{rtxt}</think>"
        content_text = message.get("content") or ""
        if isinstance(content_text, str):
            message["content"] = think_block + (content_text or "")

    return message


# =============================================================================
# PROVIDER IMPLEMENTATION
# =============================================================================

class CodexProvider(OpenAIOAuthBase, ProviderInterface):
    """
    OpenAI Codex Provider

    Provides access to OpenAI Codex models (GPT-5, Codex) via the Responses API.
    Uses OAuth with PKCE for authentication.
    """

    # Provider configuration
    provider_env_name: str = "codex"
    skip_cost_calculation: bool = True  # Cost calculation handled differently

    # Rotation configuration
    default_rotation_mode: str = "balanced"

    # Tier configuration
    tier_priorities: Dict[str, int] = {
        "plus": 1,
        "pro": 1,
        "team": 2,
        "free": 3,
    }
    default_tier_priority: int = 3

    # Usage reset configuration
    usage_reset_configs = {
        frozenset({1}): UsageResetConfigDef(
            window_seconds=86400,  # 24 hours
            mode="per_model",
            description="Daily per-model reset for Plus/Pro tier",
            field_name="models",
        ),
        "default": UsageResetConfigDef(
            window_seconds=86400,
            mode="per_model",
            description="Daily per-model reset",
            field_name="models",
        ),
    }

    # Model quota groups (models that share limits)
    model_quota_groups: QuotaGroupMap = {
        "gpt5": ["gpt-5", "gpt-5.1", "gpt-5.2"],
        "codex": ["gpt-5-codex", "gpt-5.1-codex", "gpt-5.2-codex", "gpt-5.1-codex-max"],
    }

    def __init__(self):
        # Initialize both parent classes
        ProviderInterface.__init__(self)
        OpenAIOAuthBase.__init__(self)

        self.model_definitions = ModelDefinitions()
        self._session_cache: Dict[str, str] = {}  # Cache session IDs per credential

    def has_custom_logic(self) -> bool:
        """This provider uses custom logic (Responses API instead of litellm)."""
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return available Codex models."""
        return [f"codex/{m}" for m in AVAILABLE_MODELS]

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """Get tier name for a credential."""
        creds = self._credentials_cache.get(credential)
        if creds:
            plan_type = creds.get("_proxy_metadata", {}).get("plan_type", "")
            if plan_type:
                return plan_type.lower()
        return None

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request using Responses API.
        """
        # Extract parameters
        model = kwargs.get("model", "gpt-5")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice", "auto")
        parallel_tool_calls = kwargs.get("parallel_tool_calls", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("credential_path", ""))
        reasoning_effort = kwargs.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
        extra_headers = kwargs.get("extra_headers", {})

        # Normalize model name
        requested_model = model
        if "/" in model:
            model = model.split("/", 1)[1]
        normalized_model = _normalize_model_name(model)

        # Build reasoning parameters
        model_reasoning = _extract_reasoning_from_model_name(requested_model)
        reasoning_overrides = kwargs.get("reasoning") or model_reasoning
        reasoning_param = _build_reasoning_param(
            reasoning_effort,
            DEFAULT_REASONING_SUMMARY,
            reasoning_overrides,
            allowed_efforts=_allowed_efforts_for_model(normalized_model),
        )

        # Convert messages to Responses API format
        input_items = _convert_messages_to_responses_input(messages)

        # Use ONLY the base opencode instructions (system messages are converted to user messages)
        # The ChatGPT backend API validates that instructions match exactly
        instructions = CODEX_SYSTEM_INSTRUCTION if INJECT_CODEX_INSTRUCTION else None

        # Convert tools
        responses_tools = _convert_tools_to_responses_format(tools)

        # Get auth headers
        auth_headers = await self.get_auth_header(credential_path)
        account_id = await self.get_account_id(credential_path)

        # Build request headers
        headers = {
            **auth_headers,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "OpenAI-Beta": "responses=experimental",
        }

        if account_id:
            headers["chatgpt-account-id"] = account_id

        # Add any extra headers
        headers.update(extra_headers)

        # Build request payload
        include = ["reasoning.encrypted_content"] if reasoning_param else []

        payload = {
            "model": normalized_model,
            "input": input_items,
            "stream": True,  # Always use streaming internally
            "store": False,
        }

        if instructions:
            payload["instructions"] = instructions

        if responses_tools:
            payload["tools"] = responses_tools
            payload["tool_choice"] = tool_choice if tool_choice in ("auto", "none") else "auto"
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)

        if reasoning_param:
            payload["reasoning"] = reasoning_param

        if include:
            payload["include"] = include

        lib_logger.debug(f"Codex request to {normalized_model}: {json.dumps(payload, default=str)[:500]}...")

        if stream:
            return self._stream_response(
                client, headers, payload, requested_model, kwargs.get("reasoning_compat", DEFAULT_REASONING_COMPAT)
            )
        else:
            return await self._non_stream_response(
                client, headers, payload, requested_model, kwargs.get("reasoning_compat", DEFAULT_REASONING_COMPAT)
            )

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        reasoning_compat: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming response from Responses API."""
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Track state for tool calls
        current_tool_calls: Dict[int, Dict[str, Any]] = {}
        reasoning_summary_text = ""
        reasoning_full_text = ""
        sent_reasoning = False

        async with client.stream(
            "POST",
            CODEX_RESPONSES_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(f"Codex API error {response.status_code}: {error_text[:500]}")
                raise httpx.HTTPStatusError(
                    f"Codex API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                kind = evt.get("type")

                # Handle response ID
                if isinstance(evt.get("response"), dict):
                    resp_id = evt["response"].get("id")
                    if resp_id:
                        response_id = resp_id

                # Handle text delta
                if kind == "response.output_text.delta":
                    delta_text = evt.get("delta", "")
                    if delta_text:
                        # If we have reasoning and haven't sent it yet, prepend it
                        if not sent_reasoning and (reasoning_summary_text or reasoning_full_text):
                            rtxt = "\n\n".join(filter(None, [reasoning_summary_text, reasoning_full_text]))
                            if rtxt and reasoning_compat == "think-tags":
                                delta_text = f"<think>{rtxt}</think>{delta_text}"
                            sent_reasoning = True

                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {"content": delta_text, "role": "assistant"},
                                "finish_reason": None,
                            }],
                        )
                        yield chunk

                # Handle reasoning delta
                elif kind == "response.reasoning_summary_text.delta":
                    reasoning_summary_text += evt.get("delta", "")

                elif kind == "response.reasoning_text.delta":
                    reasoning_full_text += evt.get("delta", "")

                # Handle function call arguments delta
                elif kind == "response.function_call_arguments.delta":
                    output_index = evt.get("output_index", 0)
                    delta = evt.get("delta", "")

                    if output_index not in current_tool_calls:
                        current_tool_calls[output_index] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    current_tool_calls[output_index]["arguments"] += delta

                # Handle output item added (start of tool call)
                elif kind == "response.output_item.added":
                    item = evt.get("item", {})
                    output_index = evt.get("output_index", 0)

                    if item.get("type") == "function_call":
                        current_tool_calls[output_index] = {
                            "id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "arguments": "",
                        }

                # Handle output item done (complete tool call)
                elif kind == "response.output_item.done":
                    item = evt.get("item", {})
                    output_index = evt.get("output_index", 0)

                    if item.get("type") == "function_call":
                        call_id = item.get("call_id") or item.get("id", "")
                        name = item.get("name", "")
                        arguments = item.get("arguments", "")

                        # Update from tracked state
                        if output_index in current_tool_calls:
                            tc = current_tool_calls[output_index]
                            if not call_id:
                                call_id = tc["id"]
                            if not name:
                                name = tc["name"]
                            if not arguments:
                                arguments = tc["arguments"]

                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": output_index,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments,
                                        },
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        )
                        yield chunk

                # Handle completion
                elif kind == "response.completed":
                    # Determine finish reason
                    finish_reason = "stop"
                    if current_tool_calls:
                        finish_reason = "tool_calls"

                    # Send final chunk with reasoning if not sent
                    if not sent_reasoning and (reasoning_summary_text or reasoning_full_text):
                        rtxt = "\n\n".join(filter(None, [reasoning_summary_text, reasoning_full_text]))
                        if rtxt and reasoning_compat == "think-tags":
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                object="chat.completion.chunk",
                                choices=[{
                                    "index": 0,
                                    "delta": {"content": f"<think>{rtxt}</think>", "role": "assistant"},
                                    "finish_reason": None,
                                }],
                            )
                            yield chunk
                            sent_reasoning = True

                    # Extract usage if available
                    usage = None
                    resp_data = evt.get("response", {})
                    if isinstance(resp_data.get("usage"), dict):
                        u = resp_data["usage"]
                        usage = litellm.Usage(
                            prompt_tokens=u.get("input_tokens", 0),
                            completion_tokens=u.get("output_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        )

                    # Send final chunk
                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                    )
                    if usage:
                        final_chunk.usage = usage
                    yield final_chunk
                    break

                # Handle errors
                elif kind == "response.failed":
                    error = evt.get("response", {}).get("error", {})
                    error_msg = error.get("message", "Response failed")
                    lib_logger.error(f"Codex response failed: {error_msg}")
                    raise Exception(f"Codex response failed: {error_msg}")

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        reasoning_compat: str,
    ) -> litellm.ModelResponse:
        """Handle non-streaming response by collecting stream."""
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        full_text = ""
        reasoning_summary_text = ""
        reasoning_full_text = ""
        tool_calls: List[Dict[str, Any]] = []
        usage = None
        error_message = None

        async with client.stream(
            "POST",
            CODEX_RESPONSES_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(f"Codex API error {response.status_code}: {error_text[:500]}")
                raise httpx.HTTPStatusError(
                    f"Codex API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    break

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                kind = evt.get("type")

                # Handle response ID
                if isinstance(evt.get("response"), dict):
                    resp_id = evt["response"].get("id")
                    if resp_id:
                        response_id = resp_id

                # Collect text
                if kind == "response.output_text.delta":
                    full_text += evt.get("delta", "")

                # Collect reasoning
                elif kind == "response.reasoning_summary_text.delta":
                    reasoning_summary_text += evt.get("delta", "")

                elif kind == "response.reasoning_text.delta":
                    reasoning_full_text += evt.get("delta", "")

                # Collect tool calls
                elif kind == "response.output_item.done":
                    item = evt.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id") or item.get("id", "")
                        name = item.get("name", "")
                        arguments = item.get("arguments", "")
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        })

                # Extract usage
                elif kind == "response.completed":
                    resp_data = evt.get("response", {})
                    if isinstance(resp_data.get("usage"), dict):
                        u = resp_data["usage"]
                        usage = litellm.Usage(
                            prompt_tokens=u.get("input_tokens", 0),
                            completion_tokens=u.get("output_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        )

                # Handle errors
                elif kind == "response.failed":
                    error = evt.get("response", {}).get("error", {})
                    error_message = error.get("message", "Response failed")

        if error_message:
            raise Exception(f"Codex response failed: {error_message}")

        # Build message
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": full_text if full_text else None,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        # Apply reasoning
        message = _apply_reasoning_to_message(
            message, reasoning_summary_text, reasoning_full_text, reasoning_compat
        )

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else "stop"

        # Build response
        response_obj = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=model,
            object="chat.completion",
            choices=[{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        )

        if usage:
            response_obj.usage = usage

        return response_obj

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse quota/rate-limit errors from Codex API."""
        if not error_body:
            return None

        try:
            error_data = json.loads(error_body)
            error_info = error_data.get("error", {})

            if error_info.get("code") == "rate_limit_exceeded":
                # Look for retry-after information
                message = error_info.get("message", "")
                retry_after = 60  # Default

                # Try to extract from message
                import re
                match = re.search(r"try again in (\d+)s", message)
                if match:
                    retry_after = int(match.group(1))

                return {
                    "retry_after": retry_after,
                    "reason": "RATE_LIMITED",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }

            if error_info.get("code") == "quota_exceeded":
                return {
                    "retry_after": 3600,  # 1 hour default
                    "reason": "QUOTA_EXHAUSTED",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }

        except Exception:
            pass

        return None
