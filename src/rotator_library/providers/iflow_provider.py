# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/iflow_provider.py

import copy
import hmac
import hashlib
import json
import time
import os
import re
import threading
import asyncio
import httpx
import logging
from typing import Union, AsyncGenerator, List, Dict, Any, Optional, Tuple
from .provider_interface import ProviderInterface
from .iflow_auth_base import IFlowAuthBase
from .provider_cache import ProviderCache
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
import litellm
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
)
from pathlib import Path
import uuid
from datetime import datetime

lib_logger = logging.getLogger("rotator_library")


# Model list can be expanded as iFlow supports more models
HARDCODED_MODELS = [
    "glm-5",
    "glm-4.6",
    "glm-4.7",
    "minimax-m2",
    "minimax-m2.1",
    "qwen3-coder-plus",
    "kimi-k2",
    "kimi-k2-0905",
    "kimi-k2-thinking",  # Seems to not work, but should
    "kimi-k2.5",  # Seems to not work, but should
    "qwen3-max",
    "qwen3-max-preview",
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3.2-reasoner",
    "deepseek-v3.2-chat",
    "deepseek-v3.2",  # seems to not work, but should. Use above variants instead
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-vl-plus",
    "qwen3-235b-a22b-instruct",
    "qwen3-235b",
]

# OpenAI-compatible parameters supported by iFlow API
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_new_tokens",
    "max_completion_tokens",
    "stream",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "response_format",
    "thinking",
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_split",
}

IFLOW_USER_AGENT = "iFlow-Cli"
IFLOW_HEADER_SESSION_ID = "session-id"
IFLOW_HEADER_CONVERSATION_ID = "conversation-id"
IFLOW_HEADER_TIMESTAMP = "x-iflow-timestamp"
IFLOW_HEADER_SIGNATURE = "x-iflow-signature"
IFLOW_HEADER_API_KEY = "x-api-key"
IFLOW_HEADER_TRACEPARENT = "traceparent"
IFLOW_HEADER_X_BIZ_INFO = "x-biz-info"
IFLOW_HEADER_EAGLEEYE_USERDATA = "EagleEye-UserData"
IFLOW_HEADER_PRIORITY = "priority"

TRACEPARENT_PATTERN = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
DEFAULT_IFLOW_STICKY_MODE = "auto"
DEFAULT_IFLOW_STICKY_TTL_SECONDS = 86400
DEFAULT_IFLOW_STICKY_MAX_ENTRIES = 10000


def _is_truthy_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Connection error types that should trigger retries.
CONNECTION_ERROR_TYPES = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    httpx.NetworkError,
)

# Context window error patterns to detect in provider error bodies.
CONTEXT_WINDOW_ERROR_PATTERNS = (
    "context_length",
    "token limit",
    "context window",
    "too many tokens",
    "too long",
    "max_tokens",
)

MAX_CONNECTION_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0

# =============================================================================
# THINKING MODE CONFIGURATION
# =============================================================================
# Models using chat_template_kwargs.enable_thinking (boolean toggle)
# Based on Go implementation: internal/thinking/provider/iflow/apply.go
ENABLE_THINKING_MODELS = {
    "glm-4.6",
    "glm-4.7",
    "qwen3-max-preview",
    "deepseek-v3.2",
    "deepseek-v3.1",
}

# GLM models need additional clear_thinking=false when thinking is enabled
GLM_MODELS = {"glm-4.6", "glm-4.7"}

# Models using reasoning_split (boolean) instead of enable_thinking
REASONING_SPLIT_MODELS = {"minimax-m2", "minimax-m2.1"}

# Models that benefit from reasoning_content preservation in message history
# (for multi-turn conversations)
REASONING_PRESERVATION_MODELS_PREFIXES = ("glm-4", "minimax-m2")

# Cache file path for reasoning content preservation
_REASONING_CACHE_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "cache"
    / "iflow_reasoning.json"
)


class IFlowProvider(IFlowAuthBase, ProviderInterface):
    """
    iFlow provider using OAuth authentication with local callback server.
    API requests use the derived API key (NOT OAuth access_token).
    """

    skip_cost_calculation = True

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

        # Initialize reasoning cache for multi-turn conversation support
        # Created in __init__ (not module level) to ensure event loop exists
        self._reasoning_cache = ProviderCache(
            cache_file=_REASONING_CACHE_FILE,
            memory_ttl_seconds=3600,  # 1 hour in memory
            disk_ttl_seconds=86400,  # 24 hours on disk
            env_prefix="IFLOW_REASONING_CACHE",
        )

        self._sticky_mode = os.getenv(
            "IFLOW_STICKY_SESSION_MODE", DEFAULT_IFLOW_STICKY_MODE
        ).strip().lower()
        self._sticky_ttl_seconds = max(
            60,
            int(
                os.getenv(
                    "IFLOW_STICKY_SESSION_TTL_SECONDS",
                    str(DEFAULT_IFLOW_STICKY_TTL_SECONDS),
                )
            ),
        )
        self._sticky_max_entries = max(
            100,
            int(
                os.getenv(
                    "IFLOW_STICKY_SESSION_MAX_ENTRIES",
                    str(DEFAULT_IFLOW_STICKY_MAX_ENTRIES),
                )
            ),
        )
        self._sticky_lock = threading.Lock()
        self._sticky_session_cache: Dict[str, Tuple[str, float]] = {}
        self._sticky_conversation_cache: Dict[str, Tuple[str, float]] = {}

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of iFlow models from three sources:
        1. Environment variable models (via IFLOW_MODELS) - ALWAYS included, take priority
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        3. Dynamic discovery from iFlow API (if supported) - added only if ID not in env vars

        Environment variable models always win and are never deduplicated, even if they
        share the same ID (to support different configs like temperature, etc.)

        Validates OAuth credentials if applicable.
        """
        models = []
        env_var_ids = (
            set()
        )  # Track IDs from env vars to prevent hardcoded/dynamic duplicates

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
        static_models = self.model_definitions.get_all_provider_models("iflow")
        if static_models:
            for model in static_models:
                # Extract model name from "iflow/ModelName" format
                model_name = model.split("/")[-1] if "/" in model else model
                # Get the actual model ID from definitions (which may differ from the name)
                model_id = self.model_definitions.get_model_id("iflow", model_name)

                # ALWAYS add env var models (no deduplication)
                models.append(model)
                # Track the ID to prevent hardcoded/dynamic duplicates
                if model_id:
                    env_var_ids.add(model_id)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for iflow from environment variables"
            )

        # Source 2: Add hardcoded models (only if ID not already in env vars)
        for model_id in HARDCODED_MODELS:
            if model_id not in env_var_ids:
                models.append(f"iflow/{model_id}")
                env_var_ids.add(model_id)

        # Source 3: Try dynamic discovery from iFlow API (only if ID not already in env vars)
        try:
            # Validate OAuth credentials and get API details
            if os.path.isfile(credential):
                await self.initialize_token(credential)

            _, api_key = await self.get_api_details(credential)
            api_bases = self.get_api_base_candidates()
            request_ids = self._extract_iflow_ids({})

            last_error: Optional[Exception] = None
            fetched = False

            for idx, api_base in enumerate(api_bases):
                models_url = f"{api_base.rstrip('/')}/models"
                try:
                    headers = self._build_iflow_headers(
                        api_key=api_key,
                        stream=False,
                        request_ids=request_ids,
                        include_signature=True,
                    )
                    response = await client.get(
                        models_url,
                        headers=headers,
                        timeout=TimeoutConfig.non_streaming(),
                    )

                    if response.status_code == 406:
                        unsigned_headers = self._build_iflow_headers(
                            api_key=api_key,
                            stream=False,
                            request_ids=request_ids,
                            include_signature=False,
                        )
                        response = await client.get(
                            models_url,
                            headers=unsigned_headers,
                            timeout=TimeoutConfig.non_streaming(),
                        )

                    response.raise_for_status()

                    dynamic_data = response.json()
                    # Handle both {data: [...]} and direct [...] formats
                    model_list = (
                        dynamic_data.get("data", dynamic_data)
                        if isinstance(dynamic_data, dict)
                        else dynamic_data
                    )

                    dynamic_count = 0
                    for model in model_list:
                        model_id = extract_model_id(model)
                        if model_id and model_id not in env_var_ids:
                            models.append(f"iflow/{model_id}")
                            env_var_ids.add(model_id)
                            dynamic_count += 1

                    if dynamic_count > 0:
                        lib_logger.debug(
                            f"Discovered {dynamic_count} additional models for iflow from API"
                        )

                    fetched = True
                    break

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_error = e
                    if idx + 1 < len(api_bases):
                        lib_logger.warning(
                            f"iFlow models fetch network error on {api_base}, trying fallback base"
                        )
                        continue
                    raise
                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code
                    if self._should_fallback_base(status_code) and idx + 1 < len(api_bases):
                        lib_logger.warning(
                            f"iFlow models fetch got HTTP {status_code} on {api_base}, trying fallback base"
                        )
                        continue
                    raise

            if not fetched and last_error:
                raise last_error

        except Exception as e:
            # Silently ignore dynamic discovery errors
            lib_logger.debug(f"Dynamic model discovery failed for iflow: {e}")
            pass

        return models

    def _clean_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes unsupported properties from tool schemas to prevent API errors.
        Similar to Qwen Code implementation.
        """
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

    # =========================================================================
    # THINKING MODE SUPPORT
    # =========================================================================

    def _should_enable_thinking(self, kwargs: Dict[str, Any]) -> Optional[bool]:
        """
        Check if thinking should be enabled based on request parameters.

        Uses OpenAI-compatible format. Checks for reasoning_effort parameter.
        Thinking is enabled for any value except "none", "disabled", or "0".

        Returns:
            True: Enable thinking
            False: Disable thinking explicitly
            None: No thinking params (passthrough - don't modify payload)
        """
        # Check explicit iFlow thinking fields first
        direct_thinking = kwargs.get("thinking")
        if direct_thinking is not None:
            if isinstance(direct_thinking, dict):
                thinking_type = str(direct_thinking.get("type", "")).lower().strip()
                if thinking_type == "disabled":
                    return False
                if thinking_type == "enabled":
                    budget = direct_thinking.get("budget_tokens")
                    if budget is None:
                        return True
                    try:
                        return int(budget) != 0
                    except Exception:
                        return True
            return bool(direct_thinking)

        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is not None:
            if isinstance(enable_thinking, str):
                lowered = enable_thinking.lower().strip()
                return lowered not in ("0", "false", "off", "none", "disabled")
            return bool(enable_thinking)

        chat_template_kwargs = kwargs.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            ctk_enable = chat_template_kwargs.get("enable_thinking")
            if ctk_enable is not None:
                if isinstance(ctk_enable, str):
                    lowered = ctk_enable.lower().strip()
                    return lowered not in (
                        "0",
                        "false",
                        "off",
                        "none",
                        "disabled",
                    )
                return bool(ctk_enable)

        # Check reasoning_effort (OpenAI-style)
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            effort_lower = str(reasoning_effort).lower().strip()
            # Disabled values
            if effort_lower in ("none", "disabled", "0", "off", "false"):
                # lib_logger.info(
                #    f"iFlow: Detected reasoning_effort='{reasoning_effort}' → thinking DISABLED"
                # )
                return False
            # Any other value enables thinking
            # lib_logger.info(
            #    f"iFlow: Detected reasoning_effort='{reasoning_effort}' → thinking ENABLED"
            # )
            return True

        # Check extra_body for thinking config (Claude-style, for compatibility)
        extra_body = kwargs.get("extra_body", {})
        if extra_body and "thinking" in extra_body:
            thinking = extra_body["thinking"]
            if isinstance(thinking, dict):
                budget = thinking.get("budget_tokens", 0)
                return budget != 0
            return bool(thinking)

        return None  # No thinking params specified

    def _apply_thinking_config(
        self, payload: Dict[str, Any], model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply thinking configuration for supported iFlow models.

        Logic matches Go implementation (internal/thinking/provider/iflow/apply.go):
        - GLM models: enable_thinking + clear_thinking=false (when enabled)
        - Qwen/DeepSeek: enable_thinking only
        - MiniMax: reasoning_split

        Args:
            payload: The request payload to modify
            model_name: Model name (without provider prefix)
            kwargs: Original request kwargs containing reasoning_effort, etc.

        Returns:
            Modified payload with thinking config applied
        """
        model_lower = model_name.lower()
        enable_thinking = self._should_enable_thinking(kwargs)

        if enable_thinking is None:
            return payload  # No thinking params, passthrough

        # Check model type
        is_glm = model_lower in GLM_MODELS
        is_enable_thinking = model_lower in ENABLE_THINKING_MODELS or is_glm
        is_minimax = model_lower in REASONING_SPLIT_MODELS

        if is_enable_thinking:
            # Models using chat_template_kwargs.enable_thinking
            if "chat_template_kwargs" not in payload:
                payload["chat_template_kwargs"] = {}
            payload["chat_template_kwargs"]["enable_thinking"] = enable_thinking

            # GLM models: strip clear_thinking first (like Go does with DeleteBytes),
            # then set it to false only when thinking is enabled
            if is_glm:
                payload["chat_template_kwargs"].pop("clear_thinking", None)
                if enable_thinking:
                    payload["chat_template_kwargs"]["clear_thinking"] = False

            lib_logger.info(
                f"iFlow: Applied enable_thinking={enable_thinking} for {model_name}"
            )
        elif is_minimax:
            # MiniMax models use reasoning_split
            payload["reasoning_split"] = enable_thinking
            lib_logger.info(
                f"iFlow: Applied reasoning_split={enable_thinking} for {model_name}"
            )

        return payload

    # =========================================================================
    # REASONING CONTENT PRESERVATION
    # =========================================================================

    def _get_conversation_signature(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a stable conversation signature from the first user message.

        This provides conversation-level uniqueness for cache keys.
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    # Use first 100 chars of first user message as conversation signature
                    return hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return "default"

    def _get_message_cache_key(self, message: Dict[str, Any], conv_sig: str) -> str:
        """
        Generate a cache key for a message to look up cached reasoning.

        Combines:
        - Conversation signature (stable per conversation)
        - Message content hash (identifies specific message)
        """
        content = message.get("content", "") or ""
        role = message.get("role", "")
        # Use content[:200] + role for message identity
        msg_hash = hashlib.md5(f"{role}:{content[:200]}".encode()).hexdigest()[:12]
        return f"{conv_sig}:{msg_hash}"

    def _store_reasoning_content(self, message: Dict[str, Any], conv_sig: str) -> None:
        """
        Store reasoning_content from an assistant message for later retrieval.

        Args:
            message: The assistant message dict containing reasoning_content
            conv_sig: Conversation signature for the cache key
        """
        reasoning = message.get("reasoning_content")
        if reasoning and message.get("role") == "assistant":
            key = self._get_message_cache_key(message, conv_sig)
            self._reasoning_cache.store(key, reasoning)
            lib_logger.debug(f"iFlow: Cached reasoning_content for message {key}")

    def _inject_reasoning_content(
        self, messages: List[Dict[str, Any]], model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Inject cached reasoning_content into assistant messages.

        Only for models that benefit from reasoning preservation (GLM-4.x, MiniMax-M2.x).
        This is helpful for multi-turn conversations where the model may benefit
        from seeing its previous reasoning to maintain coherent thought chains.

        Args:
            messages: List of messages in the conversation
            model_name: Model name (without provider prefix)

        Returns:
            Messages list with reasoning_content restored where available
        """
        model_lower = model_name.lower()

        # Only for models that benefit from reasoning preservation
        if not any(
            model_lower.startswith(prefix)
            for prefix in REASONING_PRESERVATION_MODELS_PREFIXES
        ):
            return messages

        # Get conversation signature
        conv_sig = self._get_conversation_signature(messages)

        result = []
        restored_count = 0
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                key = self._get_message_cache_key(msg, conv_sig)
                cached = self._reasoning_cache.retrieve(key)
                if cached:
                    msg = {**msg, "reasoning_content": cached}
                    restored_count += 1
            result.append(msg)

        if restored_count > 0:
            lib_logger.debug(
                f"iFlow: Restored reasoning_content for {restored_count} messages in {model_name}"
            )

        return result

    def _build_request_payload(
        self, model_name: str, full_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Builds a clean request payload with only supported parameters.
        Also applies thinking mode and reasoning content preservation.

        Args:
            model_name: Model name without provider prefix (for thinking/reasoning logic)
            full_kwargs: Original kwargs (for extracting reasoning_effort, etc.)
            **kwargs: Filtered kwargs with stripped model name

        Returns:
            Complete payload ready for iFlow API
        """
        # Extract only supported OpenAI parameters
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        # Always force streaming for internal processing
        payload["stream"] = True

        # NOTE: iFlow API does not support stream_options parameter
        # Unlike other providers, we don't include it to avoid HTTP 406 errors

        # Handle tool schema cleaning
        if "tools" in payload and payload["tools"]:
            payload["tools"] = self._clean_tool_schemas(payload["tools"])
            lib_logger.debug(f"Cleaned {len(payload['tools'])} tool schemas")
        elif (
            "tools" in payload
            and isinstance(payload["tools"], list)
            and len(payload["tools"]) == 0
        ):
            # Inject dummy tool for empty arrays to prevent streaming issues (similar to Qwen's behavior)
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Placeholder tool to stabilise streaming",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            lib_logger.debug("Injected placeholder tool for empty tools array")

        # Inject cached reasoning_content into messages for multi-turn conversations
        if "messages" in payload:
            payload["messages"] = self._inject_reasoning_content(
                payload["messages"], model_name
            )

        # Apply thinking mode configuration based on reasoning_effort
        payload = self._apply_thinking_config(payload, model_name, full_kwargs)
        explicit_thinking = self._should_enable_thinking(full_kwargs)

        # CLI-like defaults when absent
        payload.setdefault("temperature", 1)
        payload.setdefault("top_p", 0.95)

        # Align token field naming with native iFlow CLI payload shape
        if "max_new_tokens" not in payload:
            if "max_completion_tokens" in payload:
                payload["max_new_tokens"] = payload["max_completion_tokens"]
            elif "max_tokens" in payload:
                payload["max_new_tokens"] = payload["max_tokens"]
            else:
                payload["max_new_tokens"] = 32000

        # Enable thinking by default unless caller explicitly disables it.
        if "enable_thinking" not in payload:
            if explicit_thinking is not None:
                payload["enable_thinking"] = bool(explicit_thinking)
            else:
                payload["enable_thinking"] = _is_truthy_env(
                    os.getenv("IFLOW_ENABLE_THINKING_BY_DEFAULT", "true")
                )

        if "thinking" not in payload:
            payload["thinking"] = {
                "type": "enabled" if bool(payload["enable_thinking"]) else "disabled"
            }

        model_lower = model_name.lower()
        has_enable_thinking = "enable_thinking" in payload
        enable_thinking_value = bool(payload.get("enable_thinking"))
        if (
            model_lower.startswith("glm-")
            or model_lower in ENABLE_THINKING_MODELS
            or model_lower in GLM_MODELS
            or isinstance(payload.get("chat_template_kwargs"), dict)
        ):
            chat_template = payload.get("chat_template_kwargs")
            if not isinstance(chat_template, dict):
                chat_template = {}
            if has_enable_thinking:
                chat_template.setdefault("enable_thinking", enable_thinking_value)
            if model_lower in GLM_MODELS:
                if has_enable_thinking and chat_template.get("enable_thinking"):
                    chat_template["clear_thinking"] = False
                else:
                    chat_template.pop("clear_thinking", None)
            if chat_template:
                payload["chat_template_kwargs"] = chat_template

        if (
            model_lower in REASONING_SPLIT_MODELS
            and "reasoning_split" not in payload
            and has_enable_thinking
        ):
            payload["reasoning_split"] = enable_thinking_value

        return payload

    def _create_iflow_signature(
        self, user_agent: str, session_id: str, timestamp_ms: int, api_key: str
    ) -> str:
        """Generate iFlow HMAC-SHA256 signature: userAgent:sessionId:timestamp."""
        if not api_key:
            return ""

        payload = f"{user_agent}:{session_id}:{timestamp_ms}"
        return hmac.new(
            api_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _sticky_enabled(self) -> bool:
        return self._sticky_mode not in {"off", "false", "0", "none", "disabled"}

    def _sticky_mode_value(self) -> str:
        mode = self._sticky_mode
        if mode in {"conversation", "conv"}:
            return "conversation"
        if mode in {"client", "user"}:
            return "client"
        return "auto"

    def _prune_sticky_cache(self, cache: Dict[str, Tuple[str, float]], now: float) -> None:
        expired = [
            key for key, (_, ts) in cache.items() if now - ts > self._sticky_ttl_seconds
        ]
        for key in expired:
            cache.pop(key, None)

        overflow = len(cache) - self._sticky_max_entries
        if overflow > 0:
            oldest = sorted(cache.items(), key=lambda item: item[1][1])[:overflow]
            for key, _ in oldest:
                cache.pop(key, None)

    def _get_sticky_value(
        self,
        cache: Dict[str, Tuple[str, float]],
        keys: List[str],
        now: float,
    ) -> Optional[str]:
        self._prune_sticky_cache(cache, now)
        for key in keys:
            cached = cache.get(key)
            if cached is None:
                continue
            value, _ = cached
            cache[key] = (value, now)
            return value
        return None

    def _set_sticky_value(
        self,
        cache: Dict[str, Tuple[str, float]],
        keys: List[str],
        value: str,
        now: float,
    ) -> None:
        self._prune_sticky_cache(cache, now)
        for key in keys:
            cache[key] = (value, now)

    def _build_sticky_keys(
        self,
        sources: List[Dict[str, Any]],
        messages: Any,
        conversation_id: str,
    ) -> List[str]:
        mode = self._sticky_mode_value()

        def _pick_case_insensitive(*keys: str) -> str:
            for source in sources:
                for key in keys:
                    for source_key, value in source.items():
                        if (
                            isinstance(source_key, str)
                            and source_key.lower() == key.lower()
                            and value not in (None, "")
                        ):
                            return str(value)
            return ""

        keys: List[str] = []
        if conversation_id:
            conv_hash = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:32]
            keys.append(f"conv:{conv_hash}")

        client_identifier = _pick_case_insensitive(
            "client_id",
            "clientId",
            "session_key",
            "thread_id",
            "threadId",
            "user",
            "user_id",
            "userId",
            "x-user-id",
            "x-client-id",
        )
        if client_identifier:
            client_hash = hashlib.sha256(client_identifier.encode("utf-8")).hexdigest()[:32]
            keys.append(f"client:{client_hash}")

        if _is_truthy_env(os.getenv("IFLOW_STICKY_HISTORY_FALLBACK", "false")) and isinstance(
            messages, list
        ):
            conv_sig = self._get_conversation_signature(messages)
            if conv_sig and conv_sig != "default":
                keys.append(f"history:{conv_sig}")

        if mode == "conversation":
            keys = [k for k in keys if k.startswith("conv:") or k.startswith("history:")]
        elif mode == "client":
            keys = [k for k in keys if k.startswith("client:") or k.startswith("history:")]

        return list(dict.fromkeys(keys))

    def _extract_iflow_ids(
        self, request_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Extract iFlow routing + tracing metadata from request args."""
        request_args = request_args or {}

        metadata = request_args.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        extra_headers = request_args.get("extra_headers")
        if not isinstance(extra_headers, dict):
            extra_headers = {}

        sources: List[Dict[str, Any]] = [request_args, metadata, extra_headers]

        def _pick(*keys: str) -> str:
            for source in sources:
                for key in keys:
                    for source_key, value in source.items():
                        if (
                            isinstance(source_key, str)
                            and source_key.lower() == key.lower()
                            and value not in (None, "")
                        ):
                            return str(value)
            return ""

        def _generate_traceparent() -> str:
            trace_id = uuid.uuid4().hex
            span_id = uuid.uuid4().hex[:16]
            return f"00-{trace_id}-{span_id}-01"

        def _normalize_traceparent(raw_value: str) -> str:
            lowered = raw_value.strip().lower()
            if lowered and TRACEPARENT_PATTERN.match(lowered):
                return lowered
            return _generate_traceparent()

        explicit_session_id = _pick(
            "session_id",
            "sessionId",
            "litellm_session_id",
            "session-id",
            "x-litellm-session-id",
        )
        explicit_conversation_id = _pick(
            "conversation_id",
            "conversationId",
            "litellm_conversation_id",
            "conversation-id",
            "x-litellm-conversation-id",
        )

        messages = request_args.get("messages")

        conversation_id = explicit_conversation_id
        conversation_keys = self._build_sticky_keys(
            sources=sources,
            messages=messages,
            conversation_id=conversation_id,
        )
        if not conversation_id:
            if self._sticky_enabled() and conversation_keys:
                now = time.time()
                with self._sticky_lock:
                    cached_conversation = self._get_sticky_value(
                        cache=self._sticky_conversation_cache,
                        keys=conversation_keys,
                        now=now,
                    )
                    if cached_conversation:
                        conversation_id = cached_conversation
                    else:
                        conversation_id = str(uuid.uuid4())
                        self._set_sticky_value(
                            cache=self._sticky_conversation_cache,
                            keys=conversation_keys,
                            value=conversation_id,
                            now=now,
                        )
            else:
                conversation_id = str(uuid.uuid4())

        session_keys = self._build_sticky_keys(
            sources=sources,
            messages=messages,
            conversation_id=conversation_id,
        )
        session_id = explicit_session_id
        if not session_id:
            if self._sticky_enabled() and session_keys:
                now = time.time()
                with self._sticky_lock:
                    cached_session = self._get_sticky_value(
                        cache=self._sticky_session_cache,
                        keys=session_keys,
                        now=now,
                    )
                    if cached_session:
                        session_id = cached_session
                    else:
                        session_id = f"session-{uuid.uuid4()}"
                        self._set_sticky_value(
                            cache=self._sticky_session_cache,
                            keys=session_keys,
                            value=session_id,
                            now=now,
                        )
            else:
                session_id = f"session-{uuid.uuid4()}"
        elif self._sticky_enabled() and session_keys:
            now = time.time()
            with self._sticky_lock:
                self._set_sticky_value(
                    cache=self._sticky_session_cache,
                    keys=session_keys,
                    value=str(session_id),
                    now=now,
                )

        if self._sticky_enabled() and conversation_keys and conversation_id:
            now = time.time()
            with self._sticky_lock:
                self._set_sticky_value(
                    cache=self._sticky_conversation_cache,
                    keys=conversation_keys,
                    value=str(conversation_id),
                    now=now,
                )

        traceparent = _normalize_traceparent(
            _pick("traceparent", IFLOW_HEADER_TRACEPARENT)
        )

        return {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "traceparent": traceparent,
            "x_biz_info": _pick(
                "iflow_x_biz_info", "x_biz_info", IFLOW_HEADER_X_BIZ_INFO
            ),
            "eagleeye_userdata": _pick(
                "iflow_eagleeye_userdata",
                "eagleeye_userdata",
                IFLOW_HEADER_EAGLEEYE_USERDATA,
            ),
            "priority": _pick("iflow_priority", IFLOW_HEADER_PRIORITY),
        }

    def _build_iflow_headers(
        self,
        api_key: str,
        stream: bool,
        request_ids: Optional[Dict[str, str]] = None,
        include_signature: bool = True,
    ) -> Dict[str, str]:
        """Build iFlow request headers, with optional anti-block signature headers."""
        request_ids = request_ids or self._extract_iflow_ids()
        session_id = request_ids["session_id"]
        conversation_id = request_ids["conversation_id"]
        traceparent = request_ids.get("traceparent", "")
        timestamp_ms = int(time.time() * 1000)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": IFLOW_USER_AGENT,
            IFLOW_HEADER_SESSION_ID: session_id,
            IFLOW_HEADER_CONVERSATION_ID: conversation_id,
            IFLOW_HEADER_TRACEPARENT: traceparent,
            "Accept": "*/*",
            "Accept-Language": "*",
            "Sec-Fetch-Mode": "cors",
            "Accept-Encoding": "br, gzip, deflate",
        }

        if _is_truthy_env(os.getenv("IFLOW_SEND_X_API_KEY", "false")):
            headers[IFLOW_HEADER_API_KEY] = api_key

        if request_ids.get("x_biz_info"):
            headers[IFLOW_HEADER_X_BIZ_INFO] = request_ids["x_biz_info"]
        if request_ids.get("eagleeye_userdata"):
            headers[IFLOW_HEADER_EAGLEEYE_USERDATA] = request_ids[
                "eagleeye_userdata"
            ]
        if request_ids.get("priority"):
            headers[IFLOW_HEADER_PRIORITY] = request_ids["priority"]

        if include_signature:
            signature = self._create_iflow_signature(
                IFLOW_USER_AGENT, session_id, timestamp_ms, api_key
            )
            headers[IFLOW_HEADER_TIMESTAMP] = str(timestamp_ms)
            headers[IFLOW_HEADER_SIGNATURE] = signature

        return headers

    def _should_fallback_base(self, status_code: int) -> bool:
        """Return True for block-like status codes worth base fallback."""
        if status_code in {403, 406, 408, 423, 451, 502, 503, 504}:
            return True
        return 520 <= status_code <= 530

    def _extract_finish_reason_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """
        Extract finish_reason from a raw iFlow chunk by searching all possible locations.

        Args:
            chunk: Raw chunk from iFlow API

        Returns:
            The finish_reason if found, None otherwise
        """
        choices = chunk.get("choices", [])
        for choice in choices:
            if isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    return finish_reason
        return None

    def _convert_chunk_to_openai(
        self,
        chunk: Dict[str, Any],
        model_id: str,
        stream_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Converts a raw iFlow SSE chunk to an OpenAI-compatible chunk.
        Since iFlow is OpenAI-compatible, minimal conversion is needed.

        ROBUST FINISH_REASON HANDLING:
        - Tracks finish_reason across all chunks in stream_state
        - tool_calls takes priority over stop
        - Always sets finish_reason on final chunks (with usage)
        - Logs warning if no finish_reason found

        Args:
            chunk: Raw chunk from iFlow API
            model_id: Model identifier for response
            stream_state: Mutable dict to track state across chunks
        """
        if not isinstance(chunk, dict):
            return

        # Initialize stream_state if not provided
        if stream_state is None:
            stream_state = {}

        # Get choices and usage data
        choices = chunk.get("choices", [])
        usage_data = chunk.get("usage")

        # IMPORTANT: Empty dict {} is falsy in Python, but "usage": {} still indicates final chunk
        # Use "is not None" check instead of truthiness
        has_usage = usage_data is not None
        is_final_chunk = has_usage

        # Extract and track finish_reason from raw chunk
        raw_finish_reason = self._extract_finish_reason_from_chunk(chunk)
        if raw_finish_reason:
            stream_state["last_finish_reason"] = raw_finish_reason
            # lib_logger.debug(
            #    f"iFlow: Found finish_reason='{raw_finish_reason}' in raw chunk"
            # )

        def normalize_choices(
            choices_list: List[Dict[str, Any]],
            force_final: bool = False,
        ) -> List[Dict[str, Any]]:
            """
            Normalizes choices array with robust finish_reason handling.

            Priority for finish_reason:
            1. tool_calls (if any tool_calls were seen in the stream)
            2. Explicit finish_reason from this chunk
            3. Last tracked finish_reason from stream_state
            4. Default to 'stop' (with warning)
            """
            normalized = []
            for choice in choices_list:
                choice_copy = dict(choice) if isinstance(choice, dict) else choice
                delta = choice_copy.get("delta", {})

                # Track tool_calls presence
                if delta.get("tool_calls"):
                    stream_state["has_tool_calls"] = True

                # Track reasoning_content presence (for logging)
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content and reasoning_content.strip():
                    if not stream_state.get("has_reasoning_logged"):
                        # lib_logger.debug(
                        #    f"iFlow: Chunk contains reasoning_content "
                        #    f"({len(reasoning_content)} chars)"
                        # )
                        stream_state["has_reasoning_logged"] = True

                # Get current finish_reason
                finish_reason = choice_copy.get("finish_reason")

                # Track any finish_reason we see
                if finish_reason:
                    stream_state["last_finish_reason"] = finish_reason

                # For final chunks, ensure finish_reason is ALWAYS set
                if force_final:
                    # Priority: tool_calls > explicit > tracked > stop (with warning)
                    if stream_state.get("has_tool_calls"):
                        # Tool calls take highest priority
                        final_reason = "tool_calls"
                        if finish_reason and finish_reason != "tool_calls":
                            pass  # Silently override - tool_calls takes priority
                            # lib_logger.debug(
                            #    f"iFlow: Overriding finish_reason '{finish_reason}' "
                            #    f"with 'tool_calls' (tool_calls present)"
                            # )
                    elif finish_reason:
                        # Use explicit finish_reason from this chunk
                        final_reason = finish_reason
                    elif stream_state.get("last_finish_reason"):
                        # Use tracked finish_reason from earlier chunk
                        final_reason = stream_state["last_finish_reason"]
                        # lib_logger.debug(
                        #    f"iFlow: Using tracked finish_reason '{final_reason}' "
                        #    f"for final chunk"
                        # )
                    else:
                        # No finish_reason found anywhere - default to stop with warning
                        final_reason = "stop"
                        lib_logger.warning(
                            f"iFlow: No finish_reason found in stream, defaulting to 'stop'"
                        )

                    choice_copy = {**choice_copy, "finish_reason": final_reason}
                    # lib_logger.debug(
                    #    f"iFlow: Final chunk finish_reason set to '{final_reason}'"
                    # )
                else:
                    # For non-final chunks, normalize tool_calls if needed
                    if (
                        finish_reason
                        and stream_state.get("has_tool_calls")
                        and finish_reason != "tool_calls"
                    ):
                        choice_copy = {**choice_copy, "finish_reason": "tool_calls"}

                normalized.append(choice_copy)
            return normalized

        # Handle chunks with usage (final chunk indicator)
        # Note: "usage": {} (empty dict) still indicates final chunk
        if choices and has_usage:
            # Normalize choices for final chunk - MUST set finish_reason
            normalized_choices = normalize_choices(choices, force_final=True)
            # Build usage dict, handling empty usage gracefully
            usage_dict = dict(usage_data) if isinstance(usage_data, dict) else {}
            usage_dict.setdefault("prompt_tokens", 0)
            usage_dict.setdefault("completion_tokens", 0)
            usage_dict.setdefault("total_tokens", 0)

            # CRITICAL FIX: If usage is empty/all-zeros (e.g., MiniMax sends "usage": {}),
            # set placeholder non-zero values to ensure downstream processing
            # (litellm/client) recognizes this as a final chunk and preserves finish_reason
            if not any(
                usage_dict.get(k, 0) > 0
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
            ):
                usage_dict = {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                }
                # lib_logger.debug(
                #    "iFlow: Empty usage detected, using placeholder values for final chunk"
                # )

            yield {
                "choices": normalized_choices,
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-iflow-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
                "usage": usage_dict,
            }
            return

        # Handle usage-only chunks (no choices)
        if has_usage and not choices:
            usage_dict = dict(usage_data) if isinstance(usage_data, dict) else {}
            usage_dict.setdefault("prompt_tokens", 0)
            usage_dict.setdefault("completion_tokens", 0)
            usage_dict.setdefault("total_tokens", 0)
            yield {
                "choices": [],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-iflow-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
                "usage": usage_dict,
            }
            return

        # Handle content-only chunks (no usage)
        if choices:
            # Normalize choices - not final, so finish_reason not forced
            normalized_choices = normalize_choices(choices, force_final=False)
            yield {
                "choices": normalized_choices,
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-iflow-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
            }

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.

        Key improvements:
        - Determines finish_reason based on accumulated state (tool_calls vs stop)
        - Properly initializes tool_calls with type field
        - Handles usage data extraction from chunks
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        # Initialize the final response structure
        final_message = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        chunk_finish_reason = (
            None  # Track finish_reason from chunks (but we'll override)
        )

        # Get the first chunk for basic response metadata
        first_chunk = chunks[0]

        # Process each chunk to aggregate content
        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            # Handle both dict and object access patterns for choice.delta
            if hasattr(choice, "get"):
                delta = choice.get("delta", {})
                choice_finish = choice.get("finish_reason")
            elif hasattr(choice, "delta"):
                delta = choice.delta if choice.delta else {}
                # Convert delta to dict if it's an object
                if hasattr(delta, "__dict__") and not isinstance(delta, dict):
                    delta = {
                        k: v
                        for k, v in delta.__dict__.items()
                        if not k.startswith("_") and v is not None
                    }
                elif hasattr(delta, "model_dump"):
                    delta = delta.model_dump(exclude_none=True)
                choice_finish = getattr(choice, "finish_reason", None)
            else:
                delta = {}
                choice_finish = None

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

            # Aggregate tool calls with proper initialization
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_chunk in delta["tool_calls"]:
                    index = tc_chunk.get("index", 0)
                    if index not in aggregated_tool_calls:
                        # Initialize with type field for OpenAI compatibility
                        aggregated_tool_calls[index] = {
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if "id" in tc_chunk:
                        aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                    if "type" in tc_chunk:
                        aggregated_tool_calls[index]["type"] = tc_chunk["type"]
                    if "function" in tc_chunk:
                        if (
                            "name" in tc_chunk["function"]
                            and tc_chunk["function"]["name"] is not None
                        ):
                            aggregated_tool_calls[index]["function"]["name"] += (
                                tc_chunk["function"]["name"]
                            )
                        if (
                            "arguments" in tc_chunk["function"]
                            and tc_chunk["function"]["arguments"] is not None
                        ):
                            aggregated_tool_calls[index]["function"]["arguments"] += (
                                tc_chunk["function"]["arguments"]
                            )

            # Aggregate function calls (legacy format)
            if "function_call" in delta and delta["function_call"] is not None:
                if "function_call" not in final_message:
                    final_message["function_call"] = {"name": "", "arguments": ""}
                if (
                    "name" in delta["function_call"]
                    and delta["function_call"]["name"] is not None
                ):
                    final_message["function_call"]["name"] += delta["function_call"][
                        "name"
                    ]
                if (
                    "arguments" in delta["function_call"]
                    and delta["function_call"]["arguments"] is not None
                ):
                    final_message["function_call"]["arguments"] += delta[
                        "function_call"
                    ]["arguments"]

            # Track finish_reason from chunks (for reference only)
            if choice_finish:
                chunk_finish_reason = choice_finish

        # Handle usage data from the last chunk that has it
        for chunk in reversed(chunks):
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                break

        # Add tool calls to final message if any
        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())

        # Ensure standard fields are present for consistent logging
        for field in ["content", "tool_calls", "function_call"]:
            if field not in final_message:
                final_message[field] = None

        # Remove MiniMax-specific reasoning_details - we have the full reasoning_content
        # The reasoning_details array only contains partial data from the last chunk
        final_message.pop("reasoning_details", None)

        # Determine finish_reason based on accumulated state
        # Priority: tool_calls wins if present, then chunk's finish_reason, then default to "stop"
        if aggregated_tool_calls:
            finish_reason = "tool_calls"
        elif chunk_finish_reason:
            finish_reason = chunk_finish_reason
        else:
            finish_reason = "stop"

        # Construct the final response
        final_choice = {
            "index": 0,
            "message": final_message,
            "finish_reason": finish_reason,
        }

        # Create the final ModelResponse
        final_response_data = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": first_chunk.model,
            "choices": [final_choice],
            "usage": usage_data,
        }

        return litellm.ModelResponse(**final_response_data)

    def _get_usage_token_count(self, usage: Any, token_key: str) -> int:
        """Extract usage token count from dict/object usage payloads."""
        if usage is None:
            return 0

        if isinstance(usage, dict):
            value = usage.get(token_key, 0)
        else:
            value = getattr(usage, token_key, 0)

        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """Normalize response message object to dict."""
        if message is None:
            return {}
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=False)
        if hasattr(message, "dict"):
            return message.dict()
        if hasattr(message, "__dict__"):
            return {k: v for k, v in message.__dict__.items() if not k.startswith("_")}
        return {}

    def _raise_silent_context_failure(
        self,
        *,
        model: str,
        reason: str,
        file_logger: ProviderLogger,
    ) -> None:
        """Raise a non-retryable context-window style error for silent 200 failures."""
        error_msg = f"iFlow silent context failure detected for {model}: {reason}"
        file_logger.log_error(error_msg)
        lib_logger.warning(error_msg)
        request = httpx.Request("POST", "https://iflow.invalid/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            text=f"context window exceeded: {error_msg}",
        )
        raise httpx.HTTPStatusError(
            f"Context window exceeded: {error_msg}",
            request=request,
            response=response,
        )

    def _validate_final_response(
        self,
        *,
        final_response: litellm.ModelResponse,
        model: str,
        file_logger: ProviderLogger,
    ) -> None:
        """Detect empty/invalid 200 responses that indicate silent context failures."""
        choices = getattr(final_response, "choices", None) or []
        if not choices:
            self._raise_silent_context_failure(
                model=model,
                reason="HTTP 200 response had no choices",
                file_logger=file_logger,
            )

        usage = getattr(final_response, "usage", None)
        prompt_tokens = self._get_usage_token_count(usage, "prompt_tokens")
        completion_tokens = self._get_usage_token_count(usage, "completion_tokens")
        if prompt_tokens > 0 and completion_tokens == 0:
            self._raise_silent_context_failure(
                model=model,
                reason=(
                    "completion_tokens=0 with non-zero prompt_tokens "
                    f"(prompt_tokens={prompt_tokens})"
                ),
                file_logger=file_logger,
            )

        first_choice = choices[0]
        message_obj = (
            first_choice.get("message")
            if isinstance(first_choice, dict)
            else getattr(first_choice, "message", None)
        )
        message = self._message_to_dict(message_obj)

        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        function_call = message.get("function_call")

        has_content = isinstance(content, str) and bool(content.strip())
        has_reasoning = isinstance(reasoning_content, str) and bool(
            reasoning_content.strip()
        )
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        has_function_call = function_call not in (None, {}, [])

        if not (has_content or has_reasoning or has_tool_calls or has_function_call):
            self._raise_silent_context_failure(
                model=model,
                reason="HTTP 200 response completed with empty assistant message",
                file_logger=file_logger,
            )

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        # Create provider logger from transaction context
        file_logger = ProviderLogger(transaction_context)
        request_ids = self._extract_iflow_ids(kwargs)
        api_bases = self.get_api_base_candidates()

        class _NextBaseError(Exception):
            def __init__(self, message: str, cause: Optional[Exception] = None):
                super().__init__(message)
                self.cause = cause

        async def make_request(api_base: str, include_signature: bool = True):
            """Prepares and makes the actual API call."""
            # CRITICAL: get_api_details returns api_key, NOT access_token
            _, api_key = await self.get_api_details(credential_path)

            # Strip provider prefix from model name (e.g., "iflow/Qwen3-Coder-Plus" -> "Qwen3-Coder-Plus")
            model_name = model.split("/")[-1]
            kwargs_with_stripped_model = {**kwargs, "model": model_name}

            # Build clean payload with only supported parameters
            # Pass original kwargs for thinking detection (reasoning_effort, etc.)
            payload = self._build_request_payload(
                model_name, kwargs, **kwargs_with_stripped_model
            )

            headers = self._build_iflow_headers(
                api_key=api_key,
                stream=bool(payload.get("stream")),
                request_ids=request_ids,
                include_signature=include_signature,
            )

            url = f"{api_base.rstrip('/')}/chat/completions"

            # Log request to dedicated file
            file_logger.log_request(payload)
            # lib_logger.debug(f"iFlow Request URL: {url}")

            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(
            response_stream,
            api_base: str,
            attempt: int = 1,
            include_signature: bool = True,
            allow_unsigned_retry: bool = True,
        ):
            """Handles streaming response with retries and context-failure detection."""
            stream_state: Dict[str, Any] = {}
            saw_data_chunk = False

            connection_retry_count = 0
            current_stream = response_stream

            while connection_retry_count <= MAX_CONNECTION_RETRIES:
                try:
                    async with current_stream as response:
                        if response.status_code >= 400:
                            error_text = await response.aread()
                            error_text = (
                                error_text.decode("utf-8")
                                if isinstance(error_text, bytes)
                                else error_text
                            )
                            error_text_lower = error_text.lower()

                            if response.status_code == 401 and attempt == 1:
                                lib_logger.warning(
                                    "iFlow returned 401. Forcing token refresh and retrying once."
                                )
                                await self._refresh_token(credential_path, force=True)
                                retry_stream = await make_request(
                                    api_base, include_signature=include_signature
                                )
                                async for chunk in stream_handler(
                                    retry_stream,
                                    api_base=api_base,
                                    attempt=2,
                                    include_signature=include_signature,
                                    allow_unsigned_retry=allow_unsigned_retry,
                                ):
                                    yield chunk
                                return

                            if (
                                response.status_code == 406
                                and include_signature
                                and allow_unsigned_retry
                            ):
                                lib_logger.warning(
                                    f"iFlow returned 406 on {api_base}, retrying once without signature headers"
                                )
                                retry_stream = await make_request(
                                    api_base, include_signature=False
                                )
                                async for chunk in stream_handler(
                                    retry_stream,
                                    api_base=api_base,
                                    attempt=attempt,
                                    include_signature=False,
                                    allow_unsigned_retry=False,
                                ):
                                    yield chunk
                                return

                            if (
                                response.status_code == 429
                                or "slow_down" in error_text_lower
                            ):
                                raise RateLimitError(
                                    f"iFlow rate limit exceeded: {error_text}",
                                    llm_provider="iflow",
                                    model=model,
                                    response=response,
                                )

                            if any(
                                pattern in error_text_lower
                                for pattern in CONTEXT_WINDOW_ERROR_PATTERNS
                            ):
                                error_msg = f"iFlow context window exceeded: {error_text}"
                                file_logger.log_error(error_msg)
                                lib_logger.warning(
                                    f"iFlow context window error detected: {error_text}"
                                )
                                raise httpx.HTTPStatusError(
                                    f"Context window exceeded: {error_text}",
                                    request=response.request,
                                    response=response,
                                )

                            if self._should_fallback_base(response.status_code):
                                raise _NextBaseError(
                                    f"iFlow HTTP {response.status_code} on {api_base}",
                                    cause=httpx.HTTPStatusError(
                                        f"HTTP {response.status_code}: {error_text}",
                                        request=response.request,
                                        response=response,
                                    ),
                                )

                            if not error_text:
                                content_type = response.headers.get("content-type", "")
                                error_text = (
                                    f"(empty response body, content-type={content_type})"
                                )
                            error_msg = (
                                f"iFlow HTTP {response.status_code} error: {error_text}"
                            )
                            file_logger.log_error(error_msg)
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response,
                            )

                        async for line in response.aiter_lines():
                            file_logger.log_response_chunk(line)

                            if line.startswith("data:"):
                                data_str = line[6:] if line.startswith("data: ") else line[5:]

                                if data_str.strip() == "[DONE]":
                                    break

                                saw_data_chunk = True
                                try:
                                    chunk = json.loads(data_str)
                                    for openai_chunk in self._convert_chunk_to_openai(
                                        chunk, model, stream_state
                                    ):
                                        yield litellm.ModelResponse(**openai_chunk)
                                except json.JSONDecodeError:
                                    lib_logger.warning(
                                        f"Could not decode JSON from iFlow: {line}"
                                    )

                    if not saw_data_chunk:
                        self._raise_silent_context_failure(
                            model=model,
                            reason="stream completed without any data chunks",
                            file_logger=file_logger,
                        )
                    return

                except httpx.HTTPStatusError:
                    raise
                except _NextBaseError:
                    raise
                except CONNECTION_ERROR_TYPES as e:
                    connection_retry_count += 1
                    error_type_name = type(e).__name__

                    if connection_retry_count > MAX_CONNECTION_RETRIES:
                        raise _NextBaseError(
                            f"Network error on iFlow base {api_base}: {error_type_name}: {e}",
                            cause=e,
                        )

                    backoff = RETRY_BACKOFF_BASE * (2 ** (connection_retry_count - 1))
                    lib_logger.warning(
                        f"iFlow connection error ({error_type_name}) on attempt "
                        f"{connection_retry_count}/{MAX_CONNECTION_RETRIES}: {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    current_stream = await make_request(
                        api_base,
                        include_signature=include_signature,
                    )
                except Exception as e:
                    file_logger.log_error(f"Error during iFlow stream processing: {e}")
                    lib_logger.error(
                        f"Error during iFlow stream processing: {e}", exc_info=True
                    )
                    raise

        async def logging_stream_wrapper():
            """Wraps the stream to log the final reassembled response and cache reasoning."""
            openai_chunks = []
            last_fallback_error: Optional[Exception] = None
            stream_completed = False
            try:
                for idx, api_base in enumerate(api_bases):
                    try:
                        async for chunk in stream_handler(
                            await make_request(api_base, include_signature=True),
                            api_base=api_base,
                            include_signature=True,
                            allow_unsigned_retry=True,
                        ):
                            openai_chunks.append(chunk)
                            yield chunk
                        stream_completed = True
                        return
                    except _NextBaseError as e:
                        if openai_chunks:
                            raise e.cause or e

                        last_fallback_error = e.cause or e
                        if idx + 1 < len(api_bases):
                            lib_logger.warning(
                                f"iFlow base {api_base} failed ({e}); trying fallback base"
                            )
                            continue
                        raise last_fallback_error

                if last_fallback_error:
                    raise last_fallback_error
            finally:
                if stream_completed:
                    if not openai_chunks:
                        self._raise_silent_context_failure(
                            model=model,
                            reason="HTTP 200 stream ended without any data chunks",
                            file_logger=file_logger,
                        )

                    final_response = self._stream_to_completion_response(openai_chunks)
                    self._validate_final_response(
                        final_response=final_response,
                        model=model,
                        file_logger=file_logger,
                    )
                    file_logger.log_final_response(final_response.dict())

                    # Store reasoning_content from the response for future multi-turn conversations
                    # This enables reasoning preservation in subsequent requests
                    model_name = model.split("/")[-1]
                    messages = kwargs.get("messages", [])
                    if messages:
                        conv_sig = self._get_conversation_signature(messages)
                        # Get the assistant message from the final response
                        if final_response.choices and len(final_response.choices) > 0:
                            choice = final_response.choices[0]
                            message = getattr(choice, "message", None)
                            if message:
                                # Convert to dict if needed
                                if hasattr(message, "model_dump"):
                                    msg_dict = message.model_dump()
                                elif hasattr(message, "__dict__"):
                                    msg_dict = {
                                        k: v
                                        for k, v in message.__dict__.items()
                                        if not k.startswith("_")
                                    }
                                else:
                                    msg_dict = (
                                        dict(message)
                                        if isinstance(message, dict)
                                        else {}
                                    )
                                self._store_reasoning_content(msg_dict, conv_sig)

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:

            async def non_stream_wrapper():
                chunks = [chunk async for chunk in logging_stream_wrapper()]
                return self._stream_to_completion_response(chunks)

            return await non_stream_wrapper()
