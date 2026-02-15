# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/iflow_provider.py

import copy
import gzip
import hmac
import hashlib
import json
import re
import time
import os
import httpx
import logging
import asyncio
import mimetypes
import base64
from typing import Union, AsyncGenerator, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import unquote
from .provider_interface import ProviderInterface
from .iflow_auth_base import IFlowAuthBase
from .provider_cache import ProviderCache
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
import litellm
from litellm.exceptions import RateLimitError, AuthenticationError
from pathlib import Path

from ..core.errors import StreamedAPIError
import uuid
from datetime import datetime

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# CONSTANTS (Avoid magic strings, enable configuration)
# =============================================================================

# iFlow API payload field names
IFLOW_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
IFLOW_ENABLE_THINKING = "enable_thinking"
IFLOW_CLEAR_THINKING = "clear_thinking"
IFLOW_REASONING_SPLIT = "reasoning_split"
IFLOW_REASONING_EFFORT = "reasoning_effort"
IFLOW_REASONING_CONTENT = "reasoning_content"

# Default vision model for two-stage processing (configurable via env var)
FORCED_VISION_MODEL = os.environ.get("IFLOW_FORCED_VISION_MODEL", "qwen3-vl-plus")


# =============================================================================
# THINKING MODE TYPES AND ENUMS (Based on CLIProxyAPI Go implementation)
# =============================================================================


class ThinkingMode(Enum):
    """Thinking configuration mode."""

    BUDGET = "budget"  # Numeric token budget
    LEVEL = "level"  # Discrete level (high, medium, low)
    NONE = "none"  # Disabled
    AUTO = "auto"  # Automatic


class ThinkingLevel(Enum):
    """Discrete thinking levels."""

    NONE = "none"
    AUTO = "auto"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class ThinkingConfig:
    """Unified thinking configuration."""

    mode: ThinkingMode
    budget: int = 0
    level: Optional[ThinkingLevel] = None


@dataclass
class SuffixResult:
    """Result of parsing model name suffix."""

    model_name: str
    has_suffix: bool
    raw_suffix: str = ""


# =============================================================================
# SUFFIX PARSING (Based on CLIProxyAPI Go implementation)
# =============================================================================


def parse_suffix(model: str) -> SuffixResult:
    """
    Extract thinking suffix from model name.

    Examples:
        "glm-4.7(8192)" -> model_name="glm-4.7", raw_suffix="8192"
        "glm-4.7(high)" -> model_name="glm-4.7", raw_suffix="high"
        "glm-4.7" -> model_name="glm-4.7", has_suffix=False
    """
    # Find the last opening parenthesis
    last_open = model.rfind("(")
    if last_open == -1:
        return SuffixResult(model_name=model, has_suffix=False)

    # Check if the string ends with a closing parenthesis
    if not model.endswith(")"):
        return SuffixResult(model_name=model, has_suffix=False)

    # Extract components
    model_name = model[:last_open]
    raw_suffix = model[last_open + 1 : -1]

    return SuffixResult(model_name=model_name, has_suffix=True, raw_suffix=raw_suffix)


def parse_special_suffix(raw_suffix: str) -> Tuple[ThinkingMode, bool]:
    """
    Parse special suffix values: none, auto, -1.

    Returns (mode, ok) where ok indicates if parsing succeeded.
    """
    if not raw_suffix:
        return ThinkingMode.BUDGET, False

    lower = raw_suffix.strip().lower()
    if lower == "none":
        return ThinkingMode.NONE, True
    if lower in ("auto", "-1"):
        return ThinkingMode.AUTO, True
    return ThinkingMode.BUDGET, False


def parse_level_suffix(raw_suffix: str) -> Tuple[ThinkingLevel, bool]:
    """
    Parse level suffix: minimal, low, medium, high, xhigh.

    Returns (level, ok) where ok indicates if parsing succeeded.
    """
    if not raw_suffix:
        return ThinkingLevel.NONE, False

    lower = raw_suffix.strip().lower()
    try:
        return ThinkingLevel(lower), True
    except ValueError:
        return ThinkingLevel.NONE, False


def parse_numeric_suffix(raw_suffix: str) -> Tuple[int, bool]:
    """
    Parse numeric suffix as budget value.

    Only non-negative integers are valid.
    Returns (budget, ok) where ok indicates if parsing succeeded.
    """
    if not raw_suffix:
        return 0, False

    try:
        value = int(raw_suffix)
        if value < 0:
            return 0, False
        return value, True
    except ValueError:
        return 0, False


def parse_suffix_to_config(raw_suffix: str) -> Optional[ThinkingConfig]:
    """
    Convert raw suffix to ThinkingConfig.

    Priority:
        1. Special values: "none", "auto", "-1"
        2. Level names: "minimal", "low", "medium", "high", "xhigh"
        3. Numeric values: positive integers
    """
    if not raw_suffix:
        return None

    # 1. Try special values first
    mode, ok = parse_special_suffix(raw_suffix)
    if ok:
        if mode == ThinkingMode.NONE:
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)
        if mode == ThinkingMode.AUTO:
            return ThinkingConfig(mode=ThinkingMode.AUTO, budget=-1)

    # 2. Try level parsing
    level, ok = parse_level_suffix(raw_suffix)
    if ok:
        return ThinkingConfig(mode=ThinkingMode.LEVEL, level=level)

    # 3. Try numeric parsing
    budget, ok = parse_numeric_suffix(raw_suffix)
    if ok:
        if budget == 0:
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)
        return ThinkingConfig(mode=ThinkingMode.BUDGET, budget=budget)

    return None


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF_FACTOR = 2.0  # exponential backoff multiplier


# =============================================================================
# VISION/MULTIMODAL CONFIGURATION
# =============================================================================

# Maximum size for local image files (10MB)
MAX_LOCAL_IMAGE_BYTES = 10 * 1024 * 1024

# Note: FORCED_VISION_MODEL is defined above with env var support

# Known multimodal models that support images natively
KNOWN_MULTIMODAL_MODELS = {
    "qwen3-vl-plus",
    "tstars2.0",
}

# Model series that need two-stage vision processing (GLM/MiniMax don't support images directly)
FORCE_VISION_MODEL_SERIES_PREFIXES = ("glm", "minimax")


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: ~4 chars per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_content_tokens(content: Any) -> int:
    """Estimate token count for message content (string or list of parts)."""
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += estimate_text_tokens(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                total += estimate_text_tokens(str(item.get("text", "")))
            elif item_type == "reasoning":
                total += estimate_text_tokens(str(item.get("text", "")))
            elif item_type == "tool_calls":
                tool_calls = item.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        if isinstance(func, dict):
                            total += estimate_text_tokens(
                                str(func.get("arguments", ""))
                            )
            elif item_type == "tool_use":
                try:
                    total += estimate_text_tokens(
                        json.dumps(item.get("input", {}), ensure_ascii=False)
                    )
                except (TypeError, ValueError):
                    # Invalid input for JSON serialization
                    pass
            elif item_type == "tool_result":
                total += estimate_content_tokens(item.get("content", ""))
        return total
    return 0


def estimate_openai_prompt_tokens(body: Dict[str, Any]) -> int:
    """Estimate prompt tokens from OpenAI-format request body."""
    total = 0
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += estimate_content_tokens(msg.get("content", ""))
        if "reasoning_content" in msg:
            total += estimate_content_tokens(msg.get("reasoning_content"))
    return total


def estimate_openai_completion_tokens(response_data: Dict[str, Any]) -> int:
    """Estimate completion tokens from OpenAI-format response."""
    choices = response_data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return 0

    choice = choices[0]
    if not isinstance(choice, dict):
        return 0

    total = 0
    message = choice.get("message", {})
    if isinstance(message, dict):
        total += estimate_content_tokens(message.get("content", ""))
        if "reasoning_content" in message:
            total += estimate_content_tokens(message.get("reasoning_content"))

    delta = choice.get("delta", {})
    if isinstance(delta, dict):
        total += estimate_content_tokens(delta.get("content", ""))
        if "reasoning_content" in delta:
            total += estimate_content_tokens(delta.get("reasoning_content"))

    return total


# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================


def normalize_base64_data(data: str) -> str:
    """Normalize base64 data by removing whitespace."""
    if not data:
        return ""
    return "".join(data.split())


def build_data_url(media_type: str, data: str) -> str:
    """Build a data URL from media type and base64 data."""
    if not data:
        return ""
    if data.startswith("data:"):
        return data
    media_type = media_type or "image/png"
    data = normalize_base64_data(data)
    if not data:
        return ""
    return f"data:{media_type};base64,{data}"


def to_local_path(url: str) -> str:
    """Convert file:// URL to local path."""
    if not url:
        return ""
    value = url.strip()
    if value.startswith("file://"):
        path = unquote(value[7:])
        if path.startswith("/"):
            path = path.lstrip("/")
        path = path.replace("/", os.sep)
        # UNC path support: check if path is not a local drive path
        if not re.match(r"^[A-Za-z]:\\", path) and not path.startswith("\\"):
            # Path looks like a network path or relative path
            if "\\" not in path and "/" not in path:
                # Simple filename, return as-is
                return path
            # Reconstruct as UNC path
            return "\\\\" + path.lstrip("\\")
        return path
    if re.match(r"^[A-Za-z]:\\", value) or re.match(r"^[A-Za-z]:/", value):
        return unquote(value).replace("/", os.sep)
    return ""


def load_local_image_as_data_url(
    url: str, max_bytes: int = MAX_LOCAL_IMAGE_BYTES
) -> str:
    """Load a local image file and return as data URL."""
    path = to_local_path(url)
    if not path:
        return ""
    if not os.path.exists(path):
        lib_logger.warning(f"[iFlow] Local image path does not exist: {path}")
        return ""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            lib_logger.warning(
                f"[iFlow] Local image too large, skipping: {path} ({size} bytes)"
            )
            return ""
        with open(path, "rb") as f:
            data = f.read()
        media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
    except (OSError, IOError, PermissionError) as e:
        lib_logger.warning(f"[iFlow] Failed to read local image: {path} ({e})")
        return ""


def normalize_image_url(url: str, allow_local: bool = False) -> str:
    """Normalize an image URL to a standard format."""
    if not url:
        return ""
    value = url.strip()
    if (
        value.startswith("data:")
        or value.startswith("http://")
        or value.startswith("https://")
    ):
        return value
    if allow_local:
        local_data = load_local_image_as_data_url(value)
        if local_data:
            return local_data
    return value


def extract_image_url_from_part(part: Dict[str, Any]) -> str:
    """Extract image URL from various content part formats."""
    if not isinstance(part, dict):
        return ""

    image_url_field = part.get("image_url")
    if isinstance(image_url_field, dict):
        url = image_url_field.get("url", "")
        if url:
            return url
    elif isinstance(image_url_field, str):
        return image_url_field

    url_field = part.get("url")
    if isinstance(url_field, str) and url_field:
        return url_field

    source = part.get("source")
    if isinstance(source, dict):
        source_type = source.get("type", "")
        if source_type == "url":
            return source.get("url", "")
        if source_type == "base64":
            media_type = (
                source.get("media_type") or part.get("media_type") or "image/png"
            )
            return build_data_url(media_type, source.get("data", ""))

    return ""


def message_has_image(message: Dict[str, Any]) -> bool:
    """Check if a message contains image content."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type in ("image", "image_url", "input_image"):
            return True
        if "image_url" in part:
            return True
        source = part.get("source")
        if isinstance(source, dict) and source.get("type") in ("base64", "url"):
            return True
    return False


def request_has_images(body: Dict[str, Any]) -> bool:
    """Check if a request body contains any images."""
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return False
    return any(message_has_image(msg) for msg in messages)


def is_image_part(part: Dict[str, Any]) -> bool:
    """Check if a content part is an image."""
    if not isinstance(part, dict):
        return False
    part_type = part.get("type", "")
    if part_type in ("image", "image_url", "input_image"):
        return True
    if "image_url" in part:
        return True
    source = part.get("source")
    return isinstance(source, dict) and source.get("type") in ("base64", "url")


def strip_images_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove image content from messages, keeping text only."""
    sanitized: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        msg = copy.deepcopy(message)
        content = msg.get("content")
        if isinstance(content, list):
            filtered: List[Any] = []
            for part in content:
                if isinstance(part, dict):
                    if is_image_part(part):
                        continue
                    if part.get("type") == "input_text":
                        filtered.append({"type": "text", "text": part.get("text", "")})
                        continue
                filtered.append(part)
            msg["content"] = filtered
        sanitized.append(msg)
    return sanitized


def is_vision_model(model: str) -> bool:
    """Check if a model supports vision/images natively."""
    if not model:
        return False
    model_lower = model.lower()
    return (
        model_lower in KNOWN_MULTIMODAL_MODELS
        or "vl" in model_lower
        or "vision" in model_lower
        or "4v" in model_lower
        or "multimodal" in model_lower
    )


def should_force_vision_for_series(model: str) -> bool:
    """Check if model belongs to a series that needs two-stage vision processing."""
    if not model:
        return False
    model_lower = model.lower()
    return any(
        model_lower.startswith(prefix) for prefix in FORCE_VISION_MODEL_SERIES_PREFIXES
    )


def looks_like_image_capability_error(exc: Exception) -> bool:
    """Check if an error indicates the model doesn't support images."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False

    status = exc.response.status_code
    if status not in (400, 415, 422):
        return False

    try:
        text = (exc.response.text or "").lower()
    except (AttributeError, UnicodeDecodeError):
        return False

    if not text:
        return False

    image_tokens = [
        "image",
        "image_url",
        "input_image",
        "vision",
        "multimodal",
        "图片",
        "图像",
        "视觉",
        "多模态",
    ]
    unsupported_tokens = [
        "not support",
        "unsupported",
        "invalid",
        "must be text",
        "only text",
        "不支持",
        "无效",
        "仅支持文本",
    ]
    return any(token in text for token in image_tokens) and any(
        token in text for token in unsupported_tokens
    )


def extract_text_from_result(result: Dict[str, Any]) -> str:
    """Extract text content from an OpenAI-format response."""
    choices = result.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message", {})
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text = part.get("text", "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


# Model list can be expanded as iFlow supports more models
HARDCODED_MODELS = [
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "iflow-rome-30ba3b",
    "kimi-k2",
    "kimi-k2-0905",
    "kimi-k2-thinking",  # Seems to not work, but should
    "kimi-k2.5",  # Seems to not work, but should
    "minimax-m2",
    "minimax-m2.1",
    "minimax-m2.5",
    "qwen3-32b",
    "qwen3-235b",
    "qwen3-235b-a22b-instruct",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-coder-plus",
    "qwen3-max",
    "qwen3-max-preview",
    "qwen3-vl-plus",
    "deepseek-v3.2-reasoner",
    "deepseek-v3.2-chat",
    "deepseek-v3.2",  # seems to not work, but should. Use above variants instead
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-r1",
    "tstars2.0",
]

# OpenAI-compatible parameters supported by iFlow API
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "response_format",
}

IFLOW_USER_AGENT = "iFlow-Cli"
IFLOW_HEADER_SESSION_ID = "session-id"
IFLOW_HEADER_TIMESTAMP = "x-iflow-timestamp"
IFLOW_HEADER_SIGNATURE = "x-iflow-signature"

# =============================================================================
# THINKING MODE CONFIGURATION
# =============================================================================
# Models using chat_template_kwargs.enable_thinking (boolean toggle)
# Based on Go implementation: internal/thinking/provider/iflow/apply.go
ENABLE_THINKING_MODELS = {
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "qwen3-max-preview",
    "qwen3-32b",
    "deepseek-v3.2",
    "deepseek-v3.1",
}

# GLM models need additional clear_thinking=false when thinking is enabled
GLM_MODELS = {"glm-4.6", "glm-4.7", "glm-5"}

# Models using reasoning_split (boolean) instead of enable_thinking
REASONING_SPLIT_MODELS = {"minimax-m2", "minimax-m2.1", "minimax-m2.5"}

# Models that benefit from reasoning_content preservation in message history
# (for multi-turn conversations)
REASONING_PRESERVATION_MODELS_PREFIXES = ("glm-4", "glm-5", "minimax-m2", "tstars")

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

            api_base, api_key = await self.get_api_details(credential)
            models_url = f"{api_base.rstrip('/')}/models"

            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
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

        except Exception as e:
            # Silently ignore dynamic discovery errors - non-critical feature
            lib_logger.debug(f"Dynamic model discovery failed for iflow: {e}")

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
    # THINKING MODE SUPPORT (Enhanced with Suffix Parsing)
    # =========================================================================

    def _extract_iflow_config_from_payload(
        self, payload: Dict[str, Any]
    ) -> Optional[ThinkingConfig]:
        """
        Extract thinking config from iFlow-format request body.

        iFlow formats:
            - GLM: chat_template_kwargs.enable_thinking (boolean)
            - MiniMax: reasoning_split (boolean)
        """
        # GLM format: chat_template_kwargs.enable_thinking
        ctk = payload.get(IFLOW_CHAT_TEMPLATE_KWARGS, {})
        if isinstance(ctk, dict) and IFLOW_ENABLE_THINKING in ctk:
            if ctk[IFLOW_ENABLE_THINKING]:
                return ThinkingConfig(mode=ThinkingMode.BUDGET, budget=1)
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)

        # MiniMax format: reasoning_split
        if IFLOW_REASONING_SPLIT in payload:
            if payload[IFLOW_REASONING_SPLIT]:
                return ThinkingConfig(mode=ThinkingMode.BUDGET, budget=1)
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)

        return None

    def _extract_openai_config_from_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Optional[ThinkingConfig]:
        """
        Extract thinking config from OpenAI-format request.

        OpenAI format: reasoning_effort = "none" | "low" | "medium" | "high"
        """
        effort = kwargs.get(IFLOW_REASONING_EFFORT)
        if not effort:
            return None

        effort_lower = str(effort).strip().lower()

        if effort_lower == "none":
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)

        try:
            level = ThinkingLevel(effort_lower)
            return ThinkingConfig(mode=ThinkingMode.LEVEL, level=level)
        except ValueError:
            return None

    def _config_to_boolean(self, config: ThinkingConfig) -> bool:
        """
        Convert ThinkingConfig to boolean for iFlow models.

        Conversion rules:
            - ModeNone: false
            - ModeAuto: true
            - ModeBudget + Budget=0: false
            - ModeBudget + Budget>0: true
            - ModeLevel + Level=none: false
            - ModeLevel + any other level: true
        """
        if config.mode == ThinkingMode.NONE:
            return False
        if config.mode == ThinkingMode.AUTO:
            return True
        if config.mode == ThinkingMode.BUDGET:
            return config.budget > 0
        if config.mode == ThinkingMode.LEVEL:
            return config.level != ThinkingLevel.NONE
        return True

    def _get_thinking_config(
        self, model_name: str, kwargs: Dict[str, Any], payload: Dict[str, Any]
    ) -> Tuple[str, Optional[ThinkingConfig]]:
        """
        Get thinking configuration with suffix priority.

        Priority:
            1. Suffix from model name (e.g., "glm-4.7(8192)")
            2. iFlow format from payload (chat_template_kwargs.enable_thinking)
            3. OpenAI format from kwargs (reasoning_effort)

        Returns:
            (base_model_name, thinking_config or None)
        """
        # Parse suffix from model name
        suffix_result = parse_suffix(model_name)
        base_model = suffix_result.model_name

        # Get config: suffix priority over body
        config = None
        if suffix_result.has_suffix:
            config = parse_suffix_to_config(suffix_result.raw_suffix)
            if config:
                lib_logger.debug(
                    f"iFlow: Thinking config from suffix '{suffix_result.raw_suffix}' "
                    f"-> mode={config.mode.value}, budget={config.budget}, level={config.level}"
                )

        if config is None:
            # Try iFlow format from payload first
            config = self._extract_iflow_config_from_payload(payload)

        if config is None:
            # Fall back to OpenAI format from kwargs
            config = self._extract_openai_config_from_kwargs(kwargs)

        return base_model, config

    def _apply_thinking_config(
        self, payload: Dict[str, Any], model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply thinking configuration for supported iFlow models.

        Enhanced to support:
        - Suffix parsing (e.g., "glm-4.7(8192)" or "glm-4.7(high)")
        - Level-based thinking (minimal, low, medium, high, xhigh)
        - Auto mode (-1 / auto)

        Logic matches Go implementation (internal/thinking/provider/iflow/apply.go):
        - GLM models: enable_thinking + clear_thinking=false (when enabled)
        - Qwen/DeepSeek: enable_thinking only
        - MiniMax: reasoning_split

        Args:
            payload: The request payload to modify
            model_name: Model name (may include suffix)
            kwargs: Original request kwargs containing reasoning_effort, etc.

        Returns:
            Modified payload with thinking config applied
        """
        # Get base model and thinking config (with suffix priority)
        base_model, config = self._get_thinking_config(model_name, kwargs, payload)
        model_lower = base_model.lower()

        # Update model in payload to base model (strip suffix)
        payload["model"] = base_model

        # No config found - check if we need to strip thinking params
        if config is None:
            return payload

        # Remove OpenAI format fields (will be replaced with iFlow format)
        payload.pop(IFLOW_REASONING_EFFORT, None)
        payload.pop("thinking", None)

        # Check model type
        is_glm = any(model_lower.startswith(prefix) for prefix in ("glm-4", "glm-5"))
        is_enable_thinking_model = model_lower in ENABLE_THINKING_MODELS or is_glm
        is_minimax = any(
            model_lower.startswith(prefix) for prefix in ("minimax-m2", "minimax-m1")
        )

        # Determine if thinking should be enabled
        enable_thinking = self._config_to_boolean(config)

        if is_enable_thinking_model:
            # Models using chat_template_kwargs.enable_thinking
            if IFLOW_CHAT_TEMPLATE_KWARGS not in payload:
                payload[IFLOW_CHAT_TEMPLATE_KWARGS] = {}
            payload[IFLOW_CHAT_TEMPLATE_KWARGS][IFLOW_ENABLE_THINKING] = enable_thinking

            # GLM models: strip clear_thinking first, then set to false when enabled
            if is_glm:
                payload[IFLOW_CHAT_TEMPLATE_KWARGS].pop(IFLOW_CLEAR_THINKING, None)
                if enable_thinking:
                    payload[IFLOW_CHAT_TEMPLATE_KWARGS][IFLOW_CLEAR_THINKING] = False

            lib_logger.info(
                f"iFlow: Applied enable_thinking={enable_thinking} for {base_model} "
                f"(mode={config.mode.value})"
            )
        elif is_minimax:
            # MiniMax models use reasoning_split
            payload[IFLOW_REASONING_SPLIT] = enable_thinking
            lib_logger.info(
                f"iFlow: Applied reasoning_split={enable_thinking} for {base_model} "
                f"(mode={config.mode.value})"
            )
        else:
            # Model doesn't support thinking - strip config
            if IFLOW_CHAT_TEMPLATE_KWARGS in payload:
                payload[IFLOW_CHAT_TEMPLATE_KWARGS].pop(IFLOW_ENABLE_THINKING, None)
                payload[IFLOW_CHAT_TEMPLATE_KWARGS].pop(IFLOW_CLEAR_THINKING, None)
                if not payload[IFLOW_CHAT_TEMPLATE_KWARGS]:
                    payload.pop(IFLOW_CHAT_TEMPLATE_KWARGS)
            payload.pop(IFLOW_REASONING_SPLIT, None)
            lib_logger.debug(
                f"iFlow: Model {base_model} doesn't support thinking, stripped config"
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

    # =========================================================================
    # VISION/MULTIMODAL SUPPORT
    # =========================================================================

    def _normalize_openai_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize image content blocks to OpenAI image_url format.

        Handles various input formats:
        - Anthropic-style: {"type": "image", "source": {"type": "base64", ...}}
        - OpenAI-style: {"type": "image_url", "image_url": {"url": "..."}}
        """
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            normalized = []
            changed = False

            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    # Convert input_text to text
                    if part_type == "input_text" and "text" in part:
                        normalized.append(
                            {"type": "text", "text": part.get("text", "")}
                        )
                        changed = True
                        continue
                    # Check for image-like parts
                    is_image_like = (
                        part_type in ("image", "image_url", "input_image")
                        or "image_url" in part
                        or "source" in part
                    )
                    if is_image_like:
                        url = extract_image_url_from_part(part)
                        if url:
                            # Build normalized image_url part
                            image_url = {"url": url}
                            detail = None
                            image_url_field = part.get("image_url")
                            if isinstance(image_url_field, dict):
                                detail = image_url_field.get("detail")
                            if detail is None:
                                detail = part.get("detail")
                            if detail:
                                image_url["detail"] = detail
                            normalized.append(
                                {"type": "image_url", "image_url": image_url}
                            )
                            changed = True
                            continue

                normalized.append(part)

            if changed:
                msg["content"] = normalized

        return messages

    async def _generate_vision_summary(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        api_base: str,
        body: Dict[str, Any],
        model: str,
    ) -> str:
        """
        Generate a text summary of images using a vision model.

        This is used for two-stage vision processing where GLM/MiniMax models
        don't support images directly, so we first use a vision model to
        describe the images, then pass that description to the text model.

        Args:
            client: HTTP client to use
            api_key: API key for iFlow
            api_base: Base URL for iFlow API
            body: Original request body with images
            model: Target model name (for context in prompt)

        Returns:
            Text summary of the images
        """
        vision_body = copy.deepcopy(body)
        vision_body["stream"] = False
        vision_body.pop("stream_options", None)
        vision_body.pop("tools", None)
        vision_body.pop("tool_choice", None)
        vision_body.pop("response_format", None)
        vision_body.pop("thinking", None)
        vision_body.pop("reasoning_effort", None)
        vision_body["max_tokens"] = 1024

        messages = vision_body.get("messages", [])
        if not isinstance(messages, list):
            return ""

        # Add instruction for vision model
        instruction = (
            f"You are an image analysis assistant. The current model is {model}, "
            "which does not process images. Please analyze the user-uploaded images "
            "and output a structured summary for the text model to continue reasoning. "
            "Must include: 1) Main subject and scene; 2) Text in the image; "
            "3) Key details relevant to the user's question. "
            "Do not say you cannot see the image. Be concise."
        )
        vision_body["messages"] = [
            {"role": "system", "content": instruction}
        ] + messages
        vision_body["model"] = FORCED_VISION_MODEL

        # Build headers
        headers = self._build_iflow_headers(api_key, stream=False)
        url = f"{api_base.rstrip('/')}/chat/completions"

        try:
            response = await client.post(
                url,
                headers=headers,
                json=vision_body,
                timeout=TimeoutConfig.non_streaming(),
            )
            response.raise_for_status()

            # Handle potential gzip response
            content = response.content
            if response.headers.get("content-encoding") == "gzip":
                content = gzip.decompress(content)

            result = json.loads(content)
            return extract_text_from_result(result)

        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            lib_logger.warning(f"[iFlow] Vision summary generation failed: {e}")
            return ""
        except (json.JSONDecodeError, gzip.BadGzipFile) as e:
            lib_logger.warning(f"[iFlow] Vision summary parsing failed: {e}")
            return ""

    def _build_two_stage_main_body(
        self, body: Dict[str, Any], vision_summary: str
    ) -> Dict[str, Any]:
        """
        Build the main request body for two-stage vision processing.

        After vision model processes images, we build a new request with
        the vision summary as a system message, and images stripped from
        the original messages.
        """
        main_body = copy.deepcopy(body)
        messages = main_body.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        # Strip images from messages
        sanitized_messages = strip_images_from_messages(messages)

        # Add vision summary as bridge system message
        bridge_system_msg = {
            "role": "system",
            "content": (
                "Below is the vision model's analysis of the user's images. "
                "Use this as the factual image content to continue answering.\n\n"
                f"{vision_summary}"
            ),
        }

        main_body["messages"] = [bridge_system_msg] + sanitized_messages
        return main_body

    def _should_fallback_to_forced_vision(
        self, exc: Exception, model: str, has_images: bool
    ) -> bool:
        """
        Check if we should fall back to forced vision model.

        This happens when:
        1. Request has images
        2. Model is GLM/MiniMax (needs two-stage vision)
        3. Error indicates model doesn't support images
        """
        if not has_images:
            return False
        if not model:
            return False
        if not should_force_vision_for_series(model):
            return False
        if model.lower() == FORCED_VISION_MODEL.lower():
            return False
        if is_vision_model(model):
            return False
        if not is_vision_model(FORCED_VISION_MODEL):
            return False
        return looks_like_image_capability_error(exc)

    # =========================================================================
    # GZIP DECOMPRESSION
    # =========================================================================

    def _decompress_if_gzip(self, content: bytes, headers: Dict[str, str]) -> bytes:
        """
        Decompress content if it's gzip-encoded.

        Checks both Content-Encoding header and magic bytes for detection.
        """
        # Check Content-Encoding header first
        if headers.get("content-encoding", "").lower() == "gzip":
            try:
                decompressed = gzip.decompress(content)
                lib_logger.debug(
                    f"[iFlow] Decompressed gzip response ({len(content)} -> {len(decompressed)} bytes)"
                )
                return decompressed
            except gzip.BadGzipFile as e:
                lib_logger.warning(
                    f"[iFlow] Gzip decompression failed for Content-Encoding header: {e}"
                )
                return content

        # Check for gzip magic bytes (0x1f 0x8b)
        if len(content) >= 2 and content[0] == 0x1F and content[1] == 0x8B:
            try:
                decompressed = gzip.decompress(content)
                lib_logger.debug(
                    f"[iFlow] Decompressed gzip response by magic bytes ({len(content)} -> {len(decompressed)} bytes)"
                )
                return decompressed
            except gzip.BadGzipFile as e:
                lib_logger.warning(
                    f"[iFlow] Gzip decompression by magic bytes failed, "
                    f"returning raw content: {e}"
                )

        return content

    # =========================================================================
    # RETRY LOGIC
    # =========================================================================

    async def _make_request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_body: Dict[str, Any],
        max_retries: int = MAX_RETRIES,
    ) -> httpx.Response:
        """
        Make an HTTP request with automatic retry for transient failures.

        Retry conditions:
        - 5xx server errors
        - 429 rate limits (with Retry-After header)
        - Network errors

        Does NOT retry:
        - 4xx client errors (except 429)
        - Authentication errors
        """
        if max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {max_retries}")
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                if method.upper() == "POST":
                    response = await client.post(
                        url,
                        headers=headers,
                        json=json_body,
                        timeout=TimeoutConfig.streaming(),
                    )
                else:
                    response = await client.get(
                        url,
                        headers=headers,
                        timeout=TimeoutConfig.non_streaming(),
                    )

                # Check for retryable status codes
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                if response.status_code == 429:
                    # Parse Retry-After header (may be seconds or date string)
                    retry_after_str = response.headers.get("Retry-After", "5")
                    try:
                        retry_after = int(retry_after_str)
                    except ValueError:
                        # Header might be a date string; use default
                        retry_after = 5
                        lib_logger.debug(
                            f"[iFlow] Non-numeric Retry-After header: {retry_after_str}, using default {retry_after}s"
                        )
                    if attempt < max_retries - 1:
                        lib_logger.warning(
                            f"[iFlow] Rate limited, waiting {retry_after}s before retry "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    raise httpx.HTTPStatusError(
                        f"Rate limited: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0

                if 500 <= status_code < 600 and attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (RETRY_BACKOFF_FACTOR**attempt)
                    lib_logger.warning(
                        f"[iFlow] Server error {status_code}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                raise

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (RETRY_BACKOFF_FACTOR**attempt)
                    lib_logger.warning(
                        f"[iFlow] Network error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

        # Unreachable: loop always returns or raises before exhausting
        # This is a defensive assertion for static analysis
        assert False, "Unreachable: retry loop should always return or raise"

    def _build_request_payload(
        self, model_name: str, full_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Builds a clean request payload with only supported parameters.
        Also applies thinking mode, image normalization, and reasoning content preservation.

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

        # Ensure max_tokens is at least 1024 as recommended by iflow2api logic
        if "max_tokens" in payload:
            payload["max_tokens"] = max(payload["max_tokens"], 1024)
        else:
            # Default to 4096 if not specified
            payload["max_tokens"] = 4096

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

        # Normalize image content blocks to OpenAI format
        if "messages" in payload:
            payload["messages"] = self._normalize_openai_messages(payload["messages"])

        # Inject cached reasoning_content into messages for multi-turn conversations
        if "messages" in payload:
            payload["messages"] = self._inject_reasoning_content(
                payload["messages"], model_name
            )

        # Apply thinking mode configuration based on reasoning_effort and suffix
        payload = self._apply_thinking_config(payload, model_name, full_kwargs)

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

    def _build_iflow_headers(self, api_key: str, stream: bool) -> Dict[str, str]:
        """Build iFlow request headers, including signed auth headers."""
        session_id = f"session-{uuid.uuid4()}"
        timestamp_ms = int(time.time() * 1000)
        signature = self._create_iflow_signature(
            IFLOW_USER_AGENT, session_id, timestamp_ms, api_key
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,  # Dual auth for better compatibility
            "Content-Type": "application/json",
            "User-Agent": IFLOW_USER_AGENT,
            IFLOW_HEADER_SESSION_ID: session_id,
            IFLOW_HEADER_TIMESTAMP: str(timestamp_ms),
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if signature:
            headers[IFLOW_HEADER_SIGNATURE] = signature

        return headers

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
                choice_copy = dict(choice) if isinstance(choice, dict) else {}
                delta = choice_copy.get("delta")
                if not isinstance(delta, dict):
                    delta = {}

                # Track tool_calls presence
                if delta.get("tool_calls"):
                    stream_state["has_tool_calls"] = True

                # Track reasoning_content presence (for logging)
                reasoning_content = delta.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
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
            usage_dict = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0)
                if usage_data
                else 0,
                "completion_tokens": usage_data.get("completion_tokens", 0)
                if usage_data
                else 0,
                "total_tokens": usage_data.get("total_tokens", 0) if usage_data else 0,
            }

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
            usage_dict = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0)
                if usage_data
                else 0,
                "completion_tokens": usage_data.get("completion_tokens", 0)
                if usage_data
                else 0,
                "total_tokens": usage_data.get("total_tokens", 0) if usage_data else 0,
            }
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
        self,
        chunks: List[litellm.ModelResponse],
        request_body: Optional[Dict[str, Any]] = None,
    ) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.

        Key improvements:
        - Determines finish_reason based on accumulated state (tool_calls vs stop)
        - Properly initializes tool_calls with type field
        - Handles usage data extraction from chunks
        - Estimates token usage when not provided by API
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        # Initialize the final response structure
        final_message: Dict[str, Any] = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        chunk_finish_reason = (
            None  # Track finish_reason from chunks (but we'll override)
        )

        # Get the first chunk for basic response metadata
        first_chunk = chunks[0]

        # Process each chunk to aggregate content
        for chunk in chunks:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue

            choice = choices[0]
            # Handle both dict and object access patterns for choice.delta
            delta = getattr(choice, "delta", None)
            if delta is None and hasattr(choice, "get"):
                delta = choice.get("delta", {})

            if delta is None:
                delta = {}

            # Convert delta to dict if it's an object to silence LSP and handle consistently
            if hasattr(delta, "model_dump"):
                delta_dict = delta.model_dump(exclude_none=True)
            elif hasattr(delta, "__dict__") and not isinstance(delta, dict):
                delta_dict = {
                    k: v
                    for k, v in delta.__dict__.items()
                    if not k.startswith("_") and v is not None
                }
            else:
                delta_dict = dict(delta) if isinstance(delta, dict) else {}

            choice_finish = getattr(choice, "finish_reason", None)
            if choice_finish is None and hasattr(choice, "get"):
                choice_finish = choice.get("finish_reason")

            # Aggregate content
            content = delta_dict.get("content")
            if content is not None:
                if "content" not in final_message:
                    final_message["content"] = ""
                final_message["content"] += content

            # Aggregate reasoning content (if supported by iFlow)
            reasoning = delta_dict.get("reasoning_content")
            if reasoning is not None:
                if "reasoning_content" not in final_message:
                    final_message["reasoning_content"] = ""
                final_message["reasoning_content"] += reasoning

            # Aggregate tool calls with proper initialization
            tool_calls = delta_dict.get("tool_calls")
            if tool_calls:
                for tc_chunk in tool_calls:
                    index = tc_chunk.get("index", 0)
                    if index not in aggregated_tool_calls:
                        aggregated_tool_calls[index] = {
                            "id": tc_chunk.get("id"),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    tc = aggregated_tool_calls[index]
                    if tc_chunk.get("id"):
                        tc["id"] = tc_chunk.get("id")

                    fn_chunk = tc_chunk.get("function", {})
                    if fn_chunk.get("name"):
                        tc["function"]["name"] += fn_chunk.get("name")
                    if fn_chunk.get("arguments"):
                        tc["function"]["arguments"] += fn_chunk.get("arguments")

            # Update finish reason if present
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

        # Estimate token usage if not provided by API
        if usage_data is None and request_body:
            prompt_tokens = estimate_openai_prompt_tokens(request_body)
            response_data = {"choices": [{"message": final_message}]}
            completion_tokens = estimate_openai_completion_tokens(response_data)
            usage_data = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            lib_logger.debug(
                f"iFlow: Estimated token usage - prompt={prompt_tokens}, completion={completion_tokens}"
            )

        # Create the final ModelResponse
        # Use getattr to avoid LSP errors on first_chunk attributes
        final_response_data = {
            "id": getattr(first_chunk, "id", f"chatcmpl-iflow-{time.time()}"),
            "object": "chat.completion",
            "created": getattr(first_chunk, "created", int(time.time())),
            "model": getattr(first_chunk, "model", "iflow-model"),
            "choices": [final_choice],
            "usage": usage_data,
        }

        return litellm.ModelResponse(**final_response_data)

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        # Create provider logger from transaction context
        file_logger = ProviderLogger(transaction_context)

        # Check if request has images for vision processing
        has_images = request_has_images(kwargs)

        async def make_request(vision_fallback: bool = False):
            """Prepares and makes the actual API call."""
            # CRITICAL: get_api_details returns api_key, NOT access_token
            api_base, api_key = await self.get_api_details(credential_path)

            # Strip provider prefix from model name
            model_name = model.split("/")[-1]

            # Parse suffix from model name for thinking config
            suffix_result = parse_suffix(model_name)
            base_model = suffix_result.model_name

            # Handle vision fallback
            effective_model = FORCED_VISION_MODEL if vision_fallback else base_model

            kwargs_with_model = {**kwargs, "model": effective_model}
            payload = self._build_request_payload(
                effective_model, kwargs, **kwargs_with_model
            )

            # If vision fallback, strip images from payload and add vision summary context
            if vision_fallback:
                messages = payload.get("messages", [])
                if messages:
                    sanitized = strip_images_from_messages(messages)
                    bridge_msg = {
                        "role": "system",
                        "content": (
                            "The original model does not support images. "
                            "Please continue with the text content only."
                        ),
                    }
                    payload["messages"] = [bridge_msg] + sanitized

            headers = self._build_iflow_headers(
                api_key=api_key,
                stream=bool(payload.get("stream")),
            )

            url = f"{api_base.rstrip('/')}/chat/completions"

            # Log request to dedicated file
            file_logger.log_request(payload)

            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            ), payload

        async def stream_handler(response_stream, attempt=1, vision_fallback=False):
            """Handles the streaming response and converts chunks."""
            # Track state across chunks for finish_reason normalization
            stream_state: Dict[str, Any] = {}
            chunk_count = 0
            try:
                async with response_stream as response:
                    # Check for HTTP errors before processing stream
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        # Try gzip decompression
                        error_text_bytes = self._decompress_if_gzip(
                            error_text, dict(response.headers)
                        )
                        error_text = (
                            error_text_bytes.decode("utf-8")
                            if isinstance(error_text_bytes, bytes)
                            else error_text
                        )

                        # Handle 401: Force token refresh and retry once
                        if response.status_code == 401 and attempt == 1:
                            lib_logger.warning(
                                "iFlow returned 401. Forcing token refresh and retrying once."
                            )
                            await self._refresh_token(credential_path, force=True)
                            retry_stream, _ = await make_request(vision_fallback)
                            async for chunk in stream_handler(
                                retry_stream, attempt=2, vision_fallback=vision_fallback
                            ):
                                yield chunk
                            return

                        # Handle 429: Rate limit
                        elif (
                            response.status_code == 429
                            or "slow_down" in error_text.lower()
                        ):
                            raise RateLimitError(
                                f"iFlow rate limit exceeded: {error_text}",
                                llm_provider="iflow",
                                model=model,
                                response=response,
                            )

                        # Handle image capability errors - fallback to vision model
                        elif (
                            response.status_code in (400, 415, 422)
                            and has_images
                            and not vision_fallback
                        ):
                            exc = httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response,
                            )
                            if self._should_fallback_to_forced_vision(
                                exc, model.split("/")[-1], has_images
                            ):
                                lib_logger.info(
                                    f"[iFlow] Model doesn't support images, falling back to {FORCED_VISION_MODEL}"
                                )
                                fallback_stream, _ = await make_request(
                                    vision_fallback=True
                                )
                                async for chunk in stream_handler(
                                    fallback_stream, attempt=1, vision_fallback=True
                                ):
                                    yield chunk
                                return

                        # Handle other errors
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
                            request=response,
                            response=response,
                        )

                    # Handle non-streaming JSON responses (often used for errors with 200 OK)
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/json" in content_type:
                        body = await response.aread()
                        # Try gzip decompression
                        body = self._decompress_if_gzip(body, dict(response.headers))
                        body_text = body.decode("utf-8", errors="replace")
                        file_logger.log_response_chunk(body_text)
                        try:
                            error_data = json.loads(body_text)
                            if (
                                "error" in error_data
                                or error_data.get("success") is False
                            ):
                                error_msg = (
                                    error_data.get("message")
                                    or str(error_data.get("error"))
                                    or "Unknown iFlow JSON error"
                                )
                                raise StreamedAPIError(
                                    f"iFlow JSON Error: {error_msg}", data=error_data
                                )

                            # If it's valid non-error JSON, convert and yield as chunks
                            for openai_chunk in self._convert_chunk_to_openai(
                                error_data, model, stream_state
                            ):
                                yield litellm.ModelResponse(**openai_chunk)
                                chunk_count += 1
                            return
                        except json.JSONDecodeError:
                            # Fall through to line-by-line if not valid JSON
                            pass

                    # Process successful streaming response
                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        # CRITICAL FIX: Handle both "data:" (no space) and "data: " (with space)
                        if line.startswith("data:"):
                            # Extract data after "data:" prefix, handling both formats
                            if line.startswith("data: "):
                                data_str = line[6:]  # Skip "data: "
                            else:
                                data_str = line[5:]  # Skip "data:"

                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)

                                # Check for error JSON in chunk
                                if isinstance(chunk, dict) and (
                                    "error" in chunk or chunk.get("success") is False
                                ):
                                    error_msg = (
                                        chunk.get("message")
                                        or str(chunk.get("error"))
                                        or "Unknown iFlow error in stream"
                                    )
                                    raise StreamedAPIError(
                                        f"iFlow stream error: {error_msg}", data=chunk
                                    )

                                for openai_chunk in self._convert_chunk_to_openai(
                                    chunk, model, stream_state
                                ):
                                    yield litellm.ModelResponse(**openai_chunk)
                                    chunk_count += 1
                            except json.JSONDecodeError:
                                lib_logger.warning(
                                    f"Could not decode JSON from iFlow: {line}"
                                )

                # Detect empty responses
                if chunk_count == 0:
                    raise StreamedAPIError(
                        "iFlow returned an empty response body with 200 OK",
                        data={"status_code": 200},
                    )

            except httpx.HTTPStatusError:
                raise  # Re-raise HTTP errors we already handled
            except StreamedAPIError:
                raise  # Re-raise our custom streamed errors
            except (
                httpx.RequestError,
                httpx.TimeoutException,
                json.JSONDecodeError,
            ) as e:
                file_logger.log_error(f"Error during iFlow stream processing: {e}")
                lib_logger.error(
                    f"Error during iFlow stream processing: {e}", exc_info=True
                )
                raise

        async def logging_stream_wrapper(request_body: Optional[Dict[str, Any]] = None):
            """Wraps the stream to log the final reassembled response and cache reasoning."""
            openai_chunks = []
            try:
                stream, _ = await make_request()
                async for chunk in stream_handler(stream):
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(
                        openai_chunks, request_body
                    )
                    file_logger.log_final_response(final_response.dict())

                    # Store reasoning_content from the response for future multi-turn conversations
                    model_name = model.split("/")[-1]
                    messages = kwargs.get("messages", [])
                    if messages:
                        conv_sig = self._get_conversation_signature(messages)
                        if final_response.choices and len(final_response.choices) > 0:
                            choice = final_response.choices[0]
                            message = getattr(choice, "message", None)
                            if message:
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

        # Build request body for token estimation
        request_body = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        if kwargs.get("stream"):
            return logging_stream_wrapper(request_body)
        else:

            async def non_stream_wrapper():
                chunks = [chunk async for chunk in logging_stream_wrapper(request_body)]
                return self._stream_to_completion_response(chunks, request_body)

            return await non_stream_wrapper()
