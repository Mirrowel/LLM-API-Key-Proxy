# src/rotator_library/providers/antigravity_provider_v2.py
"""
Antigravity Provider - Refactored Implementation

A clean, well-structured provider for Google's Antigravity API, supporting:
- Gemini 2.5 (Pro/Flash) with thinkingBudget
- Gemini 3 (Pro/Image) with thinkingLevel
- Claude (Sonnet 4.5) via Antigravity proxy
- Claude (Opus 4.5) via Antigravity proxy

Key Features:
- Unified streaming/non-streaming handling
- Server-side thought signature caching
- Automatic base URL fallback
- Gemini 3 tool hallucination prevention
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import litellm

from .provider_interface import ProviderInterface, UsageResetConfigDef, QuotaGroupMap
from .antigravity_auth_base import AntigravityAuthBase
from .provider_cache import ProviderCache
from ..model_definitions import ModelDefinitions


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

lib_logger = logging.getLogger("rotator_library")

# Antigravity base URLs with fallback order
# Priority: daily (sandbox) → autopush (sandbox) → production
BASE_URLS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    # "https://autopush-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://cloudcode-pa.googleapis.com/v1internal",  # Production fallback
]

# Available models via Antigravity
AVAILABLE_MODELS = [
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
    # "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",  # Internally mapped to -low/-high variant based on thinkingLevel
    # "gemini-3-pro-image-preview",
    # "gemini-2.5-computer-use-preview-10-2025",
    "claude-sonnet-4-5",  # Internally mapped to -thinking variant when reasoning_effort is provided
    "claude-opus-4-5",  # ALWAYS uses -thinking variant (non-thinking doesn't exist)
]

# Default max output tokens (including thinking) - can be overridden per request
DEFAULT_MAX_OUTPUT_TOKENS = 64000

# Model alias mappings (internal ↔ public)
MODEL_ALIAS_MAP = {
    "rev19-uic3-1p": "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-pro-image": "gemini-3-pro-image-preview",
    "gemini-3-pro-low": "gemini-3-pro-preview",
    "gemini-3-pro-high": "gemini-3-pro-preview",
}
MODEL_ALIAS_REVERSE = {v: k for k, v in MODEL_ALIAS_MAP.items()}

# Models to exclude from dynamic discovery
EXCLUDED_MODELS = {
    "chat_20706",
    "chat_23310",
    "gemini-2.5-flash-thinking",
    "gemini-2.5-pro",
}

# Gemini finish reason mapping
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}

# Default safety settings - disable content filtering for all categories
# Per CLIProxyAPI: these are attached to prevent safety blocks during API calls
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
]

# Directory paths
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = _BASE_DIR / "logs" / "antigravity_logs"
CACHE_DIR = _BASE_DIR / "cache" / "antigravity"
GEMINI3_SIGNATURE_CACHE_FILE = CACHE_DIR / "gemini3_signatures.json"
CLAUDE_THINKING_CACHE_FILE = CACHE_DIR / "claude_thinking.json"

# Gemini 3 tool fix system instruction (prevents hallucination)
DEFAULT_GEMINI3_SYSTEM_INSTRUCTION = """<CRITICAL_TOOL_USAGE_INSTRUCTIONS>
You are operating in a CUSTOM ENVIRONMENT where tool definitions COMPLETELY DIFFER from your training data.
VIOLATION OF THESE RULES WILL CAUSE IMMEDIATE SYSTEM FAILURE.

## ABSOLUTE RULES - NO EXCEPTIONS

1. **SCHEMA IS LAW**: The JSON schema in each tool definition is the ONLY source of truth.
   - Your pre-trained knowledge about tools like 'read_file', 'apply_diff', 'write_to_file', 'bash', etc. is INVALID here.
   - Every tool has been REDEFINED with different parameters than what you learned during training.

2. **PARAMETER NAMES ARE EXACT**: Use ONLY the parameter names from the schema.
   - WRONG: 'suggested_answers', 'file_path', 'files_to_read', 'command_to_run'
   - RIGHT: Check the 'properties' field in the schema for the exact names
   - The schema's 'required' array tells you which parameters are mandatory

3. **ARRAY PARAMETERS**: When a parameter has "type": "array", check the 'items' field:
   - If items.type is "object", you MUST provide an array of objects with the EXACT properties listed
   - If items.type is "string", you MUST provide an array of strings
   - NEVER provide a single object when an array is expected
   - NEVER provide an array when a single value is expected

4. **NESTED OBJECTS**: When items.type is "object":
   - Check items.properties for the EXACT field names required
   - Check items.required for which nested fields are mandatory
   - Include ALL required nested fields in EVERY array element

5. **STRICT PARAMETERS HINT**: Tool descriptions contain "STRICT PARAMETERS: ..." which lists:
   - Parameter name, type, and whether REQUIRED
   - For arrays of objects: the nested structure in brackets like [field: type REQUIRED, ...]
   - USE THIS as your quick reference, but the JSON schema is authoritative

6. **BEFORE EVERY TOOL CALL**:
   a. Read the tool's 'parametersJsonSchema' or 'parameters' field completely
   b. Identify ALL required parameters
   c. Verify your parameter names match EXACTLY (case-sensitive)
   d. For arrays, verify you're providing the correct item structure
   e. Do NOT add parameters that don't exist in the schema

## COMMON FAILURE PATTERNS TO AVOID

- Using 'path' when schema says 'filePath' (or vice versa)
- Using 'content' when schema says 'text' (or vice versa)  
- Providing {"file": "..."} when schema wants [{"path": "...", "line_ranges": [...]}]
- Omitting required nested fields in array items
- Adding 'additionalProperties' that the schema doesn't define
- Guessing parameter names from similar tools you know from training

## REMEMBER
Your training data about function calling is OUTDATED for this environment.
The tool names may look familiar, but the schemas are DIFFERENT.
When in doubt, RE-READ THE SCHEMA before making the call.
</CRITICAL_TOOL_USAGE_INSTRUCTIONS>
"""

# Claude tool fix system instruction (prevents hallucination)
DEFAULT_CLAUDE_SYSTEM_INSTRUCTION = """CRITICAL TOOL USAGE INSTRUCTIONS:
You are operating in a custom environment where tool definitions differ from your training data.
You MUST follow these rules strictly:

1. DO NOT use your internal training data to guess tool parameters
2. ONLY use the exact parameter structure defined in the tool schema
3. Parameter names in schemas are EXACT - do not substitute with similar names from your training (e.g., use 'follow_up' not 'suggested_answers')
4. Array parameters have specific item types - check the schema's 'items' field for the exact structure
5. When you see "STRICT PARAMETERS" in a tool description, those type definitions override any assumptions
6. Tool use in agentic workflows is REQUIRED - you must call tools with the exact parameters specified in the schema

If you are unsure about a tool's parameters, YOU MUST read the schema definition carefully.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


def _generate_request_id() -> str:
    """Generate Antigravity request ID: agent-{uuid}"""
    return f"agent-{uuid.uuid4()}"


def _generate_session_id() -> str:
    """Generate Antigravity session ID: -{random_number}"""
    n = random.randint(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)
    return f"-{n}"


def _generate_project_id() -> str:
    """Generate fake project ID: {adj}-{noun}-{random}"""
    adjectives = ["useful", "bright", "swift", "calm", "bold"]
    nouns = ["fuze", "wave", "spark", "flow", "core"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:5]}"


def _normalize_type_arrays(schema: Any) -> Any:
    """
    Normalize type arrays in JSON Schema for Proto-based Antigravity API.
    Converts `"type": ["string", "null"]` → `"type": "string"`.
    """
    if isinstance(schema, dict):
        normalized = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                normalized[key] = non_null[0] if non_null else value[0]
            else:
                normalized[key] = _normalize_type_arrays(value)
        return normalized
    elif isinstance(schema, list):
        return [_normalize_type_arrays(item) for item in schema]
    return schema


def _recursively_parse_json_strings(obj: Any) -> Any:
    """
    Recursively parse JSON strings in nested data structures.

    Antigravity sometimes returns tool arguments with JSON-stringified values:
    {"files": "[{...}]"} instead of {"files": [{...}]}.

    Additionally handles:
    - Malformed double-encoded JSON (extra trailing '}' or ']')
    - Escaped string content (\n, \t, \", etc.)
    """
    if isinstance(obj, dict):
        return {k: _recursively_parse_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursively_parse_json_strings(item) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()

        # Check if string contains control character escape sequences that need unescaping
        # This handles cases where diff content has literal \n or \t instead of actual newlines/tabs
        #
        # IMPORTANT: We intentionally do NOT unescape strings containing \" or \\
        # because these are typically intentional escapes in code/config content
        # (e.g., JSON embedded in YAML: BOT_NAMES_JSON: '["mirrobot", ...]')
        # Unescaping these would corrupt the content and cause issues like
        # oldString and newString becoming identical when they should differ.
        has_control_char_escapes = "\\n" in obj or "\\t" in obj
        has_intentional_escapes = '\\"' in obj or "\\\\" in obj

        if has_control_char_escapes and not has_intentional_escapes:
            try:
                # Use json.loads with quotes to properly unescape the string
                # This converts \n -> newline, \t -> tab
                unescaped = json.loads(f'"{obj}"')
                # Log the fix with a snippet for debugging
                snippet = obj[:80] + "..." if len(obj) > 80 else obj
                lib_logger.debug(
                    f"[Antigravity] Unescaped control chars in string: "
                    f"{len(obj) - len(unescaped)} chars changed. Snippet: {snippet!r}"
                )
                return unescaped
            except (json.JSONDecodeError, ValueError):
                # If unescaping fails, continue with original processing
                pass

        # Check if it looks like JSON (starts with { or [)
        if stripped and stripped[0] in ("{", "["):
            # Try standard parsing first
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    parsed = json.loads(obj)
                    return _recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass

            # Handle malformed JSON: array that doesn't end with ]
            # e.g., '[{"path": "..."}]}' instead of '[{"path": "..."}]'
            if stripped.startswith("[") and not stripped.endswith("]"):
                try:
                    # Find the last ] and truncate there
                    last_bracket = stripped.rfind("]")
                    if last_bracket > 0:
                        cleaned = stripped[: last_bracket + 1]
                        parsed = json.loads(cleaned)
                        lib_logger.warning(
                            f"[Antigravity] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return _recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass

            # Handle malformed JSON: object that doesn't end with }
            if stripped.startswith("{") and not stripped.endswith("}"):
                try:
                    # Find the last } and truncate there
                    last_brace = stripped.rfind("}")
                    if last_brace > 0:
                        cleaned = stripped[: last_brace + 1]
                        parsed = json.loads(cleaned)
                        lib_logger.warning(
                            f"[Antigravity] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return _recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
    return obj


def _clean_claude_schema(schema: Any) -> Any:
    """
    Recursively clean JSON Schema for Antigravity/Google's Proto-based API.
    - Removes unsupported fields ($schema, additionalProperties, etc.)
    - Converts 'const' to 'enum' with single value (supported equivalent)
    - Converts 'anyOf'/'oneOf' to the first option (Claude doesn't support these)
    """
    if not isinstance(schema, dict):
        return schema

    # Fields not supported by Antigravity/Google's Proto-based API
    # Note: Claude via Antigravity rejects JSON Schema draft 2020-12 validation keywords
    incompatible = {
        "$schema",
        "additionalProperties",
        "minItems",
        "maxItems",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "default",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
        "minProperties",
        "maxProperties",
        "uniqueItems",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "title",
    }

    # Handle 'anyOf' by taking the first option (Claude doesn't support anyOf)
    if "anyOf" in schema and isinstance(schema["anyOf"], list) and schema["anyOf"]:
        first_option = _clean_claude_schema(schema["anyOf"][0])
        if isinstance(first_option, dict):
            return first_option

    # Handle 'oneOf' similarly
    if "oneOf" in schema and isinstance(schema["oneOf"], list) and schema["oneOf"]:
        first_option = _clean_claude_schema(schema["oneOf"][0])
        if isinstance(first_option, dict):
            return first_option

    cleaned = {}

    # Handle 'const' by converting to 'enum' with single value
    if "const" in schema:
        const_value = schema["const"]
        cleaned["enum"] = [const_value]

    for key, value in schema.items():
        if key in incompatible or key == "const":
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_claude_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_claude_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned


# =============================================================================
# FILE LOGGER
# =============================================================================


class AntigravityFileLogger:
    """Transaction file logger for debugging Antigravity requests/responses."""

    __slots__ = ("enabled", "log_dir")

    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        self.log_dir: Optional[Path] = None

        if not enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_model = model_name.replace("/", "_").replace(":", "_")
        self.log_dir = LOGS_DIR / f"{timestamp}_{safe_model}_{uuid.uuid4()}"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create log directory: {e}")
            self.enabled = False

    def log_request(self, payload: Dict[str, Any]) -> None:
        """Log the request payload."""
        self._write_json("request_payload.json", payload)

    def log_response_chunk(self, chunk: str) -> None:
        """Append a raw chunk to the response stream log."""
        self._append_text("response_stream.log", chunk)

    def log_error(self, error_message: str) -> None:
        """Log an error message."""
        self._append_text(
            "error.log", f"[{datetime.utcnow().isoformat()}] {error_message}"
        )

    def log_final_response(self, response: Dict[str, Any]) -> None:
        """Log the final response."""
        self._write_json("final_response.json", response)

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"Failed to write {filename}: {e}")

    def _append_text(self, filename: str, text: str) -> None:
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            lib_logger.error(f"Failed to append to {filename}: {e}")


# =============================================================================
# MAIN PROVIDER CLASS
# =============================================================================


class AntigravityProvider(AntigravityAuthBase, ProviderInterface):
    """
    Antigravity provider for Gemini and Claude models via Google's internal API.

    Supports:
    - Gemini 2.5 (Pro/Flash) with thinkingBudget
    - Gemini 3 (Pro/Image) with thinkingLevel
    - Claude Sonnet 4.5 via Antigravity proxy
    - Claude Opus 4.5 via Antigravity proxy

    Features:
    - Unified streaming/non-streaming handling
    - ThoughtSignature caching for multi-turn conversations
    - Automatic base URL fallback
    - Gemini 3 tool hallucination prevention
    """

    skip_cost_calculation = True

    # Sequential mode by default - preserves thinking signature caches between requests
    default_rotation_mode: str = "sequential"

    # =========================================================================
    # TIER & USAGE CONFIGURATION
    # =========================================================================

    # Provider name for env var lookups (QUOTA_GROUPS_ANTIGRAVITY_*)
    provider_env_name: str = "antigravity"

    # Tier name -> priority mapping (Single Source of Truth)
    # Lower numbers = higher priority
    tier_priorities = {
        # Priority 1: Highest paid tier (Google AI Ultra - name unconfirmed)
        # "google-ai-ultra": 1,  # Uncomment when tier name is confirmed
        # Priority 2: Standard paid tier
        "standard-tier": 2,
        # Priority 3: Free tier
        "free-tier": 3,
        # Priority 10: Legacy/Unknown (lowest)
        "legacy-tier": 10,
        "unknown": 10,
    }

    # Default priority for tiers not in the mapping
    default_tier_priority: int = 10

    # Usage reset configs keyed by priority sets
    # Priorities 1-2 (paid tiers) get 5h window, others get 7d window
    usage_reset_configs = {
        frozenset({1, 2}): UsageResetConfigDef(
            window_seconds=5 * 60 * 60,  # 5 hours
            mode="per_model",
            description="5-hour per-model window (paid tier)",
            field_name="models",
        ),
        "default": UsageResetConfigDef(
            window_seconds=7 * 24 * 60 * 60,  # 7 days
            mode="per_model",
            description="7-day per-model window (free/unknown tier)",
            field_name="models",
        ),
    }

    # Model quota groups (can be overridden via QUOTA_GROUPS_ANTIGRAVITY_CLAUDE)
    # Models in the same group share quota - when one is exhausted, all are
    model_quota_groups: QuotaGroupMap = {
        "claude": ["claude-sonnet-4-5", "claude-opus-4-5"],
    }

    # Model usage weights for grouped usage calculation
    # Opus consumes more quota per request, so its usage counts 2x when
    # comparing credentials for selection
    model_usage_weights = {
        "claude-opus-4-5": 2,
    }

    # Priority-based concurrency multipliers
    # Higher priority credentials (lower number) get higher multipliers
    # Priority 1 (paid ultra): 5x concurrent requests
    # Priority 2 (standard paid): 3x concurrent requests
    # Others: Use sequential fallback (2x) or balanced default (1x)
    default_priority_multipliers = {1: 5, 2: 3}

    # For sequential mode, lower priority tiers still get 2x to maintain stickiness
    # For balanced mode, this doesn't apply (falls back to 1x)
    default_sequential_fallback_multiplier = 2

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Antigravity/Google RPC quota errors.

        Handles the Google Cloud API error format with ErrorInfo and RetryInfo details.

        Example error format:
        {
          "error": {
            "code": 429,
            "details": [
              {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "QUOTA_EXHAUSTED",
                "metadata": {
                  "quotaResetDelay": "143h4m52.730699158s",
                  "quotaResetTimeStamp": "2025-12-11T22:53:16Z"
                }
              },
              {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "515092.730699158s"
              }
            ]
          }
        }

        Args:
            error: The caught exception
            error_body: Optional raw response body string

        Returns:
            None if not a parseable quota error, otherwise:
            {
                "retry_after": int,
                "reason": str,
                "reset_timestamp": str | None,
            }
        """
        import re as regex_module

        def parse_duration(duration_str: str) -> Optional[int]:
            """Parse duration strings like '143h4m52.73s' or '515092.73s' to seconds."""
            if not duration_str:
                return None

            # Handle pure seconds format: "515092.730699158s"
            pure_seconds_match = regex_module.match(r"^([\d.]+)s$", duration_str)
            if pure_seconds_match:
                return int(float(pure_seconds_match.group(1)))

            # Handle compound format: "143h4m52.730699158s"
            total_seconds = 0
            patterns = [
                (r"(\d+)h", 3600),  # hours
                (r"(\d+)m", 60),  # minutes
                (r"([\d.]+)s", 1),  # seconds
            ]
            for pattern, multiplier in patterns:
                match = regex_module.search(pattern, duration_str)
                if match:
                    total_seconds += float(match.group(1)) * multiplier

            return int(total_seconds) if total_seconds > 0 else None

        # Get error body from exception if not provided
        body = error_body
        if not body:
            # Try to extract from various exception attributes
            if hasattr(error, "response") and hasattr(error.response, "text"):
                body = error.response.text
            elif hasattr(error, "body"):
                body = str(error.body)
            elif hasattr(error, "message"):
                body = str(error.message)
            else:
                body = str(error)

        # Try to find JSON in the body
        try:
            # Handle cases where JSON is embedded in a larger string
            json_match = regex_module.search(r"\{[\s\S]*\}", body)
            if not json_match:
                return None

            data = json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

        # Navigate to error.details
        error_obj = data.get("error", data)
        details = error_obj.get("details", [])

        if not details:
            return None

        result = {
            "retry_after": None,
            "reason": None,
            "reset_timestamp": None,
            "quota_reset_timestamp": None,  # Unix timestamp for quota reset
        }

        for detail in details:
            detail_type = detail.get("@type", "")

            # Parse RetryInfo - most authoritative source for retry delay
            if "RetryInfo" in detail_type:
                retry_delay = detail.get("retryDelay")
                if retry_delay:
                    parsed = parse_duration(retry_delay)
                    if parsed:
                        result["retry_after"] = parsed

            # Parse ErrorInfo - contains reason and quota reset metadata
            elif "ErrorInfo" in detail_type:
                result["reason"] = detail.get("reason")
                metadata = detail.get("metadata", {})

                # Get quotaResetDelay as fallback if RetryInfo not present
                if not result["retry_after"]:
                    quota_delay = metadata.get("quotaResetDelay")
                    if quota_delay:
                        parsed = parse_duration(quota_delay)
                        if parsed:
                            result["retry_after"] = parsed

                # Capture reset timestamp for logging and authoritative reset time
                reset_ts_str = metadata.get("quotaResetTimeStamp")
                result["reset_timestamp"] = reset_ts_str

                # Parse ISO timestamp to Unix timestamp for usage tracking
                if reset_ts_str:
                    try:
                        # Handle ISO format: "2025-12-11T22:53:16Z"
                        reset_dt = datetime.fromisoformat(
                            reset_ts_str.replace("Z", "+00:00")
                        )
                        result["quota_reset_timestamp"] = reset_dt.timestamp()
                    except (ValueError, AttributeError) as e:
                        lib_logger.warning(
                            f"Failed to parse quota reset timestamp '{reset_ts_str}': {e}"
                        )

        # Return None if we couldn't extract retry_after
        if not result["retry_after"]:
            return None

        return result

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        # NOTE: project_id_cache and project_tier_cache are inherited from AntigravityAuthBase

        # Base URL management
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

        # Configuration from environment
        memory_ttl = _env_int("ANTIGRAVITY_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = _env_int("ANTIGRAVITY_SIGNATURE_DISK_TTL", 86400)

        # Initialize caches using shared ProviderCache
        self._signature_cache = ProviderCache(
            GEMINI3_SIGNATURE_CACHE_FILE,
            memory_ttl,
            disk_ttl,
            env_prefix="ANTIGRAVITY_SIGNATURE",
        )
        self._thinking_cache = ProviderCache(
            CLAUDE_THINKING_CACHE_FILE,
            memory_ttl,
            disk_ttl,
            env_prefix="ANTIGRAVITY_THINKING",
        )

        # Feature flags
        self._preserve_signatures_in_client = _env_bool(
            "ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES", True
        )
        self._enable_signature_cache = _env_bool(
            "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE", True
        )
        self._enable_dynamic_models = _env_bool(
            "ANTIGRAVITY_ENABLE_DYNAMIC_MODELS", False
        )
        self._enable_gemini3_tool_fix = _env_bool("ANTIGRAVITY_GEMINI3_TOOL_FIX", True)
        self._enable_claude_tool_fix = _env_bool("ANTIGRAVITY_CLAUDE_TOOL_FIX", True)
        self._enable_thinking_sanitization = _env_bool(
            "ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION", True
        )

        # Gemini 3 tool fix configuration
        self._gemini3_tool_prefix = os.getenv(
            "ANTIGRAVITY_GEMINI3_TOOL_PREFIX", "gemini3_"
        )
        self._gemini3_description_prompt = os.getenv(
            "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT",
            "\n\n⚠️ STRICT PARAMETERS (use EXACTLY as shown): {params}. Do NOT use parameters from your training data - use ONLY these parameter names.",
        )
        self._gemini3_enforce_strict_schema = _env_bool(
            "ANTIGRAVITY_GEMINI3_STRICT_SCHEMA", True
        )
        self._gemini3_system_instruction = os.getenv(
            "ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION", DEFAULT_GEMINI3_SYSTEM_INSTRUCTION
        )

        # Claude tool fix configuration (separate from Gemini 3)
        self._claude_description_prompt = os.getenv(
            "ANTIGRAVITY_CLAUDE_DESCRIPTION_PROMPT", "\n\nSTRICT PARAMETERS: {params}."
        )
        self._claude_system_instruction = os.getenv(
            "ANTIGRAVITY_CLAUDE_SYSTEM_INSTRUCTION", DEFAULT_CLAUDE_SYSTEM_INSTRUCTION
        )

        # Log configuration
        self._log_config()

    def _log_config(self) -> None:
        """Log provider configuration."""
        lib_logger.debug(
            f"Antigravity config: signatures_in_client={self._preserve_signatures_in_client}, "
            f"cache={self._enable_signature_cache}, dynamic_models={self._enable_dynamic_models}, "
            f"gemini3_fix={self._enable_gemini3_tool_fix}, gemini3_strict_schema={self._gemini3_enforce_strict_schema}, "
            f"claude_fix={self._enable_claude_tool_fix}, thinking_sanitization={self._enable_thinking_sanitization}"
        )

    def _load_tier_from_file(self, credential_path: str) -> Optional[str]:
        """
        Load tier from credential file's _proxy_metadata and cache it.

        This is used as a fallback when the tier isn't in the memory cache,
        typically on first access before initialize_credentials() has run.

        Args:
            credential_path: Path to the credential file

        Returns:
            Tier string if found, None otherwise
        """
        # Skip env:// paths (environment-based credentials)
        if self._parse_env_credential_path(credential_path) is not None:
            return None

        try:
            with open(credential_path, "r") as f:
                creds = json.load(f)

            metadata = creds.get("_proxy_metadata", {})
            tier = metadata.get("tier")
            project_id = metadata.get("project_id")

            if tier:
                self.project_tier_cache[credential_path] = tier
                lib_logger.debug(
                    f"Lazy-loaded tier '{tier}' for credential: {Path(credential_path).name}"
                )

            if project_id and credential_path not in self.project_id_cache:
                self.project_id_cache[credential_path] = project_id

            return tier
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            lib_logger.debug(f"Could not lazy-load tier from {credential_path}: {e}")
            return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the human-readable tier name for a credential.

        Args:
            credential: The credential path

        Returns:
            Tier name string (e.g., "free-tier") or None if unknown
        """
        tier = self.project_tier_cache.get(credential)
        if not tier:
            tier = self._load_tier_from_file(credential)
        return tier

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        Returns the minimum priority tier required for a model.
        Antigravity has no model-tier restrictions - all models work on all tiers.

        Args:
            model: The model name (with or without provider prefix)

        Returns:
            None - no restrictions for any model
        """
        return None

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Load persisted tier information from credential files at startup.

        This ensures all credential priorities are known before any API calls,
        preventing unknown credentials from getting priority 999.
        """
        await self._load_persisted_tiers(credential_paths)

    async def _load_persisted_tiers(
        self, credential_paths: List[str]
    ) -> Dict[str, str]:
        """
        Load persisted tier information from credential files into memory cache.

        Args:
            credential_paths: List of credential file paths

        Returns:
            Dict mapping credential path to tier name for logging purposes
        """
        loaded = {}
        for path in credential_paths:
            # Skip env:// paths (environment-based credentials)
            if self._parse_env_credential_path(path) is not None:
                continue

            # Skip if already in cache
            if path in self.project_tier_cache:
                continue

            try:
                with open(path, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                tier = metadata.get("tier")
                project_id = metadata.get("project_id")

                if tier:
                    self.project_tier_cache[path] = tier
                    loaded[path] = tier
                    lib_logger.debug(
                        f"Loaded persisted tier '{tier}' for credential: {Path(path).name}"
                    )

                if project_id:
                    self.project_id_cache[path] = project_id

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                lib_logger.debug(f"Could not load persisted tier from {path}: {e}")

        if loaded:
            # Log summary at debug level
            tier_counts: Dict[str, int] = {}
            for tier in loaded.values():
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            lib_logger.debug(
                f"Antigravity: Loaded {len(loaded)} credential tiers from disk: "
                + ", ".join(
                    f"{tier}={count}" for tier, count in sorted(tier_counts.items())
                )
            )

        return loaded

    # NOTE: _post_auth_discovery() is inherited from AntigravityAuthBase

    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================

    def _alias_to_internal(self, alias: str) -> str:
        """Convert public alias to internal model name."""
        return MODEL_ALIAS_REVERSE.get(alias, alias)

    def _internal_to_alias(self, internal: str) -> str:
        """Convert internal model name to public alias."""
        if internal in EXCLUDED_MODELS:
            return ""
        return MODEL_ALIAS_MAP.get(internal, internal)

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        internal = self._alias_to_internal(model)
        return internal.startswith("gemini-3-") or model.startswith("gemini-3-")

    def _is_claude(self, model: str) -> bool:
        """Check if model is Claude."""
        return "claude" in model.lower()

    def _strip_provider_prefix(self, model: str) -> str:
        """Strip provider prefix from model name."""
        return model.split("/")[-1] if "/" in model else model

    # =========================================================================
    # BASE URL MANAGEMENT
    # =========================================================================

    def _get_base_url(self) -> str:
        """Get current base URL."""
        return self._current_base_url

    def _try_next_base_url(self) -> bool:
        """Switch to next base URL in fallback list. Returns True if successful."""
        if self._base_url_index < len(BASE_URLS) - 1:
            self._base_url_index += 1
            self._current_base_url = BASE_URLS[self._base_url_index]
            lib_logger.info(f"Switching to fallback URL: {self._current_base_url}")
            return True
        return False

    def _reset_base_url(self) -> None:
        """Reset to primary base URL."""
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

    # =========================================================================
    # THINKING CACHE KEY GENERATION
    # =========================================================================

    def _generate_thinking_cache_key(
        self, text_content: str, tool_calls: List[Dict]
    ) -> Optional[str]:
        """
        Generate stable cache key from response content for Claude thinking preservation.

        Uses composite key:
        - Tool call IDs (most stable)
        - Text hash (for text-only responses)
        """
        key_parts = []

        if tool_calls:
            first_id = tool_calls[0].get("id", "")
            if first_id:
                key_parts.append(f"tool_{first_id.replace('call_', '')}")

        if text_content:
            text_hash = hashlib.md5(text_content[:200].encode()).hexdigest()[:16]
            key_parts.append(f"text_{text_hash}")

        return "thinking_" + "_".join(key_parts) if key_parts else None

    # NOTE: _discover_project_id() and _persist_project_metadata() are inherited from AntigravityAuthBase

    # =========================================================================
    # THINKING MODE SANITIZATION
    # =========================================================================

    def _analyze_conversation_state(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation state to detect tool use loops and thinking mode issues.

        Key insight: A "turn" can span multiple assistant messages in a tool-use loop.
        We need to find the TURN START (first assistant message after last real user message)
        and check if THAT message had thinking, not just the last assistant message.

        Returns:
            {
                "in_tool_loop": bool - True if we're in an incomplete tool use loop
                "turn_start_idx": int - Index of first model message in current turn
                "turn_has_thinking": bool - Whether the TURN started with thinking
                "last_model_idx": int - Index of last model message
                "last_model_has_thinking": bool - Whether last model msg has thinking
                "last_model_has_tool_calls": bool - Whether last model msg has tool calls
                "pending_tool_results": bool - Whether there are tool results after last model
                "thinking_block_indices": List[int] - Indices of messages with thinking/reasoning
            }

        NOTE: This now operates on Gemini-format messages (after transformation):
        - Role "model" instead of "assistant"
        - Role "user" for both user messages AND tool results (with functionResponse)
        - "parts" array with "thought": true for thinking
        - "parts" array with "functionCall" for tool calls
        - "parts" array with "functionResponse" for tool results
        """
        state = {
            "in_tool_loop": False,
            "turn_start_idx": -1,
            "turn_has_thinking": False,
            "last_assistant_idx": -1,  # Keep name for compatibility
            "last_assistant_has_thinking": False,
            "last_assistant_has_tool_calls": False,
            "pending_tool_results": False,
            "thinking_block_indices": [],
        }

        # First pass: Find the last "real" user message (not a tool result)
        # In Gemini format, tool results are "user" role with functionResponse parts
        last_real_user_idx = -1
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "user":
                # Check if this is a real user message or a tool result container
                parts = msg.get("parts", [])
                is_tool_result_msg = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )

                if not is_tool_result_msg:
                    last_real_user_idx = i

        # Second pass: Analyze conversation and find turn boundaries
        for i, msg in enumerate(messages):
            role = msg.get("role")

            if role == "model":
                # Check for thinking/reasoning content (Gemini format)
                has_thinking = self._message_has_thinking(msg)

                # Check for tool calls (functionCall in parts)
                parts = msg.get("parts", [])
                has_tool_calls = any(
                    isinstance(p, dict) and "functionCall" in p for p in parts
                )

                # Track if this is the turn start
                if i > last_real_user_idx and state["turn_start_idx"] == -1:
                    state["turn_start_idx"] = i
                    state["turn_has_thinking"] = has_thinking

                state["last_assistant_idx"] = i
                state["last_assistant_has_tool_calls"] = has_tool_calls
                state["last_assistant_has_thinking"] = has_thinking

                if has_thinking:
                    state["thinking_block_indices"].append(i)

            elif role == "user":
                # Check if this is a tool result (functionResponse in parts)
                parts = msg.get("parts", [])
                is_tool_result = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )

                if is_tool_result and state["last_assistant_has_tool_calls"]:
                    state["pending_tool_results"] = True

        # We're in a tool loop if:
        # 1. There are pending tool results
        # 2. The conversation ends with tool results (last message is user with functionResponse)
        if state["pending_tool_results"] and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                parts = last_msg.get("parts", [])
                ends_with_tool_result = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )
                if ends_with_tool_result:
                    state["in_tool_loop"] = True

        return state

    def _message_has_thinking(self, msg: Dict[str, Any]) -> bool:
        """
        Check if a message contains thinking/reasoning content.

        Handles GEMINI format (after transformation):
        - "parts" array with items having "thought": true
        """
        parts = msg.get("parts", [])
        for part in parts:
            if isinstance(part, dict) and part.get("thought") is True:
                return True
        return False

    def _message_has_tool_calls(self, msg: Dict[str, Any]) -> bool:
        """Check if a message contains tool calls (Gemini format)."""
        parts = msg.get("parts", [])
        return any(isinstance(p, dict) and "functionCall" in p for p in parts)

    def _sanitize_thinking_for_claude(
        self, messages: List[Dict[str, Any]], thinking_enabled: bool
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Sanitize thinking blocks in conversation history for Claude compatibility.

        Handles the following scenarios per Claude docs:
        1. If thinking is disabled, remove all thinking blocks from conversation
        2. If thinking is enabled:
           a. In a tool use loop WITH thinking: preserve it (same mode continues)
           b. In a tool use loop WITHOUT thinking: this is INVALID toggle - force disable
           c. Not in tool loop: strip old thinking, new response adds thinking naturally

        Per Claude docs:
        - "If thinking is enabled, the final assistant turn must start with a thinking block"
        - "If thinking is disabled, the final assistant turn must not contain any thinking blocks"
        - Tool use loops are part of a single assistant turn
        - You CANNOT toggle thinking mid-turn

        The key insight: We only force-disable thinking when TOGGLING it ON mid-turn.
        If thinking was already enabled (assistant has thinking), we preserve.
        If thinking was disabled (assistant has no thinking), enabling it now is invalid.

        Returns:
            Tuple of (sanitized_messages, force_disable_thinking)
            - sanitized_messages: The cleaned message list
            - force_disable_thinking: If True, thinking must be disabled for this request
        """
        messages = copy.deepcopy(messages)
        state = self._analyze_conversation_state(messages)

        lib_logger.debug(
            f"[Thinking Sanitization] thinking_enabled={thinking_enabled}, "
            f"in_tool_loop={state['in_tool_loop']}, "
            f"turn_has_thinking={state['turn_has_thinking']}, "
            f"turn_start_idx={state['turn_start_idx']}, "
            f"last_assistant_has_thinking={state['last_assistant_has_thinking']}, "
            f"last_assistant_has_tool_calls={state['last_assistant_has_tool_calls']}"
        )

        if not thinking_enabled:
            # CASE 1: Thinking is disabled - strip ALL thinking blocks
            return self._strip_all_thinking_blocks(messages), False

        # CASE 2: Thinking is enabled
        if state["in_tool_loop"]:
            # We're in a tool use loop (conversation ends with tool_result)
            # Per Claude docs: entire assistant turn must operate in single thinking mode
            #
            # KEY FIX: Check turn_has_thinking (thinking at turn START), not last_assistant_has_thinking.
            # In multi-message tool loops, thinking is at the FIRST assistant message of the turn,
            # not necessarily the last one (which might just have tool_calls).

            if state["turn_has_thinking"]:
                # The TURN started with thinking - this is valid!
                # Thinking was enabled when tool was called, continue with thinking enabled.
                # Preserve thinking for the turn start message.
                lib_logger.debug(
                    "[Thinking Sanitization] Tool loop with thinking at turn start - preserving. "
                    f"turn_start_idx={state['turn_start_idx']}, last_assistant_idx={state['last_assistant_idx']}"
                )
                return self._preserve_turn_start_thinking(
                    messages, state["turn_start_idx"]
                ), False
            else:
                # The TURN did NOT start with thinking, but thinking is NOW enabled
                # This is the INVALID case: toggling thinking ON mid-turn
                #
                # Per Claude docs, this causes:
                # "Expected `thinking` or `redacted_thinking`, but found `tool_use`."
                #
                # There are TWO possible scenarios:
                # 1. Original turn was made WITHOUT thinking (e.g., by Gemini or non-thinking Claude)
                #    → Solution: Close the tool loop with synthetic message
                # 2. Original turn HAD thinking but compaction stripped it
                #    → Solution: Try to inject cached thinking, fallback to synthetic closure

                turn_start_msg = (
                    messages[state["turn_start_idx"]]
                    if state["turn_start_idx"] >= 0
                    else None
                )

                # Check if this looks like a compacted thinking turn
                if turn_start_msg and self._looks_like_compacted_thinking_turn(
                    turn_start_msg
                ):
                    # Try to recover cached thinking block
                    recovered = self._try_recover_thinking_from_cache(
                        messages, state["turn_start_idx"]
                    )
                    if recovered:
                        lib_logger.info(
                            "[Thinking Sanitization] Recovered thinking from cache for compacted turn."
                        )
                        return self._preserve_turn_start_thinking(
                            messages, state["turn_start_idx"]
                        ), False
                    else:
                        # Can't recover from cache - close the loop with synthetic messages
                        # This allows Claude to start a fresh turn with thinking
                        lib_logger.info(
                            "[Thinking Sanitization] Compacted thinking turn detected in tool loop. "
                            "Cache miss - closing loop with synthetic messages to enable fresh thinking turn."
                        )
                        return self._close_tool_loop_for_thinking(messages), False
                else:
                    # Not a compacted turn - genuinely no thinking. Close the loop.
                    lib_logger.info(
                        "[Thinking Sanitization] Closing tool loop with synthetic response. "
                        "Turn did not start with thinking (turn_has_thinking=False). "
                        "This allows thinking to be enabled on the new turn."
                    )
                    return self._close_tool_loop_for_thinking(messages), False
        else:
            # Not in a tool loop - this is the simple case
            # The conversation doesn't end with tool_result, so we're starting fresh.
            #
            # HOWEVER, there's a special case: compaction might have removed the thinking
            # block from the turn start, but Claude still expects it.
            # We detect this by checking if there's an assistant message with tool_calls
            # but no thinking, and the conversation structure suggests thinking was expected.

            # Check if we need to inject a fake thinking block for compaction recovery
            if state["last_assistant_idx"] >= 0:
                last_assistant = messages[state["last_assistant_idx"]]

                if (
                    state["last_assistant_has_tool_calls"]
                    and not state["turn_has_thinking"]
                ):
                    # The turn has functionCall but no thinking at turn start.
                    # This could be:
                    # 1. Compaction removed the thinking block
                    # 2. The original call was made without thinking
                    #
                    # For case 1, we need to close the turn and start fresh.
                    # For case 2, we let the model respond naturally.
                    #
                    # We can detect case 1 if there's evidence thinking was expected:
                    # - The turn_start message has functionCall (typical thinking-enabled flow)
                    # - The content structure suggests a thinking block was stripped

                    # Check if turn_start has the hallmarks of a compacted thinking response
                    turn_start_msg = (
                        messages[state["turn_start_idx"]]
                        if state["turn_start_idx"] >= 0
                        else None
                    )
                    if turn_start_msg and self._looks_like_compacted_thinking_turn(
                        turn_start_msg
                    ):
                        # Try cache recovery first
                        recovered = self._try_recover_thinking_from_cache(
                            messages, state["turn_start_idx"]
                        )
                        if recovered:
                            lib_logger.info(
                                "[Thinking Sanitization] Recovered thinking from cache for compacted turn (not in tool loop)."
                            )
                            return self._strip_old_turn_thinking(
                                messages, state["turn_start_idx"]
                            ), False
                        else:
                            # Can't recover - add synthetic user to start fresh turn (Gemini format)
                            lib_logger.info(
                                "[Thinking Sanitization] Detected compacted turn missing thinking block. "
                                "Adding synthetic user message to start fresh thinking turn."
                            )
                            # Add synthetic user message to trigger new turn with thinking
                            synthetic_user = {
                                "role": "user",
                                "parts": [{"text": "[Continue]"}],
                            }
                            messages.append(synthetic_user)
                            return self._strip_all_thinking_blocks(messages), False
                    else:
                        lib_logger.debug(
                            "[Thinking Sanitization] Last model has functionCall but no thinking. "
                            "This is likely from context compression or non-thinking model. "
                            "New response will include thinking naturally."
                        )

            # Strip thinking from old turns, let new response add thinking naturally
            return self._strip_old_turn_thinking(
                messages, state["last_assistant_idx"]
            ), False

    def _strip_all_thinking_blocks(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove all thinking/reasoning content from messages.

        Handles GEMINI format (after transformation):
        - Role "model" instead of "assistant"
        - "parts" array with "thought": true for thinking
        """
        for msg in messages:
            if msg.get("role") == "model":
                parts = msg.get("parts", [])
                if parts:
                    # Filter out thinking parts (those with "thought": true)
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    # Check if there are still functionCalls remaining
                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        # All parts were thinking - need placeholder for valid structure
                        if not has_function_calls:
                            msg["parts"] = [{"text": ""}]
                        else:
                            msg["parts"] = []  # Will be invalid, but shouldn't happen
                    else:
                        msg["parts"] = filtered
        return messages

    def _strip_old_turn_thinking(
        self, messages: List[Dict[str, Any]], last_model_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Strip thinking from old turns but preserve for the last model turn.

        Per Claude docs: "thinking blocks from previous turns are removed from context"
        This mimics the API behavior and prevents issues.

        Handles GEMINI format: role "model", "parts" with "thought": true
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "model" and i < last_model_idx:
                # Old turn - strip thinking parts
                parts = msg.get("parts", [])
                if parts:
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        msg["parts"] = [{"text": ""}] if not has_function_calls else []
                    else:
                        msg["parts"] = filtered
        return messages

    def _preserve_current_turn_thinking(
        self, messages: List[Dict[str, Any]], last_model_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Preserve thinking only for the current (last) model turn.
        Strip from all previous turns.
        """
        # Same as strip_old_turn_thinking - we keep the last turn intact
        return self._strip_old_turn_thinking(messages, last_model_idx)

    def _preserve_turn_start_thinking(
        self, messages: List[Dict[str, Any]], turn_start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Preserve thinking at the turn start message.

        In multi-message tool loops, the thinking block is at the FIRST model
        message of the turn (turn_start_idx), not the last one. We need to preserve
        thinking from the turn start, and strip it from all older turns.

        Handles GEMINI format: role "model", "parts" with "thought": true
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "model" and i < turn_start_idx:
                # Old turn - strip thinking parts
                parts = msg.get("parts", [])
                if parts:
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        msg["parts"] = [{"text": ""}] if not has_function_calls else []
                    else:
                        msg["parts"] = filtered
        return messages

    def _looks_like_compacted_thinking_turn(self, msg: Dict[str, Any]) -> bool:
        """
        Detect if a message looks like it was compacted from a thinking-enabled turn.

        Heuristics (GEMINI format):
        1. Has functionCall parts (typical thinking flow produces tool calls)
        2. No thinking parts (thought: true)
        3. No text content before functionCall (thinking responses usually have text)

        This is imperfect but helps catch common compaction scenarios.
        """
        parts = msg.get("parts", [])
        if not parts:
            return False

        has_function_call = any(
            isinstance(p, dict) and "functionCall" in p for p in parts
        )

        if not has_function_call:
            return False

        # Check for text content (not thinking)
        has_text = any(
            isinstance(p, dict)
            and "text" in p
            and p.get("text", "").strip()
            and not p.get("thought")  # Exclude thinking text
            for p in parts
        )

        # If we have functionCall but no non-thinking text, likely compacted
        if not has_text:
            return True

        return False

    def _try_recover_thinking_from_cache(
        self, messages: List[Dict[str, Any]], turn_start_idx: int
    ) -> bool:
        """
        Try to recover thinking content from cache for a compacted turn.

        Handles GEMINI format: extracts functionCall for cache key lookup,
        injects thinking as a part with thought: true.

        Returns True if thinking was successfully recovered and injected, False otherwise.
        """
        if turn_start_idx < 0 or turn_start_idx >= len(messages):
            return False

        msg = messages[turn_start_idx]
        parts = msg.get("parts", [])

        # Extract text content and build tool_calls structure for cache key lookup
        text_content = ""
        tool_calls = []

        for part in parts:
            if isinstance(part, dict):
                if "text" in part and not part.get("thought"):
                    text_content = part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Convert to OpenAI tool_calls format for cache key compatibility
                    tool_calls.append(
                        {
                            "id": fc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }
                    )

        # Generate cache key and try to retrieve
        cache_key = self._generate_thinking_cache_key(text_content, tool_calls)
        if not cache_key:
            return False

        cached_json = self._thinking_cache.retrieve(cache_key)
        if not cached_json:
            lib_logger.debug(
                f"[Thinking Sanitization] No cached thinking found for key: {cache_key}"
            )
            return False

        try:
            thinking_data = json.loads(cached_json)
            thinking_text = thinking_data.get("thinking_text", "")
            signature = thinking_data.get("thought_signature", "")

            if not thinking_text or not signature:
                lib_logger.debug(
                    "[Thinking Sanitization] Cached thinking missing text or signature"
                )
                return False

            # Inject the recovered thinking part at the beginning (Gemini format)
            thinking_part = {
                "text": thinking_text,
                "thought": True,
                "thoughtSignature": signature,
            }

            msg["parts"] = [thinking_part] + parts

            lib_logger.debug(
                f"[Thinking Sanitization] Recovered thinking from cache: {len(thinking_text)} chars"
            )
            return True

        except json.JSONDecodeError:
            lib_logger.warning(
                f"[Thinking Sanitization] Failed to parse cached thinking"
            )
            return False

    def _close_tool_loop_for_thinking(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Close an incomplete tool loop by injecting synthetic messages to start a new turn.

        This is used when:
        - We're in a tool loop (conversation ends with functionResponse)
        - The tool call was made WITHOUT thinking (e.g., by Gemini, non-thinking Claude, or compaction stripped it)
        - We NOW want to enable thinking

        Per Claude docs on toggling thinking modes:
        - "If thinking is enabled, the final assistant turn must start with a thinking block"
        - "To toggle thinking, you must complete the assistant turn first"
        - A non-tool-result user message ends the turn and allows a fresh start

        Solution (GEMINI format):
        1. Add synthetic MODEL message to complete the non-thinking turn
        2. Add synthetic USER message to start a NEW turn
        3. Claude will generate thinking for its response to the new turn

        The synthetic messages are minimal and unobtrusive - they just satisfy the
        turn structure requirements without influencing model behavior.
        """
        # Strip any old thinking first
        messages = self._strip_all_thinking_blocks(messages)

        # Count tool results from the end of the conversation (Gemini format)
        tool_result_count = 0
        for msg in reversed(messages):
            if msg.get("role") == "user":
                parts = msg.get("parts", [])
                has_function_response = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )
                if has_function_response:
                    tool_result_count += len(
                        [
                            p
                            for p in parts
                            if isinstance(p, dict) and "functionResponse" in p
                        ]
                    )
                else:
                    break  # Real user message, stop counting
            elif msg.get("role") == "model":
                break  # Stop at the model that made the tool calls

        # Safety check: if no tool results found, this shouldn't have been called
        # But handle gracefully with a generic message
        if tool_result_count == 0:
            lib_logger.warning(
                "[Thinking Sanitization] _close_tool_loop_for_thinking called but no tool results found. "
                "This may indicate malformed conversation history."
            )
            synthetic_model_content = "[Processing previous context.]"
        elif tool_result_count == 1:
            synthetic_model_content = "[Tool execution completed.]"
        else:
            synthetic_model_content = (
                f"[{tool_result_count} tool executions completed.]"
            )

        # Step 1: Inject synthetic MODEL message to complete the non-thinking turn (Gemini format)
        synthetic_model = {
            "role": "model",
            "parts": [{"text": synthetic_model_content}],
        }
        messages.append(synthetic_model)

        # Step 2: Inject synthetic USER message to start a NEW turn (Gemini format)
        # This allows Claude to generate thinking for its response
        # The message is minimal and unobtrusive - just triggers a new turn
        synthetic_user = {
            "role": "user",
            "parts": [{"text": "[Continue]"}],
        }
        messages.append(synthetic_user)

        lib_logger.info(
            f"[Thinking Sanitization] Closed tool loop with synthetic messages. "
            f"Model: '{synthetic_model_content}', User: '[Continue]'. "
            f"Claude will now start a fresh turn with thinking enabled."
        )

        return messages

    # =========================================================================
    # REASONING CONFIGURATION
    # =========================================================================

    def _get_thinking_config(
        self, reasoning_effort: Optional[str], model: str, custom_budget: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration.

        - Gemini 2.5 & Claude: thinkingBudget (integer tokens)
        - Gemini 3: thinkingLevel (string: "low"/"high")
        """
        internal = self._alias_to_internal(model)
        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = internal.startswith("gemini-3-")
        is_claude = self._is_claude(model)

        if not (is_gemini_25 or is_gemini_3 or is_claude):
            return None

        # Gemini 3: String-based thinkingLevel
        if is_gemini_3:
            if reasoning_effort == "low":
                return {"thinkingLevel": "low", "include_thoughts": True}
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 2.5 & Claude: Integer thinkingBudget
        if not reasoning_effort:
            return {"thinkingBudget": -1, "include_thoughts": True}  # Auto

        if reasoning_effort == "disable":
            return {"thinkingBudget": 0, "include_thoughts": False}

        # Model-specific budgets
        if "gemini-2.5-pro" in model or is_claude:
            budgets = {"low": 8192, "medium": 16384, "high": 32768}
        elif "gemini-2.5-flash" in model:
            budgets = {"low": 6144, "medium": 12288, "high": 24576}
        else:
            budgets = {"low": 1024, "medium": 2048, "high": 4096}

        budget = budgets.get(reasoning_effort, -1)
        if not custom_budget:
            budget = budget // 4  # Default to 25% of max output tokens

        return {"thinkingBudget": budget, "include_thoughts": True}

    # =========================================================================
    # MESSAGE TRANSFORMATION (OpenAI → Gemini)
    # =========================================================================

    def _transform_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.

        Handles:
        - System instruction extraction
        - Multi-part content (text, images)
        - Tool calls and responses
        - Claude thinking injection from cache
        - Gemini 3 thoughtSignature preservation
        """
        messages = copy.deepcopy(messages)
        system_instruction = None
        gemini_contents = []

        # Extract system prompts (handle multiple consecutive system messages)
        system_parts = []
        while messages and messages[0].get("role") == "system":
            system_content = messages.pop(0).get("content", "")
            if system_content:
                new_parts = self._parse_content_parts(
                    system_content, _strip_cache_control=True
                )
                system_parts.extend(new_parts)

        if system_parts:
            system_instruction = {"role": "user", "parts": system_parts}

        # Build tool_call_id → name mapping
        tool_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function":
                        tc_id = tc["id"]
                        tc_name = tc["function"]["name"]
                        tool_id_to_name[tc_id] = tc_name
                        # lib_logger.debug(f"[ID Mapping] Registered tool_call: id={tc_id}, name={tc_name}")

        # Convert each message, consolidating consecutive tool responses
        # Per Gemini docs: parallel function responses must be in a single user message
        pending_tool_parts = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []

            # Flush pending tool parts before non-tool message
            if pending_tool_parts and role != "tool":
                gemini_contents.append({"role": "user", "parts": pending_tool_parts})
                pending_tool_parts = []

            if role == "user":
                parts = self._transform_user_message(content)
            elif role == "assistant":
                parts = self._transform_assistant_message(msg, model, tool_id_to_name)
            elif role == "tool":
                tool_parts = self._transform_tool_message(msg, model, tool_id_to_name)
                # Accumulate tool responses instead of adding individually
                pending_tool_parts.extend(tool_parts)
                continue

            if parts:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({"role": gemini_role, "parts": parts})

        # Flush any remaining tool parts
        if pending_tool_parts:
            gemini_contents.append({"role": "user", "parts": pending_tool_parts})

        return system_instruction, gemini_contents

    def _parse_content_parts(
        self, content: Any, _strip_cache_control: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse content into Gemini parts format."""
        parts = []

        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append({"text": text})
                elif item.get("type") == "image_url":
                    image_part = self._parse_image_url(item.get("image_url", {}))
                    if image_part:
                        parts.append(image_part)

        return parts

    def _parse_image_url(self, image_url: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse image URL into Gemini inlineData format."""
        url = image_url.get("url", "")
        if not url.startswith("data:"):
            return None

        try:
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inlineData": {"mimeType": mime_type, "data": data}}
        except Exception as e:
            lib_logger.warning(f"Failed to parse image URL: {e}")
            return None

    def _transform_user_message(self, content: Any) -> List[Dict[str, Any]]:
        """Transform user message content to Gemini parts."""
        return self._parse_content_parts(content)

    def _transform_assistant_message(
        self, msg: Dict[str, Any], model: str, _tool_id_to_name: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform assistant message including tool calls and thinking injection."""
        parts = []
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        reasoning_content = msg.get("reasoning_content")

        # Handle reasoning_content if present (from original Claude response with thinking)
        if reasoning_content and self._is_claude(model):
            # Add thinking part with cached signature
            thinking_part = {
                "text": reasoning_content,
                "thought": True,
            }
            # Try to get signature from cache
            cache_key = self._generate_thinking_cache_key(
                content if isinstance(content, str) else "", tool_calls
            )
            cached_sig = None
            if cache_key:
                cached_json = self._thinking_cache.retrieve(cache_key)
                if cached_json:
                    try:
                        cached_data = json.loads(cached_json)
                        cached_sig = cached_data.get("thought_signature", "")
                    except json.JSONDecodeError:
                        pass

            if cached_sig:
                thinking_part["thoughtSignature"] = cached_sig
                parts.append(thinking_part)
                lib_logger.debug(
                    f"Added reasoning_content with cached signature ({len(reasoning_content)} chars)"
                )
            else:
                # No cached signature - skip the thinking block
                # This can happen if context was compressed and signature was lost
                lib_logger.warning(
                    f"Skipping reasoning_content - no valid signature found. "
                    f"This may cause issues if thinking is enabled."
                )
        elif (
            self._is_claude(model)
            and self._enable_signature_cache
            and not reasoning_content
        ):
            # Fallback: Try to inject cached thinking for Claude (original behavior)
            thinking_parts = self._get_cached_thinking(content, tool_calls)
            parts.extend(thinking_parts)

        # Add regular content
        if isinstance(content, str) and content:
            parts.append({"text": content})

        # Add tool calls
        # Track if we've seen the first function call in this message
        # Per Gemini docs: Only the FIRST parallel function call gets a signature
        first_func_in_msg = True
        for tc in tool_calls:
            if tc.get("type") != "function":
                continue

            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}

            tool_id = tc.get("id", "")
            func_name = tc["function"]["name"]

            # lib_logger.debug(
            #    f"[ID Transform] Converting assistant tool_call to functionCall: "
            #    f"id={tool_id}, name={func_name}"
            # )

            # Add prefix for Gemini 3
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                func_name = f"{self._gemini3_tool_prefix}{func_name}"

            func_part = {
                "functionCall": {"name": func_name, "args": args, "id": tool_id}
            }

            # Add thoughtSignature for Gemini 3
            # Per Gemini docs: Only the FIRST parallel function call gets a signature.
            # Subsequent parallel calls should NOT have a thoughtSignature field.
            if self._is_gemini_3(model):
                sig = tc.get("thought_signature")
                if not sig and tool_id and self._enable_signature_cache:
                    sig = self._signature_cache.retrieve(tool_id)

                if sig:
                    func_part["thoughtSignature"] = sig
                elif first_func_in_msg:
                    # Only add bypass to the first function call if no sig available
                    func_part["thoughtSignature"] = "skip_thought_signature_validator"
                    lib_logger.debug(
                        f"Missing thoughtSignature for first func call {tool_id}, using bypass"
                    )
                # Subsequent parallel calls: no signature field at all

                first_func_in_msg = False

            parts.append(func_part)

        # Safety: ensure we return at least one part to maintain role alternation
        # This handles edge cases like assistant messages that had only thinking content
        # which got stripped, leaving the message otherwise empty
        if not parts:
            # Use a minimal text part - can happen after thinking is stripped
            parts.append({"text": ""})
            lib_logger.debug(
                "[Transform] Added empty text part to maintain role alternation"
            )

        return parts

    def _get_cached_thinking(
        self, content: Any, tool_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Retrieve and format cached thinking content for Claude."""
        parts = []
        msg_text = content if isinstance(content, str) else ""
        cache_key = self._generate_thinking_cache_key(msg_text, tool_calls)

        if not cache_key:
            return parts

        cached_json = self._thinking_cache.retrieve(cache_key)
        if not cached_json:
            return parts

        try:
            thinking_data = json.loads(cached_json)
            thinking_text = thinking_data.get("thinking_text", "")
            sig = thinking_data.get("thought_signature", "")

            if thinking_text:
                thinking_part = {
                    "text": thinking_text,
                    "thought": True,
                    "thoughtSignature": sig or "skip_thought_signature_validator",
                }
                parts.append(thinking_part)
                lib_logger.debug(f"Injected {len(thinking_text)} chars of thinking")
        except json.JSONDecodeError:
            lib_logger.warning(f"Failed to parse cached thinking: {cache_key}")

        return parts

    def _transform_tool_message(
        self, msg: Dict[str, Any], model: str, tool_id_to_name: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform tool response message."""
        tool_id = msg.get("tool_call_id", "")
        func_name = tool_id_to_name.get(tool_id, "unknown_function")
        content = msg.get("content", "{}")

        # Log ID lookup
        if tool_id not in tool_id_to_name:
            lib_logger.warning(
                f"[ID Mismatch] Tool response has ID '{tool_id}' which was not found in tool_id_to_name map. "
                f"Available IDs: {list(tool_id_to_name.keys())}"
            )
        # else:
        # lib_logger.debug(f"[ID Mapping] Tool response matched: id={tool_id}, name={func_name}")

        # Add prefix for Gemini 3
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
            func_name = f"{self._gemini3_tool_prefix}{func_name}"

        try:
            parsed_content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed_content = content

        return [
            {
                "functionResponse": {
                    "name": func_name,
                    "response": {"result": parsed_content},
                    "id": tool_id,
                }
            }
        ]

    # =========================================================================
    # TOOL RESPONSE GROUPING
    # =========================================================================

    def _fix_tool_response_grouping(
        self, contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Antigravity compatibility.

        Converts linear format (call, response, call, response)
        to grouped format (model with calls, user with all responses).

        IMPORTANT: Preserves ID-based pairing to prevent mismatches.
        When IDs don't match, attempts recovery by:
        1. Matching by function name first
        2. Matching by order if names don't match
        3. Inserting placeholder responses if responses are missing
        4. Inserting responses at the CORRECT position (after their corresponding call)
        """
        new_contents = []
        # Each pending group tracks:
        # - ids: expected response IDs
        # - func_names: expected function names (for orphan matching)
        # - insert_after_idx: position in new_contents where model message was added
        pending_groups = []
        collected_responses = {}  # Dict mapping ID -> response_part

        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])

            response_parts = [p for p in parts if "functionResponse" in p]

            if response_parts:
                # Collect responses by ID (ignore duplicates - keep first occurrence)
                for resp in response_parts:
                    resp_id = resp.get("functionResponse", {}).get("id", "")
                    if resp_id:
                        if resp_id in collected_responses:
                            lib_logger.warning(
                                f"[Grouping] Duplicate response ID detected: {resp_id}. "
                                f"Ignoring duplicate - this may indicate malformed conversation history."
                            )
                            continue
                        # lib_logger.debug(
                        #    f"[Grouping] Collected response for ID: {resp_id}"
                        # )
                        collected_responses[resp_id] = resp

                # Try to satisfy pending groups (newest first)
                for i in range(len(pending_groups) - 1, -1, -1):
                    group = pending_groups[i]
                    group_ids = group["ids"]

                    # Check if we have ALL responses for this group
                    if all(gid in collected_responses for gid in group_ids):
                        # Extract responses in the same order as the function calls
                        group_responses = [
                            collected_responses.pop(gid) for gid in group_ids
                        ]
                        new_contents.append({"parts": group_responses, "role": "user"})
                        # lib_logger.debug(
                        #    f"[Grouping] Satisfied group with {len(group_responses)} responses: "
                        #    f"ids={group_ids}"
                        # )
                        pending_groups.pop(i)
                        break
                continue

            if role == "model":
                func_calls = [p for p in parts if "functionCall" in p]
                new_contents.append(content)
                if func_calls:
                    call_ids = [
                        fc.get("functionCall", {}).get("id", "") for fc in func_calls
                    ]
                    call_ids = [cid for cid in call_ids if cid]  # Filter empty IDs

                    # Also extract function names for orphan matching
                    func_names = [
                        fc.get("functionCall", {}).get("name", "") for fc in func_calls
                    ]

                    if call_ids:
                        # lib_logger.debug(
                        #    f"[Grouping] Created pending group expecting {len(call_ids)} responses: "
                        #    f"ids={call_ids}, names={func_names}"
                        # )
                        pending_groups.append(
                            {
                                "ids": call_ids,
                                "func_names": func_names,
                                "insert_after_idx": len(new_contents) - 1,
                            }
                        )
            else:
                new_contents.append(content)

        # Handle remaining groups (shouldn't happen in well-formed conversations)
        # Attempt recovery by matching orphans to unsatisfied calls
        # Process in REVERSE order of insert_after_idx so insertions don't shift indices
        pending_groups.sort(key=lambda g: g["insert_after_idx"], reverse=True)

        for group in pending_groups:
            group_ids = group["ids"]
            group_func_names = group.get("func_names", [])
            insert_idx = group["insert_after_idx"] + 1
            group_responses = []

            lib_logger.debug(
                f"[Grouping Recovery] Processing unsatisfied group: "
                f"ids={group_ids}, names={group_func_names}, insert_at={insert_idx}"
            )

            for i, expected_id in enumerate(group_ids):
                expected_name = group_func_names[i] if i < len(group_func_names) else ""

                if expected_id in collected_responses:
                    # Direct ID match
                    group_responses.append(collected_responses.pop(expected_id))
                    lib_logger.debug(
                        f"[Grouping Recovery] Direct ID match for '{expected_id}'"
                    )
                elif collected_responses:
                    # Try to find orphan with matching function name first
                    matched_orphan_id = None

                    # First pass: match by function name
                    for orphan_id, orphan_resp in collected_responses.items():
                        orphan_name = orphan_resp.get("functionResponse", {}).get(
                            "name", ""
                        )
                        # Match if names are equal, or if orphan has "unknown_function" (can be fixed)
                        if orphan_name == expected_name:
                            matched_orphan_id = orphan_id
                            lib_logger.debug(
                                f"[Grouping Recovery] Matched orphan '{orphan_id}' by name '{orphan_name}'"
                            )
                            break

                    # Second pass: if no name match, try "unknown_function" orphans
                    if not matched_orphan_id:
                        for orphan_id, orphan_resp in collected_responses.items():
                            orphan_name = orphan_resp.get("functionResponse", {}).get(
                                "name", ""
                            )
                            if orphan_name == "unknown_function":
                                matched_orphan_id = orphan_id
                                lib_logger.debug(
                                    f"[Grouping Recovery] Matched unknown_function orphan '{orphan_id}' "
                                    f"to expected '{expected_name}'"
                                )
                                break

                    # Third pass: if still no match, take first available (order-based)
                    if not matched_orphan_id:
                        matched_orphan_id = next(iter(collected_responses))
                        lib_logger.debug(
                            f"[Grouping Recovery] No name match, using first available orphan '{matched_orphan_id}'"
                        )

                    if matched_orphan_id:
                        orphan_resp = collected_responses.pop(matched_orphan_id)

                        # Fix the ID in the response to match the call
                        old_id = orphan_resp["functionResponse"].get("id", "")
                        orphan_resp["functionResponse"]["id"] = expected_id

                        # Fix the name if it was "unknown_function"
                        if (
                            orphan_resp["functionResponse"].get("name")
                            == "unknown_function"
                            and expected_name
                        ):
                            orphan_resp["functionResponse"]["name"] = expected_name
                            lib_logger.info(
                                f"[Grouping Recovery] Fixed function name from 'unknown_function' to '{expected_name}'"
                            )

                        lib_logger.warning(
                            f"[Grouping] Auto-repaired ID mismatch: mapped response '{old_id}' "
                            f"to call '{expected_id}' (function: {expected_name})"
                        )
                        group_responses.append(orphan_resp)
                else:
                    # No responses available - create placeholder
                    placeholder_resp = {
                        "functionResponse": {
                            "name": expected_name or "unknown_function",
                            "response": {
                                "result": {
                                    "error": "Tool response was lost during context processing. "
                                    "This is a recovered placeholder.",
                                    "recovered": True,
                                }
                            },
                            "id": expected_id,
                        }
                    }
                    lib_logger.warning(
                        f"[Grouping Recovery] Created placeholder response for missing tool: "
                        f"id='{expected_id}', name='{expected_name}'"
                    )
                    group_responses.append(placeholder_resp)

            if group_responses:
                # Insert at the correct position (right after the model message with the calls)
                new_contents.insert(
                    insert_idx, {"parts": group_responses, "role": "user"}
                )
                lib_logger.info(
                    f"[Grouping Recovery] Inserted {len(group_responses)} responses at position {insert_idx} "
                    f"(expected {len(group_ids)})"
                )

        # Warn about unmatched responses
        if collected_responses:
            lib_logger.warning(
                f"[Grouping] {len(collected_responses)} unmatched responses remaining: "
                f"ids={list(collected_responses.keys())}"
            )

        return new_contents

    # =========================================================================
    # GEMINI 3 TOOL TRANSFORMATIONS
    # =========================================================================

    def _apply_gemini3_namespace(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add namespace prefix to tool names for Gemini 3."""
        if not tools:
            return tools

        modified = copy.deepcopy(tools)
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                name = func_decl.get("name", "")
                if name:
                    func_decl["name"] = f"{self._gemini3_tool_prefix}{name}"

        return modified

    def _enforce_strict_schema(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforce strict JSON schema for Gemini 3 to prevent hallucinated parameters.

        Adds 'additionalProperties: false' recursively to all object schemas,
        which tells the model it CANNOT add properties not in the schema.
        """
        if not tools:
            return tools

        def enforce_strict(schema: Any) -> Any:
            if not isinstance(schema, dict):
                return schema

            result = {}
            for key, value in schema.items():
                if isinstance(value, dict):
                    result[key] = enforce_strict(value)
                elif isinstance(value, list):
                    result[key] = [
                        enforce_strict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value

            # Add additionalProperties: false to object schemas
            if result.get("type") == "object" and "properties" in result:
                result["additionalProperties"] = False

            return result

        modified = copy.deepcopy(tools)
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                if "parametersJsonSchema" in func_decl:
                    func_decl["parametersJsonSchema"] = enforce_strict(
                        func_decl["parametersJsonSchema"]
                    )

        return modified

    def _inject_signature_into_descriptions(
        self, tools: List[Dict[str, Any]], description_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Inject parameter signatures into tool descriptions for Gemini 3 & Claude."""
        if not tools:
            return tools

        # Use provided prompt or default to Gemini 3 prompt
        prompt_template = description_prompt or self._gemini3_description_prompt

        modified = copy.deepcopy(tools)
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                schema = func_decl.get("parametersJsonSchema", {})
                if not schema:
                    continue

                required = schema.get("required", [])
                properties = schema.get("properties", {})

                if not properties:
                    continue

                param_list = []
                for prop_name, prop_data in properties.items():
                    if not isinstance(prop_data, dict):
                        continue

                    type_hint = self._format_type_hint(prop_data)
                    is_required = prop_name in required
                    param_list.append(
                        f"{prop_name} ({type_hint}{', REQUIRED' if is_required else ''})"
                    )

                if param_list:
                    sig_str = prompt_template.replace("{params}", ", ".join(param_list))
                    func_decl["description"] = (
                        func_decl.get("description", "") + sig_str
                    )

        return modified

    def _format_type_hint(self, prop_data: Dict[str, Any], depth: int = 0) -> str:
        """Format a detailed type hint for a property schema."""
        type_hint = prop_data.get("type", "unknown")

        # Handle enum values - show allowed options
        if "enum" in prop_data:
            enum_vals = prop_data["enum"]
            if len(enum_vals) <= 5:
                return f"string ENUM[{', '.join(repr(v) for v in enum_vals)}]"
            return f"string ENUM[{len(enum_vals)} options]"

        # Handle const values
        if "const" in prop_data:
            return f"string CONST={repr(prop_data['const'])}"

        if type_hint == "array":
            items = prop_data.get("items", {})
            if isinstance(items, dict):
                item_type = items.get("type", "unknown")
                if item_type == "object":
                    nested_props = items.get("properties", {})
                    nested_req = items.get("required", [])
                    if nested_props:
                        nested_list = []
                        for n, d in nested_props.items():
                            if isinstance(d, dict):
                                # Recursively format nested types (limit depth)
                                if depth < 1:
                                    t = self._format_type_hint(d, depth + 1)
                                else:
                                    t = d.get("type", "unknown")
                                req = " REQUIRED" if n in nested_req else ""
                                nested_list.append(f"{n}: {t}{req}")
                        return f"ARRAY_OF_OBJECTS[{', '.join(nested_list)}]"
                    return "ARRAY_OF_OBJECTS"
                return f"ARRAY_OF_{item_type.upper()}"
            return "ARRAY"

        if type_hint == "object":
            nested_props = prop_data.get("properties", {})
            nested_req = prop_data.get("required", [])
            if nested_props and depth < 1:
                nested_list = []
                for n, d in nested_props.items():
                    if isinstance(d, dict):
                        t = d.get("type", "unknown")
                        req = " REQUIRED" if n in nested_req else ""
                        nested_list.append(f"{n}: {t}{req}")
                return f"object{{{', '.join(nested_list)}}}"

        return type_hint

    def _strip_gemini3_prefix(self, name: str) -> str:
        """Strip the Gemini 3 namespace prefix from a tool name."""
        if name and name.startswith(self._gemini3_tool_prefix):
            return name[len(self._gemini3_tool_prefix) :]
        return name

    def _translate_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]], model: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Translates OpenAI's `tool_choice` to Gemini's `toolConfig`.
        Handles Gemini 3 namespace prefixes for specific tool selection.
        """
        if not tool_choice:
            return None

        config = {}
        mode = "AUTO"  # Default to auto
        is_gemini_3 = self._is_gemini_3(model)

        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                mode = "AUTO"
            elif tool_choice == "none":
                mode = "NONE"
            elif tool_choice == "required":
                mode = "ANY"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                # Add Gemini 3 prefix if needed
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = f"{self._gemini3_tool_prefix}{function_name}"

                mode = "ANY"  # Force a call, but only to this function
                config["functionCallingConfig"] = {
                    "mode": mode,
                    "allowedFunctionNames": [function_name],
                }
                return config

        config["functionCallingConfig"] = {"mode": mode}
        return config

    # =========================================================================
    # REQUEST TRANSFORMATION
    # =========================================================================

    def _build_tools_payload(
        self, tools: Optional[List[Dict[str, Any]]], _model: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Build Gemini-format tools from OpenAI tools."""
        if not tools:
            return None

        gemini_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            params = func.get("parameters")

            func_decl = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }

            if params and isinstance(params, dict):
                schema = dict(params)
                schema.pop("$schema", None)
                schema.pop("strict", None)
                schema = _normalize_type_arrays(schema)

                # Workaround: Antigravity/Gemini fails to emit functionCall
                # when tool has empty properties {}. Inject a dummy optional
                # parameter to ensure the tool call is emitted.
                props = schema.get("properties", {})
                if not props:
                    schema["properties"] = {
                        "_": {"type": "string", "description": "Unused"}
                    }

                func_decl["parametersJsonSchema"] = schema
            else:
                # No parameters provided - use default with dummy param
                func_decl["parametersJsonSchema"] = {
                    "type": "object",
                    "properties": {"_": {"type": "string", "description": "Unused"}},
                }

            gemini_tools.append({"functionDeclarations": [func_decl]})

        return gemini_tools or None

    def _transform_to_antigravity_format(
        self,
        gemini_payload: Dict[str, Any],
        model: str,
        project_id: str,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Transform Gemini CLI payload to complete Antigravity format.

        Args:
            gemini_payload: Request in Gemini CLI format
            model: Model name (public alias)
            max_tokens: Max output tokens (including thinking)
            reasoning_effort: Reasoning effort level (determines -thinking variant for Claude)
        """
        internal_model = self._alias_to_internal(model)

        # Map Claude models to their -thinking variant
        # claude-opus-4-5: ALWAYS use -thinking (non-thinking variant doesn't exist)
        # claude-sonnet-4-5: only use -thinking when reasoning_effort is provided
        if self._is_claude(internal_model) and not internal_model.endswith("-thinking"):
            if internal_model == "claude-opus-4-5":
                # Opus 4.5 ALWAYS requires -thinking variant
                internal_model = "claude-opus-4-5-thinking"
            elif internal_model == "claude-sonnet-4-5" and reasoning_effort:
                # Sonnet 4.5 uses -thinking only when reasoning_effort is provided
                internal_model = "claude-sonnet-4-5-thinking"

        # Map gemini-3-pro-preview to -low/-high variant based on thinking config
        if model == "gemini-3-pro-preview" or internal_model == "gemini-3-pro-preview":
            # Check thinking config to determine variant
            thinking_config = gemini_payload.get("generationConfig", {}).get(
                "thinkingConfig", {}
            )
            thinking_level = thinking_config.get("thinkingLevel", "high")
            if thinking_level == "low":
                internal_model = "gemini-3-pro-low"
            else:
                internal_model = "gemini-3-pro-high"

        # Wrap in Antigravity envelope
        antigravity_payload = {
            "project": project_id,  # Will be passed as parameter
            "userAgent": "antigravity",
            "requestId": _generate_request_id(),
            "model": internal_model,
            "request": copy.deepcopy(gemini_payload),
        }

        # Add session ID
        antigravity_payload["request"]["sessionId"] = _generate_session_id()

        # Add default safety settings to prevent content filtering
        # Only add if not already present in the payload
        if "safetySettings" not in antigravity_payload["request"]:
            antigravity_payload["request"]["safetySettings"] = copy.deepcopy(
                DEFAULT_SAFETY_SETTINGS
            )

        # Handle max_tokens - only apply to Claude, or if explicitly set for others
        gen_config = antigravity_payload["request"].get("generationConfig", {})
        is_claude = self._is_claude(model)

        if max_tokens is not None:
            # Explicitly set in request - apply to all models
            gen_config["maxOutputTokens"] = max_tokens
        elif is_claude:
            # Claude model without explicit max_tokens - use default
            gen_config["maxOutputTokens"] = DEFAULT_MAX_OUTPUT_TOKENS
        # For non-Claude models without explicit max_tokens, don't set it

        antigravity_payload["request"]["generationConfig"] = gen_config

        # Set toolConfig based on tool_choice parameter
        tool_config_result = self._translate_tool_choice(tool_choice, model)
        if tool_config_result:
            antigravity_payload["request"]["toolConfig"] = tool_config_result
        else:
            # Default to AUTO if no tool_choice specified
            tool_config = antigravity_payload["request"].setdefault("toolConfig", {})
            func_config = tool_config.setdefault("functionCallingConfig", {})
            func_config["mode"] = "AUTO"

        # Handle Gemini 3 thinking logic
        if not internal_model.startswith("gemini-3-"):
            thinking_config = gen_config.get("thinkingConfig", {})
            if "thinkingLevel" in thinking_config:
                del thinking_config["thinkingLevel"]
                thinking_config["thinkingBudget"] = -1

        # Ensure first function call in each model message has a thoughtSignature for Gemini 3
        # Per Gemini docs: Only the FIRST parallel function call gets a signature
        if internal_model.startswith("gemini-3-"):
            for content in antigravity_payload["request"].get("contents", []):
                if content.get("role") == "model":
                    first_func_seen = False
                    for part in content.get("parts", []):
                        if "functionCall" in part:
                            if not first_func_seen:
                                # First function call in this message - needs a signature
                                if "thoughtSignature" not in part:
                                    part["thoughtSignature"] = (
                                        "skip_thought_signature_validator"
                                    )
                                first_func_seen = True
                            # Subsequent parallel calls: leave as-is (no signature)

        # Claude-specific tool schema transformation
        if internal_model.startswith("claude-sonnet-") or internal_model.startswith(
            "claude-opus-"
        ):
            self._apply_claude_tool_transform(antigravity_payload)

        return antigravity_payload

    def _apply_claude_tool_transform(self, payload: Dict[str, Any]) -> None:
        """Apply Claude-specific tool schema transformations.

        Converts parametersJsonSchema to parameters and applies Claude-specific
        schema cleaning (removes unsupported JSON Schema fields).
        """
        tools = payload["request"].get("tools", [])
        for tool in tools:
            for func_decl in tool.get("functionDeclarations", []):
                if "parametersJsonSchema" in func_decl:
                    params = func_decl["parametersJsonSchema"]
                    params = (
                        _clean_claude_schema(params)
                        if isinstance(params, dict)
                        else params
                    )
                    func_decl["parameters"] = params
                    del func_decl["parametersJsonSchema"]

    # =========================================================================
    # RESPONSE TRANSFORMATION
    # =========================================================================

    def _unwrap_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Gemini response from Antigravity envelope."""
        return response.get("response", response)

    def _gemini_to_openai_chunk(
        self,
        chunk: Dict[str, Any],
        model: str,
        accumulator: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert Gemini response chunk to OpenAI streaming format.

        Args:
            chunk: Gemini API response chunk
            model: Model name
            accumulator: Optional dict to accumulate data for post-processing
        """
        candidates = chunk.get("candidates", [])
        if not candidates:
            return {}

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        reasoning_content = ""
        tool_calls = []
        # Use accumulator's tool_idx if available, otherwise use local counter
        tool_idx = accumulator.get("tool_idx", 0) if accumulator else 0

        for part in content_parts:
            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = (
                part.get("thought") is True
                or str(part.get("thought")).lower() == "true"
            )

            # Accumulate signature for Claude caching
            if has_sig and is_thought and accumulator is not None:
                accumulator["thought_signature"] = part["thoughtSignature"]

            # Skip standalone signature parts
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue

            if has_text:
                text = part["text"]
                if is_thought:
                    reasoning_content += text
                    if accumulator is not None:
                        accumulator["reasoning_content"] += text
                else:
                    text_content += text
                    if accumulator is not None:
                        accumulator["text_content"] += text

            if has_func:
                tool_call = self._extract_tool_call(part, model, tool_idx, accumulator)

                # Store signature for each tool call (needed for parallel tool calls)
                if has_sig:
                    self._handle_tool_signature(tool_call, part["thoughtSignature"])

                tool_calls.append(tool_call)
                tool_idx += 1

        # Build delta
        delta = {}
        if text_content:
            delta["content"] = text_content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["role"] = "assistant"
            # Update tool_idx for next chunk
            if accumulator is not None:
                accumulator["tool_idx"] = tool_idx
        elif text_content or reasoning_content:
            delta["role"] = "assistant"

        # Build usage if present
        usage = self._build_usage(chunk.get("usageMetadata", {}))

        # Mark completion when we see usageMetadata
        if chunk.get("usageMetadata") and accumulator is not None:
            accumulator["is_complete"] = True

        # Build choice - just translate, don't include finish_reason
        # Client will handle finish_reason logic
        choice = {"index": 0, "delta": delta}

        response = {
            "id": chunk.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
        }

        if usage:
            response["usage"] = usage

        return response

    def _gemini_to_openai_non_streaming(
        self, response: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI non-streaming format."""
        candidates = response.get("candidates", [])
        if not candidates:
            return {}

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        reasoning_content = ""
        tool_calls = []
        thought_sig = ""

        for part in content_parts:
            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = (
                part.get("thought") is True
                or str(part.get("thought")).lower() == "true"
            )

            if has_sig and is_thought:
                thought_sig = part["thoughtSignature"]

            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue

            if has_text:
                if is_thought:
                    reasoning_content += part["text"]
                else:
                    text_content += part["text"]

            if has_func:
                tool_call = self._extract_tool_call(part, model, len(tool_calls))

                # Store signature for each tool call (needed for parallel tool calls)
                if has_sig:
                    self._handle_tool_signature(tool_call, part["thoughtSignature"])

                tool_calls.append(tool_call)

        # Cache Claude thinking
        if (
            reasoning_content
            and self._is_claude(model)
            and self._enable_signature_cache
        ):
            self._cache_thinking(
                reasoning_content, thought_sig, text_content, tool_calls
            )

        # Build message
        message = {"role": "assistant"}
        if text_content:
            message["content"] = text_content
        elif not tool_calls:
            message["content"] = ""
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
            message.pop("content", None)

        finish_reason = self._map_finish_reason(
            candidate.get("finishReason"), bool(tool_calls)
        )
        usage = self._build_usage(response.get("usageMetadata", {}))

        # For non-streaming, always include finish_reason (should always be present)
        result = {
            "id": response.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }

        if usage:
            result["usage"] = usage

        return result

    def _extract_tool_call(
        self,
        part: Dict[str, Any],
        model: str,
        index: int,
        accumulator: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract and format a tool call from a response part."""
        func_call = part["functionCall"]
        tool_id = func_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"

        # lib_logger.debug(f"[ID Extraction] Extracting tool call: id={tool_id}, raw_id={func_call.get('id')}")

        tool_name = func_call.get("name", "")
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
            tool_name = self._strip_gemini3_prefix(tool_name)

        raw_args = func_call.get("args", {})
        parsed_args = _recursively_parse_json_strings(raw_args)

        tool_call = {
            "id": tool_id,
            "type": "function",
            "index": index,
            "function": {"name": tool_name, "arguments": json.dumps(parsed_args)},
        }

        if accumulator is not None:
            accumulator["tool_calls"].append(tool_call)

        return tool_call

    def _handle_tool_signature(self, tool_call: Dict, signature: str) -> None:
        """Handle thoughtSignature for a tool call."""
        tool_id = tool_call["id"]

        if self._enable_signature_cache:
            self._signature_cache.store(tool_id, signature)
            lib_logger.debug(f"Stored signature for {tool_id}")

        if self._preserve_signatures_in_client:
            tool_call["thought_signature"] = signature

    def _map_finish_reason(
        self, gemini_reason: Optional[str], has_tool_calls: bool
    ) -> Optional[str]:
        """Map Gemini finish reason to OpenAI format."""
        if not gemini_reason:
            return None
        reason = FINISH_REASON_MAP.get(gemini_reason, "stop")
        return "tool_calls" if has_tool_calls else reason

    def _build_usage(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build usage dict from Gemini usage metadata."""
        if not metadata:
            return None

        prompt = metadata.get("promptTokenCount", 0)
        thoughts = metadata.get("thoughtsTokenCount", 0)
        completion = metadata.get("candidatesTokenCount", 0)

        usage = {
            "prompt_tokens": prompt + thoughts,
            "completion_tokens": completion,
            "total_tokens": metadata.get("totalTokenCount", 0),
        }

        if thoughts > 0:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}

        return usage

    def _cache_thinking(
        self, reasoning: str, signature: str, text: str, tool_calls: List[Dict]
    ) -> None:
        """Cache Claude thinking content."""
        cache_key = self._generate_thinking_cache_key(text, tool_calls)
        if not cache_key:
            return

        data = {
            "thinking_text": reasoning,
            "thought_signature": signature,
            "text_preview": text[:100] if text else "",
            "tool_ids": [tc.get("id", "") for tc in tool_calls],
            "timestamp": time.time(),
        }

        self._thinking_cache.store(cache_key, json.dumps(data))
        lib_logger.debug(f"Cached thinking: {cache_key[:50]}...")

    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================

    async def get_valid_token(self, credential_identifier: str) -> str:
        """Get a valid access token for the credential."""
        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_identifier, creds)
        return creds["access_token"]

    def has_custom_logic(self) -> bool:
        """Antigravity uses custom translation logic."""
        return True

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Get OAuth authorization header."""
        token = await self.get_valid_token(credential_identifier)
        return {"Authorization": f"Bearer {token}"}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models from Antigravity."""
        if not self._enable_dynamic_models:
            lib_logger.debug("Using hardcoded model list")
            return [f"antigravity/{m}" for m in AVAILABLE_MODELS]

        try:
            token = await self.get_valid_token(api_key)
            url = f"{self._get_base_url()}/fetchAvailableModels"

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {
                "project": _generate_project_id(),
                "requestId": _generate_request_id(),
                "userAgent": "antigravity",
            }

            response = await client.post(
                url, json=payload, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model_info in data.get("models", []):
                internal = model_info.get("name", "").replace("models/", "")
                if internal:
                    public = self._internal_to_alias(internal)
                    if public:
                        models.append(f"antigravity/{public}")

            if models:
                lib_logger.info(f"Discovered {len(models)} models")
                return models
        except Exception as e:
            lib_logger.warning(f"Dynamic model discovery failed: {e}")

        return [f"antigravity/{m}" for m in AVAILABLE_MODELS]

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests for Antigravity.

        Main entry point that:
        1. Extracts parameters and transforms messages
        2. Builds Antigravity request payload
        3. Makes API call with fallback logic
        4. Transforms response to OpenAI format
        """
        # Extract parameters
        model = self._strip_provider_prefix(kwargs.get("model", "gemini-2.5-pro"))
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("api_key", ""))
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        reasoning_effort = kwargs.get("reasoning_effort")
        top_p = kwargs.get("top_p")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        custom_budget = kwargs.get("custom_reasoning_budget", False)
        enable_logging = kwargs.pop("enable_request_logging", False)

        # Create logger
        file_logger = AntigravityFileLogger(model, enable_logging)

        # Determine if thinking is enabled for this request
        # Thinking is enabled if reasoning_effort is set (and not "disable") for Claude
        thinking_enabled = False
        if self._is_claude(model):
            # For Claude, thinking is enabled when reasoning_effort is provided and not "disable"
            thinking_enabled = (
                reasoning_effort is not None and reasoning_effort != "disable"
            )

        # Transform messages to Gemini format FIRST
        # This restores thinking from cache if reasoning_content was stripped by client
        system_instruction, gemini_contents = self._transform_messages(messages, model)
        gemini_contents = self._fix_tool_response_grouping(gemini_contents)

        # Sanitize thinking blocks for Claude AFTER transformation
        # Now we can see the full picture including cached thinking that was restored
        # This handles: context compression, model switching, mid-turn thinking toggle
        force_disable_thinking = False
        if self._is_claude(model) and self._enable_thinking_sanitization:
            gemini_contents, force_disable_thinking = (
                self._sanitize_thinking_for_claude(gemini_contents, thinking_enabled)
            )

            # If we're in a mid-turn thinking toggle situation, we MUST disable thinking
            # for this request. Thinking will naturally resume on the next turn.
            if force_disable_thinking:
                thinking_enabled = False
                reasoning_effort = "disable"  # Force disable for this request

        # Build payload
        gemini_payload = {"contents": gemini_contents}

        if system_instruction:
            gemini_payload["system_instruction"] = system_instruction

        # Inject tool usage hardening system instructions
        if tools:
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._gemini3_system_instruction
                )
            elif self._is_claude(model) and self._enable_claude_tool_fix:
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._claude_system_instruction
                )

        # Add generation config
        gen_config = {}
        if top_p is not None:
            gen_config["topP"] = top_p

        # Handle temperature - Gemini 3 defaults to 1 if not explicitly set
        if temperature is not None:
            gen_config["temperature"] = temperature
        elif self._is_gemini_3(model):
            # Gemini 3 performs better with temperature=1 for tool use
            gen_config["temperature"] = 1.0

        thinking_config = self._get_thinking_config(
            reasoning_effort, model, custom_budget
        )
        if thinking_config:
            gen_config.setdefault("thinkingConfig", {}).update(thinking_config)

        if gen_config:
            gemini_payload["generationConfig"] = gen_config

        # Add tools
        gemini_tools = self._build_tools_payload(tools, model)
        if gemini_tools:
            gemini_payload["tools"] = gemini_tools

            # Apply tool transformations
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                # Gemini 3: namespace prefix + strict schema + parameter signatures
                gemini_payload["tools"] = self._apply_gemini3_namespace(
                    gemini_payload["tools"]
                )
                if self._gemini3_enforce_strict_schema:
                    gemini_payload["tools"] = self._enforce_strict_schema(
                        gemini_payload["tools"]
                    )
                gemini_payload["tools"] = self._inject_signature_into_descriptions(
                    gemini_payload["tools"], self._gemini3_description_prompt
                )
            elif self._is_claude(model) and self._enable_claude_tool_fix:
                # Claude: parameter signatures only (no namespace prefix)
                gemini_payload["tools"] = self._inject_signature_into_descriptions(
                    gemini_payload["tools"], self._claude_description_prompt
                )

        # Get access token first (needed for project discovery)
        token = await self.get_valid_token(credential_path)

        # Discover real project ID
        litellm_params = kwargs.get("litellm_params", {}) or {}
        project_id = await self._discover_project_id(
            credential_path, token, litellm_params
        )

        # Transform to Antigravity format with real project ID
        payload = self._transform_to_antigravity_format(
            gemini_payload, model, project_id, max_tokens, reasoning_effort, tool_choice
        )
        file_logger.log_request(payload)

        # Make API call
        base_url = self._get_base_url()
        endpoint = ":streamGenerateContent" if stream else ":generateContent"
        url = f"{base_url}{endpoint}"

        if stream:
            url = f"{url}?alt=sse"

        parsed = urlparse(base_url)
        host = parsed.netloc or base_url.replace("https://", "").replace(
            "http://", ""
        ).rstrip("/")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Host": host,
            "User-Agent": "antigravity/1.11.9",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        try:
            if stream:
                return self._handle_streaming(
                    client, url, headers, payload, model, file_logger
                )
            else:
                return await self._handle_non_streaming(
                    client, url, headers, payload, model, file_logger
                )
        except httpx.HTTPStatusError as e:
            # 429 = Rate limit/quota exhausted - tied to credential, not URL
            # Do NOT retry on different URL, just raise immediately
            if e.response.status_code == 429:
                lib_logger.debug(f"429 quota error - not retrying on fallback URL: {e}")
                raise

            # For other HTTP errors (403, 500, etc.), try fallback URL
            if self._try_next_base_url():
                lib_logger.warning(f"Retrying with fallback URL: {e}")
                url = f"{self._get_base_url()}{endpoint}"
                if stream:
                    return self._handle_streaming(
                        client, url, headers, payload, model, file_logger
                    )
                else:
                    return await self._handle_non_streaming(
                        client, url, headers, payload, model, file_logger
                    )
            raise
        except Exception as e:
            # Non-HTTP errors (network issues, timeouts, etc.) - try fallback URL
            if self._try_next_base_url():
                lib_logger.warning(f"Retrying with fallback URL: {e}")
                url = f"{self._get_base_url()}{endpoint}"
                if stream:
                    return self._handle_streaming(
                        client, url, headers, payload, model, file_logger
                    )
                else:
                    return await self._handle_non_streaming(
                        client, url, headers, payload, model, file_logger
                    )
            raise

    def _inject_tool_hardening_instruction(
        self, payload: Dict[str, Any], instruction_text: str
    ) -> None:
        """Inject tool usage hardening system instruction for Gemini 3 & Claude."""
        if not instruction_text:
            return

        instruction_part = {"text": instruction_text}

        if "system_instruction" in payload:
            existing = payload["system_instruction"]
            if isinstance(existing, dict) and "parts" in existing:
                existing["parts"].insert(0, instruction_part)
            else:
                payload["system_instruction"] = {
                    "role": "user",
                    "parts": [instruction_part, {"text": str(existing)}],
                }
        else:
            payload["system_instruction"] = {
                "role": "user",
                "parts": [instruction_part],
            }

    async def _handle_non_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityFileLogger] = None,
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        response = await client.post(url, headers=headers, json=payload, timeout=600.0)
        response.raise_for_status()

        data = response.json()
        if file_logger:
            file_logger.log_final_response(data)

        gemini_response = self._unwrap_response(data)
        openai_response = self._gemini_to_openai_non_streaming(gemini_response, model)

        return litellm.ModelResponse(**openai_response)

    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityFileLogger] = None,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        # Accumulator tracks state across chunks for caching and tool indexing
        accumulator = {
            "reasoning_content": "",
            "thought_signature": "",
            "text_content": "",
            "tool_calls": [],
            "tool_idx": 0,  # Track tool call index across chunks
            "is_complete": False,  # Track if we received usageMetadata
        }

        async with client.stream(
            "POST", url, headers=headers, json=payload, timeout=600.0
        ) as response:
            if response.status_code >= 400:
                # Read error body so it's available in response.text for logging
                # The actual logging happens in failure_logger via _extract_response_body
                try:
                    await response.aread()
                    # lib_logger.error(
                    #     f"API error {response.status_code}: {error_body.decode()}"
                    # )
                except Exception:
                    pass

            response.raise_for_status()

            async for line in response.aiter_lines():
                if file_logger:
                    file_logger.log_response_chunk(line)

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        gemini_chunk = self._unwrap_response(chunk)
                        openai_chunk = self._gemini_to_openai_chunk(
                            gemini_chunk, model, accumulator
                        )

                        yield litellm.ModelResponse(**openai_chunk)
                    except json.JSONDecodeError:
                        if file_logger:
                            file_logger.log_error(f"Parse error: {data_str[:100]}")
                        continue

        # If stream ended without usageMetadata chunk, emit a final chunk with finish_reason
        # Emit final chunk if stream ended without usageMetadata
        # Client will determine the correct finish_reason based on accumulated state
        if not accumulator.get("is_complete"):
            final_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                # Include minimal usage to signal this is the final chunk
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 1,
                    "total_tokens": 1,
                },
            }
            yield litellm.ModelResponse(**final_chunk)

        # Cache Claude thinking after stream completes
        if (
            self._is_claude(model)
            and self._enable_signature_cache
            and accumulator.get("reasoning_content")
        ):
            self._cache_thinking(
                accumulator["reasoning_content"],
                accumulator["thought_signature"],
                accumulator["text_content"],
                accumulator["tool_calls"],
            )

    async def count_tokens(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Count tokens for the given prompt using Antigravity :countTokens endpoint."""
        try:
            token = await self.get_valid_token(credential_path)
            internal_model = self._alias_to_internal(model)

            # Discover project ID
            project_id = await self._discover_project_id(
                credential_path, token, litellm_params or {}
            )

            system_instruction, contents = self._transform_messages(
                messages, internal_model
            )
            contents = self._fix_tool_response_grouping(contents)

            gemini_payload = {"contents": contents}
            if system_instruction:
                gemini_payload["systemInstruction"] = system_instruction

            gemini_tools = self._build_tools_payload(tools, model)
            if gemini_tools:
                gemini_payload["tools"] = gemini_tools

            antigravity_payload = {
                "project": project_id,
                "userAgent": "antigravity",
                "requestId": _generate_request_id(),
                "model": internal_model,
                "request": gemini_payload,
            }

            url = f"{self._get_base_url()}:countTokens"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            response = await client.post(
                url, headers=headers, json=antigravity_payload, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            unwrapped = self._unwrap_response(data)
            total = unwrapped.get("totalTokens", 0)

            return {"prompt_tokens": total, "total_tokens": total}
        except Exception as e:
            lib_logger.error(f"Token counting failed: {e}")
            return {"prompt_tokens": 0, "total_tokens": 0}
