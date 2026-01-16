# src/rotator_library/providers/utilities/gemini_shared_utils.py
"""
Shared utility functions and constants for Gemini-based providers.

This module contains helper functions used by both GeminiCliProvider and
AntigravityProvider, extracted to reduce code duplication.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================


def env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


# =============================================================================
# API ENDPOINTS
# =============================================================================

# Google Code Assist API endpoint (used by Gemini CLI and Antigravity providers)
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

# Gemini CLI endpoint fallback chain
# Sandbox endpoints may have separate/higher rate limits than production
# Order: sandbox daily -> production (fallback)
GEMINI_CLI_ENDPOINT_FALLBACKS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",  # Sandbox daily
    "https://cloudcode-pa.googleapis.com/v1internal",  # Production fallback
]


# =============================================================================
# GEMINI 3 TOOL RENAMING CONSTANTS
# =============================================================================

# Gemini 3 tool name remapping
# Some tool names trigger internal Gemini behavior that causes issues
# Rename them to avoid conflicts
GEMINI3_TOOL_RENAMES: Dict[str, str] = {
    # "batch": "multi_tool",  # "batch" triggers internal format: call:default_api:...
}
GEMINI3_TOOL_RENAMES_REVERSE: Dict[str, str] = {
    v: k for k, v in GEMINI3_TOOL_RENAMES.items()
}

# Gemini finish reason mapping to OpenAI format
FINISH_REASON_MAP: Dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}

# Default safety settings - disable content filtering for all categories
DEFAULT_SAFETY_SETTINGS: List[Dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
]


# =============================================================================
# SCHEMA TRANSFORMATION FUNCTIONS
# =============================================================================


def inline_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inline local $ref definitions before sanitization.

    Handles JSON Schema $ref resolution for local definitions in $defs or definitions.
    Prevents circular references by tracking seen refs.

    Args:
        schema: JSON schema that may contain $ref references

    Returns:
        Schema with all local $refs inlined
    """
    if not isinstance(schema, dict):
        return schema

    defs = schema.get("$defs", schema.get("definitions", {}))
    if not defs:
        return schema

    def resolve(node, seen=()):
        if not isinstance(node, dict):
            return [resolve(x, seen) for x in node] if isinstance(node, list) else node
        if "$ref" in node:
            ref = node["$ref"]
            if ref in seen:  # Circular - drop it
                return {k: resolve(v, seen) for k, v in node.items() if k != "$ref"}
            for prefix in ("#/$defs/", "#/definitions/"):
                if isinstance(ref, str) and ref.startswith(prefix):
                    name = ref[len(prefix) :]
                    if name in defs:
                        return resolve(copy.deepcopy(defs[name]), seen + (ref,))
            return {k: resolve(v, seen) for k, v in node.items() if k != "$ref"}
        return {k: resolve(v, seen) for k, v in node.items()}

    return resolve(schema)


def normalize_type_arrays(schema: Any) -> Any:
    """
    Normalize type arrays in JSON Schema for Proto-based Gemini API.

    Converts `"type": ["string", "null"]` → `"type": "string", "nullable": true`.
    This is required because Gemini's Proto-based API doesn't support type arrays.

    Args:
        schema: JSON schema that may contain type arrays

    Returns:
        Schema with type arrays normalized to single type + nullable flag
    """
    if isinstance(schema, dict):
        normalized = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                types = value
                if "null" in types:
                    normalized["nullable"] = True
                    remaining_types = [t for t in types if t != "null"]
                    if len(remaining_types) == 1:
                        normalized[key] = remaining_types[0]
                    elif len(remaining_types) > 1:
                        normalized[key] = remaining_types
                    # If no types remain, don't add "type" key
                else:
                    normalized[key] = value[0] if len(value) == 1 else value
            else:
                normalized[key] = normalize_type_arrays(value)
        return normalized
    elif isinstance(schema, list):
        return [normalize_type_arrays(item) for item in schema]
    return schema


def clean_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively clean JSON Schema for Gemini CLI endpoint compatibility.

    Handles:
    - Converts `type: ["type", "null"]` to `type: "type", nullable: true`
    - Removes unsupported properties like `strict`
    - Preserves `additionalProperties` for strict schema enforcement

    Args:
        schema: JSON schema to clean

    Returns:
        Cleaned schema compatible with Gemini CLI API
    """
    if not isinstance(schema, dict):
        return schema

    # Handle nullable types
    if "type" in schema and isinstance(schema["type"], list):
        types = schema["type"]
        if "null" in types:
            schema["nullable"] = True
            remaining_types = [t for t in types if t != "null"]
            if len(remaining_types) == 1:
                schema["type"] = remaining_types[0]
            elif len(remaining_types) > 1:
                schema["type"] = remaining_types
            else:
                del schema["type"]

    # Recurse into properties
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop_schema in schema["properties"].values():
            clean_gemini_schema(prop_schema)

    # Recurse into items (for arrays)
    if "items" in schema and isinstance(schema["items"], dict):
        clean_gemini_schema(schema["items"])

    # Clean up unsupported properties
    schema.pop("strict", None)
    # Note: additionalProperties is preserved for _enforce_strict_schema to handle

    return schema


def recursively_parse_json_strings(
    obj: Any,
    schema: Optional[Dict[str, Any]] = None,
    parse_json_objects: bool = False,
    log_prefix: str = "Gemini",
) -> Any:
    """
    Recursively parse JSON strings in nested data structures.

    Gemini sometimes returns tool arguments with JSON-stringified values:
    {"files": "[{...}]"} instead of {"files": [{...}]}.

    Args:
        obj: The object to process
        schema: Optional JSON schema for the current level (used for schema-aware parsing)
        parse_json_objects: If False (default), don't parse JSON-looking strings into objects.
                           This prevents corrupting string content like write tool's "content" field.
                           If True, parse strings that look like JSON objects/arrays.
        log_prefix: Prefix for log messages (e.g., "GeminiCli", "Antigravity")

    Additionally handles:
    - Malformed double-encoded JSON (extra trailing '}' or ']') - only when parse_json_objects=True
    - Escaped string content (\n, \t, etc.) - always processed
    """
    if isinstance(obj, dict):
        # Get properties schema for looking up field types
        properties_schema = schema.get("properties", {}) if schema else {}
        return {
            k: recursively_parse_json_strings(
                v,
                properties_schema.get(k),
                parse_json_objects,
                log_prefix,
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        # Get items schema for array elements
        items_schema = schema.get("items") if schema else None
        return [
            recursively_parse_json_strings(
                item, items_schema, parse_json_objects, log_prefix
            )
            for item in obj
        ]
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
                    f"[{log_prefix}] Unescaped control chars in string: "
                    f"{len(obj) - len(unescaped)} chars changed. Snippet: {snippet!r}"
                )
                return unescaped
            except (json.JSONDecodeError, ValueError):
                # If unescaping fails, continue with original processing
                pass

        # Only parse JSON strings if explicitly enabled
        if not parse_json_objects:
            return obj

        # Schema-aware parsing: only parse if schema expects object/array, not string
        if schema:
            schema_type = schema.get("type")
            if schema_type == "string":
                # Schema says this should be a string - don't parse it
                return obj
            # Only parse if schema expects object or array
            if schema_type not in ("object", "array", None):
                return obj

        # Check if it looks like JSON (starts with { or [)
        if stripped and stripped[0] in ("{", "["):
            # Try standard parsing first
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    parsed = json.loads(obj)
                    return recursively_parse_json_strings(
                        parsed, schema, parse_json_objects, log_prefix
                    )
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
                            f"[{log_prefix}] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return recursively_parse_json_strings(
                            parsed, schema, parse_json_objects, log_prefix
                        )
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
                            f"[{log_prefix}] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return recursively_parse_json_strings(
                            parsed, schema, parse_json_objects, log_prefix
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
    return obj
