# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

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
# ANTIGRAVITY ENDPOINTS
# =============================================================================

# Antigravity API endpoint constants
# Sandbox endpoints often have different rate limits or newer features
ANTIGRAVITY_ENDPOINT_DAILY = (
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
)
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com/v1internal"
# ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com/v1internal"  # Reserved for future use

# Antigravity endpoint fallback chain for API requests
# Order: sandbox daily -> production (matches CLIProxy/Vibeproxy behavior)
ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    ANTIGRAVITY_ENDPOINT_DAILY,  # Daily sandbox first
    ANTIGRAVITY_ENDPOINT_PROD,  # Production fallback
]

# Endpoint order for loadCodeAssist (project discovery)
# Production first for better project resolution, then fallback to sandbox
ANTIGRAVITY_LOAD_ENDPOINT_ORDER = [
    ANTIGRAVITY_ENDPOINT_PROD,  # Prod first for discovery
    ANTIGRAVITY_ENDPOINT_DAILY,  # Daily fallback
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


# =============================================================================
# TIER NAMING AND PRIORITY CONSTANTS
# =============================================================================
# Shared tier handling for Google/Gemini-based providers (Gemini CLI, Antigravity)
#
# Canonical tier names are uppercase: ULTRA, PRO, FREE
# API returns various formats: g1-pro-tier, standard-tier, free-tier, etc.
# This module normalizes all tier names and provides priority ordering.

# Canonical tier names (short)
TIER_ULTRA = "ULTRA"
TIER_PRO = "PRO"
TIER_FREE = "FREE"

# Full tier names for display (based on tier source/subscription type)
# Used for one-time displays like credential discovery logging
TIER_ID_TO_FULL_NAME: Dict[str, str] = {
    # Google One AI subscription tiers (from paidTier API response)
    "g1-pro-tier": "Google One AI PRO",
    "g1-ultra-tier": "Google One AI ULTRA",
    "g1-free-tier": TIER_FREE,  # Free tiers are just "FREE"
    # Gemini Code Assist subscription tiers
    "gemini-code-assist-pro": "Code Assist PRO",
    "gemini-code-assist-ultra": "Code Assist ULTRA",
    "gemini-code-assist-free": TIER_FREE,
    # Legacy/standard tier names (no special prefix)
    "standard-tier": TIER_PRO,
    "pro-tier": TIER_PRO,
    "ultra-tier": TIER_ULTRA,
    "enterprise-tier": TIER_ULTRA,
    "free-tier": TIER_FREE,
    "legacy-tier": TIER_FREE,
    # Already canonical - return as-is
    TIER_FREE: TIER_FREE,
    TIER_PRO: TIER_PRO,
    TIER_ULTRA: TIER_ULTRA,
}

# Mapping from API/legacy tier names to canonical names
# Handles all known tier name formats from various API responses
TIER_NAME_TO_CANONICAL: Dict[str, str] = {
    # Legacy Python names
    "free-tier": TIER_FREE,
    "legacy-tier": TIER_FREE,  # Legacy is treated as free
    "standard-tier": TIER_PRO,
    "pro-tier": TIER_PRO,
    "ultra-tier": TIER_ULTRA,
    "enterprise-tier": TIER_ULTRA,
    # Google One AI tier names (from paidTier API response)
    "g1-pro-tier": TIER_PRO,
    "g1-ultra-tier": TIER_ULTRA,
    "g1-free-tier": TIER_FREE,
    # Gemini Code Assist tier names
    "gemini-code-assist-pro": TIER_PRO,
    "gemini-code-assist-ultra": TIER_ULTRA,
    "gemini-code-assist-free": TIER_FREE,
    # Already canonical (uppercase)
    TIER_FREE: TIER_FREE,
    TIER_PRO: TIER_PRO,
    TIER_ULTRA: TIER_ULTRA,
}

# Reverse mapping for backwards compatibility (canonical -> legacy)
CANONICAL_TO_LEGACY: Dict[str, str] = {
    TIER_FREE: "free-tier",
    TIER_PRO: "standard-tier",
    TIER_ULTRA: "enterprise-tier",
}

# Free tier identifiers (all naming conventions)
FREE_TIER_IDS: set = {TIER_FREE, "free-tier", "legacy-tier", "g1-free-tier"}

# Tier priorities for credential selection (lower number = higher priority)
# ULTRA (Google One AI Premium) > PRO (Google One AI / Paid) > FREE
TIER_PRIORITIES: Dict[str, int] = {
    # Canonical names
    TIER_ULTRA: 1,  # Highest priority - Google One AI Premium
    TIER_PRO: 2,  # Standard paid tier - Google One AI
    TIER_FREE: 3,  # Free tier
    # API/legacy names mapped to same priorities for backwards compatibility
    "g1-ultra-tier": 1,
    "g1-pro-tier": 2,
    "standard-tier": 2,
    "free-tier": 3,
    "legacy-tier": 10,  # Legacy/unknown treated as lowest
    "unknown": 10,
}

# Default priority for tiers not in the mapping
DEFAULT_TIER_PRIORITY: int = 10


# =============================================================================
# TIER HELPER FUNCTIONS
# =============================================================================


def normalize_tier_name(tier_id: Optional[str]) -> Optional[str]:
    """
    Normalize tier name to canonical format (ULTRA, PRO, FREE).

    Supports all tier name formats:
    - Legacy Python names: free-tier, standard-tier, legacy-tier
    - Google One AI names: g1-pro-tier, g1-ultra-tier
    - Gemini Code Assist names: gemini-code-assist-pro
    - Already canonical: FREE, PRO, ULTRA

    Args:
        tier_id: Tier identifier from API response or config

    Returns:
        Canonical tier name (ULTRA, PRO, FREE) or original if unknown
    """
    if not tier_id:
        return None
    return TIER_NAME_TO_CANONICAL.get(tier_id, tier_id)


def is_free_tier(tier_id: Optional[str]) -> bool:
    """
    Check if tier is a free tier (any naming convention).

    Args:
        tier_id: Tier identifier to check

    Returns:
        True if tier is free, False otherwise
    """
    if not tier_id:
        return False
    return tier_id in FREE_TIER_IDS or normalize_tier_name(tier_id) == TIER_FREE


def is_paid_tier(tier_id: Optional[str]) -> bool:
    """
    Check if tier is a paid tier (PRO or ULTRA).

    Args:
        tier_id: Tier identifier to check

    Returns:
        True if tier is paid (PRO or ULTRA), False otherwise
    """
    if not tier_id or tier_id == "unknown":
        return False
    canonical = normalize_tier_name(tier_id)
    return canonical in (TIER_PRO, TIER_ULTRA)


def get_tier_priority(tier_id: Optional[str]) -> int:
    """
    Get priority for a tier (lower number = higher priority).

    Priority order: ULTRA (1) > PRO (2) > FREE (3) > unknown (10)

    Args:
        tier_id: Tier identifier

    Returns:
        Priority number (1-10), lower is better
    """
    if not tier_id:
        return DEFAULT_TIER_PRIORITY
    # Try direct lookup first (handles both canonical and API names)
    if tier_id in TIER_PRIORITIES:
        return TIER_PRIORITIES[tier_id]
    # Normalize and try again
    canonical = normalize_tier_name(tier_id)
    return TIER_PRIORITIES.get(canonical, DEFAULT_TIER_PRIORITY)


def format_tier_for_display(tier_id: Optional[str]) -> str:
    """
    Format tier name for display (lowercase canonical).

    Args:
        tier_id: Tier identifier

    Returns:
        Display-friendly tier name: "ultra", "pro", "free", or "unknown"
    """
    if not tier_id:
        return "unknown"
    canonical = normalize_tier_name(tier_id)
    if canonical in (TIER_ULTRA, TIER_PRO, TIER_FREE):
        return canonical.lower()
    return "unknown"


def get_tier_full_name(tier_id: Optional[str]) -> str:
    """
    Get the full/descriptive tier name for display.

    Used for one-time displays like credential discovery logging where
    we want to show the subscription source (e.g., "Google One AI PRO").

    Args:
        tier_id: Original tier identifier from API response (e.g., "g1-pro-tier")

    Returns:
        Full tier name (e.g., "Google One AI PRO") or canonical short name as fallback
    """
    if not tier_id:
        return "unknown"
    # Try direct lookup for full name
    if tier_id in TIER_ID_TO_FULL_NAME:
        return TIER_ID_TO_FULL_NAME[tier_id]
    # Fallback to canonical short name
    canonical = normalize_tier_name(tier_id)
    return canonical if canonical else tier_id


# =============================================================================
# PROJECT ID EXTRACTION
# =============================================================================


def extract_project_id_from_response(
    data: Dict[str, Any], key: str = "cloudaicompanionProject"
) -> Optional[str]:
    """
    Extract project ID from API response, handling both string and object formats.

    The API may return cloudaicompanionProject as either:
    - A string: "project-id-123"
    - An object: {"id": "project-id-123", ...}

    Args:
        data: API response data
        key: Key to extract from (default: "cloudaicompanionProject")

    Returns:
        Project ID string or None if not found
    """
    value = data.get(key)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        return value.get("id")
    return None


# =============================================================================
# CREDENTIAL LOADING HELPERS
# =============================================================================


def load_persisted_project_metadata(
    credential_path: str,
    credential_index: Optional[int],
    credentials_cache: Dict[str, Any],
    project_id_cache: Dict[str, str],
    project_tier_cache: Dict[str, str],
    tier_full_cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Load persisted project_id and tier from credential file or env cache.

    This helper handles the common pattern of checking for already-persisted
    project metadata in both file-based and env-based credentials.

    Args:
        credential_path: Path to the credential (file path or env:// path)
        credential_index: Result of _parse_env_credential_path() - None for file, int for env
        credentials_cache: Dict of loaded credentials (for env-based)
        project_id_cache: Dict to populate with project_id
        project_tier_cache: Dict to populate with tier
        tier_full_cache: Optional dict to populate with tier_full (full display name)

    Returns:
        Project ID if found and cached, None otherwise (caller should do discovery)
    """
    if credential_index is None:
        # File-based credentials: load from file
        try:
            with open(credential_path, "r") as f:
                creds = json.load(f)

            metadata = creds.get("_proxy_metadata", {})
            persisted_project_id = metadata.get("project_id")
            persisted_tier = metadata.get("tier")
            persisted_tier_full = metadata.get("tier_full")

            if persisted_project_id:
                lib_logger.debug(
                    f"Loaded persisted project ID from credential file: {persisted_project_id}"
                )
                project_id_cache[credential_path] = persisted_project_id

                # Also load tier if available
                if persisted_tier:
                    project_tier_cache[credential_path] = persisted_tier
                    lib_logger.debug(f"Loaded persisted tier: {persisted_tier}")

                # Load tier_full if available and cache provided
                if persisted_tier_full and tier_full_cache is not None:
                    tier_full_cache[credential_path] = persisted_tier_full
                    lib_logger.debug(
                        f"Loaded persisted tier_full: {persisted_tier_full}"
                    )

                return persisted_project_id
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            lib_logger.debug(f"Could not load persisted project ID from file: {e}")
    else:
        # Env-based credentials: load from credentials cache
        # The credentials were already loaded by _load_from_env() which reads
        # {PREFIX}_{N}_PROJECT_ID and {PREFIX}_{N}_TIER into _proxy_metadata
        if credential_path in credentials_cache:
            creds = credentials_cache[credential_path]
            metadata = creds.get("_proxy_metadata", {})
            env_project_id = metadata.get("project_id")
            env_tier = metadata.get("tier")
            env_tier_full = metadata.get("tier_full")

            if env_project_id:
                lib_logger.debug(
                    f"Loaded project ID from env credential metadata: {env_project_id}"
                )
                project_id_cache[credential_path] = env_project_id

                if env_tier:
                    project_tier_cache[credential_path] = env_tier
                    lib_logger.debug(
                        f"Loaded tier from env credential metadata: {env_tier}"
                    )

                # Load tier_full if available and cache provided
                if env_tier_full and tier_full_cache is not None:
                    tier_full_cache[credential_path] = env_tier_full
                    lib_logger.debug(
                        f"Loaded tier_full from env credential metadata: {env_tier_full}"
                    )

                return env_project_id

    return None


# =============================================================================
# ENV FILE HELPERS
# =============================================================================


def build_project_tier_env_lines(
    creds: Dict[str, Any], env_prefix: str, cred_number: int
) -> List[str]:
    """
    Build env lines for project_id and tier from credential metadata.

    Used by Google OAuth providers (Gemini CLI, Antigravity) to generate
    environment variable lines for project and tier information.

    Args:
        creds: Credential dict containing _proxy_metadata
        env_prefix: Environment variable prefix (e.g., "GEMINI_CLI", "ANTIGRAVITY")
        cred_number: Credential number for env var naming

    Returns:
        List of env lines like ["PREFIX_N_PROJECT_ID=...", "PREFIX_N_TIER=..."]
    """
    lines = []
    metadata = creds.get("_proxy_metadata", {})
    prefix = f"{env_prefix}_{cred_number}"

    project_id = metadata.get("project_id", "")
    tier = metadata.get("tier", "")
    tier_full = metadata.get("tier_full", "")

    if project_id:
        lines.append(f"{prefix}_PROJECT_ID={project_id}")
    if tier:
        lines.append(f"{prefix}_TIER={tier}")
    if tier_full:
        lines.append(f"{prefix}_TIER_FULL={tier_full}")

    return lines
