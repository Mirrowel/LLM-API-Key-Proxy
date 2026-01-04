# src/rotator_library/providers/gemini_cli_provider.py

import copy
import json
import httpx
import logging
import time
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple
from .provider_interface import ProviderInterface
from .gemini_auth_base import GeminiAuthBase
from .provider_cache import ProviderCache
from .utilities.gemini_cli_quota_tracker import GeminiCliQuotaTracker
from .antigravity_provider import GEMINI3_TOOL_RENAMES, GEMINI3_TOOL_RENAMES_REVERSE
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..utils.paths import get_logs_dir, get_cache_dir
import litellm
from litellm.exceptions import RateLimitError
from ..error_handler import extract_retry_after_from_body
import os
from pathlib import Path
import uuid
from datetime import datetime

lib_logger = logging.getLogger("rotator_library")


def _get_gemini_cli_logs_dir() -> Path:
    """Get the Gemini CLI logs directory."""
    logs_dir = get_logs_dir() / "gemini_cli_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _get_gemini_cli_cache_dir() -> Path:
    """Get the Gemini CLI cache directory."""
    return get_cache_dir(subdir="gemini_cli")


def _get_gemini3_signature_cache_file() -> Path:
    """Get the Gemini 3 signature cache file path."""
    return _get_gemini_cli_cache_dir() / "gemini3_signatures.json"


class _GeminiCliFileLogger:
    """A simple file logger for a single Gemini CLI transaction."""

    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = str(uuid.uuid4())
        # Sanitize model name for directory
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        self.log_dir = (
            _get_gemini_cli_logs_dir() / f"{timestamp}_{safe_model_name}_{request_id}"
        )
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create Gemini CLI log directory: {e}")
            self.enabled = False

    def log_request(self, payload: Dict[str, Any]):
        """Logs the request payload sent to Gemini."""
        if not self.enabled:
            return
        try:
            with open(
                self.log_dir / "request_payload.json", "w", encoding="utf-8"
            ) as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"_GeminiCliFileLogger: Failed to write request: {e}")

    def log_response_chunk(self, chunk: str):
        """Logs a raw chunk from the Gemini response stream."""
        if not self.enabled:
            return
        try:
            with open(self.log_dir / "response_stream.log", "a", encoding="utf-8") as f:
                f.write(chunk + "\n")
        except Exception as e:
            lib_logger.error(
                f"_GeminiCliFileLogger: Failed to write response chunk: {e}"
            )

    def log_error(self, error_message: str):
        """Logs an error message."""
        if not self.enabled:
            return
        try:
            with open(self.log_dir / "error.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] {error_message}\n")
        except Exception as e:
            lib_logger.error(f"_GeminiCliFileLogger: Failed to write error: {e}")

    def log_final_response(self, response_data: Dict[str, Any]):
        """Logs the final, reassembled response."""
        if not self.enabled:
            return
        try:
            with open(self.log_dir / "final_response.json", "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(
                f"_GeminiCliFileLogger: Failed to write final response: {e}"
            )


CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]

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

# Gemini finish reason mapping
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}


def _recursively_parse_json_strings(
    obj: Any,
    schema: Optional[Dict[str, Any]] = None,
    parse_json_objects: bool = False,
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

    Additionally handles:
    - Malformed double-encoded JSON (extra trailing '}' or ']') - only when parse_json_objects=True
    - Escaped string content (\n, \t, etc.) - always processed
    """
    if isinstance(obj, dict):
        # Get properties schema for looking up field types
        properties_schema = schema.get("properties", {}) if schema else {}
        return {
            k: _recursively_parse_json_strings(
                v,
                properties_schema.get(k),
                parse_json_objects,
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        # Get items schema for array elements
        items_schema = schema.get("items") if schema else None
        return [
            _recursively_parse_json_strings(item, items_schema, parse_json_objects)
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
                    f"[GeminiCli] Unescaped control chars in string: "
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
                    return _recursively_parse_json_strings(
                        parsed, schema, parse_json_objects
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
                            f"[GeminiCli] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return _recursively_parse_json_strings(
                            parsed, schema, parse_json_objects
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
                            f"[GeminiCli] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return _recursively_parse_json_strings(
                            parsed, schema, parse_json_objects
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
    return obj


def _inline_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline local $ref definitions before sanitization."""
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


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


class GeminiCliProvider(GeminiAuthBase, GeminiCliQuotaTracker, ProviderInterface):
    skip_cost_calculation = True

    # Sequential mode - stick with one credential until it gets a 429, then switch
    default_rotation_mode: str = "sequential"

    # =========================================================================
    # TIER CONFIGURATION
    # =========================================================================

    # Provider name for env var lookups (QUOTA_GROUPS_GEMINI_CLI_*)
    provider_env_name: str = "gemini_cli"

    # Tier name -> priority mapping (Single Source of Truth)
    # Same tier names as Antigravity (coincidentally), but defined separately
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

    # Gemini CLI uses default daily reset - no custom usage_reset_configs
    # (Empty dict means inherited get_usage_reset_config returns None)

    # No quota groups defined for Gemini CLI
    # (Models don't share quotas)

    # Priority-based concurrency multipliers
    # Same structure as Antigravity (by coincidence, tiers share naming)
    # Priority 1 (paid ultra): 5x concurrent requests
    # Priority 2 (standard paid): 3x concurrent requests
    # Others: 1x (no sequential fallback, uses global default)
    default_priority_multipliers = {1: 5, 2: 3}

    # No sequential fallback for Gemini CLI (uses balanced mode default)
    # default_sequential_fallback_multiplier = 1  (inherited from ProviderInterface)

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Gemini CLI rate limit/quota errors.

        Handles the Gemini CLI error format which embeds reset time in the message:
        "You have exhausted your capacity on this model. Your quota will reset after 2s."

        Unlike Antigravity which uses structured RetryInfo/quotaResetDelay metadata,
        Gemini CLI embeds the reset time in a human-readable message.

        Example error format:
        {
          "error": {
            "code": 429,
            "message": "You have exhausted your capacity on this model. Your quota will reset after 2s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
              {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "RATE_LIMIT_EXCEEDED",
                "domain": "cloudcode-pa.googleapis.com",
                "metadata": { "uiMessage": "true", "model": "gemini-3-pro-preview" }
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
                "reason": str | None,
                "reset_timestamp": str | None,
                "quota_reset_timestamp": float | None,
            }
        """
        import re as regex_module

        # Get error body from exception if not provided
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, "body"):
                body = str(error.body)
            if not body and hasattr(error, "message"):
                body = str(error.message)
            if not body:
                body = str(error)

        if not body:
            return None

        result = {
            "retry_after": None,
            "reason": None,
            "reset_timestamp": None,
            "quota_reset_timestamp": None,
        }

        # 1. Try to extract retry time from human-readable message
        # Pattern: "Your quota will reset after 2s." or "quota will reset after 156h14m36s"
        retry_after = extract_retry_after_from_body(body)
        if retry_after:
            result["retry_after"] = retry_after

        # 2. Try to parse JSON to get structured details (reason, any RetryInfo fallback)
        try:
            json_match = regex_module.search(r"\{[\s\S]*\}", body)
            if json_match:
                data = json.loads(json_match.group(0))
                error_obj = data.get("error", data)
                details = error_obj.get("details", [])

                for detail in details:
                    detail_type = detail.get("@type", "")

                    # Extract reason from ErrorInfo
                    if "ErrorInfo" in detail_type:
                        if not result["reason"]:
                            result["reason"] = detail.get("reason")
                        # Check metadata for any additional timing info
                        metadata = detail.get("metadata", {})
                        quota_delay = metadata.get("quotaResetDelay")
                        if quota_delay and not result["retry_after"]:
                            parsed = GeminiCliProvider._parse_duration(quota_delay)
                            if parsed:
                                result["retry_after"] = parsed

                    # Check for RetryInfo (fallback, in case format changes)
                    if "RetryInfo" in detail_type and not result["retry_after"]:
                        retry_delay = detail.get("retryDelay")
                        if retry_delay:
                            parsed = GeminiCliProvider._parse_duration(retry_delay)
                            if parsed:
                                result["retry_after"] = parsed

        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        # Return None if we couldn't extract retry_after
        if not result["retry_after"]:
            return None

        return result

    @staticmethod
    def _parse_duration(duration_str: str) -> Optional[int]:
        """
        Parse duration strings like '2s', '156h14m36.73s', '515092.73s' to seconds.

        Args:
            duration_str: Duration string to parse

        Returns:
            Total seconds as integer, or None if parsing fails
        """
        import re as regex_module

        if not duration_str:
            return None

        # Handle pure seconds format: "515092.730699158s" or "2s"
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

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        # NOTE: project_id_cache and project_tier_cache are inherited from GeminiAuthBase

        # Quota refresh interval (mirrors Antigravity pattern)
        self._quota_refresh_interval = _env_int(
            "GEMINI_CLI_QUOTA_REFRESH_INTERVAL", 300
        )

        # Track whether initial quota fetch has been done (for background job)
        self._initial_quota_fetch_done = False

        # Gemini 3 configuration from environment
        memory_ttl = _env_int("GEMINI_CLI_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = _env_int("GEMINI_CLI_SIGNATURE_DISK_TTL", 86400)

        # Initialize signature cache for Gemini 3 thoughtSignatures
        self._signature_cache = ProviderCache(
            _get_gemini3_signature_cache_file(),
            memory_ttl,
            disk_ttl,
            env_prefix="GEMINI_CLI_SIGNATURE",
        )

        # Gemini 3 feature flags
        self._preserve_signatures_in_client = _env_bool(
            "GEMINI_CLI_PRESERVE_THOUGHT_SIGNATURES", True
        )
        self._enable_signature_cache = _env_bool(
            "GEMINI_CLI_ENABLE_SIGNATURE_CACHE", True
        )
        self._enable_gemini3_tool_fix = _env_bool("GEMINI_CLI_GEMINI3_TOOL_FIX", True)
        self._gemini3_enforce_strict_schema = _env_bool(
            "GEMINI_CLI_GEMINI3_STRICT_SCHEMA", True
        )
        # Toggle for JSON string parsing in tool call arguments
        # NOTE: This is possibly redundant - modern Gemini models may not need this fix.
        # Disabled by default. Enable if you see JSON-stringified values in tool args.
        self._enable_json_string_parsing = _env_bool(
            "GEMINI_CLI_ENABLE_JSON_STRING_PARSING", False
        )

        # Gemini 3 tool fix configuration
        self._gemini3_tool_prefix = os.getenv(
            "GEMINI_CLI_GEMINI3_TOOL_PREFIX", "gemini3_"
        )
        self._gemini3_description_prompt = os.getenv(
            "GEMINI_CLI_GEMINI3_DESCRIPTION_PROMPT",
            "\n\n⚠️ STRICT PARAMETERS (use EXACTLY as shown): {params}. Do NOT use parameters from your training data - use ONLY these parameter names.",
        )
        self._gemini3_system_instruction = os.getenv(
            "GEMINI_CLI_GEMINI3_SYSTEM_INSTRUCTION", DEFAULT_GEMINI3_SYSTEM_INSTRUCTION
        )

        lib_logger.debug(
            f"GeminiCli config: signatures_in_client={self._preserve_signatures_in_client}, "
            f"cache={self._enable_signature_cache}, gemini3_fix={self._enable_gemini3_tool_fix}, "
            f"gemini3_strict_schema={self._gemini3_enforce_strict_schema}"
        )

        # Quota tracking instance variables (required by GeminiCliQuotaTracker mixin)
        self._learned_costs: Dict[str, Dict[str, float]] = {}
        self._learned_costs_loaded: bool = False

    # =========================================================================
    # CREDENTIAL TIER LOOKUP (Provider-specific - uses cache)
    # =========================================================================
    #
    # NOTE: get_credential_priority() is now inherited from ProviderInterface.
    # It uses get_credential_tier_name() to get the tier and resolve priority
    # from the tier_priorities class attribute.
    # =========================================================================

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
        Gemini 3 requires paid tier (priority 1).

        Args:
            model: The model name (with or without provider prefix)

        Returns:
            Minimum required priority level or None if no restrictions
        """
        model_name = model.split("/")[-1].replace(":thinking", "")

        # Gemini 3 requires paid tier
        if model_name.startswith("gemini-3-"):
            return 2  # Only priority 2 (paid) credentials

        return None  # All other models have no restrictions

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Load persisted tier information from credential files at startup.

        This ensures all credential priorities are known before any API calls,
        preventing unknown credentials from getting priority 999.

        For credentials without persisted tier info (new or corrupted), performs
        full discovery to ensure proper prioritization in sequential rotation mode.
        """
        # Step 1: Load persisted tiers from files
        await self._load_persisted_tiers(credential_paths)

        # Step 2: Identify credentials still missing tier info
        credentials_needing_discovery = [
            path
            for path in credential_paths
            if path not in self.project_tier_cache
            and self._parse_env_credential_path(path) is None  # Skip env:// paths
        ]

        if not credentials_needing_discovery:
            return  # All credentials have tier info

        lib_logger.info(
            f"GeminiCli: Discovering tier info for {len(credentials_needing_discovery)} credential(s)..."
        )

        # Step 3: Perform discovery for each missing credential (sequential to avoid rate limits)
        for credential_path in credentials_needing_discovery:
            try:
                auth_header = await self.get_auth_header(credential_path)
                access_token = auth_header["Authorization"].split(" ")[1]
                await self._discover_project_id(
                    credential_path, access_token, litellm_params={}
                )
                discovered_tier = self.project_tier_cache.get(
                    credential_path, "unknown"
                )
                lib_logger.debug(
                    f"Discovered tier '{discovered_tier}' for {Path(credential_path).name}"
                )
            except Exception as e:
                lib_logger.warning(
                    f"Failed to discover tier for {Path(credential_path).name}: {e}. "
                    f"Credential will use default priority."
                )

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
                f"GeminiCli: Loaded {len(loaded)} credential tiers from disk: "
                + ", ".join(
                    f"{tier}={count}" for tier, count in sorted(tier_counts.items())
                )
            )

        return loaded

    # =========================================================================
    # BACKGROUND JOB INTERFACE (Quota Baseline Refresh)
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return background job configuration for quota baseline refresh.

        The quota baseline refresh fetches current quota status from the
        Google Code Assist API and stores it in UsageManager for accurate
        quota display in the viewer.

        Returns:
            Dict with job configuration, or None to disable background jobs.
        """
        return {
            "interval": self._quota_refresh_interval,  # default 300s (5 min)
            "name": "gemini_cli_quota_refresh",
            "run_on_start": True,  # fetch baselines immediately at startup
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota baselines for credentials.

        On first run (startup): Fetches quota for ALL credentials to establish baselines.
        On subsequent runs: Only fetches for credentials used since last refresh.

        Handles both file paths and env:// credential formats.

        Args:
            usage_manager: UsageManager instance to store baselines
            credentials: List of credential paths (file paths or env:// URIs)
        """
        if not credentials:
            return

        if not self._initial_quota_fetch_done:
            # First run: fetch ALL credentials to establish baselines
            lib_logger.info(
                f"GeminiCli: Fetching initial quota baselines for {len(credentials)} credentials..."
            )
            quota_results = await self.fetch_initial_baselines(credentials)
            self._initial_quota_fetch_done = True
        else:
            # Subsequent runs: only recently used credentials (incremental updates)
            usage_data = await usage_manager._get_usage_data_snapshot()
            quota_results = await self.refresh_active_quota_baselines(
                credentials, usage_data
            )

        if not quota_results:
            return

        # Store new baselines in UsageManager
        stored = await self._store_baselines_to_usage_manager(
            quota_results, usage_manager
        )
        if stored > 0:
            lib_logger.debug(
                f"GeminiCli quota refresh: updated {stored} model baselines"
            )

    # NOTE: _post_auth_discovery() is inherited from GeminiAuthBase

    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        model_name = model.split("/")[-1].replace(":thinking", "")
        return model_name.startswith("gemini-3-")

    def _strip_gemini3_prefix(self, name: str) -> str:
        """
        Strip the Gemini 3 namespace prefix from a tool name.

        Also reverses any tool renames that were applied to avoid Gemini conflicts.
        """
        if name and name.startswith(self._gemini3_tool_prefix):
            stripped = name[len(self._gemini3_tool_prefix) :]
            # Reverse any renames
            return GEMINI3_TOOL_RENAMES_REVERSE.get(stripped, stripped)
        return name

    # NOTE: _discover_project_id() and _persist_project_metadata() are inherited from GeminiAuthBase

    def _check_mixed_tier_warning(self):
        """Check if mixed free/paid tier credentials are loaded and emit warning."""
        if not self.project_tier_cache:
            return  # No tiers loaded yet

        tiers = set(self.project_tier_cache.values())
        if len(tiers) <= 1:
            return  # All same tier or only one credential

        # Define paid vs free tiers
        free_tiers = {"free-tier", "legacy-tier", "unknown"}
        paid_tiers = tiers - free_tiers

        # Check if we have both free and paid
        has_free = bool(tiers & free_tiers)
        has_paid = bool(paid_tiers)

        if has_free and has_paid:
            lib_logger.warning(
                f"Mixed Gemini tier credentials detected! You have both free-tier and paid-tier "
                f"(e.g., gemini-advanced) credentials loaded. Tiers found: {', '.join(sorted(tiers))}. "
                f"This may cause unexpected behavior with model availability and rate limits."
            )

    def has_custom_logic(self) -> bool:
        return True

    def _cli_preview_fallback_order(self, model: str) -> List[str]:
        """
        Returns a list of model names to try in order for rate limit fallback.
        First model in list is the original model, subsequent models are fallback options.

        Since all fallbacks have been deprecated, this now only returns the base model.
        The fallback logic will check if there are actual fallbacks available.
        """
        # Remove provider prefix if present
        model_name = model.split("/")[-1].replace(":thinking", "")

        # Define fallback chains for models with preview versions
        # All fallbacks have been deprecated, so only base models are returned
        fallback_chains = {
            "gemini-2.5-pro": ["gemini-2.5-pro"],
            "gemini-2.5-flash": ["gemini-2.5-flash"],
            # Add more fallback chains as needed
        }

        # Return fallback chain if available, otherwise just return the original model
        return fallback_chains.get(model_name, [model_name])

    def _transform_messages(
        self, messages: List[Dict[str, Any]], model: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.

        Handles:
        - System instruction extraction
        - Multi-part content (text, images)
        - Tool calls and responses
        - Gemini 3 thoughtSignature preservation
        """
        messages = copy.deepcopy(messages)  # Don't mutate original
        system_instruction = None
        gemini_contents = []
        is_gemini_3 = self._is_gemini_3(model)

        # Separate system prompt from other messages
        if messages and messages[0].get("role") == "system":
            system_prompt_content = messages.pop(0).get("content", "")
            if system_prompt_content:
                system_instruction = {
                    "role": "user",
                    "parts": [{"text": system_prompt_content}],
                }

        tool_call_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("type") == "function":
                        tool_call_id_to_name[tool_call["id"]] = tool_call["function"][
                            "name"
                        ]

        # Process messages and consolidate consecutive tool responses
        # Per Gemini docs: parallel function responses must be in a single user message,
        # not interleaved as separate messages
        pending_tool_parts = []  # Accumulate tool responses

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            gemini_role = (
                "model" if role == "assistant" else "user"
            )  # tool -> user in Gemini

            # If we have pending tool parts and hit a non-tool message, flush them first
            if pending_tool_parts and role != "tool":
                gemini_contents.append({"role": "user", "parts": pending_tool_parts})
                pending_tool_parts = []

            if role == "user":
                if isinstance(content, str):
                    # Simple text content
                    if content:
                        parts.append({"text": content})
                elif isinstance(content, list):
                    # Multi-part content (text, images, etc.)
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                parts.append({"text": text})
                        elif item.get("type") == "image_url":
                            # Handle image data URLs
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                try:
                                    # Parse: data:image/png;base64,iVBORw0KG...
                                    header, data = image_url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": data,
                                            }
                                        }
                                    )
                                except Exception as e:
                                    lib_logger.warning(
                                        f"Failed to parse image data URL: {e}"
                                    )
                            else:
                                lib_logger.warning(
                                    f"Non-data-URL images not supported: {image_url[:50]}..."
                                )

            elif role == "assistant":
                if isinstance(content, str):
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    # Track if we've seen the first function call in this message
                    # Per Gemini docs: Only the FIRST parallel function call gets a signature
                    first_func_in_msg = True
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            try:
                                args_dict = json.loads(
                                    tool_call["function"]["arguments"]
                                )
                            except (json.JSONDecodeError, TypeError):
                                args_dict = {}

                            tool_id = tool_call.get("id", "")
                            func_name = tool_call["function"]["name"]

                            # Add prefix for Gemini 3 (and rename problematic tools)
                            if is_gemini_3 and self._enable_gemini3_tool_fix:
                                func_name = GEMINI3_TOOL_RENAMES.get(
                                    func_name, func_name
                                )
                                func_name = f"{self._gemini3_tool_prefix}{func_name}"

                            func_part = {
                                "functionCall": {
                                    "name": func_name,
                                    "args": args_dict,
                                    "id": tool_id,
                                }
                            }

                            # Add thoughtSignature for Gemini 3
                            # Per Gemini docs: Only the FIRST parallel function call gets a signature.
                            # Subsequent parallel calls should NOT have a thoughtSignature field.
                            if is_gemini_3:
                                sig = tool_call.get("thought_signature")
                                if not sig and tool_id and self._enable_signature_cache:
                                    sig = self._signature_cache.retrieve(tool_id)

                                if sig:
                                    func_part["thoughtSignature"] = sig
                                elif first_func_in_msg:
                                    # Only add bypass to the first function call if no sig available
                                    func_part["thoughtSignature"] = (
                                        "skip_thought_signature_validator"
                                    )
                                    lib_logger.debug(
                                        f"Missing thoughtSignature for first func call {tool_id}, using bypass"
                                    )
                                # Subsequent parallel calls: no signature field at all

                                first_func_in_msg = False

                            parts.append(func_part)

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                function_name = tool_call_id_to_name.get(tool_call_id)

                # Log warning if tool_call_id not found in mapping (can happen after context compaction)
                if not function_name:
                    lib_logger.warning(
                        f"[ID Mismatch] Tool response has ID '{tool_call_id}' which was not found in tool_id_to_name map. "
                        f"Available IDs: {list(tool_call_id_to_name.keys())}. Using 'unknown_function' as fallback."
                    )
                    function_name = "unknown_function"

                # Add prefix for Gemini 3 (and rename problematic tools)
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = GEMINI3_TOOL_RENAMES.get(
                        function_name, function_name
                    )
                    function_name = f"{self._gemini3_tool_prefix}{function_name}"

                # Try to parse content as JSON first, fall back to string
                try:
                    parsed_content = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                except (json.JSONDecodeError, TypeError):
                    parsed_content = content

                # Wrap the tool response in a 'result' object
                response_content = {"result": parsed_content}
                # Accumulate tool responses - they'll be combined into one user message
                pending_tool_parts.append(
                    {
                        "functionResponse": {
                            "name": function_name,
                            "response": response_content,
                            "id": tool_call_id,
                        }
                    }
                )
                # Don't add parts here - tool responses are handled via pending_tool_parts
                continue

            if parts:
                gemini_contents.append({"role": gemini_role, "parts": parts})

        # Flush any remaining tool parts at end of messages
        if pending_tool_parts:
            gemini_contents.append({"role": "user", "parts": pending_tool_parts})

        if not gemini_contents or gemini_contents[0]["role"] != "user":
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})

        return system_instruction, gemini_contents

    def _fix_tool_response_grouping(
        self, contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Gemini CLI compatibility.

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
                        # Match if names are equal
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

    def _handle_reasoning_parameters(
        self, payload: Dict[str, Any], model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration.

        - Gemini 2.5: thinkingBudget (integer tokens)
        - Gemini 3 Pro: thinkingLevel (string: "low"/"high")
        - Gemini 3 Flash: thinkingLevel (string: "minimal"/"low"/"medium"/"high")
        """
        custom_reasoning_budget = payload.get("custom_reasoning_budget", False)
        reasoning_effort = payload.get("reasoning_effort")

        if "thinkingConfig" in payload.get("generationConfig", {}):
            return None

        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = self._is_gemini_3(model)
        is_gemini_3_flash = "gemini-3-flash" in model

        # Only apply reasoning logic to supported models
        if not (is_gemini_25 or is_gemini_3):
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)
            return None

        # Gemini 3 Flash: Supports minimal/low/medium/high thinkingLevel
        if is_gemini_3_flash:
            # Clean up the original payload
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)

            if reasoning_effort == "disable":
                # "minimal" matches "no thinking" for most queries
                return {"thinkingLevel": "minimal", "include_thoughts": True}
            elif reasoning_effort == "low":
                return {"thinkingLevel": "low", "include_thoughts": True}
            elif reasoning_effort == "medium":
                return {"thinkingLevel": "medium", "include_thoughts": True}
            # Default to high for Flash
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 3 Pro: Only supports low/high thinkingLevel
        if is_gemini_3:
            # Clean up the original payload
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)

            if reasoning_effort == "low":
                return {"thinkingLevel": "low", "include_thoughts": True}
            # medium maps to high for Pro (not supported)
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 2.5: Integer thinkingBudget
        if not reasoning_effort:
            # Clean up the original payload
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)
            return {"thinkingBudget": -1, "include_thoughts": True}

        # If reasoning_effort is provided, calculate the budget
        budget = -1  # Default for 'auto' or invalid values
        if "gemini-2.5-pro" in model:
            budgets = {"low": 8192, "medium": 16384, "high": 32768}
        elif "gemini-2.5-flash" in model:
            budgets = {"low": 6144, "medium": 12288, "high": 24576}
        else:
            # Fallback for other gemini-2.5 models
            budgets = {"low": 1024, "medium": 2048, "high": 4096}

        budget = budgets.get(reasoning_effort, -1)
        if reasoning_effort == "disable":
            budget = 0

        if not custom_reasoning_budget:
            budget = budget // 4

        # Clean up the original payload
        payload.pop("reasoning_effort", None)
        payload.pop("custom_reasoning_budget", None)

        return {"thinkingBudget": budget, "include_thoughts": True}

    def _convert_chunk_to_openai(
        self,
        chunk: Dict[str, Any],
        model_id: str,
        accumulator: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert Gemini response chunk to OpenAI streaming format.

        Args:
            chunk: Gemini API response chunk
            model_id: Model name
            accumulator: Optional dict to accumulate data for post-processing (signatures, etc.)
        """
        response_data = chunk.get("response", chunk)
        candidates = response_data.get("candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        is_gemini_3 = self._is_gemini_3(model_id)

        for part in parts:
            delta = {}

            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = part.get("thought") is True or (
                isinstance(part.get("thought"), str)
                and str(part.get("thought")).lower() == "true"
            )

            # Skip standalone signature parts (no function, no meaningful text)
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue

            if has_func:
                function_call = part["functionCall"]
                function_name = function_call.get("name", "unknown")

                # Strip Gemini 3 prefix from tool name
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = self._strip_gemini3_prefix(function_name)

                # Use provided ID or generate unique one with nanosecond precision
                tool_call_id = (
                    function_call.get("id")
                    or f"call_{function_name}_{int(time.time() * 1_000_000_000)}"
                )

                # Get current tool index from accumulator (default 0) and increment
                current_tool_idx = accumulator.get("tool_idx", 0) if accumulator else 0

                # Optionally parse JSON strings in tool args
                # NOTE: This is very possibly redundant
                raw_args = function_call.get("args", {})
                if self._enable_json_string_parsing:
                    tool_args = _recursively_parse_json_strings(raw_args)
                else:
                    tool_args = raw_args

                # Strip _confirm ONLY if it's the sole parameter
                # This ensures we only strip our injection, not legitimate user params
                if isinstance(tool_args, dict) and "_confirm" in tool_args:
                    if len(tool_args) == 1:
                        # _confirm is the only param - this was our injection
                        tool_args.pop("_confirm")

                tool_call = {
                    "index": current_tool_idx,
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(tool_args),
                    },
                }

                # Handle thoughtSignature for Gemini 3
                # Store signature for each tool call (needed for parallel tool calls)
                if is_gemini_3 and has_sig:
                    sig = part["thoughtSignature"]

                    if self._enable_signature_cache:
                        self._signature_cache.store(tool_call_id, sig)
                        lib_logger.debug(f"Stored signature for {tool_call_id}")

                    if self._preserve_signatures_in_client:
                        tool_call["thought_signature"] = sig

                delta["tool_calls"] = [tool_call]
                # Mark that we've sent tool calls and increment tool_idx
                if accumulator is not None:
                    accumulator["has_tool_calls"] = True
                    accumulator["tool_idx"] = current_tool_idx + 1

            elif has_text:
                # Use an explicit check for the 'thought' flag, as its type can be inconsistent
                if is_thought:
                    delta["reasoning_content"] = part["text"]
                else:
                    delta["content"] = part["text"]

            if not delta:
                continue

            # Mark that we have tool calls for accumulator tracking
            # finish_reason determination is handled by the client

            # Mark stream complete if we have usageMetadata
            is_final_chunk = "usageMetadata" in response_data
            if is_final_chunk and accumulator is not None:
                accumulator["is_complete"] = True

            # Build choice - don't include finish_reason, let client handle it
            choice = {"index": 0, "delta": delta}

            openai_chunk = {
                "choices": [choice],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("responseId", f"chatcmpl-geminicli-{time.time()}"),
                "created": int(time.time()),
            }

            if "usageMetadata" in response_data:
                usage = response_data["usageMetadata"]
                prompt_tokens = usage.get("promptTokenCount", 0)
                thoughts_tokens = usage.get("thoughtsTokenCount", 0)
                candidate_tokens = usage.get("candidatesTokenCount", 0)

                openai_chunk["usage"] = {
                    "prompt_tokens": prompt_tokens
                    + thoughts_tokens,  # Include thoughts in prompt tokens
                    "completion_tokens": candidate_tokens,
                    "total_tokens": usage.get("totalTokenCount", 0),
                }

                # Add reasoning tokens details if present (OpenAI o1 format)
                if thoughts_tokens > 0:
                    if "completion_tokens_details" not in openai_chunk["usage"]:
                        openai_chunk["usage"]["completion_tokens_details"] = {}
                    openai_chunk["usage"]["completion_tokens_details"][
                        "reasoning_tokens"
                    ] = thoughts_tokens

            yield openai_chunk

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.

        Key improvements:
        - Determines finish_reason based on accumulated state
        - Priority: tool_calls > chunk's finish_reason (length, content_filter, etc.) > stop
        - Properly initializes tool_calls with type field
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        # Initialize the final response structure
        final_message = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        chunk_finish_reason = None  # Track finish_reason from chunks

        # Get the first chunk for basic response metadata
        first_chunk = chunks[0]

        # Process each chunk to aggregate content
        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
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
                    index = tc_chunk.get("index", 0)
                    if index not in aggregated_tool_calls:
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

            # Track finish_reason from chunks (respects length, content_filter, etc.)
            if choice.get("finish_reason"):
                chunk_finish_reason = choice["finish_reason"]

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

        # Determine finish_reason based on accumulated state
        # Priority: tool_calls wins if present, then chunk's finish_reason (length, content_filter, etc.), then default to "stop"
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

    def _gemini_cli_transform_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively transforms a JSON schema to be compatible with the Gemini CLI endpoint.
        - Converts `type: ["type", "null"]` to `type: "type", nullable: true`
        - Removes unsupported properties like `strict`.
        - Preserves `additionalProperties` for _enforce_strict_schema to handle.
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
                    schema["type"] = (
                        remaining_types  # Let's see if Gemini supports this
                    )
                else:
                    del schema["type"]

        # Recurse into properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_schema in schema["properties"].values():
                self._gemini_cli_transform_schema(prop_schema)

        # Recurse into items (for arrays)
        if "items" in schema and isinstance(schema["items"], dict):
            self._gemini_cli_transform_schema(schema["items"])

        # Clean up unsupported properties
        schema.pop("strict", None)
        # Note: additionalProperties is preserved for _enforce_strict_schema to handle

        return schema

    def _enforce_strict_schema(self, schema: Any) -> Any:
        """
        Enforce strict JSON schema for Gemini 3 to prevent hallucinated parameters.

        Adds 'additionalProperties: false' to object schemas with 'properties',
        which tells the model it CANNOT add properties not in the schema.

        IMPORTANT: Preserves 'additionalProperties: true' (or {}) when explicitly
        set in the original schema. This is critical for "freeform" parameter objects
        like batch/multi_tool's nested parameters which need to accept arbitrary
        tool parameters that aren't pre-defined in the schema.
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        preserved_additional_props = None

        for key, value in schema.items():
            # Preserve additionalProperties as-is if it's truthy
            # This is critical for "freeform" parameter objects like batch's
            # nested parameters which need to accept arbitrary tool parameters
            if key == "additionalProperties":
                if value is not False:
                    # Preserve the original value (true, {}, {"type": "string"}, etc.)
                    preserved_additional_props = value
                continue
            if isinstance(value, dict):
                result[key] = self._enforce_strict_schema(value)
            elif isinstance(value, list):
                result[key] = [
                    self._enforce_strict_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        # Add additionalProperties: false to object schemas with properties,
        # BUT only if we didn't preserve a value from the original schema
        if result.get("type") == "object" and "properties" in result:
            if preserved_additional_props is not None:
                result["additionalProperties"] = preserved_additional_props
            else:
                result["additionalProperties"] = False

        return result

    def _transform_tool_schemas(
        self, tools: List[Dict[str, Any]], model: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Transforms a list of OpenAI-style tool schemas into the format required by the Gemini CLI API.
        This uses a custom schema transformer instead of litellm's generic one.

        For Gemini 3 models, also applies:
        - Namespace prefix to tool names
        - Parameter signature injection into descriptions
        - Strict schema enforcement (additionalProperties: false)
        """
        transformed_declarations = []
        is_gemini_3 = self._is_gemini_3(model)

        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                new_function = json.loads(json.dumps(tool["function"]))

                # The Gemini CLI API does not support the 'strict' property.
                new_function.pop("strict", None)

                # Gemini CLI expects 'parametersJsonSchema' instead of 'parameters'
                if "parameters" in new_function:
                    # Inline $ref definitions first
                    schema = _inline_schema_refs(new_function["parameters"])
                    schema = self._gemini_cli_transform_schema(schema)
                    # Workaround: Gemini fails to emit functionCall for tools
                    # with empty properties {}. Inject a required confirmation param.
                    # Using a required parameter forces the model to commit to
                    # the tool call rather than just thinking about it.
                    props = schema.get("properties", {})
                    if not props:
                        schema["properties"] = {
                            "_confirm": {
                                "type": "string",
                                "description": "Enter 'yes' to proceed",
                            }
                        }
                        schema["required"] = ["_confirm"]
                    new_function["parametersJsonSchema"] = schema
                    del new_function["parameters"]
                elif "parametersJsonSchema" not in new_function:
                    # Set default schema with required confirm param if neither exists
                    new_function["parametersJsonSchema"] = {
                        "type": "object",
                        "properties": {
                            "_confirm": {
                                "type": "string",
                                "description": "Enter 'yes' to proceed",
                            }
                        },
                        "required": ["_confirm"],
                    }

                # Gemini 3 specific transformations
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    # Add namespace prefix to tool names (and rename problematic tools)
                    name = new_function.get("name", "")
                    if name:
                        name = GEMINI3_TOOL_RENAMES.get(name, name)
                        new_function["name"] = f"{self._gemini3_tool_prefix}{name}"

                    # Enforce strict schema (additionalProperties: false)
                    if (
                        self._gemini3_enforce_strict_schema
                        and "parametersJsonSchema" in new_function
                    ):
                        new_function["parametersJsonSchema"] = (
                            self._enforce_strict_schema(
                                new_function["parametersJsonSchema"]
                            )
                        )

                    # Inject parameter signature into description
                    new_function = self._inject_signature_into_description(new_function)

                transformed_declarations.append(new_function)

        return transformed_declarations

    def _inject_signature_into_description(
        self, func_decl: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject parameter signatures into tool description for Gemini 3."""
        schema = func_decl.get("parametersJsonSchema", {})
        if not schema:
            return func_decl

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if not properties:
            return func_decl

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
            sig_str = self._gemini3_description_prompt.replace(
                "{params}", ", ".join(param_list)
            )
            func_decl["description"] = func_decl.get("description", "") + sig_str

        return func_decl

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

    def _inject_gemini3_system_instruction(
        self, request_payload: Dict[str, Any]
    ) -> None:
        """Inject Gemini 3 tool fix system instruction if tools are present."""
        if not request_payload.get("request", {}).get("tools"):
            return

        existing_system = request_payload.get("request", {}).get("systemInstruction")

        if existing_system:
            # Prepend to existing system instruction
            existing_parts = existing_system.get("parts", [])
            if existing_parts and existing_parts[0].get("text"):
                existing_parts[0]["text"] = (
                    self._gemini3_system_instruction
                    + "\n\n"
                    + existing_parts[0]["text"]
                )
            else:
                existing_parts.insert(0, {"text": self._gemini3_system_instruction})
        else:
            # Create new system instruction
            request_payload["request"]["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": self._gemini3_system_instruction}],
            }

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
                # Add Gemini 3 prefix if needed (and rename problematic tools)
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = GEMINI3_TOOL_RENAMES.get(
                        function_name, function_name
                    )
                    function_name = f"{self._gemini3_tool_prefix}{function_name}"

                mode = "ANY"  # Force a call, but only to this function
                config["functionCallingConfig"] = {
                    "mode": mode,
                    "allowedFunctionNames": [function_name],
                }
                return config

        config["functionCallingConfig"] = {"mode": mode}
        return config

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        enable_request_logging = kwargs.pop("enable_request_logging", False)

        # Get fallback models for rate limit handling
        fallback_models = self._cli_preview_fallback_order(model)

        async def do_call(attempt_model: str, is_fallback: bool = False):
            # Get auth header once, it's needed for the request anyway
            auth_header = await self.get_auth_header(credential_path)

            # Discover project ID only if not already cached
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                access_token = auth_header["Authorization"].split(" ")[1]
                project_id = await self._discover_project_id(
                    credential_path, access_token, kwargs.get("litellm_params", {})
                )

            # Log paid tier usage visibly on each request
            credential_tier = self.project_tier_cache.get(credential_path)
            if credential_tier and credential_tier not in [
                "free-tier",
                "legacy-tier",
                "unknown",
            ]:
                lib_logger.info(
                    f"[PAID TIER] Using Gemini '{credential_tier}' subscription for this request"
                )

            # Handle :thinking suffix
            model_name = attempt_model.split("/")[-1].replace(":thinking", "")

            # [NEW] Create a dedicated file logger for this request
            file_logger = _GeminiCliFileLogger(
                model_name=model_name, enabled=enable_request_logging
            )

            is_gemini_3 = self._is_gemini_3(model_name)

            gen_config = {
                "maxOutputTokens": kwargs.get("max_tokens", 64000),  # Increased default
                "temperature": kwargs.get(
                    "temperature", 1
                ),  # Default to 1 if not provided
            }
            if "top_k" in kwargs:
                gen_config["topK"] = kwargs["top_k"]
            if "top_p" in kwargs:
                gen_config["topP"] = kwargs["top_p"]

            # Use the sophisticated reasoning logic
            thinking_config = self._handle_reasoning_parameters(kwargs, model_name)
            if thinking_config:
                gen_config["thinkingConfig"] = thinking_config

            system_instruction, contents = self._transform_messages(
                kwargs.get("messages", []), model_name
            )
            # Fix tool response grouping (handles ID mismatches, missing responses)
            contents = self._fix_tool_response_grouping(contents)

            request_payload = {
                "model": model_name,
                "project": project_id,
                "request": {
                    "contents": contents,
                    "generationConfig": gen_config,
                },
            }

            if system_instruction:
                request_payload["request"]["systemInstruction"] = system_instruction

            if "tools" in kwargs and kwargs["tools"]:
                function_declarations = self._transform_tool_schemas(
                    kwargs["tools"], model_name
                )
                if function_declarations:
                    request_payload["request"]["tools"] = [
                        {"functionDeclarations": function_declarations}
                    ]

            # [NEW] Handle tool_choice translation
            if "tool_choice" in kwargs and kwargs["tool_choice"]:
                tool_config = self._translate_tool_choice(
                    kwargs["tool_choice"], model_name
                )
                if tool_config:
                    request_payload["request"]["toolConfig"] = tool_config

            # Inject Gemini 3 system instruction if using tools
            if is_gemini_3 and self._enable_gemini3_tool_fix:
                self._inject_gemini3_system_instruction(request_payload)

            # Add default safety settings to prevent content filtering
            if "safetySettings" not in request_payload["request"]:
                request_payload["request"]["safetySettings"] = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                    {
                        "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                        "threshold": "BLOCK_NONE",
                    },
                ]

            # Log the final payload for debugging and to the dedicated file
            # lib_logger.debug(f"Gemini CLI Request Payload: {json.dumps(request_payload, indent=2)}")
            file_logger.log_request(request_payload)

            url = f"{CODE_ASSIST_ENDPOINT}:streamGenerateContent"

            async def stream_handler():
                # Track state across chunks for tool indexing
                accumulator = {
                    "has_tool_calls": False,
                    "tool_idx": 0,
                    "is_complete": False,
                }

                final_headers = auth_header.copy()
                final_headers.update(
                    {
                        "User-Agent": "google-api-nodejs-client/9.15.1",
                        "X-Goog-Api-Client": "gl-node/22.17.0",
                        "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
                        "Accept": "application/json",
                    }
                )
                try:
                    async with client.stream(
                        "POST",
                        url,
                        headers=final_headers,
                        json=request_payload,
                        params={"alt": "sse"},
                        timeout=TimeoutConfig.streaming(),
                    ) as response:
                        # Read and log error body before raise_for_status for better debugging
                        if response.status_code >= 400:
                            try:
                                error_body = await response.aread()
                                lib_logger.error(
                                    f"Gemini CLI API error {response.status_code}: {error_body.decode()}"
                                )
                                file_logger.log_error(
                                    f"API error {response.status_code}: {error_body.decode()}"
                                )
                            except Exception:
                                pass

                        # This will raise an HTTPStatusError for 4xx/5xx responses
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            file_logger.log_response_chunk(line)
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    for openai_chunk in self._convert_chunk_to_openai(
                                        chunk, model, accumulator
                                    ):
                                        yield litellm.ModelResponse(**openai_chunk)
                                except json.JSONDecodeError:
                                    lib_logger.warning(
                                        f"Could not decode JSON from Gemini CLI: {line}"
                                    )

                        # Emit final chunk if stream ended without usageMetadata
                        # Client will determine the correct finish_reason
                        if not accumulator.get("is_complete"):
                            final_chunk = {
                                "id": f"chatcmpl-geminicli-{time.time()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": None}
                                ],
                                # Include minimal usage to signal this is the final chunk
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 1,
                                    "total_tokens": 1,
                                },
                            }
                            yield litellm.ModelResponse(**final_chunk)

                except httpx.HTTPStatusError as e:
                    error_body = None
                    if e.response is not None:
                        try:
                            error_body = e.response.text
                        except Exception:
                            pass

                    # Only log to file logger (for detailed logging)
                    if error_body:
                        file_logger.log_error(
                            f"HTTPStatusError {e.response.status_code}: {error_body}"
                        )
                    else:
                        file_logger.log_error(
                            f"HTTPStatusError {e.response.status_code}: {str(e)}"
                        )

                    if e.response.status_code == 429:
                        # Extract retry-after time from the error body
                        retry_after = extract_retry_after_from_body(error_body)
                        retry_info = (
                            f" (retry after {retry_after}s)" if retry_after else ""
                        )
                        error_msg = f"Gemini CLI rate limit exceeded{retry_info}"
                        if error_body:
                            error_msg = f"{error_msg} | {error_body}"
                        # Only log at debug level - rotation happens silently
                        lib_logger.debug(
                            f"Gemini CLI 429 rate limit: retry_after={retry_after}s"
                        )
                        raise RateLimitError(
                            message=error_msg,
                            llm_provider="gemini_cli",
                            model=model,
                            response=e.response,
                        )
                    # Re-raise other status errors to be handled by the main acompletion logic
                    raise e
                except Exception as e:
                    file_logger.log_error(f"Stream handler exception: {str(e)}")
                    raise

            async def logging_stream_wrapper():
                """Wraps the stream to log the final reassembled response."""
                openai_chunks = []
                try:
                    async for chunk in stream_handler():
                        openai_chunks.append(chunk)
                        yield chunk
                finally:
                    if openai_chunks:
                        final_response = self._stream_to_completion_response(
                            openai_chunks
                        )
                        file_logger.log_final_response(final_response.dict())

            return logging_stream_wrapper()

        # Check if there are actual fallback models available
        # If fallback_models is empty or contains only the base model (no actual fallbacks), skip fallback logic
        has_fallbacks = len(fallback_models) > 1 and any(
            model != fallback_models[0] for model in fallback_models[1:]
        )

        lib_logger.debug(f"Fallback models available: {fallback_models}")
        if not has_fallbacks:
            lib_logger.debug(
                "No actual fallback models available, proceeding with single model attempt"
            )

        last_error = None
        for idx, attempt_model in enumerate(fallback_models):
            is_fallback = idx > 0
            if is_fallback:
                # Silent rotation - only log at debug level
                lib_logger.debug(
                    f"Rate limited on previous model, trying fallback: {attempt_model}"
                )
            elif has_fallbacks:
                lib_logger.debug(
                    f"Attempting primary model: {attempt_model} (with {len(fallback_models) - 1} fallback(s) available)"
                )
            else:
                lib_logger.debug(
                    f"Attempting model: {attempt_model} (no fallbacks available)"
                )

            try:
                response_gen = await do_call(attempt_model, is_fallback)

                if kwargs.get("stream", False):
                    return response_gen
                else:
                    # Accumulate stream for non-streaming response
                    chunks = [chunk async for chunk in response_gen]
                    return self._stream_to_completion_response(chunks)

            except RateLimitError as e:
                last_error = e
                # If this is not the last model in the fallback chain, continue to next model
                if idx + 1 < len(fallback_models):
                    lib_logger.debug(
                        f"Rate limit hit on {attempt_model}, trying next fallback..."
                    )
                    continue
                # If this was the last fallback option, log error and raise
                lib_logger.warning(
                    f"Rate limit exhausted on all fallback models (tried {len(fallback_models)} models)"
                )
                raise

        # Should not reach here, but raise last error if we do
        if last_error:
            raise last_error
        raise ValueError("No fallback models available")

    async def count_tokens(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Counts tokens for the given prompt using the Gemini CLI :countTokens endpoint.

        Args:
            client: The HTTP client to use
            credential_path: Path to the credential file
            model: Model name to use for token counting
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions
            litellm_params: Optional additional parameters

        Returns:
            Dict with 'prompt_tokens' and 'total_tokens' counts
        """
        # Get auth header
        auth_header = await self.get_auth_header(credential_path)

        # Discover project ID
        project_id = self.project_id_cache.get(credential_path)
        if not project_id:
            access_token = auth_header["Authorization"].split(" ")[1]
            project_id = await self._discover_project_id(
                credential_path, access_token, litellm_params or {}
            )

        # Handle :thinking suffix
        model_name = model.split("/")[-1].replace(":thinking", "")

        # Transform messages to Gemini format
        system_instruction, contents = self._transform_messages(messages)
        # Fix tool response grouping (handles ID mismatches, missing responses)
        contents = self._fix_tool_response_grouping(contents)

        # Build request payload
        request_payload = {
            "request": {
                "contents": contents,
            },
        }

        if system_instruction:
            request_payload["request"]["systemInstruction"] = system_instruction

        if tools:
            function_declarations = self._transform_tool_schemas(tools)
            if function_declarations:
                request_payload["request"]["tools"] = [
                    {"functionDeclarations": function_declarations}
                ]

        # Make the request
        url = f"{CODE_ASSIST_ENDPOINT}:countTokens"
        headers = auth_header.copy()
        headers.update(
            {
                "User-Agent": "google-api-nodejs-client/9.15.1",
                "X-Goog-Api-Client": "gl-node/22.17.0",
                "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
                "Accept": "application/json",
            }
        )

        try:
            response = await client.post(
                url, headers=headers, json=request_payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Extract token counts from response
            total_tokens = data.get("totalTokens", 0)

            return {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            }

        except httpx.HTTPStatusError as e:
            lib_logger.error(f"Failed to count tokens: {e}")
            # Return 0 on error rather than raising
            return {"prompt_tokens": 0, "total_tokens": 0}

    # Use the shared GeminiAuthBase for auth logic
    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of Gemini CLI models from three sources:
        1. Environment variable models (via GEMINI_CLI_MODELS) - ALWAYS included, take priority
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        3. Dynamic discovery from Gemini API (if supported) - added only if ID not in env vars

        Environment variable models always win and are never deduplicated, even if they
        share the same ID (to support different configs like temperature, etc.)
        """
        # Check for mixed tier credentials and warn if detected
        self._check_mixed_tier_warning()

        models = []
        env_var_ids = (
            set()
        )  # Track IDs from env vars to prevent hardcoded/dynamic duplicates

        def extract_model_id(item) -> str:
            """Extract model ID from various formats (dict, string with/without provider prefix)."""
            if isinstance(item, dict):
                # Dict format: extract 'name' or 'id' field
                model_id = item.get("name") or item.get("id", "")
                # Gemini models often have format "models/gemini-pro", extract just the model name
                if model_id and "/" in model_id:
                    model_id = model_id.split("/")[-1]
                return model_id
            elif isinstance(item, str):
                # String format: extract ID from "provider/id" or "models/id" or just "id"
                return item.split("/")[-1] if "/" in item else item
            return str(item)

        # Source 1: Load environment variable models (ALWAYS include ALL of them)
        static_models = self.model_definitions.get_all_provider_models("gemini_cli")
        if static_models:
            for model in static_models:
                # Extract model name from "gemini_cli/ModelName" format
                model_name = model.split("/")[-1] if "/" in model else model
                # Get the actual model ID from definitions (which may differ from the name)
                model_id = self.model_definitions.get_model_id("gemini_cli", model_name)

                # ALWAYS add env var models (no deduplication)
                models.append(model)
                # Track the ID to prevent hardcoded/dynamic duplicates
                if model_id:
                    env_var_ids.add(model_id)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for gemini_cli from environment variables"
            )

        # Source 2: Add hardcoded models (only if ID not already in env vars)
        for model_id in HARDCODED_MODELS:
            if model_id not in env_var_ids:
                models.append(f"gemini_cli/{model_id}")
                env_var_ids.add(model_id)

        # Source 3: Try dynamic discovery from Gemini API (only if ID not already in env vars)
        try:
            # Get access token for API calls
            auth_header = await self.get_auth_header(credential)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Try Vertex AI models endpoint
            # Note: Gemini may not support a simple /models endpoint like OpenAI
            # This is a best-effort attempt that will gracefully fail if unsupported
            models_url = f"https://generativelanguage.googleapis.com/v1beta/models"

            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()

            dynamic_data = response.json()
            # Handle various response formats
            model_list = dynamic_data.get("models", dynamic_data.get("data", []))

            dynamic_count = 0
            for model in model_list:
                model_id = extract_model_id(model)
                # Only include Gemini models that aren't already in env vars
                if (
                    model_id
                    and model_id not in env_var_ids
                    and model_id.startswith("gemini")
                ):
                    models.append(f"gemini_cli/{model_id}")
                    env_var_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} additional models for gemini_cli from API"
                )

        except Exception as e:
            # Silently ignore dynamic discovery errors
            lib_logger.debug(f"Dynamic model discovery failed for gemini_cli: {e}")
            pass

        return models
