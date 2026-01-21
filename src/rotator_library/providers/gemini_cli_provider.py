# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/gemini_cli_provider.py

import copy
import json
import httpx
import logging
import time
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple
from .provider_interface import ProviderInterface, QuotaGroupMap, UsageResetConfigDef
from .gemini_auth_base import GeminiAuthBase
from .provider_cache import ProviderCache
from .utilities.gemini_cli_quota_tracker import GeminiCliQuotaTracker
from .utilities.gemini_shared_utils import (
    env_bool,
    env_int,
    inline_schema_refs,
    recursively_parse_json_strings,
    GEMINI3_TOOL_RENAMES,
    GEMINI3_TOOL_RENAMES_REVERSE,
    FINISH_REASON_MAP,
    CODE_ASSIST_ENDPOINT,
    GEMINI_CLI_ENDPOINT_FALLBACKS,
)
from ..transaction_logger import ProviderLogger
from .utilities.gemini_tool_handler import GeminiToolHandler
from .utilities.gemini_credential_manager import GeminiCredentialManager
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..utils.paths import get_cache_dir
import litellm
from litellm.exceptions import RateLimitError
from ..error_handler import extract_retry_after_from_body
import os
from pathlib import Path
import uuid
import secrets
import hashlib
from datetime import datetime

lib_logger = logging.getLogger("rotator_library")


def _get_gemini_cli_cache_dir() -> Path:
    """Get the Gemini CLI cache directory."""
    return get_cache_dir(subdir="gemini_cli")


def _get_gemini3_signature_cache_file() -> Path:
    """Get the Gemini 3 signature cache file path."""
    return _get_gemini_cli_cache_dir() / "gemini3_signatures.json"


# NOTE: _GeminiCliFileLogger has been moved to utilities.gemini_file_logger
# and is imported as GeminiCliFileLogger


AVAILABLE_MODELS = [
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

# NOTE: FINISH_REASON_MAP has been moved to utilities.gemini_shared_utils

# NOTE: _recursively_parse_json_strings, _inline_schema_refs, _env_bool, _env_int
# have been moved to utilities.gemini_shared_utils and are imported at top of file


class GeminiCliProvider(
    GeminiAuthBase,
    GeminiCliQuotaTracker,
    GeminiToolHandler,
    GeminiCredentialManager,
    ProviderInterface,
):
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

    # Usage reset configs for Gemini CLI
    # Verified 2026-01-07: 24-hour fixed window from first request for ALL tiers
    # The reset time is set when the first request is made and does NOT roll forward
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=24 * 60 * 60,  # 24 hours
            mode="per_model",
            description="24-hour per-model window (all tiers)",
            field_name="models",
        ),
    }

    # Model quota groups - models that share quota/cooldown timing
    # Verified 2026-01-07 via quota verification tests
    # Can be overridden via env: QUOTA_GROUPS_GEMINI_CLI_{GROUP}="model1,model2"
    model_quota_groups: QuotaGroupMap = {
        # Pro models share a quota pool (verified: gemini-2.5-pro and gemini-3-pro-preview)
        "pro": ["gemini-2.5-pro", "gemini-3-pro-preview"],
        # All 2.x Flash models share a quota pool (verified: 2.0 shares with 2.5)
        # Note: contrary to PR #62 which claimed 2.0-flash was standalone
        "25-flash": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        # Gemini 3 Flash is standalone (verified)
        "3-flash": ["gemini-3-flash-preview"],
    }

    # Priority-based concurrency multipliers
    # Same structure as Antigravity (by coincidence, tiers share naming)
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
        self._quota_refresh_interval = env_int("GEMINI_CLI_QUOTA_REFRESH_INTERVAL", 300)

        # Track whether initial quota fetch has been done (for background job)
        self._initial_quota_fetch_done = False

        # Gemini 3 configuration from environment
        memory_ttl = env_int("GEMINI_CLI_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = env_int("GEMINI_CLI_SIGNATURE_DISK_TTL", 86400)

        # Initialize signature cache for Gemini 3 thoughtSignatures
        self._signature_cache = ProviderCache(
            _get_gemini3_signature_cache_file(),
            memory_ttl,
            disk_ttl,
            env_prefix="GEMINI_CLI_SIGNATURE",
        )

        # Gemini 3 feature flags
        self._preserve_signatures_in_client = env_bool(
            "GEMINI_CLI_PRESERVE_THOUGHT_SIGNATURES", True
        )
        self._enable_signature_cache = env_bool(
            "GEMINI_CLI_ENABLE_SIGNATURE_CACHE", True
        )
        self._enable_gemini3_tool_fix = env_bool("GEMINI_CLI_GEMINI3_TOOL_FIX", True)
        self._gemini3_enforce_strict_schema = env_bool(
            "GEMINI_CLI_GEMINI3_STRICT_SCHEMA", True
        )
        # Toggle for JSON string parsing in tool call arguments
        # NOTE: This is possibly redundant - modern Gemini models may not need this fix.
        # Disabled by default. Enable if you see JSON-stringified values in tool args.
        self._enable_json_string_parsing = env_bool(
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
    #
    # NOTE: _load_tier_from_file(), get_credential_tier_name(), initialize_credentials(),
    # _load_persisted_tiers(), get_background_job_config(), and run_background_job()
    # are now inherited from GeminiCredentialManager mixin.
    # =========================================================================

    # NOTE: _load_tier_from_file() is inherited from GeminiCredentialManager

    # NOTE: get_credential_tier_name() is inherited from GeminiCredentialManager

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        Returns the minimum priority tier required for a model.

        Args:
            model: The model name (with or without provider prefix)

        Returns:
            Minimum required priority level or None if no restrictions
        """
        # No model-specific priority restrictions
        # (Gemini 3 is now public and available to all tiers)
        return None

    # NOTE: initialize_credentials() is inherited from GeminiCredentialManager

    # NOTE: _load_persisted_tiers() is inherited from GeminiCredentialManager

    # =========================================================================
    # BACKGROUND JOB INTERFACE (Quota Baseline Refresh)
    # =========================================================================

    # NOTE: get_background_job_config() is inherited from GeminiCredentialManager

    # NOTE: run_background_job() is inherited from GeminiCredentialManager

    # NOTE: _post_auth_discovery() is inherited from GeminiAuthBase

    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        model_name = model.split("/")[-1].replace(":thinking", "")
        return model_name.startswith("gemini-3-")

    def _generate_user_prompt_id(self) -> str:
        """
        Generate a unique prompt ID matching native gemini-cli format.

        Native JS: Math.random().toString(16).slice(2) produces 13-14 hex chars.
        Python equivalent using secrets for cryptographic randomness.
        """
        return secrets.token_hex(7)  # 14 hex characters

    def _generate_stable_session_id(self, contents: List[Dict[str, Any]]) -> str:
        """
        Generate a stable session ID based on the first user message.

        This ensures:
        - Same conversation = same session_id (even across server restarts)
        - Different conversations = different session_ids
        - Multi-user scenarios are properly isolated

        Uses SHA256 hash of the first user message to create a deterministic
        UUID-formatted session ID. Falls back to random UUID if no user message.

        This approach mirrors Antigravity's _generate_stable_session_id() but
        uses UUID format instead of the -{number} format to match native
        gemini-cli's crypto.randomUUID() output format.

        Args:
            contents: List of message contents in Gemini format

        Returns:
            UUID-formatted session ID string
        """
        # Find first user message text
        for content in contents:
            if content.get("role") == "user":
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if text:
                            # SHA256 hash and use first 16 bytes to create UUID
                            h = hashlib.sha256(text.encode("utf-8")).digest()
                            # Format as UUID (8-4-4-4-12 hex chars)
                            return f"{h[:4].hex()}-{h[4:6].hex()}-{h[6:8].hex()}-{h[8:10].hex()}-{h[10:16].hex()}"

        # Fallback to random UUID if no user message found
        return str(uuid.uuid4())

    def _get_gemini_cli_request_headers(self, model: str) -> Dict[str, str]:
        """
        Build request headers matching native gemini-cli client.

        For the OAuth/Code Assist path, native gemini-cli only sends:
        - Content-Type: application/json (handled by httpx)
        - Authorization: Bearer <token> (handled by auth_header)
        - User-Agent: GeminiCLI/${version}/${model} (${platform}; ${arch})

        Headers NOT sent by native CLI (confirmed via explore agent analysis):
        - X-Goog-Api-Client: Not used in Code Assist path (only in SDK/API key path)
        - Client-Metadata: Not sent as HTTP header (only in request body for management endpoints)
        - X-Goog-User-Project: Only used in MCP path, causes 403 errors in Code Assist

        Source: gemini-cli/packages/core/src/code_assist/server.ts:332
        Source: gemini-cli/packages/core/src/core/contentGenerator.ts:129
        """
        model_name = model.split("/")[-1].replace(":thinking", "")

        # Hardcoded to Windows x64 platform (matching common development environment)
        # Native format: GeminiCLI/${version}/${model} (${platform}; ${arch})
        user_agent = f"GeminiCLI/0.26.0/{model_name} (win32; x64)"

        # =========================================================================
        # COMMENTED OUT HEADERS - Not sent by native gemini-cli for Code Assist path
        # Keeping these for reference as they worked well for SDK mimicry.
        # Uncomment if rate limiting issues arise and you want to try SDK fingerprinting.
        # =========================================================================

        # X-Goog-Api-Client: Mimics @google/genai SDK but native CLI doesn't send this
        # for OAuth/Code Assist path (only set when using API key authentication)
        # x_goog_api_client = "gl-node/22.17.0 gdcl/1.30.0"

        # Client-Metadata: Native CLI sends this in REQUEST BODY for management endpoints
        # (loadCodeAssist, onboardUser, listExperiments, recordCodeAssistMetrics)
        # but NOT as an HTTP header for generateContent requests.
        # client_metadata = (
        #     "ideType=IDE_UNSPECIFIED,"
        #     "pluginType=GEMINI,"
        #     "ideVersion=0.26.0,"
        #     "platform=WINDOWS_AMD64,"
        #     "updateChannel=stable"
        # )

        return {
            "User-Agent": user_agent,
            # "X-Goog-Api-Client": x_goog_api_client,  # Not sent by native CLI
            # "Client-Metadata": client_metadata,      # Not sent as header by native CLI
            # "Accept": "application/json",            # Not explicitly sent by native CLI
        }

    def _get_available_models(self) -> List[str]:
        """
        Get list of user-facing model names available via this provider.

        Used by quota tracker to filter which models to store baselines for.
        Only models in this list will have quota baselines tracked.

        Returns:
            List of user-facing model names
        """
        return AVAILABLE_MODELS

    # NOTE: _strip_gemini3_prefix() is inherited from GeminiToolHandler

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

    # NOTE: _fix_tool_response_grouping() is inherited from GeminiToolHandler mixin

    def _handle_reasoning_parameters(
        self, payload: Dict[str, Any], model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration for Gemini models.

        - Gemini 2.5: thinkingBudget (integer tokens)
        - Gemini 3 Pro: thinkingLevel (string: "low"/"high")
        - Gemini 3 Flash: thinkingLevel (string: "minimal"/"low"/"medium"/"high")
        """
        reasoning_effort = payload.pop("reasoning_effort", None)

        if "thinkingConfig" in payload.get("generationConfig", {}):
            return None

        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = self._is_gemini_3(model)
        is_gemini_3_flash = "gemini-3-flash" in model

        if not (is_gemini_25 or is_gemini_3):
            return None

        # Normalize and validate upfront
        if reasoning_effort is None:
            effort = "auto"
        elif isinstance(reasoning_effort, str):
            effort = reasoning_effort.strip().lower() or "auto"
        else:
            lib_logger.warning(
                f"[GeminiCLI] Invalid reasoning_effort type: {type(reasoning_effort).__name__}, using auto"
            )
            effort = "auto"

        valid_efforts = {
            "auto",
            "disable",
            "off",
            "none",
            "minimal",
            "low",
            "low_medium",
            "medium",
            "medium_high",
            "high",
        }
        if effort not in valid_efforts:
            lib_logger.warning(
                f"[GeminiCLI] Unknown reasoning_effort: '{reasoning_effort}', using auto"
            )
            effort = "auto"

        # Gemini 3 Flash: minimal/low/medium/high
        if is_gemini_3_flash:
            if effort in ("disable", "off", "none"):
                return {"thinkingLevel": "minimal", "include_thoughts": True}
            if effort in ("minimal", "low"):
                return {"thinkingLevel": "low", "include_thoughts": True}
            if effort in ("low_medium", "medium"):
                return {"thinkingLevel": "medium", "include_thoughts": True}
            # auto, medium_high, high → high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 3 Pro: only low/high
        if is_gemini_3:
            if effort in ("disable", "off", "none", "minimal", "low", "low_medium"):
                return {"thinkingLevel": "low", "include_thoughts": True}
            # auto, medium, medium_high, high → high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 2.5: Integer thinkingBudget
        if effort in ("disable", "off", "none"):
            return {"thinkingBudget": 0, "include_thoughts": False}

        if effort == "auto":
            return {"thinkingBudget": -1, "include_thoughts": True}

        # Model-specific budgets
        if "gemini-2.5-flash" in model:
            budgets = {
                "minimal": 3072,
                "low": 6144,
                "low_medium": 9216,
                "medium": 12288,
                "medium_high": 18432,
                "high": 24576,
            }
        else:
            budgets = {
                "minimal": 4096,
                "low": 8192,
                "low_medium": 12288,
                "medium": 16384,
                "medium_high": 24576,
                "high": 32768,
            }

        return {"thinkingBudget": budgets[effort], "include_thoughts": True}

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
                    tool_args = recursively_parse_json_strings(raw_args)
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
                prompt_tokens = usage.get("promptTokenCount", 0)  # Input
                thoughts_tokens = usage.get(
                    "thoughtsTokenCount", 0
                )  # Output (thinking)
                candidate_tokens = usage.get(
                    "candidatesTokenCount", 0
                )  # Output (content)
                cached_tokens = usage.get("cachedContentTokenCount", 0)  # Input subset

                openai_chunk["usage"] = {
                    "prompt_tokens": prompt_tokens,  # Input only
                    "completion_tokens": candidate_tokens
                    + thoughts_tokens,  # All output
                    "total_tokens": usage.get("totalTokenCount", 0),
                }

                # Add input breakdown: cached tokens
                if cached_tokens > 0:
                    openai_chunk["usage"]["prompt_tokens_details"] = {
                        "cached_tokens": cached_tokens
                    }

                # Add output breakdown: reasoning tokens
                if thoughts_tokens > 0:
                    openai_chunk["usage"]["completion_tokens_details"] = {
                        "reasoning_tokens": thoughts_tokens
                    }

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

    # NOTE: _enforce_strict_schema() is inherited from GeminiToolHandler mixin

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
                    schema = inline_schema_refs(new_function["parameters"])
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
                    new_function = self._inject_signature_into_description(
                        new_function, self._gemini3_description_prompt
                    )

                transformed_declarations.append(new_function)

        return transformed_declarations

    # NOTE: _inject_signature_into_description() is inherited from GeminiToolHandler mixin
    # The inherited version requires passing the description_prompt parameter

    # NOTE: _format_type_hint() is inherited from GeminiToolHandler mixin

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

    # NOTE: _translate_tool_choice() is inherited from GeminiToolHandler mixin

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)

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

            # Handle :thinking suffix
            model_name = attempt_model.split("/")[-1].replace(":thinking", "")

            # Create provider logger from transaction context
            file_logger = ProviderLogger(transaction_context)

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

            # Generate unique prompt ID for this request (matches native gemini-cli)
            # Source: gemini-cli/packages/cli/src/gemini.tsx line 668
            user_prompt_id = self._generate_user_prompt_id()

            # Build payload matching native gemini-cli structure
            # Source: gemini-cli/packages/core/src/code_assist/converter.ts lines 31-48
            request_payload = {
                "model": model_name,
                "project": project_id,
                "user_prompt_id": user_prompt_id,
                "request": {
                    "contents": contents,
                    "generationConfig": gen_config,
                    "session_id": self._generate_stable_session_id(contents),
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

            async def stream_handler():
                # Track state across chunks for tool indexing
                accumulator = {
                    "has_tool_calls": False,
                    "tool_idx": 0,
                    "is_complete": False,
                }

                # Build headers matching native gemini-cli client fingerprint
                final_headers = auth_header.copy()
                final_headers.update(self._get_gemini_cli_request_headers(model_name))

                # Endpoint fallback loop: try sandbox first, then production
                # This mirrors the opencode-antigravity-auth plugin behavior
                last_endpoint_error = None
                for endpoint_idx, base_endpoint in enumerate(
                    GEMINI_CLI_ENDPOINT_FALLBACKS
                ):
                    url = f"{base_endpoint}:streamGenerateContent"
                    is_fallback = endpoint_idx > 0

                    if is_fallback:
                        lib_logger.debug(
                            f"Endpoint fallback: trying {base_endpoint} after previous endpoint failed"
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
                                        for (
                                            openai_chunk
                                        ) in self._convert_chunk_to_openai(
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

                            # Success - exit the endpoint fallback loop
                            return

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

                        # 429 rate limit - don't fallback to next endpoint, let rotator handle it
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

                        # 5xx server errors - try next endpoint if available
                        if e.response.status_code >= 500:
                            last_endpoint_error = e
                            if endpoint_idx < len(GEMINI_CLI_ENDPOINT_FALLBACKS) - 1:
                                lib_logger.warning(
                                    f"Endpoint {base_endpoint} returned {e.response.status_code}, trying fallback"
                                )
                                continue
                            # No more endpoints to try
                            raise e

                        # Other 4xx errors - don't fallback, re-raise
                        raise e

                    except (httpx.ConnectError, httpx.TimeoutException) as e:
                        # Connection/timeout errors - try next endpoint if available
                        last_endpoint_error = e
                        file_logger.log_error(
                            f"Connection error to {base_endpoint}: {str(e)}"
                        )
                        if endpoint_idx < len(GEMINI_CLI_ENDPOINT_FALLBACKS) - 1:
                            lib_logger.warning(
                                f"Connection error to {base_endpoint}, trying fallback endpoint"
                            )
                            continue
                        # No more endpoints to try
                        raise e

                    except Exception as e:
                        file_logger.log_error(f"Stream handler exception: {str(e)}")
                        raise

                # If we get here, all endpoints failed (shouldn't happen due to raise in loop)
                if last_endpoint_error:
                    raise last_endpoint_error

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

        # Build request payload matching native gemini-cli structure
        request_payload = {
            "model": model_name,
            "project": project_id,
            "user_prompt_id": self._generate_user_prompt_id(),
            "request": {
                "contents": contents,
                "session_id": self._generate_stable_session_id(contents),
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

        # Build headers matching native gemini-cli client fingerprint
        headers = auth_header.copy()
        headers.update(self._get_gemini_cli_request_headers(model_name))

        # Endpoint fallback loop: try sandbox first, then production
        for endpoint_idx, base_endpoint in enumerate(GEMINI_CLI_ENDPOINT_FALLBACKS):
            url = f"{base_endpoint}:countTokens"
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
                # 5xx errors - try next endpoint if available
                if (
                    e.response.status_code >= 500
                    and endpoint_idx < len(GEMINI_CLI_ENDPOINT_FALLBACKS) - 1
                ):
                    lib_logger.warning(
                        f"countTokens: endpoint {base_endpoint} returned {e.response.status_code}, trying fallback"
                    )
                    continue
                lib_logger.error(f"Failed to count tokens: {e}")
                # Return 0 on error rather than raising
                return {"prompt_tokens": 0, "total_tokens": 0}

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Connection errors - try next endpoint if available
                if endpoint_idx < len(GEMINI_CLI_ENDPOINT_FALLBACKS) - 1:
                    lib_logger.warning(
                        f"countTokens: connection error to {base_endpoint}, trying fallback"
                    )
                    continue
                lib_logger.error(f"Failed to count tokens: {e}")
                return {"prompt_tokens": 0, "total_tokens": 0}

        # Shouldn't reach here, but return 0 as fallback
        return {"prompt_tokens": 0, "total_tokens": 0}

    # Use the shared GeminiAuthBase for auth logic
    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of Gemini CLI models from three sources:
        1. Environment variable models (via model definitions) - ALWAYS included, take priority
        2. Available models (AVAILABLE_MODELS fallback list) - added only if ID not in env vars
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

        # Source 2: Add available models (only if ID not already in env vars)
        for model_id in AVAILABLE_MODELS:
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
