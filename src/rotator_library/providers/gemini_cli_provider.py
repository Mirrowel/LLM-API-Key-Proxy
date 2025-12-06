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
from ..model_definitions import ModelDefinitions
import litellm
from litellm.exceptions import RateLimitError
from ..error_handler import extract_retry_after_from_body
import os
from pathlib import Path
import uuid
from datetime import datetime

lib_logger = logging.getLogger("rotator_library")

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
GEMINI_CLI_LOGS_DIR = LOGS_DIR / "gemini_cli_logs"


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
            GEMINI_CLI_LOGS_DIR / f"{timestamp}_{safe_model_name}_{request_id}"
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
]

# Cache directory for Gemini CLI
CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "cache" / "gemini_cli"
)
GEMINI3_SIGNATURE_CACHE_FILE = CACHE_DIR / "gemini3_signatures.json"

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


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


class GeminiCliProvider(GeminiAuthBase, ProviderInterface):
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        self.project_id_cache: Dict[
            str, str
        ] = {}  # Cache project ID per credential path
        self.project_tier_cache: Dict[
            str, str
        ] = {}  # Cache project tier per credential path

        # Gemini 3 configuration from environment
        memory_ttl = _env_int("GEMINI_CLI_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = _env_int("GEMINI_CLI_SIGNATURE_DISK_TTL", 86400)

        # Initialize signature cache for Gemini 3 thoughtSignatures
        self._signature_cache = ProviderCache(
            GEMINI3_SIGNATURE_CACHE_FILE,
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

    # =========================================================================
    # CREDENTIAL PRIORITIZATION
    # =========================================================================

    def get_credential_priority(self, credential: str) -> Optional[int]:
        """
        Returns priority based on Gemini tier.
        Paid tiers: priority 1 (highest)
        Free/Legacy tiers: priority 2
        Unknown: priority 10 (lowest)

        Args:
            credential: The credential path

        Returns:
            Priority level (1-10) or None if tier not yet discovered
        """
        tier = self.project_tier_cache.get(credential)

        # Lazy load from file if not in cache
        if not tier:
            tier = self._load_tier_from_file(credential)

        if not tier:
            return None  # Not yet discovered

        # Paid tiers get highest priority
        if tier not in ["free-tier", "legacy-tier", "unknown"]:
            return 1

        # Free tier gets lower priority
        if tier == "free-tier":
            return 2

        # Legacy and unknown get even lower
        return 10

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
            return 1  # Only priority 1 (paid) credentials

        return None  # All other models have no restrictions

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
                f"GeminiCli: Loaded {len(loaded)} credential tiers from disk: "
                + ", ".join(
                    f"{tier}={count}" for tier, count in sorted(tier_counts.items())
                )
            )

        return loaded

    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        model_name = model.split("/")[-1].replace(":thinking", "")
        return model_name.startswith("gemini-3-")

    def _strip_gemini3_prefix(self, name: str) -> str:
        """Strip the Gemini 3 namespace prefix from a tool name."""
        if name and name.startswith(self._gemini3_tool_prefix):
            return name[len(self._gemini3_tool_prefix) :]
        return name

    async def _discover_project_id(
        self, credential_path: str, access_token: str, litellm_params: Dict[str, Any]
    ) -> str:
        """
        Discovers the Google Cloud Project ID, with caching and onboarding for new accounts.

        This follows the official Gemini CLI discovery flow:
        1. Check in-memory cache
        2. Check configured project_id override (litellm_params or env var)
        3. Check persisted project_id in credential file
        4. Call loadCodeAssist to check if user is already known (has currentTier)
           - If currentTier exists AND cloudaicompanionProject returned: use server's project
           - If currentTier exists but NO cloudaicompanionProject: use configured project_id (paid tier requires this)
           - If no currentTier: user needs onboarding
        5. Onboard user based on tier:
           - FREE tier: pass cloudaicompanionProject=None (server-managed)
           - PAID tier: pass cloudaicompanionProject=configured_project_id
        6. Fallback to GCP Resource Manager project listing
        """
        lib_logger.debug(
            f"Starting project discovery for credential: {credential_path}"
        )

        # Check in-memory cache first
        if credential_path in self.project_id_cache:
            cached_project = self.project_id_cache[credential_path]
            lib_logger.debug(f"Using cached project ID: {cached_project}")
            return cached_project

        # Check for configured project ID override (from litellm_params or env var)
        # This is REQUIRED for paid tier users per the official CLI behavior
        configured_project_id = litellm_params.get("project_id")
        if configured_project_id:
            lib_logger.debug(
                f"Found configured project_id override: {configured_project_id}"
            )

        # Load credentials from file to check for persisted project_id and tier
        # Skip for env:// paths (environment-based credentials don't persist to files)
        credential_index = self._parse_env_credential_path(credential_path)
        if credential_index is None:
            # Only try to load from file if it's not an env:// path
            try:
                with open(credential_path, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                persisted_project_id = metadata.get("project_id")
                persisted_tier = metadata.get("tier")

                if persisted_project_id:
                    lib_logger.info(
                        f"Loaded persisted project ID from credential file: {persisted_project_id}"
                    )
                    self.project_id_cache[credential_path] = persisted_project_id

                    # Also load tier if available
                    if persisted_tier:
                        self.project_tier_cache[credential_path] = persisted_tier
                        lib_logger.debug(f"Loaded persisted tier: {persisted_tier}")

                    return persisted_project_id
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                lib_logger.debug(f"Could not load persisted project ID from file: {e}")

        lib_logger.debug(
            "No cached or configured project ID found, initiating discovery..."
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        discovered_project_id = None
        discovered_tier = None

        async with httpx.AsyncClient() as client:
            # 1. Try discovery endpoint with loadCodeAssist
            lib_logger.debug(
                "Attempting project discovery via Code Assist loadCodeAssist endpoint..."
            )
            try:
                # Build metadata - include duetProject only if we have a configured project
                core_client_metadata = {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                }
                if configured_project_id:
                    core_client_metadata["duetProject"] = configured_project_id

                # Build load request - pass configured_project_id if available, otherwise None
                load_request = {
                    "cloudaicompanionProject": configured_project_id,  # Can be None
                    "metadata": core_client_metadata,
                }

                lib_logger.debug(
                    f"Sending loadCodeAssist request with cloudaicompanionProject={configured_project_id}"
                )
                response = await client.post(
                    f"{CODE_ASSIST_ENDPOINT}:loadCodeAssist",
                    headers=headers,
                    json=load_request,
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()

                # Log full response for debugging
                lib_logger.debug(
                    f"loadCodeAssist full response keys: {list(data.keys())}"
                )

                # Extract and log ALL tier information for debugging
                allowed_tiers = data.get("allowedTiers", [])
                current_tier = data.get("currentTier")

                lib_logger.debug(f"=== Tier Information ===")
                lib_logger.debug(f"currentTier: {current_tier}")
                lib_logger.debug(f"allowedTiers count: {len(allowed_tiers)}")
                for i, tier in enumerate(allowed_tiers):
                    tier_id = tier.get("id", "unknown")
                    is_default = tier.get("isDefault", False)
                    user_defined = tier.get("userDefinedCloudaicompanionProject", False)
                    lib_logger.debug(
                        f"  Tier {i + 1}: id={tier_id}, isDefault={is_default}, userDefinedProject={user_defined}"
                    )
                lib_logger.debug(f"========================")

                # Determine the current tier ID
                current_tier_id = None
                if current_tier:
                    current_tier_id = current_tier.get("id")
                    lib_logger.debug(f"User has currentTier: {current_tier_id}")

                # Check if user is already known to server (has currentTier)
                if current_tier_id:
                    # User is already onboarded - check for project from server
                    server_project = data.get("cloudaicompanionProject")

                    # Check if this tier requires user-defined project (paid tiers)
                    requires_user_project = any(
                        t.get("id") == current_tier_id
                        and t.get("userDefinedCloudaicompanionProject", False)
                        for t in allowed_tiers
                    )
                    is_free_tier = current_tier_id == "free-tier"

                    if server_project:
                        # Server returned a project - use it (server wins)
                        # This is the normal case for FREE tier users
                        project_id = server_project
                        lib_logger.debug(f"Server returned project: {project_id}")
                    elif configured_project_id:
                        # No server project but we have configured one - use it
                        # This is the PAID TIER case where server doesn't return a project
                        project_id = configured_project_id
                        lib_logger.debug(
                            f"No server project, using configured: {project_id}"
                        )
                    elif is_free_tier:
                        # Free tier user without server project - this shouldn't happen normally
                        # but let's not fail, just proceed to onboarding
                        lib_logger.debug(
                            "Free tier user with currentTier but no project - will try onboarding"
                        )
                        project_id = None
                    elif requires_user_project:
                        # Paid tier requires a project ID to be set
                        raise ValueError(
                            f"Paid tier '{current_tier_id}' requires setting GEMINI_CLI_PROJECT_ID environment variable. "
                            "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
                        )
                    else:
                        # Unknown tier without project - proceed carefully
                        lib_logger.warning(
                            f"Tier '{current_tier_id}' has no project and none configured - will try onboarding"
                        )
                        project_id = None

                    if project_id:
                        # Cache tier info
                        self.project_tier_cache[credential_path] = current_tier_id
                        discovered_tier = current_tier_id

                        # Log appropriately based on tier
                        is_paid = current_tier_id and current_tier_id not in [
                            "free-tier",
                            "legacy-tier",
                            "unknown",
                        ]
                        if is_paid:
                            lib_logger.info(
                                f"Using Gemini paid tier '{current_tier_id}' with project: {project_id}"
                            )
                        else:
                            lib_logger.info(
                                f"Discovered Gemini project ID via loadCodeAssist: {project_id}"
                            )

                        self.project_id_cache[credential_path] = project_id
                        discovered_project_id = project_id

                        # Persist to credential file
                        await self._persist_project_metadata(
                            credential_path, project_id, discovered_tier
                        )

                        return project_id

                # 2. User needs onboarding - no currentTier
                lib_logger.info(
                    "No existing Gemini session found (no currentTier), attempting to onboard user..."
                )

                # Determine which tier to onboard with
                onboard_tier = None
                for tier in allowed_tiers:
                    if tier.get("isDefault"):
                        onboard_tier = tier
                        break

                # Fallback to LEGACY tier if no default (requires user project)
                if not onboard_tier and allowed_tiers:
                    # Look for legacy-tier as fallback
                    for tier in allowed_tiers:
                        if tier.get("id") == "legacy-tier":
                            onboard_tier = tier
                            break
                    # If still no tier, use first available
                    if not onboard_tier:
                        onboard_tier = allowed_tiers[0]

                if not onboard_tier:
                    raise ValueError("No onboarding tiers available from server")

                tier_id = onboard_tier.get("id", "free-tier")
                requires_user_project = onboard_tier.get(
                    "userDefinedCloudaicompanionProject", False
                )

                lib_logger.debug(
                    f"Onboarding with tier: {tier_id}, requiresUserProject: {requires_user_project}"
                )

                # Build onboard request based on tier type (following official CLI logic)
                # FREE tier: cloudaicompanionProject = None (server-managed)
                # PAID tier: cloudaicompanionProject = configured_project_id (user must provide)
                is_free_tier = tier_id == "free-tier"

                if is_free_tier:
                    # Free tier uses server-managed project
                    onboard_request = {
                        "tierId": tier_id,
                        "cloudaicompanionProject": None,  # Server will create/manage
                        "metadata": core_client_metadata,
                    }
                    lib_logger.debug(
                        "Free tier onboarding: using server-managed project"
                    )
                else:
                    # Paid/legacy tier requires user-provided project
                    if not configured_project_id and requires_user_project:
                        raise ValueError(
                            f"Tier '{tier_id}' requires setting GEMINI_CLI_PROJECT_ID environment variable. "
                            "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
                        )
                    onboard_request = {
                        "tierId": tier_id,
                        "cloudaicompanionProject": configured_project_id,
                        "metadata": {
                            **core_client_metadata,
                            "duetProject": configured_project_id,
                        }
                        if configured_project_id
                        else core_client_metadata,
                    }
                    lib_logger.debug(
                        f"Paid tier onboarding: using project {configured_project_id}"
                    )

                lib_logger.debug("Initiating onboardUser request...")
                lro_response = await client.post(
                    f"{CODE_ASSIST_ENDPOINT}:onboardUser",
                    headers=headers,
                    json=onboard_request,
                    timeout=30,
                )
                lro_response.raise_for_status()
                lro_data = lro_response.json()
                lib_logger.debug(
                    f"Initial onboarding response: done={lro_data.get('done')}"
                )

                for i in range(150):  # Poll for up to 5 minutes (150 × 2s)
                    if lro_data.get("done"):
                        lib_logger.debug(
                            f"Onboarding completed after {i} polling attempts"
                        )
                        break
                    await asyncio.sleep(2)
                    if (i + 1) % 15 == 0:  # Log every 30 seconds
                        lib_logger.info(
                            f"Still waiting for onboarding completion... ({(i + 1) * 2}s elapsed)"
                        )
                    lib_logger.debug(
                        f"Polling onboarding status... (Attempt {i + 1}/150)"
                    )
                    lro_response = await client.post(
                        f"{CODE_ASSIST_ENDPOINT}:onboardUser",
                        headers=headers,
                        json=onboard_request,
                        timeout=30,
                    )
                    lro_response.raise_for_status()
                    lro_data = lro_response.json()

                if not lro_data.get("done"):
                    lib_logger.error("Onboarding process timed out after 5 minutes")
                    raise ValueError(
                        "Onboarding process timed out after 5 minutes. Please try again or contact support."
                    )

                # Extract project ID from LRO response
                # Note: onboardUser returns response.cloudaicompanionProject as an object with .id
                lro_response_data = lro_data.get("response", {})
                lro_project_obj = lro_response_data.get("cloudaicompanionProject", {})
                project_id = (
                    lro_project_obj.get("id")
                    if isinstance(lro_project_obj, dict)
                    else None
                )

                # Fallback to configured project if LRO didn't return one
                if not project_id and configured_project_id:
                    project_id = configured_project_id
                    lib_logger.debug(
                        f"LRO didn't return project, using configured: {project_id}"
                    )

                if not project_id:
                    lib_logger.error(
                        "Onboarding completed but no project ID in response and none configured"
                    )
                    raise ValueError(
                        "Onboarding completed, but no project ID was returned. "
                        "For paid tiers, set GEMINI_CLI_PROJECT_ID environment variable."
                    )

                lib_logger.debug(
                    f"Successfully extracted project ID from onboarding response: {project_id}"
                )

                # Cache tier info
                self.project_tier_cache[credential_path] = tier_id
                discovered_tier = tier_id
                lib_logger.debug(f"Cached tier information: {tier_id}")

                # Log concise message for paid projects
                is_paid = tier_id and tier_id not in ["free-tier", "legacy-tier"]
                if is_paid:
                    lib_logger.info(
                        f"Using Gemini paid tier '{tier_id}' with project: {project_id}"
                    )
                else:
                    lib_logger.info(
                        f"Successfully onboarded user and discovered project ID: {project_id}"
                    )

                self.project_id_cache[credential_path] = project_id
                discovered_project_id = project_id

                # Persist to credential file
                await self._persist_project_metadata(
                    credential_path, project_id, discovered_tier
                )

                return project_id

            except httpx.HTTPStatusError as e:
                error_body = ""
                try:
                    error_body = e.response.text
                except Exception:
                    pass
                if e.response.status_code == 403:
                    lib_logger.error(
                        f"Gemini Code Assist API access denied (403). Response: {error_body}"
                    )
                    lib_logger.error(
                        "Possible causes: 1) cloudaicompanion.googleapis.com API not enabled, 2) Wrong project ID for paid tier, 3) Account lacks permissions"
                    )
                elif e.response.status_code == 404:
                    lib_logger.warning(
                        f"Gemini Code Assist endpoint not found (404). Falling back to project listing."
                    )
                elif e.response.status_code == 412:
                    # Precondition Failed - often means wrong project for free tier onboarding
                    lib_logger.error(
                        f"Precondition failed (412): {error_body}. This may mean the project ID is incompatible with the selected tier."
                    )
                else:
                    lib_logger.warning(
                        f"Gemini onboarding/discovery failed with status {e.response.status_code}: {error_body}. Falling back to project listing."
                    )
            except httpx.RequestError as e:
                lib_logger.warning(
                    f"Gemini onboarding/discovery network error: {e}. Falling back to project listing."
                )

        # 3. Fallback to listing all available GCP projects (last resort)
        lib_logger.debug(
            "Attempting to discover project via GCP Resource Manager API..."
        )
        try:
            async with httpx.AsyncClient() as client:
                lib_logger.debug(
                    "Querying Cloud Resource Manager for available projects..."
                )
                response = await client.get(
                    "https://cloudresourcemanager.googleapis.com/v1/projects",
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                projects = response.json().get("projects", [])
                lib_logger.debug(f"Found {len(projects)} total projects")
                active_projects = [
                    p for p in projects if p.get("lifecycleState") == "ACTIVE"
                ]
                lib_logger.debug(f"Found {len(active_projects)} active projects")

                if not projects:
                    lib_logger.error(
                        "No GCP projects found for this account. Please create a project in Google Cloud Console."
                    )
                elif not active_projects:
                    lib_logger.error(
                        "No active GCP projects found. Please activate a project in Google Cloud Console."
                    )
                else:
                    project_id = active_projects[0]["projectId"]
                    lib_logger.info(
                        f"Discovered Gemini project ID from active projects list: {project_id}"
                    )
                    lib_logger.debug(
                        f"Selected first active project: {project_id} (out of {len(active_projects)} active projects)"
                    )
                    self.project_id_cache[credential_path] = project_id
                    discovered_project_id = project_id

                    # [NEW] Persist to credential file (no tier info from resource manager)
                    await self._persist_project_metadata(
                        credential_path, project_id, None
                    )

                    return project_id
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                lib_logger.error(
                    "Failed to list GCP projects due to a 403 Forbidden error. The Cloud Resource Manager API may not be enabled, or your account lacks the 'resourcemanager.projects.list' permission."
                )
            else:
                lib_logger.error(
                    f"Failed to list GCP projects with status {e.response.status_code}: {e}"
                )
        except httpx.RequestError as e:
            lib_logger.error(f"Network error while listing GCP projects: {e}")

        raise ValueError(
            "Could not auto-discover Gemini project ID. Possible causes:\n"
            "  1. The cloudaicompanion.googleapis.com API is not enabled (enable it in Google Cloud Console)\n"
            "  2. No active GCP projects exist for this account (create one in Google Cloud Console)\n"
            "  3. Account lacks necessary permissions\n"
            "To manually specify a project, set GEMINI_CLI_PROJECT_ID in your .env file."
        )

    async def _persist_project_metadata(
        self, credential_path: str, project_id: str, tier: Optional[str]
    ):
        """Persists project ID and tier to the credential file for faster future startups."""
        # Skip persistence for env:// paths (environment-based credentials)
        credential_index = self._parse_env_credential_path(credential_path)
        if credential_index is not None:
            lib_logger.debug(
                f"Skipping project metadata persistence for env:// credential path: {credential_path}"
            )
            return

        try:
            # Load current credentials
            with open(credential_path, "r") as f:
                creds = json.load(f)

            # Update metadata
            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}

            creds["_proxy_metadata"]["project_id"] = project_id
            if tier:
                creds["_proxy_metadata"]["tier"] = tier

            # Save back using the existing save method (handles atomic writes and permissions)
            await self._save_credentials(credential_path, creds)

            lib_logger.debug(
                f"Persisted project_id and tier to credential file: {credential_path}"
            )
        except Exception as e:
            lib_logger.warning(
                f"Failed to persist project metadata to credential file: {e}"
            )
            # Non-fatal - just means slower startup next time

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

                            # Add prefix for Gemini 3
                            if is_gemini_3 and self._enable_gemini3_tool_fix:
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
                                    lib_logger.warning(
                                        f"Missing thoughtSignature for first func call {tool_id}, using bypass"
                                    )
                                # Subsequent parallel calls: no signature field at all

                                first_func_in_msg = False

                            parts.append(func_part)

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                function_name = tool_call_id_to_name.get(tool_call_id)
                if function_name:
                    # Add prefix for Gemini 3
                    if is_gemini_3 and self._enable_gemini3_tool_fix:
                        function_name = f"{self._gemini3_tool_prefix}{function_name}"

                    # Wrap the tool response in a 'result' object
                    response_content = {"result": content}
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

    def _handle_reasoning_parameters(
        self, payload: Dict[str, Any], model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration.

        - Gemini 2.5: thinkingBudget (integer tokens)
        - Gemini 3: thinkingLevel (string: "low"/"high")
        """
        custom_reasoning_budget = payload.get("custom_reasoning_budget", False)
        reasoning_effort = payload.get("reasoning_effort")

        if "thinkingConfig" in payload.get("generationConfig", {}):
            return None

        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = self._is_gemini_3(model)

        # Only apply reasoning logic to supported models
        if not (is_gemini_25 or is_gemini_3):
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)
            return None

        # Gemini 3: String-based thinkingLevel
        if is_gemini_3:
            # Clean up the original payload
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)

            if reasoning_effort == "low":
                return {"thinkingLevel": "low", "include_thoughts": True}
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

                tool_call = {
                    "index": current_tool_idx,
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_call.get("args", {})),
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
        - Removes unsupported properties like `strict` and `additionalProperties`.
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
        schema.pop("additionalProperties", None)

        return schema

    def _enforce_strict_schema(self, schema: Any) -> Any:
        """
        Enforce strict JSON schema for Gemini 3 to prevent hallucinated parameters.

        Adds 'additionalProperties: false' recursively to all object schemas,
        which tells the model it CANNOT add properties not in the schema.
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
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

        # Add additionalProperties: false to object schemas
        if result.get("type") == "object" and "properties" in result:
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
                    schema = self._gemini_cli_transform_schema(
                        new_function["parameters"]
                    )
                    new_function["parametersJsonSchema"] = schema
                    del new_function["parameters"]
                elif "parametersJsonSchema" not in new_function:
                    # Set default empty schema if neither exists
                    new_function["parametersJsonSchema"] = {
                        "type": "object",
                        "properties": {},
                    }

                # Gemini 3 specific transformations
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    # Add namespace prefix to tool names
                    name = new_function.get("name", "")
                    if name:
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
                        timeout=600,
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
