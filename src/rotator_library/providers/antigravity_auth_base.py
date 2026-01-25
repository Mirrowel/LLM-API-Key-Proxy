# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/antigravity_auth_base.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx

from .google_oauth_base import GoogleOAuthBase

# Import tier utilities from shared module
# These are re-exported here for backwards compatibility with existing imports
from .utilities.gemini_shared_utils import (
    # Tier constants
    TIER_ULTRA,
    TIER_PRO,
    TIER_FREE,
    TIER_NAME_TO_CANONICAL,
    CANONICAL_TO_LEGACY,
    FREE_TIER_IDS,
    TIER_PRIORITIES,
    DEFAULT_TIER_PRIORITY,
    # Tier functions
    normalize_tier_name,
    is_free_tier,
    is_paid_tier,
    get_tier_priority,
    format_tier_for_display,
    get_tier_full_name,
    # Project ID extraction
    extract_project_id_from_response,
    # Credential loading helpers
    load_persisted_project_metadata,
    # Env file helpers
    build_project_tier_env_lines,
    # Endpoint constants
    ANTIGRAVITY_LOAD_ENDPOINT_ORDER,
    ANTIGRAVITY_ENDPOINT_FALLBACKS,
)

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# FALLBACK PROJECT ID
# =============================================================================
# When loadCodeAssist returns no project, uses this fallback unconditionally
# See: quota.rs:135 - let final_project_id = project_id.unwrap_or("bamboo-precept-lgxtn");
FALLBACK_PROJECT_ID = "bamboo-precept-lgxtn"


# Headers for Antigravity auth/discovery calls (loadCodeAssist, onboardUser)
# CRITICAL: User-Agent MUST be google-api-nodejs-client/* for standard-tier detection.
# Using antigravity/* UA causes server to return free-tier only (tested via matrix test).
# X-Goog-Api-Client value doesn't affect tier detection.
ANTIGRAVITY_AUTH_HEADERS = {
    "User-Agent": "google-api-nodejs-client/10.3.0",
    "X-Goog-Api-Client": "gl-node/22.18.0",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}


class AntigravityAuthBase(GoogleOAuthBase):
    """
    Antigravity OAuth2 authentication implementation.

    Inherits all OAuth functionality from GoogleOAuthBase with Antigravity-specific configuration.
    Uses Antigravity's OAuth credentials and includes additional scopes for cclog and experimentsandconfigs.

    Also provides project/tier discovery functionality that runs during authentication,
    ensuring credentials have their tier and project_id cached before any API requests.
    """

    CLIENT_ID = (
        "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
    )
    CLIENT_SECRET = "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"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",  # Antigravity-specific
        "https://www.googleapis.com/auth/experimentsandconfigs",  # Antigravity-specific
    ]
    ENV_PREFIX = "ANTIGRAVITY"
    CALLBACK_PORT = 51121
    CALLBACK_PATH = "/oauthcallback"

    def __init__(self):
        super().__init__()
        # Project and tier caches - shared between auth base and provider
        self.project_id_cache: Dict[str, str] = {}
        self.project_tier_cache: Dict[str, str] = {}
        self.tier_full_cache: Dict[str, str] = {}  # Full tier names for display
        self.tier_full_cache: Dict[str, str] = {}  # Full tier names for display

    # =========================================================================
    # POST-AUTH DISCOVERY HOOK
    # =========================================================================

    async def _post_auth_discovery(
        self, credential_path: str, access_token: str
    ) -> None:
        """
        Discover and cache tier/project information immediately after OAuth authentication.

        This is called by GoogleOAuthBase._perform_interactive_oauth() after successful auth,
        ensuring tier and project_id are cached during the authentication flow rather than
        waiting for the first API request.

        Args:
            credential_path: Path to the credential file
            access_token: The newly obtained access token
        """
        lib_logger.debug(
            f"Starting post-auth discovery for Antigravity credential: {Path(credential_path).name}"
        )

        # Skip if already discovered (shouldn't happen during fresh auth, but be defensive)
        if (
            credential_path in self.project_id_cache
            and credential_path in self.project_tier_cache
        ):
            lib_logger.debug(
                f"Tier and project already cached for {Path(credential_path).name}, skipping discovery"
            )
            return

        # Call _discover_project_id which handles tier/project discovery and persistence
        # Pass empty litellm_params since we're in auth context (no model-specific overrides)
        project_id = await self._discover_project_id(
            credential_path, access_token, litellm_params={}
        )

        # Use full tier name for post-auth log (one-time display)
        tier_full = self.tier_full_cache.get(credential_path)
        tier = tier_full or self.project_tier_cache.get(credential_path, "unknown")
        lib_logger.info(
            f"Post-auth discovery complete for {Path(credential_path).name}: "
            f"tier={tier}, project={project_id}"
        )

        # Use full tier name for post-auth log (one-time display)
        tier_full = self.tier_full_cache.get(credential_path)
        tier = tier_full or self.project_tier_cache.get(credential_path, "unknown")
        lib_logger.info(
            f"Post-auth discovery complete for {Path(credential_path).name}: "
            f"tier={tier}, project={project_id}"
        )

        # Use full tier name for post-auth log (one-time display)
        tier_full = self.tier_full_cache.get(credential_path)
        tier = tier_full or self.project_tier_cache.get(credential_path, "unknown")
        lib_logger.info(
            f"Post-auth discovery complete for {Path(credential_path).name}: "
            f"tier={tier}, project={project_id}"
        )

    # =========================================================================
    # ENDPOINT FALLBACK HELPERS
    # =========================================================================

    async def _call_load_code_assist(
        self,
        client: httpx.AsyncClient,
        access_token: str,
        configured_project_id: Optional[str],
        headers: Dict[str, str],
    ) -> tuple:
        """
        Call loadCodeAssist with endpoint fallback chain.

        Tries endpoints in ANTIGRAVITY_LOAD_ENDPOINT_ORDER (prod first for better
        project resolution, then fallback to sandbox).

        Args:
            client: httpx async client
            access_token: OAuth access token
            configured_project_id: User-configured project ID (or None)
            headers: Request headers

        Returns:
            Tuple of (response_data, successful_endpoint) or (None, None) on failure
        """
        from .utilities.gemini_shared_utils import ANTIGRAVITY_LOAD_ENDPOINT_ORDER

        core_client_metadata = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
        if configured_project_id:
            core_client_metadata["duetProject"] = configured_project_id

        load_request = {
            "cloudaicompanionProject": configured_project_id,
            "metadata": core_client_metadata,
        }

        last_error = None
        for endpoint in ANTIGRAVITY_LOAD_ENDPOINT_ORDER:
            try:
                lib_logger.debug(f"Trying loadCodeAssist at {endpoint}")
                response = await client.post(
                    f"{endpoint}:loadCodeAssist",
                    headers=headers,
                    json=load_request,
                    timeout=15,
                )
                if response.status_code == 200:
                    data = response.json()
                    lib_logger.debug(f"loadCodeAssist succeeded at {endpoint}")
                    return data, endpoint
                lib_logger.debug(
                    f"loadCodeAssist returned {response.status_code} at {endpoint}"
                )
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                lib_logger.debug(f"loadCodeAssist failed at {endpoint}: {e}")
                last_error = str(e)
                continue

        lib_logger.warning(
            f"All loadCodeAssist endpoints failed. Last error: {last_error}"
        )
        return None, None

    async def _call_onboard_user(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        onboard_request: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Call onboardUser with endpoint fallback chain.

        Tries endpoints in ANTIGRAVITY_ENDPOINT_FALLBACKS (daily first, then prod).

        Args:
            client: httpx async client
            headers: Request headers
            onboard_request: Onboarding request payload

        Returns:
            Response data dict or None on failure
        """
        from .utilities.gemini_shared_utils import ANTIGRAVITY_ENDPOINT_FALLBACKS

        last_error = None
        for endpoint in ANTIGRAVITY_ENDPOINT_FALLBACKS:
            try:
                lib_logger.debug(f"Trying onboardUser at {endpoint}")
                response = await client.post(
                    f"{endpoint}:onboardUser",
                    headers=headers,
                    json=onboard_request,
                    timeout=30,
                )
                if response.status_code == 200:
                    lib_logger.debug(f"onboardUser succeeded at {endpoint}")
                    return response.json()
                lib_logger.debug(
                    f"onboardUser returned {response.status_code} at {endpoint}"
                )
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                lib_logger.debug(f"onboardUser failed at {endpoint}: {e}")
                last_error = str(e)
                continue

        lib_logger.warning(
            f"All onboardUser endpoints failed. Last error: {last_error}"
        )
        return None

    # =========================================================================
    # PROJECT ID DISCOVERY
    # =========================================================================

    async def _discover_project_id(
        self, credential_path: str, access_token: str, litellm_params: Dict[str, Any]
    ) -> str:
        """
        Discovers the Google Cloud Project ID, with caching and onboarding for new accounts.

        This follows the official Gemini CLI discovery flow adapted for Antigravity:
        1. Check in-memory cache
        2. Check configured project_id override (litellm_params or env var)
        3. Check persisted project_id in credential file
        4. Call loadCodeAssist to check if user is already known (has currentTier)
           - If currentTier exists AND cloudaicompanionProject returned: use server's project
           - If no currentTier: user needs onboarding
        5. Onboard user (FREE tier: pass cloudaicompanionProject=None for server-managed)
        6. Fallback to GCP Resource Manager project listing

        Note: Unlike GeminiCli, Antigravity doesn't use tier-based credential prioritization,
        but we still cache tier info for debugging and consistency.
        """
        lib_logger.debug(
            f"Starting Antigravity project discovery for credential: {credential_path}"
        )

        # Check in-memory cache first
        if credential_path in self.project_id_cache:
            cached_project = self.project_id_cache[credential_path]
            lib_logger.debug(f"Using cached project ID: {cached_project}")
            return cached_project

        # Check for configured project ID override (from litellm_params or env var)
        configured_project_id = (
            litellm_params.get("project_id")
            or os.getenv("ANTIGRAVITY_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        if configured_project_id:
            lib_logger.debug(
                f"Found configured project_id override: {configured_project_id}"
            )

        # Load credentials to check for persisted/configured project_id and tier
        credential_index = self._parse_env_credential_path(credential_path)
        persisted_project_id = load_persisted_project_metadata(
            credential_path,
            credential_index,
            self._credentials_cache,
            self.project_id_cache,
            self.project_tier_cache,
            self.tier_full_cache,
        )
        if persisted_project_id:
            return persisted_project_id

        lib_logger.debug(
            "No cached or configured project ID found, initiating discovery..."
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_AUTH_HEADERS,
        }

        # Build core metadata for API requests
        core_client_metadata = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
        if configured_project_id:
            core_client_metadata["duetProject"] = configured_project_id

        discovered_project_id = None
        discovered_tier = None
        discovered_tier_full = None
        discovered_tier_full = None

        async with httpx.AsyncClient() as client:
            # 1. Try discovery endpoint with loadCodeAssist using endpoint fallback
            lib_logger.debug(
                "Attempting project discovery via Code Assist loadCodeAssist endpoint..."
            )
            try:
                # Use helper with endpoint fallback chain
                data, successful_endpoint = await self._call_load_code_assist(
                    client, access_token, configured_project_id, headers
                )

                if data is None:
                    # All endpoints failed - skip to GCP Resource Manager fallback
                    raise httpx.HTTPStatusError(
                        "All loadCodeAssist endpoints failed",
                        request=None,
                        response=None,
                    )

                lib_logger.debug(
                    f"loadCodeAssist succeeded at {successful_endpoint}, response keys: {list(data.keys())}"
                )

                # Extract tier information
                # Canonical prioritizes paidTier over currentTier for accurate subscription detection
                allowed_tiers = data.get("allowedTiers", [])
                current_tier = data.get("currentTier")
                paid_tier = data.get(
                    "paidTier"
                )  # Added: Canonical-style tier detection

                lib_logger.debug(f"=== Tier Information ===")
                lib_logger.debug(f"paidTier: {paid_tier}")
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

                # Determine tier ID with Canonical-style priority: paidTier > currentTier
                # This matches quota.rs:88-91 logic for accurate subscription detection
                effective_tier_id = None
                if paid_tier and paid_tier.get("id"):
                    effective_tier_id = paid_tier.get("id")
                    lib_logger.debug(f"Using paidTier: {effective_tier_id}")
                elif current_tier and current_tier.get("id"):
                    effective_tier_id = current_tier.get("id")
                    lib_logger.debug(
                        f"Using currentTier (paidTier not available): {effective_tier_id}"
                    )

                # Normalize to canonical tier name (ULTRA, PRO, FREE)
                if effective_tier_id:
                    canonical_tier = normalize_tier_name(effective_tier_id)
                    lib_logger.debug(
                        f"Canonical tier: {canonical_tier} (from {effective_tier_id})"
                    )

                # Check if user is already known to server (has tier info)
                if effective_tier_id:
                    # User has tier info - use Canonical-style project selection
                    server_project = extract_project_id_from_response(data)

                    # Canonical-style project selection (quota.rs:135):
                    # 1. Server project (if returned)
                    # 2. Configured project (from env)
                    # 3. Fallback project (always works)
                    if server_project:
                        project_id = server_project
                        lib_logger.debug(f"Server returned project: {project_id}")
                    elif configured_project_id:
                        project_id = configured_project_id
                        lib_logger.debug(f"Using configured project: {project_id}")
                    else:
                        # Canonical fallback: project_id.unwrap_or("bamboo-precept-lgxtn")
                        project_id = FALLBACK_PROJECT_ID
                        lib_logger.info(
                            f"No server/configured project for tier '{effective_tier_id}' - using Canonical fallback: {project_id}"
                        )

                    # We have a project now (guaranteed by fallback)
                    # Cache tier info - use canonical tier name for consistency
                    canonical = (
                        normalize_tier_name(effective_tier_id) or effective_tier_id
                    )
                    self.project_tier_cache[credential_path] = canonical
                    discovered_tier = canonical

                    # Get and cache full tier name for display
                    tier_full = get_tier_full_name(effective_tier_id)
                    self.tier_full_cache[credential_path] = tier_full
                    discovered_tier_full = tier_full

                    # Log with full tier name for discovery messages
                    lib_logger.info(
                        f"Discovered Antigravity tier '{tier_full}' with project: {project_id}"
                    )

                    self.project_id_cache[credential_path] = project_id
                    discovered_project_id = project_id

                    # Persist to credential file
                    await self._persist_project_metadata(
                        credential_path,
                        project_id,
                        discovered_tier,
                        discovered_tier_full,
                    )

                    return project_id

                # 2. User needs onboarding - no currentTier or no project found
                lib_logger.info(
                    "No existing Antigravity session found (no currentTier), attempting to onboard user..."
                )

                # Determine which tier to onboard with
                onboard_tier = None
                for tier in allowed_tiers:
                    if tier.get("isDefault"):
                        onboard_tier = tier
                        break

                # Fallback to legacy tier if no default
                if not onboard_tier and allowed_tiers:
                    for tier in allowed_tiers:
                        if tier.get("id") == "legacy-tier":
                            onboard_tier = tier
                            break
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

                # Build onboard request based on tier type
                # FREE tier: cloudaicompanionProject = None (server-managed)
                # PAID tier: cloudaicompanionProject = configured_project_id
                tier_is_free = is_free_tier(tier_id)

                if tier_is_free:
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
                            f"Tier '{tier_id}' requires setting ANTIGRAVITY_PROJECT_ID environment variable."
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

                lib_logger.debug(
                    "Initiating onboardUser request with endpoint fallback..."
                )
                lro_data = await self._call_onboard_user(
                    client, headers, onboard_request
                )

                if lro_data is None:
                    raise ValueError(
                        "All onboardUser endpoints failed. Cannot onboard user."
                    )

                lib_logger.debug(
                    f"Initial onboarding response: done={lro_data.get('done')}"
                )

                # Poll for onboarding completion (up to 60 seconds)
                for i in range(30):  # 30 × 2s = 60 seconds
                    if lro_data.get("done"):
                        lib_logger.debug(f"Onboarding completed after {i * 2}s")
                        break
                    await asyncio.sleep(2)
                    if (i + 1) % 10 == 0:  # Log every 20 seconds
                        lib_logger.info(
                            f"Still waiting for onboarding completion... ({(i + 1) * 2}s elapsed)"
                        )
                    lib_logger.debug(
                        f"Polling onboarding status... (Attempt {i + 1}/30)"
                    )
                    lro_data = await self._call_onboard_user(
                        client, headers, onboard_request
                    )
                    if lro_data is None:
                        lib_logger.warning("onboardUser endpoint failed during polling")
                        break

                if not lro_data or not lro_data.get("done"):
                    lib_logger.error("Onboarding process timed out after 60 seconds")
                    raise ValueError(
                        "Onboarding process timed out after 60 seconds. Please try again or contact support."
                    )

                # Extract project ID from LRO response using helper
                # Note: onboardUser returns response.cloudaicompanionProject as an object with .id
                lro_response_data = lro_data.get("response", {})
                project_id = extract_project_id_from_response(lro_response_data)

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
                        "For paid tiers, set ANTIGRAVITY_PROJECT_ID environment variable."
                    )

                lib_logger.debug(
                    f"Successfully extracted project ID from onboarding response: {project_id}"
                )

                # Cache tier info - use canonical tier name for consistency
                canonical_tier = normalize_tier_name(tier_id) or tier_id
                self.project_tier_cache[credential_path] = canonical_tier
                discovered_tier = canonical_tier

                # Get and cache full tier name for display
                tier_full = get_tier_full_name(tier_id)
                self.tier_full_cache[credential_path] = tier_full
                discovered_tier_full = tier_full
                lib_logger.debug(
                    f"Cached tier information: {canonical_tier} (full: {tier_full})"
                )

                # Log with full tier name for onboarding messages
                lib_logger.info(
                    f"Onboarded Antigravity credential with tier '{tier_full}', project: {project_id}"
                )

                self.project_id_cache[credential_path] = project_id
                discovered_project_id = project_id

                # Persist to credential file
                await self._persist_project_metadata(
                    credential_path, project_id, discovered_tier, discovered_tier_full
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
                        f"Antigravity Code Assist API access denied (403). Response: {error_body}"
                    )
                    lib_logger.error(
                        "Possible causes: 1) cloudaicompanion.googleapis.com API not enabled, 2) Wrong project ID for paid tier, 3) Account lacks permissions"
                    )
                elif e.response.status_code == 404:
                    lib_logger.warning(
                        f"Antigravity Code Assist endpoint not found (404). Falling back to project listing."
                    )
                elif e.response.status_code == 412:
                    # Precondition Failed - often means wrong project for free tier onboarding
                    lib_logger.error(
                        f"Precondition failed (412): {error_body}. This may mean the project ID is incompatible with the selected tier."
                    )
                else:
                    lib_logger.warning(
                        f"Antigravity onboarding/discovery failed with status {e.response.status_code}: {error_body}. Falling back to project listing."
                    )
            except httpx.RequestError as e:
                lib_logger.warning(
                    f"Antigravity onboarding/discovery network error: {e}. Falling back to project listing."
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
                        f"Discovered Antigravity project ID from active projects list: {project_id}"
                    )
                    lib_logger.debug(
                        f"Selected first active project: {project_id} (out of {len(active_projects)} active projects)"
                    )
                    self.project_id_cache[credential_path] = project_id
                    discovered_project_id = project_id

                    # Persist to credential file (no tier info from resource manager)
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
            "Could not auto-discover Antigravity project ID. Possible causes:\n"
            "  1. The cloudaicompanion.googleapis.com API is not enabled (enable it in Google Cloud Console)\n"
            "  2. No active GCP projects exist for this account (create one in Google Cloud Console)\n"
            "  3. Account lacks necessary permissions\n"
            "To manually specify a project, set ANTIGRAVITY_PROJECT_ID in your .env file."
        )

    # =========================================================================
    # CREDENTIAL MANAGEMENT OVERRIDES
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Return the file prefix for Antigravity credentials."""
        return "antigravity"

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """
        Generate .env file lines for an Antigravity credential.

        Includes tier and project_id from _proxy_metadata.
        """
        # Get base lines from parent class
        lines = super().build_env_lines(creds, cred_number)

        # Add project_id and tier using shared helper
        lines.extend(build_project_tier_env_lines(creds, self.ENV_PREFIX, cred_number))

        return lines
