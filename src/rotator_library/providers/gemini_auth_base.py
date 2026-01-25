# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/gemini_auth_base.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx

from .google_oauth_base import GoogleOAuthBase
from .utilities.gemini_shared_utils import (
    CODE_ASSIST_ENDPOINT,
    # Tier utilities
    normalize_tier_name,
    is_free_tier,
    is_paid_tier,
    get_tier_full_name,
    # Project ID extraction
    extract_project_id_from_response,
    # Credential loading helpers
    load_persisted_project_metadata,
    # Env file helpers
    build_project_tier_env_lines,
)

# Service Usage API for checking enabled APIs
SERVICE_USAGE_API = "https://serviceusage.googleapis.com/v1"

lib_logger = logging.getLogger("rotator_library")

# Headers for Gemini CLI auth/discovery calls (loadCodeAssist, onboardUser, etc.)
#
# For OAuth/Code Assist path, native gemini-cli only sends:
# - Content-Type: application/json
# - Authorization: Bearer <token>
# - User-Agent: GeminiCLI/${version} (${platform}; ${arch})
#
# Headers NOT sent by native CLI (confirmed via explore agent analysis of server.ts):
# - X-Goog-Api-Client: Not used in Code Assist path
# - Client-Metadata: Sent in REQUEST BODY for these endpoints, not as HTTP header
#
# Note: The commented headers below previously worked well for SDK fingerprinting.
# Uncomment if you want to try SDK mimicry for potential rate limit benefits.
#
# Source: gemini-cli/packages/core/src/code_assist/server.ts:284-290
GEMINI_CLI_AUTH_HEADERS = {
    "User-Agent": "GeminiCLI/0.26.0 (win32; x64)",
    # -------------------------------------------------------------------------
    # COMMENTED OUT - Not sent by native gemini-cli for OAuth/Code Assist path
    # -------------------------------------------------------------------------
    # "X-Goog-Api-Client": "gl-node/22.17.0 gdcl/1.30.0",  # SDK mimicry - not used by native CLI
    # "Client-Metadata": (                                  # Sent in body, not as header
    #     "ideType=IDE_UNSPECIFIED,"
    #     "pluginType=GEMINI,"
    #     "ideVersion=0.26.0,"
    #     "platform=WINDOWS_AMD64,"
    #     "updateChannel=stable"
    # ),
}


class GeminiAuthBase(GoogleOAuthBase):
    """
    Gemini CLI OAuth2 authentication implementation.

    Inherits all OAuth functionality from GoogleOAuthBase with Gemini-specific configuration.

    Also provides project/tier discovery functionality that runs during authentication,
    ensuring credentials have their tier and project_id cached before any API requests.
    """

    CLIENT_ID = (
        "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
    )
    CLIENT_SECRET = "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"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    ENV_PREFIX = "GEMINI_CLI"
    CALLBACK_PORT = 8085
    CALLBACK_PATH = "/oauth2callback"

    def __init__(self):
        super().__init__()
        # Project and tier caches - shared between auth base and provider
        self.project_id_cache: Dict[str, str] = {}
        self.project_tier_cache: Dict[str, str] = {}
        self.tier_full_cache: Dict[str, str] = {}  # Full tier names for display
        self.tier_full_cache: Dict[str, str] = {}  # Full tier names for display

    # =========================================================================
    # GCP PROJECT SCANNING
    # =========================================================================

    async def _scan_gcp_projects_for_code_assist(
        self, access_token: str, headers: Dict[str, str]
    ) -> Optional[tuple]:
        """
        Scan GCP projects to find one with cloudaicompanion.googleapis.com enabled.

        This is used as a fallback when loadCodeAssist doesn't return a project
        (e.g., for accounts with manually created projects that have Code Assist enabled).

        Args:
            access_token: Valid OAuth access token
            headers: Request headers for Code Assist API calls

        Returns:
            Tuple of (project_id, tier) if found, or (None, None) if not found
        """
        lib_logger.debug("Scanning GCP projects for Code Assist API...")

        async with httpx.AsyncClient() as client:
            # Step 1: List all active GCP projects
            try:
                response = await client.get(
                    "https://cloudresourcemanager.googleapis.com/v1/projects",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=20,
                )
                if response.status_code != 200:
                    lib_logger.debug(
                        f"Failed to list GCP projects: {response.status_code}"
                    )
                    return None, None

                projects = [
                    p
                    for p in response.json().get("projects", [])
                    if p.get("lifecycleState") == "ACTIVE"
                ]
                lib_logger.debug(f"Found {len(projects)} active GCP projects")

                if not projects:
                    return None, None

            except Exception as e:
                lib_logger.debug(f"Error listing GCP projects: {e}")
                return None, None

            # Step 2: Check which projects have cloudaicompanion.googleapis.com enabled
            candidate_projects = []
            for project in projects:
                project_id = project.get("projectId")
                service_url = f"{SERVICE_USAGE_API}/projects/{project_id}/services/cloudaicompanion.googleapis.com"

                try:
                    svc_response = await client.get(
                        service_url,
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=10,
                    )
                    if svc_response.status_code == 200:
                        state = svc_response.json().get("state", "")
                        if state == "ENABLED":
                            lib_logger.debug(
                                f"Project '{project_id}' has cloudaicompanion.googleapis.com ENABLED"
                            )
                            candidate_projects.append(project_id)
                        else:
                            lib_logger.debug(
                                f"Project '{project_id}' cloudaicompanion state: {state}"
                            )
                except Exception as e:
                    lib_logger.debug(
                        f"Error checking cloudaicompanion API for '{project_id}': {e}"
                    )

            if not candidate_projects:
                lib_logger.debug(
                    "No GCP projects with cloudaicompanion.googleapis.com enabled found"
                )
                return None, None

            # Step 3: Test candidate projects with loadCodeAssist to verify and get tier
            lib_logger.debug(
                f"Testing {len(candidate_projects)} candidate projects with loadCodeAssist..."
            )

            for project_id in candidate_projects:
                try:
                    test_request = {
                        "cloudaicompanionProject": project_id,
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                            "duetProject": project_id,
                        },
                    }

                    response = await client.post(
                        f"{CODE_ASSIST_ENDPOINT}:loadCodeAssist",
                        headers=headers,
                        json=test_request,
                        timeout=15,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        current_tier = data.get("currentTier", {})
                        paid_tier = data.get("paidTier", {})

                        # Determine effective tier (paidTier > currentTier)
                        effective_tier_id = None
                        if paid_tier and paid_tier.get("id"):
                            effective_tier_id = paid_tier.get("id")
                        elif current_tier and current_tier.get("id"):
                            effective_tier_id = current_tier.get("id")

                        if effective_tier_id:
                            canonical_tier = (
                                normalize_tier_name(effective_tier_id)
                                or effective_tier_id
                            )
                            lib_logger.info(
                                f"Found Code Assist project via GCP scan: {project_id} (tier={canonical_tier})"
                            )
                            # Return raw tier ID for full name lookup, not canonical
                            return project_id, effective_tier_id

                except Exception as e:
                    lib_logger.debug(
                        f"Error testing project '{project_id}' with loadCodeAssist: {e}"
                    )

            lib_logger.debug("No working Code Assist projects found via GCP scan")
            return None, None

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
            f"Starting post-auth discovery for GeminiCli credential: {Path(credential_path).name}"
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
    # PROJECT ID DISCOVERY
    # =========================================================================

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
        configured_project_id = litellm_params.get("project_id") or os.getenv(
            "GEMINI_CLI_PROJECT_ID"
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
            **GEMINI_CLI_AUTH_HEADERS,
        }

        discovered_project_id = None
        discovered_tier = None
        discovered_tier_full = None
        discovered_tier_full = None

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
                    # User has tier info - check for project from server
                    # Use helper to handle both string and object formats
                    server_project = extract_project_id_from_response(data)

                    # Project selection: server > configured > GCP scan > onboarding
                    if server_project:
                        project_id = server_project
                        lib_logger.debug(f"Server returned project: {project_id}")
                    elif configured_project_id:
                        project_id = configured_project_id
                        lib_logger.debug(f"Using configured project: {project_id}")
                    else:
                        # No project from server or config - try scanning GCP projects
                        # This handles accounts with manually created Code Assist projects
                        lib_logger.debug(
                            f"Tier '{effective_tier_id}' detected but no project - scanning GCP projects..."
                        )
                        (
                            scanned_project,
                            scanned_tier,
                        ) = await self._scan_gcp_projects_for_code_assist(
                            access_token, headers
                        )
                        if scanned_project:
                            project_id = scanned_project
                            # Use scanned tier if available, otherwise use what we have
                            if scanned_tier:
                                effective_tier_id = scanned_tier
                            lib_logger.info(
                                f"Discovered project via GCP scan: {project_id}"
                            )
                        else:
                            # No project found via GCP scan either - will fall through to onboarding
                            lib_logger.debug(
                                f"No Code Assist project found via GCP scan - will try onboarding"
                            )
                            project_id = None

                    if project_id:
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
                            f"Discovered Gemini tier '{tier_full}' with project: {project_id}"
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
                # For ALL tiers (free and paid): cloudaicompanionProject can be None
                # The server will create a project automatically if none is provided
                # If user has configured a project, use it; otherwise let server decide
                tier_is_free = is_free_tier(tier_id)

                # For paid tiers, first try to find an existing Code Assist project
                onboard_project_id = configured_project_id
                if not tier_is_free and not onboard_project_id:
                    lib_logger.debug(
                        "Paid tier with no configured project - checking for existing Code Assist projects..."
                    )
                    scanned_project, _ = await self._scan_gcp_projects_for_code_assist(
                        access_token, headers
                    )
                    if scanned_project:
                        onboard_project_id = scanned_project
                        lib_logger.info(
                            f"Found existing Code Assist project for onboarding: {scanned_project}"
                        )
                    else:
                        lib_logger.debug(
                            "No existing Code Assist project found - server will create one"
                        )

                # Build onboard request - server will create project if None
                onboard_request = {
                    "tierId": tier_id,
                    "cloudaicompanionProject": onboard_project_id,  # Can be None - server will create
                    "metadata": {
                        **core_client_metadata,
                        "duetProject": onboard_project_id,
                    }
                    if onboard_project_id
                    else core_client_metadata,
                }

                if onboard_project_id:
                    lib_logger.debug(
                        f"Onboarding with user-provided project: {onboard_project_id}"
                    )
                else:
                    lib_logger.debug(
                        "Onboarding with server-managed project (will be created by server)"
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

                # Extract project ID from LRO response using helper
                # This handles both string and object formats for cloudaicompanionProject
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
                        "For paid tiers, set GEMINI_CLI_PROJECT_ID environment variable."
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
                    f"Onboarded Gemini credential with tier '{tier_full}', project: {project_id}"
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
            "Could not auto-discover Gemini project ID. Possible causes:\n"
            "  1. The cloudaicompanion.googleapis.com API is not enabled (enable it in Google Cloud Console)\n"
            "  2. No active GCP projects exist for this account (create one in Google Cloud Console)\n"
            "  3. Account lacks necessary permissions\n"
            "To manually specify a project, set GEMINI_CLI_PROJECT_ID in your .env file."
        )

    # =========================================================================
    # CREDENTIAL MANAGEMENT OVERRIDES
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Return the file prefix for Gemini CLI credentials."""
        return "gemini_cli"

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """
        Generate .env file lines for a Gemini CLI credential.

        Includes tier and project_id from _proxy_metadata.
        """
        # Get base lines from parent class
        lines = super().build_env_lines(creds, cred_number)

        # Add project_id and tier using shared helper
        lines.extend(build_project_tier_env_lines(creds, self.ENV_PREFIX, cred_number))

        return lines
