# src/rotator_library/providers/gemini_auth_base.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx

from .google_oauth_base import GoogleOAuthBase
from .utilities.gemini_shared_utils import CODE_ASSIST_ENDPOINT

lib_logger = logging.getLogger("rotator_library")


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

        tier = self.project_tier_cache.get(credential_path, "unknown")
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
        if credential_index is None:
            # File-based credentials: load from file
            try:
                with open(credential_path, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                persisted_project_id = metadata.get("project_id")
                persisted_tier = metadata.get("tier")

                if persisted_project_id:
                    lib_logger.debug(
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
        else:
            # Env-based credentials: load from credentials cache
            # The credentials were already loaded by _load_from_env() which reads
            # {PREFIX}_{N}_PROJECT_ID and {PREFIX}_{N}_TIER into _proxy_metadata
            if credential_path in self._credentials_cache:
                creds = self._credentials_cache[credential_path]
                metadata = creds.get("_proxy_metadata", {})
                env_project_id = metadata.get("project_id")
                env_tier = metadata.get("tier")

                if env_project_id:
                    lib_logger.debug(
                        f"Loaded project ID from env credential metadata: {env_project_id}"
                    )
                    self.project_id_cache[credential_path] = env_project_id

                    if env_tier:
                        self.project_tier_cache[credential_path] = env_tier
                        lib_logger.debug(
                            f"Loaded tier from env credential metadata: {env_tier}"
                        )

                    return env_project_id

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

        # Add Gemini-specific fields (tier and project_id)
        metadata = creds.get("_proxy_metadata", {})
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        project_id = metadata.get("project_id", "")
        tier = metadata.get("tier", "")

        if project_id:
            lines.append(f"{prefix}_PROJECT_ID={project_id}")
        if tier:
            lines.append(f"{prefix}_TIER={tier}")

        return lines
