# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
GitHub Copilot OAuth2 authentication using Device Flow.

This is fundamentally different from Google/Anthropic OAuth providers:
- Uses GitHub's Device Flow instead of Authorization Code Flow
- Two-token system:
  1. GitHub OAuth token (long-lived, used as "refresh token")
  2. Copilot API token (short-lived, ~30 min, used as "access token")
- The Copilot API token contains a proxy-ep field that determines the
  correct API base URL (e.g., api.individual.githubcopilot.com)

Based on:
- https://github.com/sst/opencode-copilot-auth
- https://github.com/badlogic/pi-mono (packages/ai/src/utils/oauth/github-copilot.ts)
"""

import asyncio
import json
import logging
import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from dataclasses import dataclass, field

from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# OAUTH CONFIGURATION
# =============================================================================

# GitHub Copilot OAuth Client ID (from VS Code Copilot extension, base64-encoded)
# Decodes to "Iv1.b507a08c87ecfe98"
import base64

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()

# Headers that mimic the official Copilot client
COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# Token refresh buffer (5 minutes before expiry)
REFRESH_EXPIRY_BUFFER_SECONDS = 5 * 60


@dataclass
class CopilotCredentialSetupResult:
    """Standardized result structure for Copilot credential setup operations."""
    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    account_id: Optional[str] = None
    sku: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


def _get_base_url_from_token(token: str) -> Optional[str]:
    """
    Parse the proxy-ep from a Copilot token and convert to API base URL.

    Token format: tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...
    Returns API URL like https://api.individual.githubcopilot.com

    Based on pi-mono's getBaseUrlFromToken().
    """
    if not token:
        return None
    import re
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    # Convert proxy.xxx to api.xxx
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


class CopilotAuthBase:
    """
    GitHub Copilot OAuth2 authentication using Device Flow.

    Key differences from other OAuth providers:
    - Uses GitHub Device Flow (polls for authorization)
    - Two-token system: GitHub OAuth token + Copilot API token
    - Copilot API tokens expire quickly (~30 min) and need frequent refresh
    - Base URL is dynamically extracted from the Copilot token's proxy-ep field

    Environment variables (numbered, per-credential):
        COPILOT_N_GITHUB_TOKEN - Long-lived GitHub OAuth token (required)

    Legacy single-credential format:
        COPILOT_GITHUB_TOKEN - Single GitHub OAuth token

    Subclasses may override:
        - ENV_PREFIX: Prefix for environment variables (default: "COPILOT")
    """

    ENV_PREFIX = "COPILOT"

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    # =========================================================================
    # CREDENTIAL LOADING
    # =========================================================================

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://copilot/0" - Legacy single credential
        - "env://copilot/1" - First numbered credential
        - "env://copilot/2" - Second numbered credential
        """
        if not path.startswith("env://"):
            return None
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables.

        For Copilot, we only need:
        - COPILOT_GITHUB_TOKEN (legacy) or COPILOT_N_GITHUB_TOKEN (numbered)

        The Copilot API token is fetched dynamically and cached.
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_login = f"copilot-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_login = "copilot-user"

        # The "refresh_token" for Copilot is the GitHub OAuth token
        github_token = os.getenv(f"{prefix}_GITHUB_TOKEN")
        if not github_token:
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

        creds = {
            "refresh_token": github_token,  # GitHub OAuth token
            "access_token": "",  # Copilot API token (fetched on demand)
            "expiry_date": 0,  # Will be set when Copilot token is fetched
            "_proxy_metadata": {
                "login": os.getenv(f"{prefix}_LOGIN", default_login),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Load credentials from cache, environment, or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check for virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    lib_logger.info(
                        f"Using {self.ENV_PREFIX} credentials from environment "
                        f"(index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for {self.ENV_PREFIX} "
                        f"credential index {credential_index} not found"
                    )

            # Try file-based loading first; fall back to legacy env
            # vars only when the file doesn't exist.  Previously the
            # legacy env check came first, which silently shadowed a
            # valid file credential when COPILOT_GITHUB_TOKEN was set.
            try:
                lib_logger.debug(
                    f"Loading {self.ENV_PREFIX} credentials from file: {path}"
                )
                with open(path, "r") as f:
                    creds = json.load(f)
                self._credentials_cache[path] = creds
                return creds
            except FileNotFoundError:
                # File not present — fall back to legacy env vars
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"Using {self.ENV_PREFIX} credentials from environment variables "
                        f"(credential file not found at '{path}')"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"{self.ENV_PREFIX} OAuth credential file not found at '{path}' "
                    f"and no environment variables set"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials "
                    f"from '{path}': {e}"
                )

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials to file (no-op for env-based credentials)."""
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            self._credentials_cache[path] = creds
            return

        parent_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_dir, exist_ok=True)

        try:
            import tempfile
            import shutil

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=parent_dir, prefix=".tmp_", suffix=".json", text=True
            )
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(creds, f, indent=2)

            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass

            shutil.move(tmp_path, path)
            self._credentials_cache[path] = creds
            lib_logger.debug(
                f"Saved {self.ENV_PREFIX} OAuth credentials to '{path}'"
            )
        except Exception as e:
            lib_logger.error(f"Failed to save credentials to '{path}': {e}")
            raise

    # =========================================================================
    # TOKEN MANAGEMENT
    # =========================================================================

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if the Copilot API token is expired."""
        expiry_timestamp = creds.get("expiry_date", 0)
        if isinstance(expiry_timestamp, (int, float)) and expiry_timestamp > 0:
            # expiry_date is stored in milliseconds
            return (expiry_timestamp / 1000) < (
                time.time() + REFRESH_EXPIRY_BUFFER_SECONDS
            )
        return True

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for the given credential path."""
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    async def _refresh_copilot_token(
        self, path: Optional[str], creds: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        """
        Refresh the Copilot API token using the GitHub OAuth token.

        The GitHub OAuth token (refresh_token) is long-lived.
        The Copilot API token (access_token) expires after ~30 minutes.

        Also extracts the base URL from the token's proxy-ep field.
        """
        display_name = Path(path).name if path else "in-memory"
        lock_key = path or "in-memory"

        async with await self._get_lock(lock_key):
            # Skip if token is still valid (unless forced)
            cached_creds = self._credentials_cache.get(lock_key, creds)
            if not force and not self._is_token_expired(cached_creds):
                return cached_creds

            github_token = creds.get("refresh_token")
            if not github_token:
                raise ValueError(
                    "No GitHub OAuth token (refresh_token) found in credentials."
                )

            lib_logger.debug(
                f"Refreshing {self.ENV_PREFIX} Copilot API token for "
                f"'{display_name}' (forced: {force})..."
            )

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        "https://api.github.com/copilot_internal/v2/token",
                        headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {github_token}",
                            **COPILOT_HEADERS,
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 401:
                        lib_logger.warning(
                            f"GitHub token invalid for '{display_name}' "
                            f"(HTTP 401). Token may have been revoked."
                        )
                        raise ValueError(
                            f"GitHub OAuth token revoked or invalid for "
                            f"'{display_name}'"
                        )

                    response.raise_for_status()
                    token_data = response.json()

                    # Update credentials with new Copilot API token
                    access_token = token_data.get("token", "")
                    expires_at = token_data.get("expires_at", 0)

                    creds["access_token"] = access_token
                    creds["expiry_date"] = expires_at * 1000  # Convert to ms

                    # Extract base URL from proxy-ep field in the token
                    base_url = _get_base_url_from_token(access_token)
                    if base_url:
                        creds["copilot_base_url"] = base_url
                        lib_logger.debug(
                            f"Extracted Copilot base URL from token: {base_url}"
                        )
                    else:
                        # Fallback (should not normally happen)
                        creds["copilot_base_url"] = (
                            "https://api.individual.githubcopilot.com"
                        )
                        lib_logger.warning(
                            "Could not extract proxy-ep from Copilot token, "
                            "using default base URL"
                        )

                    # Capture SKU from token response (e.g. "free_educational_quota",
                    # "monthly", etc.)
                    sku = token_data.get("sku", "")
                    if sku:
                        if "_proxy_metadata" not in creds:
                            creds["_proxy_metadata"] = {}
                        creds["_proxy_metadata"]["sku"] = sku
                        lib_logger.info(
                            f"Copilot account SKU: {sku} "
                            f"for '{display_name}'"
                        )

                    # Update metadata
                    if "_proxy_metadata" not in creds:
                        creds["_proxy_metadata"] = {}
                    creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

                    if path:
                        await self._save_credentials(path, creds)
                    else:
                        # In-memory only (setup_credential flow)
                        self._credentials_cache[lock_key] = creds

                    lib_logger.debug(
                        f"Successfully refreshed {self.ENV_PREFIX} Copilot API "
                        f"token for '{display_name}'."
                    )
                    return creds

                except httpx.HTTPStatusError as e:
                    lib_logger.error(
                        f"Failed to refresh Copilot token "
                        f"(HTTP {e.response.status_code}): {e}"
                    )
                    raise
                except httpx.RequestError as e:
                    lib_logger.error(
                        f"Network error refreshing Copilot token: {e}"
                    )
                    raise

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a credential if it's nearing expiry."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._refresh_copilot_token(credential_path, creds)

    # =========================================================================
    # DEVICE FLOW (Interactive Login)
    # =========================================================================

    async def initialize_token(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Initialize or re-authenticate GitHub Copilot credentials using Device Flow.

        Device Flow steps:
        1. Request device code from GitHub
        2. Display user code and verification URL
        3. Poll for authorization completion
        4. Exchange device code for GitHub OAuth token
        5. Fetch Copilot API token using GitHub OAuth token
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        try:
            creds = (
                await self._load_credentials(creds_or_path)
                if path
                else creds_or_path
            )
            needs_auth = False
            reason = ""

            if not creds.get("refresh_token"):
                needs_auth = True
                reason = "GitHub OAuth token is missing"
            elif self._is_token_expired(creds):
                try:
                    return await self._refresh_copilot_token(path, creds)
                except Exception as e:
                    # For env-based credentials, don't fall through to
                    # Device Flow — the user provided a token via env var,
                    # so interactive re-auth isn't appropriate
                    is_env_credential = creds.get("_proxy_metadata", {}).get(
                        "loaded_from_env", False
                    )
                    if is_env_credential:
                        lib_logger.error(
                            f"Copilot token refresh failed for env-based "
                            f"credential '{display_name}': {e}. "
                            f"Check that COPILOT_GITHUB_TOKEN is valid."
                        )
                        raise ValueError(
                            f"Copilot token refresh failed for env-based "
                            f"credential: {e}"
                        )
                    lib_logger.warning(
                        f"Automatic token refresh for '{display_name}' failed: "
                        f"{e}. Proceeding to interactive login."
                    )
                    needs_auth = True
                    reason = "Token refresh failed"

            if not needs_auth:
                lib_logger.info(
                    f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
                )
                return creds

            lib_logger.warning(
                f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: "
                f"{reason}."
            )

            # Step 1: Request device code
            async with httpx.AsyncClient() as client:
                device_response = await client.post(
                    "https://github.com/login/device/code",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "GitHubCopilotChat/0.35.0",
                    },
                    data={
                        "client_id": CLIENT_ID,
                        "scope": "read:user",
                    },
                    timeout=30.0,
                )

                if not device_response.is_success:
                    raise Exception(
                        f"Failed to initiate device authorization: "
                        f"{device_response.text}"
                    )

                device_data = device_response.json()
                user_code = device_data.get("user_code", "")
                verification_uri = device_data.get("verification_uri", "")
                device_code = device_data.get("device_code", "")
                interval = device_data.get("interval", 5)
                expires_in = device_data.get("expires_in", 900)

                # Display instructions
                is_headless = is_headless_environment()

                if is_headless:
                    print(
                        f"\n[{self.ENV_PREFIX} OAuth] Running in headless environment. "
                        f"Open this URL in a browser on another machine:"
                    )
                else:
                    print(
                        f"\n[{self.ENV_PREFIX} OAuth] Please visit the URL below "
                        f"and enter the code to authorize:"
                    )

                print(f"  URL:  {verification_uri}")
                print(f"  Code: {user_code}\n")

                # Step 2: Poll for authorization
                max_polls = expires_in // interval
                for _ in range(max_polls):
                    await asyncio.sleep(interval)

                    token_response = await client.post(
                        "https://github.com/login/oauth/access_token",
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/x-www-form-urlencoded",
                            "User-Agent": "GitHubCopilotChat/0.35.0",
                        },
                        data={
                            "client_id": CLIENT_ID,
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        },
                        timeout=30.0,
                    )

                    if not token_response.is_success:
                        continue

                    token_data = token_response.json()

                    if "access_token" in token_data:
                        # Success! Store the GitHub OAuth token
                        github_token = token_data["access_token"]

                        # Build new credentials
                        new_creds = {
                            "refresh_token": github_token,
                            "access_token": "",
                            "expiry_date": 0,
                            "_proxy_metadata": {
                                "last_check_timestamp": time.time(),
                            },
                        }

                        # Fetch user info
                        try:
                            user_response = await client.get(
                                "https://api.github.com/user",
                                headers={
                                    "Authorization": f"Bearer {github_token}"
                                },
                                timeout=10.0,
                            )
                            if user_response.is_success:
                                user_info = user_response.json()
                                login = user_info.get("login", "unknown")

                                new_creds["_proxy_metadata"]["login"] = login
                        except Exception as e:
                            lib_logger.warning(
                                f"Failed to fetch user info: {e}"
                            )
                            new_creds["_proxy_metadata"]["login"] = "unknown"

                        if path:
                            await self._save_credentials(path, new_creds)

                        lib_logger.info(
                            f"{self.ENV_PREFIX} OAuth initialized successfully "
                            f"for '{display_name}'."
                        )

                        # Fetch the Copilot API token
                        return await self._refresh_copilot_token(
                            path, new_creds, force=True
                        )

                    if token_data.get("error") == "authorization_pending":
                        continue

                    if token_data.get("error") == "slow_down":
                        interval = min(interval + 5, 30)
                        continue

                    if token_data.get("error"):
                        raise Exception(
                            f"OAuth failed: {token_data.get('error')}"
                        )

                raise Exception("OAuth flow timed out. Please try again.")

        except Exception as e:
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            )

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Get Authorization header with fresh Copilot API token."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_copilot_token(credential_path, creds)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Get user info from cached metadata or GitHub API."""
        path = creds_or_path if isinstance(creds_or_path, str) else None
        creds = (
            await self._load_credentials(creds_or_path) if path else creds_or_path
        )

        login = creds.get("_proxy_metadata", {}).get("login")
        if login:
            return {"login": login}

        # Fetch from GitHub API
        github_token = creds.get("refresh_token")
        if github_token:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        "https://api.github.com/user",
                        headers={"Authorization": f"Bearer {github_token}"},
                        timeout=10.0,
                    )
                    if response.is_success:
                        user_info = response.json()
                        login = user_info.get("login", "unknown")

                        creds["_proxy_metadata"] = {
                            "login": login,
                            "last_check_timestamp": time.time(),
                        }
                        if path:
                            await self._save_credentials(path, creds)
                        return {"login": login}
                except Exception as e:
                    lib_logger.warning(f"Failed to fetch user info: {e}")

        return {"login": "unknown"}

    def get_copilot_base_url(self, credential_path: str) -> str:
        """
        Get the Copilot API base URL for a credential.

        Returns the base URL extracted from the Copilot token's proxy-ep field,
        or the default if not yet resolved.
        """
        creds = self._credentials_cache.get(credential_path, {})
        return creds.get(
            "copilot_base_url",
            "https://api.individual.githubcopilot.com",
        )

    # =========================================================================
    # CREDENTIAL MANAGEMENT (for credential_tool.py integration)
    # =========================================================================

    def delete_credential(self, credential_path: str) -> bool:
        """Delete a credential file and remove it from cache."""
        try:
            cred_path = Path(credential_path)

            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be a Copilot credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            self._credentials_cache.pop(credential_path, None)
            cred_path.unlink()
            lib_logger.info(f"Deleted Copilot credential: {credential_path}")
            return True

        except Exception as e:
            lib_logger.error(f"Failed to delete Copilot credential: {e}")
            return False

    def _get_provider_file_prefix(self) -> str:
        """Return the filename prefix for credential files."""
        return "copilot"

    def _get_oauth_base_dir(self) -> Path:
        """Return the default directory for credential files."""
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_login(
        self, login: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Find an existing credential file by login username."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_login = creds.get("_proxy_metadata", {}).get("login")
                if existing_login == login:
                    return Path(cred_file)
            except Exception:
                continue

        return None

    def _get_next_credential_number(self, base_dir: Optional[Path] = None) -> int:
        """Get the next available credential file number."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        existing_numbers = []
        for cred_file in glob(pattern):
            match = re.search(r"_oauth_(\d+)\.json$", cred_file)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def _build_credential_path(
        self, base_dir: Optional[Path] = None, number: Optional[int] = None
    ) -> Path:
        """Build the file path for a new credential file."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        filename = f"{prefix}_oauth_{number}.json"
        return base_dir / filename

    async def setup_credential(
        self, base_dir: Optional[Path] = None
    ) -> CopilotCredentialSetupResult:
        """
        Complete credential setup flow: interactive Device Flow OAuth → save → return result.

        This is called by the credential tool (credential_tool.py) when the user
        selects Copilot as the provider to set up.

        Flow:
        1. Trigger GitHub Device Flow (user visits URL, enters code)
        2. Receive GitHub OAuth token
        3. Exchange for Copilot API token
        4. Fetch user info from GitHub
        5. Save credential file
        6. Return result with file path and email
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build temporary credentials to trigger Device Flow
            temp_creds: Dict[str, Any] = {
                "_proxy_metadata": {
                    "display_name": "new Copilot OAuth credential",
                },
            }

            # initialize_token() will detect no refresh_token and trigger Device Flow
            new_creds = await self.initialize_token(temp_creds)

            login = new_creds.get("_proxy_metadata", {}).get("login", "")
            sku = new_creds.get("_proxy_metadata", {}).get("sku", "")

            # Check for existing credential with same login
            existing_path = (
                self._find_existing_credential_by_login(login, base_dir)
                if login
                else None
            )
            is_update = existing_path is not None

            file_path = (
                existing_path if is_update else self._build_credential_path(base_dir)
            )

            await self._save_credentials(str(file_path), new_creds)

            return CopilotCredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=login or None,  # Reuse email field for backward compat
                is_update=is_update,
                sku=sku or None,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Copilot credential setup failed: {e}")
            return CopilotCredentialSetupResult(success=False, error=str(e))

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all Copilot credential files in the given directory.

        Returns a list of dicts with file_path, login, and number.
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append({
                    "file_path": cred_file,
                    "login": metadata.get("login", "unknown"),
                    "sku": metadata.get("sku", ""),
                    "number": number,
                })
            except Exception:
                continue

        return credentials

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """
        Generate .env file lines for a Copilot credential.

        For Copilot, only the GITHUB_TOKEN is needed (the Copilot API token
        is derived from it automatically).

        Args:
            creds: Credential dictionary loaded from JSON
            cred_number: Credential number (1, 2, 3, etc.)

        Returns:
            List of .env file lines
        """
        login = creds.get("_proxy_metadata", {}).get("login", "unknown")
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        lines = [
            f"# {self.ENV_PREFIX} Credential #{cred_number} for: {login}",
            f"# Exported from: {self._get_provider_file_prefix()}_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_GITHUB_TOKEN={creds.get('refresh_token', '')}",
        ]

        return lines