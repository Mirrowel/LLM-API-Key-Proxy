# src/rotator_library/providers/copilot_auth_base.py
"""
GitHub Copilot OAuth2 authentication implementation using Device Flow.

This is fundamentally different from Google OAuth providers:
- Uses GitHub's Device Flow instead of Authorization Code Flow
- Requires two-step token exchange:
  1. GitHub OAuth token (long-lived, used as "refresh token")
  2. Copilot API token (short-lived, ~30 min, used as "access token")

Based on: https://github.com/sst/opencode-copilot-auth
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import tempfile
import shutil

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger("rotator_library")
console = Console()


class CopilotAuthBase:
    """
    GitHub Copilot OAuth2 authentication using Device Flow.

    This provider uses GitHub's Device Authorization Grant flow, which is
    more suitable for CLI applications than the web-based Authorization Code flow.

    Key differences from GoogleOAuthBase:
    - Uses GitHub Device Flow (polls for authorization)
    - Two-token system: GitHub OAuth token + Copilot API token
    - Copilot API tokens expire quickly (~30 min) and need frequent refresh

    Subclasses may override:
        - ENV_PREFIX: Prefix for environment variables (default: "COPILOT")
        - REFRESH_EXPIRY_BUFFER_SECONDS: Time buffer before token expiry

    Supports both github.com and GitHub Enterprise deployments.
    """

    # GitHub Copilot OAuth Client ID (from VS Code Copilot extension)
    CLIENT_ID = "Iv1.b507a08c87ecfe98"

    # Headers that mimic the official Copilot client
    COPILOT_HEADERS = {
        "User-Agent": "GitHubCopilotChat/0.32.4",
        "Editor-Version": "vscode/1.105.1",
        "Editor-Plugin-Version": "copilot-chat/0.32.4",
        "Copilot-Integration-Id": "vscode-chat",
    }

    # Environment variable prefix
    ENV_PREFIX = "COPILOT"

    # Token refresh buffer (default: 5 minutes for short-lived Copilot tokens)
    REFRESH_EXPIRY_BUFFER_SECONDS = 5 * 60

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        # [QUEUE SYSTEM] Sequential refresh processing
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queued_credentials: set = set()
        self._unavailable_credentials: set = set()
        self._queue_tracking_lock = asyncio.Lock()
        self._queue_processor_task: Optional[asyncio.Task] = None

    def _normalize_domain(self, url: str) -> str:
        """Normalize GitHub domain from URL."""
        return url.replace("https://", "").replace("http://", "").rstrip("/")

    def _get_urls(self, domain: str = "github.com") -> Dict[str, str]:
        """Get GitHub OAuth URLs for the specified domain."""
        return {
            "DEVICE_CODE_URL": f"https://{domain}/login/device/code",
            "ACCESS_TOKEN_URL": f"https://{domain}/login/oauth/access_token",
            "COPILOT_API_KEY_URL": f"https://api.{domain}/copilot_internal/v2/token",
        }

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://provider/0" - Legacy single credential
        - "env://provider/1" - First numbered credential
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

        For Copilot, we need:
        - {PREFIX}_GITHUB_TOKEN or {PREFIX}_{N}_GITHUB_TOKEN (long-lived GitHub OAuth token)
        - Optionally: {PREFIX}_ENTERPRISE_URL for GitHub Enterprise

        The Copilot API token is fetched dynamically and cached.
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_email = "env-user"

        # For Copilot, the "refresh_token" is the GitHub OAuth token
        github_token = os.getenv(f"{prefix}_GITHUB_TOKEN")
        if not github_token:
            # Also check legacy naming
            github_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not github_token:
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

        # Check for enterprise URL
        enterprise_url = os.getenv(f"{prefix}_ENTERPRISE_URL", "")

        creds = {
            "refresh_token": github_token,  # GitHub OAuth token used as refresh token
            "access_token": "",  # Copilot API token (fetched on demand)
            "expiry_date": 0,  # Will be set when Copilot token is fetched
            "enterprise_url": enterprise_url,
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
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
                        f"Using {self.ENV_PREFIX} credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for {self.ENV_PREFIX} credential index {credential_index} not found"
                    )

            # Try loading from legacy env vars
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info(
                    f"Using {self.ENV_PREFIX} credentials from environment variables"
                )
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            try:
                lib_logger.debug(
                    f"Loading {self.ENV_PREFIX} credentials from file: {path}"
                )
                with open(path, "r") as f:
                    creds = json.load(f)
                self._credentials_cache[path] = creds
                return creds
            except FileNotFoundError:
                raise IOError(
                    f"{self.ENV_PREFIX} OAuth credential file not found at '{path}'"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials from '{path}': {e}"
                )

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials to file with atomic write."""
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            self._credentials_cache[path] = creds
            return

        parent_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_dir, exist_ok=True)

        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=parent_dir, prefix=".tmp_", suffix=".json", text=True
            )

            with os.fdopen(tmp_fd, "w") as f:
                json.dump(creds, f, indent=2)
                tmp_fd = None

            try:
                os.chmod(tmp_path, 0o600)
            except (OSError, AttributeError):
                pass

            shutil.move(tmp_path, path)
            tmp_path = None

            self._credentials_cache[path] = creds
            lib_logger.debug(
                f"Saved updated {self.ENV_PREFIX} OAuth credentials to '{path}' (atomic write)."
            )

        except Exception as e:
            lib_logger.error(
                f"Failed to save updated {self.ENV_PREFIX} OAuth credentials to '{path}': {e}"
            )
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            raise

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if the Copilot API token is expired."""
        expiry_timestamp = creds.get("expiry_date", 0)
        if isinstance(expiry_timestamp, (int, float)) and expiry_timestamp > 0:
            # expiry_date is stored in milliseconds (like gemini-cli format)
            return (expiry_timestamp / 1000) < (
                time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS
            )
        # If no expiry or zero, token is expired
        return True

    async def _refresh_copilot_token(
        self, path: str, creds: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        """
        Refresh the Copilot API token using the GitHub OAuth token.

        The GitHub OAuth token (refresh_token) is long-lived.
        The Copilot API token (access_token) expires after ~30 minutes.
        """
        async with await self._get_lock(path):
            # Skip if token is still valid (unless forced)
            cached_creds = self._credentials_cache.get(path, creds)
            if not force and not self._is_token_expired(cached_creds):
                return cached_creds

            github_token = creds.get("refresh_token")
            if not github_token:
                raise ValueError(
                    "No GitHub OAuth token (refresh_token) found in credentials."
                )

            enterprise_url = creds.get("enterprise_url", "")
            domain = (
                self._normalize_domain(enterprise_url)
                if enterprise_url
                else "github.com"
            )
            urls = self._get_urls(domain)

            lib_logger.debug(
                f"Refreshing {self.ENV_PREFIX} Copilot API token for '{Path(path).name}' (forced: {force})..."
            )

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        urls["COPILOT_API_KEY_URL"],
                        headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {github_token}",
                            **self.COPILOT_HEADERS,
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 401:
                        lib_logger.warning(
                            f"GitHub token invalid for '{Path(path).name}' (HTTP 401). "
                            f"Token may have been revoked. Starting re-authentication..."
                        )
                        return await self.initialize_token(path)

                    response.raise_for_status()
                    token_data = response.json()

                    # Update credentials with new Copilot API token
                    creds["access_token"] = token_data.get("token", "")
                    # expires_at is Unix timestamp in seconds
                    expires_at = token_data.get("expires_at", 0)
                    creds["expiry_date"] = expires_at * 1000  # Convert to milliseconds

                    # Update metadata
                    if "_proxy_metadata" not in creds:
                        creds["_proxy_metadata"] = {}
                    creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

                    await self._save_credentials(path, creds)
                    lib_logger.debug(
                        f"Successfully refreshed {self.ENV_PREFIX} Copilot API token for '{Path(path).name}'."
                    )
                    return creds

                except httpx.HTTPStatusError as e:
                    lib_logger.error(
                        f"Failed to refresh Copilot token (HTTP {e.response.status_code}): {e}"
                    )
                    raise
                except httpx.RequestError as e:
                    lib_logger.error(f"Network error refreshing Copilot token: {e}")
                    raise

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for the given credential path."""
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a credential if it's nearing expiry."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._refresh_copilot_token(credential_path, creds)

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation."""
        return path not in self._unavailable_credentials

    async def initialize_token(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Initialize or re-authenticate GitHub Copilot credentials using Device Flow.

        Device Flow steps:
        1. Request device code from GitHub
        2. Display user code and verification URL
        3. Poll for authorization completion
        4. Exchange device code for access token
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(
            f"Initializing {self.ENV_PREFIX} token for '{display_name}'..."
        )

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )
            needs_auth = False
            reason = ""

            if not creds.get("refresh_token"):
                needs_auth = True
                reason = "GitHub OAuth token is missing"
            elif self._is_token_expired(creds):
                # Try to refresh the Copilot API token
                try:
                    return await self._refresh_copilot_token(path, creds)
                except Exception as e:
                    lib_logger.warning(
                        f"Automatic token refresh for '{display_name}' failed: {e}. "
                        f"Proceeding to interactive login."
                    )
                    needs_auth = True
                    reason = "Token refresh failed"

            if not needs_auth:
                lib_logger.info(
                    f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
                )
                return creds

            lib_logger.warning(
                f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: {reason}."
            )

            # Check for enterprise URL in existing creds or environment
            enterprise_url = creds.get("enterprise_url", "")
            if not enterprise_url:
                enterprise_url = os.getenv(f"{self.ENV_PREFIX}_ENTERPRISE_URL", "")

            domain = (
                self._normalize_domain(enterprise_url)
                if enterprise_url
                else "github.com"
            )
            urls = self._get_urls(domain)

            is_headless = is_headless_environment()

            # Step 1: Request device code
            async with httpx.AsyncClient() as client:
                device_response = await client.post(
                    urls["DEVICE_CODE_URL"],
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "User-Agent": "GitHubCopilotChat/0.35.0",
                    },
                    json={
                        "client_id": self.CLIENT_ID,
                        "scope": "read:user",
                    },
                    timeout=30.0,
                )

                if not device_response.is_success:
                    raise Exception(
                        f"Failed to initiate device authorization: {device_response.text}"
                    )

                device_data = device_response.json()
                user_code = device_data.get("user_code", "")
                verification_uri = device_data.get("verification_uri", "")
                device_code = device_data.get("device_code", "")
                interval = device_data.get("interval", 5)
                expires_in = device_data.get("expires_in", 900)

                # Display instructions
                if is_headless:
                    auth_panel_text = Text.from_markup(
                        "Running in headless environment (no GUI detected).\n"
                        "Please open the URL below in a browser on another machine to authorize:\n"
                    )
                else:
                    auth_panel_text = Text.from_markup(
                        "Please visit the URL below and enter the code to authorize:\n"
                    )

                console.print(
                    Panel(
                        auth_panel_text,
                        title=f"{self.ENV_PREFIX} OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                        style="bold blue",
                    )
                )
                console.print(f"[bold]URL:[/bold] {verification_uri}")
                console.print(
                    f"[bold]Code:[/bold] [bold green]{user_code}[/bold green]\n"
                )

                # Step 2: Poll for authorization
                max_polls = expires_in // interval
                with console.status(
                    f"[bold green]Waiting for you to complete authentication (code: {user_code})...[/bold green]",
                    spinner="dots",
                ):
                    for _ in range(max_polls):
                        await asyncio.sleep(interval)

                        token_response = await client.post(
                            urls["ACCESS_TOKEN_URL"],
                            headers={
                                "Accept": "application/json",
                                "Content-Type": "application/json",
                                "User-Agent": "GitHubCopilotChat/0.35.0",
                            },
                            json={
                                "client_id": self.CLIENT_ID,
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
                                "access_token": "",  # Will be filled by first API call
                                "expiry_date": 0,
                                "enterprise_url": enterprise_url,
                                "_proxy_metadata": {
                                    "last_check_timestamp": time.time(),
                                },
                            }

                            # Fetch user info
                            try:
                                user_response = await client.get(
                                    f"https://api.{domain}/user",
                                    headers={"Authorization": f"Bearer {github_token}"},
                                    timeout=10.0,
                                )
                                if user_response.is_success:
                                    user_info = user_response.json()
                                    new_creds["_proxy_metadata"]["email"] = (
                                        user_info.get(
                                            "email", user_info.get("login", "unknown")
                                        )
                                    )
                            except Exception as e:
                                lib_logger.warning(f"Failed to fetch user info: {e}")
                                new_creds["_proxy_metadata"]["email"] = "unknown"

                            if path:
                                await self._save_credentials(path, new_creds)

                            lib_logger.info(
                                f"{self.ENV_PREFIX} OAuth initialized successfully for '{display_name}'."
                            )

                            # Now fetch the Copilot API token
                            return await self._refresh_copilot_token(
                                path, new_creds, force=True
                            )

                        if token_data.get("error") == "authorization_pending":
                            continue

                        if token_data.get("error"):
                            raise Exception(f"OAuth failed: {token_data.get('error')}")

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
        """Get user info from cached metadata or API."""
        path = creds_or_path if isinstance(creds_or_path, str) else None
        creds = await self._load_credentials(creds_or_path) if path else creds_or_path

        if creds.get("_proxy_metadata", {}).get("email"):
            return {"email": creds["_proxy_metadata"]["email"]}

        # Fetch from GitHub API
        github_token = creds.get("refresh_token")
        if github_token:
            enterprise_url = creds.get("enterprise_url", "")
            domain = (
                self._normalize_domain(enterprise_url)
                if enterprise_url
                else "github.com"
            )

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"https://api.{domain}/user",
                        headers={"Authorization": f"Bearer {github_token}"},
                        timeout=10.0,
                    )
                    if response.is_success:
                        user_info = response.json()
                        email = user_info.get(
                            "email", user_info.get("login", "unknown")
                        )
                        creds["_proxy_metadata"] = {
                            "email": email,
                            "last_check_timestamp": time.time(),
                        }
                        if path:
                            await self._save_credentials(path, creds)
                        return {"email": email}
                except Exception as e:
                    lib_logger.warning(f"Failed to fetch user info: {e}")

        return {"email": "unknown"}
