# src/rotator_library/providers/iflow_auth_base.py

import secrets
import base64
import json
import time
import asyncio
import logging
import webbrowser
import socket
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
from urllib.parse import urlencode, parse_qs, urlparse
import tempfile
import shutil

import httpx
from aiohttp import web
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markup import escape as rich_escape
from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger("rotator_library")

IFLOW_OAUTH_AUTHORIZE_ENDPOINT = "https://iflow.cn/oauth"
IFLOW_OAUTH_TOKEN_ENDPOINT = "https://iflow.cn/oauth/token"
IFLOW_USER_INFO_ENDPOINT = "https://iflow.cn/api/oauth/getUserInfo"
IFLOW_SUCCESS_REDIRECT_URL = "https://iflow.cn/oauth/success"
IFLOW_ERROR_REDIRECT_URL = "https://iflow.cn/oauth/error"

# Client credentials provided by iFlow
IFLOW_CLIENT_ID = "10009311001"
IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"

# Local callback server port
CALLBACK_PORT = 11451

# Refresh tokens 24 hours before expiry
REFRESH_EXPIRY_BUFFER_SECONDS = 24 * 60 * 60

console = Console()


class OAuthCallbackServer:
    """
    Minimal HTTP server for handling iFlow OAuth callbacks.
    """

    def __init__(self, port: int = CALLBACK_PORT):
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.result_future: Optional[asyncio.Future] = None
        self.expected_state: Optional[str] = None

    def _is_port_available(self) -> bool:
        """Checks if the callback port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", self.port))
            sock.close()
            return True
        except OSError:
            return False

    async def start(self, expected_state: str):
        """Starts the OAuth callback server."""
        if not self._is_port_available():
            raise RuntimeError(f"Port {self.port} is already in use")

        self.expected_state = expected_state
        self.result_future = asyncio.Future()

        # Setup route
        self.app.router.add_get("/oauth2callback", self._handle_callback)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", self.port)
        await self.site.start()

        lib_logger.debug(f"iFlow OAuth callback server started on port {self.port}")

    async def stop(self):
        """Stops the OAuth callback server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        lib_logger.debug("iFlow OAuth callback server stopped")

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handles the OAuth callback request."""
        query = request.query

        # Check for error parameter
        if "error" in query:
            error = query.get("error", "unknown_error")
            lib_logger.error(f"iFlow OAuth callback received error: {error}")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError(f"OAuth error: {error}"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Check for authorization code
        code = query.get("code")
        if not code:
            lib_logger.error("iFlow OAuth callback missing authorization code")
            if not self.result_future.done():
                self.result_future.set_exception(
                    ValueError("Missing authorization code")
                )
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Validate state parameter
        state = query.get("state", "")
        if state != self.expected_state:
            lib_logger.error(
                f"iFlow OAuth state mismatch. Expected: {self.expected_state}, Got: {state}"
            )
            if not self.result_future.done():
                self.result_future.set_exception(ValueError("State parameter mismatch"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Success - set result and redirect to success page
        if not self.result_future.done():
            self.result_future.set_result(code)

        return web.Response(
            status=302, headers={"Location": IFLOW_SUCCESS_REDIRECT_URL}
        )

    async def wait_for_callback(self, timeout: float = 300.0) -> str:
        """Waits for the OAuth callback and returns the authorization code."""
        try:
            code = await asyncio.wait_for(self.result_future, timeout=timeout)
            return code
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for OAuth callback")


class IFlowAuthBase:
    """
    iFlow OAuth authentication base class.
    Implements authorization code flow with local callback server.
    """

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = (
            asyncio.Lock()
        )  # Protects the locks dict from race conditions
        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[
            str, int
        ] = {}  # Track consecutive failures per credential
        self._next_refresh_after: Dict[
            str, float
        ] = {}  # Track backoff timers (Unix timestamp)

        # [QUEUE SYSTEM] Sequential refresh processing
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queued_credentials: set = set()  # Track credentials already in queue
        self._unavailable_credentials: set = (
            set()
        )  # Mark credentials unavailable during re-auth
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets
        self._queue_processor_task: Optional[asyncio.Task] = (
            None  # Background worker task
        )

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://provider/0" - Legacy single credential (no index in env var names)
        - "env://provider/1" - First numbered credential (IFLOW_1_ACCESS_TOKEN)

        Returns:
            The credential index as string, or None if path is not an env:// path
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
        Load OAuth credentials from environment variables for stateless deployments.

        Supports two formats:
        1. Legacy (credential_index="0" or None): IFLOW_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): IFLOW_1_ACCESS_TOKEN, etc.

        Expected environment variables (for numbered format with index N):
        - IFLOW_{N}_ACCESS_TOKEN (required)
        - IFLOW_{N}_REFRESH_TOKEN (required)
        - IFLOW_{N}_API_KEY (required - critical for iFlow!)
        - IFLOW_{N}_EXPIRY_DATE (optional, defaults to empty string)
        - IFLOW_{N}_EMAIL (optional, defaults to "env-user-{N}")
        - IFLOW_{N}_TOKEN_TYPE (optional, defaults to "Bearer")
        - IFLOW_{N}_SCOPE (optional, defaults to "read write")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            prefix = f"IFLOW_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "IFLOW"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")
        api_key = os.getenv(f"{prefix}_API_KEY")

        # All three are required for iFlow
        if not (access_token and refresh_token and api_key):
            return None

        lib_logger.debug(
            f"Loading iFlow credentials from environment variables (prefix: {prefix})"
        )

        # Parse expiry_date as string (ISO 8601 format)
        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "")

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": api_key,  # Critical for iFlow!
            "expiry_date": expiry_str,
            "email": os.getenv(f"{prefix}_EMAIL", default_email),
            "token_type": os.getenv(f"{prefix}_TOKEN_TYPE", "Bearer"),
            "scope": os.getenv(f"{prefix}_SCOPE", "read write"),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache. No locking."""
        try:
            lib_logger.debug(f"Reading iFlow credentials from file: {path}")
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"iFlow OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load iFlow OAuth credentials from '{path}': {e}")

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Loads credentials from cache, environment variables, or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            # Re-check cache after acquiring lock
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    lib_logger.info(
                        f"Using iFlow credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for iFlow credential index {credential_index} not found"
                    )

            # For file paths, try loading from legacy env vars first
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info("Using iFlow credentials from environment variables")
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            return await self._read_creds_from_file(path)

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Saves credentials to cache and file using atomic writes."""
        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            # Still update cache for in-memory consistency
            self._credentials_cache[path] = creds
            return

        # [ATOMIC WRITE] Use tempfile + move pattern to ensure atomic writes
        # This prevents credential corruption if the process is interrupted during write
        parent_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_dir, exist_ok=True)

        tmp_fd = None
        tmp_path = None
        try:
            # Create temp file in same directory as target (ensures same filesystem)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=parent_dir, prefix=".tmp_", suffix=".json", text=True
            )

            # Write JSON to temp file
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(creds, f, indent=2)
                tmp_fd = None  # fdopen closes the fd

            # Set secure permissions (0600 = owner read/write only)
            try:
                os.chmod(tmp_path, 0o600)
            except (OSError, AttributeError):
                # Windows may not support chmod, ignore
                pass

            # Atomic move (overwrites target if it exists)
            shutil.move(tmp_path, path)
            tmp_path = None  # Successfully moved

            # Update cache AFTER successful file write
            self._credentials_cache[path] = creds
            lib_logger.debug(
                f"Saved updated iFlow OAuth credentials to '{path}' (atomic write)."
            )

        except Exception as e:
            lib_logger.error(
                f"Failed to save updated iFlow OAuth credentials to '{path}': {e}"
            )
            # Clean up temp file if it still exists
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
        """Checks if the token is expired (with buffer for proactive refresh)."""
        # Try to parse expiry_date as ISO 8601 string
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        try:
            # Parse ISO 8601 format (e.g., "2025-01-17T12:00:00Z")
            from datetime import datetime

            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            expiry_timestamp = expiry_dt.timestamp()
        except (ValueError, AttributeError):
            # Fallback: treat as numeric timestamp
            try:
                expiry_timestamp = float(expiry_str)
            except (ValueError, TypeError):
                lib_logger.warning(f"Could not parse expiry_date: {expiry_str}")
                return True

        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _fetch_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Fetches user info (including API key) from iFlow API.
        This is critical: iFlow uses a separate API key for actual API calls.
        """
        if not access_token or not access_token.strip():
            raise ValueError("Access token is empty")

        url = f"{IFLOW_USER_INFO_ENDPOINT}?accessToken={access_token}"
        headers = {"Accept": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

        if not result.get("success"):
            raise ValueError("iFlow user info request not successful")

        data = result.get("data", {})
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            raise ValueError("Missing API key in user info response")

        email = data.get("email", "").strip()
        if not email:
            email = data.get("phone", "").strip()
        if not email:
            raise ValueError("Missing email/phone in user info response")

        return {"api_key": api_key, "email": email}

    async def _exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchanges authorization code for access and refresh tokens.
        Uses Basic Auth with client credentials.
        """
        # Create Basic Auth header
        auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
        basic_auth = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {basic_auth}",
        }

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": IFLOW_CLIENT_ID,
            "client_secret": IFLOW_CLIENT_SECRET,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
            )

            if response.status_code != 200:
                error_text = response.text
                lib_logger.error(
                    f"iFlow token exchange failed: {response.status_code} {error_text}"
                )
                raise ValueError(
                    f"Token exchange failed: {response.status_code} {error_text}"
                )

            token_data = response.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Missing access_token in token response")

        refresh_token = token_data.get("refresh_token", "")
        expires_in = token_data.get("expires_in", 3600)
        token_type = token_data.get("token_type", "Bearer")
        scope = token_data.get("scope", "")

        # Fetch user info to get API key
        user_info = await self._fetch_user_info(access_token)

        # Calculate expiry date
        from datetime import datetime, timedelta

        expiry_date = (
            datetime.utcnow() + timedelta(seconds=expires_in)
        ).isoformat() + "Z"

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": user_info["api_key"],
            "email": user_info["email"],
            "expiry_date": expiry_date,
            "token_type": token_type,
            "scope": scope,
        }

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        """
        Refreshes the OAuth tokens and re-fetches the API key.
        CRITICAL: Must re-fetch user info to get potentially updated API key.
        """
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # If cache is empty, read from file
            if path not in self._credentials_cache:
                await self._read_creds_from_file(path)

            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing iFlow OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in iFlow credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None
            needs_reauth = False

            # Create Basic Auth header
            auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
            basic_auth = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": IFLOW_CLIENT_ID,
                "client_secret": IFLOW_CLIENT_SECRET,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break  # Success

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text

                        lib_logger.error(
                            f"[REFRESH HTTP ERROR] HTTP {status_code} for '{Path(path).name}': {error_body}"
                        )

                        # [STATUS CODE HANDLING]
                        # [INVALID GRANT HANDLING] Handle 401/403 by triggering re-authentication
                        if status_code in (401, 403):
                            lib_logger.warning(
                                f"Refresh token invalid for '{Path(path).name}' (HTTP {status_code}). "
                                f"Token may have been revoked or expired. Starting re-authentication..."
                            )
                            needs_reauth = True
                            break  # Exit retry loop to trigger re-auth

                        elif status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(
                                f"Rate limited (HTTP 429), retry after {retry_after}s"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif 500 <= status_code < 600:
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt
                                lib_logger.warning(
                                    f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            raise

                        else:
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            lib_logger.warning(
                                f"Network error during refresh: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise

            # [INVALID GRANT RE-AUTH] Trigger OAuth flow if refresh token is invalid
            if needs_reauth:
                lib_logger.info(
                    f"Starting re-authentication for '{Path(path).name}'..."
                )
                try:
                    # Call initialize_token to trigger OAuth flow
                    new_creds = await self.initialize_token(path)
                    # Clear backoff on successful re-auth
                    self._refresh_failures.pop(path, None)
                    self._next_refresh_after.pop(path, None)
                    return new_creds
                except Exception as reauth_error:
                    lib_logger.error(
                        f"Re-authentication failed for '{Path(path).name}': {reauth_error}"
                    )
                    # [BACKOFF TRACKING] Increment failure count and set backoff timer
                    self._refresh_failures[path] = (
                        self._refresh_failures.get(path, 0) + 1
                    )
                    backoff_seconds = min(
                        300, 30 * (2 ** self._refresh_failures[path])
                    )  # Max 5 min backoff
                    self._next_refresh_after[path] = time.time() + backoff_seconds
                    lib_logger.debug(
                        f"Setting backoff for '{Path(path).name}': {backoff_seconds}s"
                    )
                    raise ValueError(
                        f"Refresh token invalid and re-authentication failed: {reauth_error}"
                    )

            if new_token_data is None:
                # [BACKOFF TRACKING] Increment failure count and set backoff timer
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                backoff_seconds = min(
                    300, 30 * (2 ** self._refresh_failures[path])
                )  # Max 5 min backoff
                self._next_refresh_after[path] = time.time() + backoff_seconds
                lib_logger.debug(
                    f"Setting backoff for '{Path(path).name}': {backoff_seconds}s"
                )
                raise last_error or Exception("Token refresh failed after all retries")

            # Update tokens
            access_token = new_token_data.get("access_token")
            if not access_token:
                raise ValueError("Missing access_token in refresh response")

            creds_from_file["access_token"] = access_token
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )

            expires_in = new_token_data.get("expires_in", 3600)
            from datetime import datetime, timedelta

            creds_from_file["expiry_date"] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat() + "Z"

            creds_from_file["token_type"] = new_token_data.get(
                "token_type", creds_from_file.get("token_type", "Bearer")
            )
            creds_from_file["scope"] = new_token_data.get(
                "scope", creds_from_file.get("scope", "")
            )

            # CRITICAL: Re-fetch user info to get potentially updated API key
            try:
                user_info = await self._fetch_user_info(access_token)
                if user_info.get("api_key"):
                    creds_from_file["api_key"] = user_info["api_key"]
                if user_info.get("email"):
                    creds_from_file["email"] = user_info["email"]
            except Exception as e:
                lib_logger.warning(
                    f"Failed to update API key during token refresh: {e}"
                )

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify required fields exist after refresh
            required_fields = ["access_token", "refresh_token", "api_key"]
            missing_fields = [
                field for field in required_fields if not creds_from_file.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Refreshed credentials missing required fields: {missing_fields}"
                )

            # [BACKOFF TRACKING] Clear failure count on successful refresh
            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            await self._save_credentials(path, creds_from_file)
            lib_logger.debug(
                f"Successfully refreshed iFlow OAuth token for '{Path(path).name}'."
            )
            return creds_from_file

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and API key (NOT access_token).
        CRITICAL: iFlow uses the api_key for API requests, not the OAuth access_token.

        Supports both credential types:
        - OAuth: credential_identifier is a file path to JSON credentials
        - API Key: credential_identifier is the API key string itself
        """
        # Detect credential type
        if os.path.isfile(credential_identifier):
            # OAuth credential: file path to JSON
            lib_logger.debug(
                f"Using OAuth credentials from file: {credential_identifier}"
            )
            creds = await self._load_credentials(credential_identifier)

            # Check if token needs refresh
            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            api_key = creds.get("api_key")
            if not api_key:
                raise ValueError("Missing api_key in iFlow OAuth credentials")
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for iFlow")
            api_key = credential_identifier

        base_url = "https://apis.iflow.cn/v1"
        return base_url, api_key

    async def proactively_refresh(self, credential_identifier: str):
        """
        Proactively refreshes tokens if they're close to expiry.
        Only applies to OAuth credentials (file paths or env:// paths). Direct API keys are skipped.
        """
        # Check if it's an env:// virtual path (OAuth credentials from environment)
        is_env_path = credential_identifier.startswith("env://")

        # Only refresh if it's an OAuth credential (file path or env:// path)
        if not is_env_path and not os.path.isfile(credential_identifier):
            return  # Direct API key, no refresh needed

        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            # Queue for refresh with needs_reauth=False (automated refresh)
            await self._queue_refresh(
                credential_identifier, force=False, needs_reauth=False
            )

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Gets or creates a lock for the given credential path."""
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation (not queued/refreshing)."""
        return path not in self._unavailable_credentials

    async def _ensure_queue_processor_running(self):
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the refresh queue if not already queued.

        Args:
            path: Credential file path
            force: Force refresh even if not expired
            needs_reauth: True if full re-authentication needed (bypasses backoff)
        """
        # IMPORTANT: Only check backoff for simple automated refreshes
        # Re-authentication (interactive OAuth) should BYPASS backoff since it needs user input
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
                    # Credential is in backoff for automated refresh, do not queue
                    remaining = int(backoff_until - now)
                    lib_logger.debug(
                        f"Skipping automated refresh for '{Path(path).name}' (in backoff for {remaining}s)"
                    )
                    return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)
                self._unavailable_credentials.add(path)  # Mark as unavailable
                await self._refresh_queue.put((path, force, needs_reauth))
                await self._ensure_queue_processor_running()

    async def _process_refresh_queue(self):
        """Background worker that processes refresh requests sequentially."""
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path, force, needs_reauth = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # No items for 60s, exit to save resources
                    self._queue_processor_task = None
                    return

                try:
                    # Perform the actual refresh (still using per-credential lock)
                    async with await self._get_lock(path):
                        # Re-check if still expired (may have changed since queueing)
                        creds = self._credentials_cache.get(path)
                        if creds and not self._is_token_expired(creds):
                            # No longer expired, mark as available
                            async with self._queue_tracking_lock:
                                self._unavailable_credentials.discard(path)
                            continue

                        # Perform refresh
                        if not creds:
                            creds = await self._load_credentials(path)
                        await self._refresh_token(path, force=force)

                        # SUCCESS: Mark as available again
                        async with self._queue_tracking_lock:
                            self._unavailable_credentials.discard(path)

                finally:
                    # Remove from queued set
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Error in queue processor: {e}")
                # Even on error, mark as available (backoff will prevent immediate retry)
                if path:
                    async with self._queue_tracking_lock:
                        self._unavailable_credentials.discard(path)

    async def initialize_token(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Initiates OAuth authorization code flow if tokens are missing or invalid.
        Uses local callback server to receive authorization code.
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing iFlow token for '{display_name}'...")

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            reason = ""
            if not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                # Try automatic refresh first if we have a refresh token
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                # Interactive OAuth flow
                lib_logger.warning(
                    f"iFlow OAuth token for '{display_name}' needs setup: {reason}."
                )

                # [HEADLESS DETECTION] Check if running in headless environment
                is_headless = is_headless_environment()

                # Generate random state for CSRF protection
                state = secrets.token_urlsafe(32)

                # Build authorization URL
                redirect_uri = f"http://localhost:{CALLBACK_PORT}/oauth2callback"
                auth_params = {
                    "loginMethod": "phone",
                    "type": "phone",
                    "redirect": redirect_uri,
                    "state": state,
                    "client_id": IFLOW_CLIENT_ID,
                }
                auth_url = f"{IFLOW_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(auth_params)}"

                # Start OAuth callback server
                callback_server = OAuthCallbackServer(port=CALLBACK_PORT)
                try:
                    await callback_server.start(expected_state=state)

                    # [HEADLESS SUPPORT] Display appropriate instructions
                    if is_headless:
                        auth_panel_text = Text.from_markup(
                            "Running in headless environment (no GUI detected).\n"
                            "Please open the URL below in a browser on another machine to authorize:\n"
                            "1. Visit the URL below to sign in with your phone number.\n"
                            "2. [bold]Authorize the application[/bold] to access your account.\n"
                            "3. You will be automatically redirected after authorization."
                        )
                    else:
                        auth_panel_text = Text.from_markup(
                            "1. Visit the URL below to sign in with your phone number.\n"
                            "2. [bold]Authorize the application[/bold] to access your account.\n"
                            "3. You will be automatically redirected after authorization."
                        )

                    console.print(
                        Panel(
                            auth_panel_text,
                            title=f"iFlow OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                            style="bold blue",
                        )
                    )
                    # [URL DISPLAY] Print URL with proper escaping to prevent Rich markup issues.
                    # IMPORTANT: OAuth URLs contain special characters (=, &, etc.) that Rich might
                    # interpret as markup in some terminal configurations. We escape the URL to
                    # ensure it displays correctly.
                    #
                    # KNOWN ISSUE: If Rich rendering fails entirely (e.g., terminal doesn't support
                    # ANSI codes, or output is piped), the escaped URL should still be valid.
                    # However, if the terminal strips or mangles the output, users should copy
                    # the URL directly from logs or use --verbose to see the raw URL.
                    #
                    # The [link=...] markup creates a clickable hyperlink in supported terminals
                    # (iTerm2, Windows Terminal, etc.), but the displayed text is the escaped URL
                    # which can be safely copied even if the hyperlink doesn't work.
                    escaped_url = rich_escape(auth_url)
                    console.print(
                        f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n"
                    )

                    # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
                    if not is_headless:
                        try:
                            webbrowser.open(auth_url)
                            lib_logger.info(
                                "Browser opened successfully for iFlow OAuth flow"
                            )
                        except Exception as e:
                            lib_logger.warning(
                                f"Failed to open browser automatically: {e}. Please open the URL manually."
                            )

                    # Wait for callback
                    with console.status(
                        "[bold green]Waiting for authorization in the browser...[/bold green]",
                        spinner="dots",
                    ):
                        code = await callback_server.wait_for_callback(timeout=300.0)

                    lib_logger.info(
                        "Received authorization code, exchanging for tokens..."
                    )

                    # Exchange code for tokens and API key
                    token_data = await self._exchange_code_for_tokens(
                        code, redirect_uri
                    )

                    # Update credentials
                    creds.update(
                        {
                            "access_token": token_data["access_token"],
                            "refresh_token": token_data["refresh_token"],
                            "api_key": token_data["api_key"],
                            "email": token_data["email"],
                            "expiry_date": token_data["expiry_date"],
                            "token_type": token_data["token_type"],
                            "scope": token_data["scope"],
                        }
                    )

                    # Create metadata object
                    if not creds.get("_proxy_metadata"):
                        creds["_proxy_metadata"] = {
                            "email": token_data["email"],
                            "last_check_timestamp": time.time(),
                        }

                    if path:
                        await self._save_credentials(path, creds)

                    lib_logger.info(
                        f"iFlow OAuth initialized successfully for '{display_name}'."
                    )
                    return creds

                finally:
                    await callback_server.stop()

            lib_logger.info(f"iFlow OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(f"Failed to initialize iFlow OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Returns auth header with API key (NOT OAuth access_token).
        CRITICAL: iFlow API requests use the api_key, not the OAuth tokens.
        """
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)

        api_key = creds.get("api_key")
        if not api_key:
            raise ValueError("Missing api_key in iFlow credentials")

        return {"Authorization": f"Bearer {api_key}"}

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Retrieves user info from the _proxy_metadata in the credential file."""
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            # Ensure the token is valid
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(path)

            email = creds.get("email") or creds.get("_proxy_metadata", {}).get("email")

            if not email:
                lib_logger.warning(
                    f"No email found in iFlow credentials for '{path or 'in-memory object'}'."
                )

            # Update timestamp on check
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get iFlow user info from credentials: {e}")
            return {"email": None}
