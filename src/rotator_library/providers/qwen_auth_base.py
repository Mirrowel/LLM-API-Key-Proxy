# src/rotator_library/providers/qwen_auth_base.py

import secrets
import hashlib
import base64
import json
import time
import asyncio
import logging
import webbrowser
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")

CLIENT_ID = (
    "f0304373b74a44d2b584a3fb70ca9e56"  # https://api.kilocode.ai/extension-config.json
)
SCOPE = "openid profile email model.completion"
TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
REFRESH_EXPIRY_BUFFER_SECONDS = 3 * 60 * 60  # 3 hours buffer before expiry

console = Console()


class QwenAuthBase:
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
        # [FIX PR#34] Changed from set to dict mapping credential path to timestamp
        # This enables TTL-based stale entry cleanup as defense in depth
        self._unavailable_credentials: Dict[
            str, float
        ] = {}  # Maps credential path -> timestamp when marked unavailable
        self._unavailable_ttl_seconds: int = 300  # 5 minutes TTL for stale entries
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets
        self._queue_processor_task: Optional[asyncio.Task] = (
            None  # Background worker task
        )

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://provider/0" - Legacy single credential (no index in env var names)
        - "env://provider/1" - First numbered credential (QWEN_CODE_1_ACCESS_TOKEN)

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
        1. Legacy (credential_index="0" or None): QWEN_CODE_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): QWEN_CODE_1_ACCESS_TOKEN, etc.

        Expected environment variables (for numbered format with index N):
        - QWEN_CODE_{N}_ACCESS_TOKEN (required)
        - QWEN_CODE_{N}_REFRESH_TOKEN (required)
        - QWEN_CODE_{N}_EXPIRY_DATE (optional, defaults to 0)
        - QWEN_CODE_{N}_RESOURCE_URL (optional, defaults to https://portal.qwen.ai/v1)
        - QWEN_CODE_{N}_EMAIL (optional, defaults to "env-user-{N}")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            prefix = f"QWEN_CODE_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "QWEN_CODE"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        # Both access and refresh tokens are required
        if not (access_token and refresh_token):
            return None

        lib_logger.debug(
            f"Loading Qwen Code credentials from environment variables (prefix: {prefix})"
        )

        # Parse expiry_date as float, default to 0 if not present
        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid {prefix}_EXPIRY_DATE value: {expiry_str}, using 0"
            )
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "resource_url": os.getenv(
                f"{prefix}_RESOURCE_URL", "https://portal.qwen.ai/v1"
            ),
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
            lib_logger.debug(f"Reading Qwen credentials from file: {path}")
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"Qwen OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load Qwen OAuth credentials from '{path}': {e}")

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
                        f"Using Qwen Code credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for Qwen Code credential index {credential_index} not found"
                    )

            # For file paths, try loading from legacy env vars first
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info(
                    "Using Qwen Code credentials from environment variables"
                )
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            return await self._read_creds_from_file(path)

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials with in-memory fallback if disk unavailable."""
        # Always update cache first (memory is reliable)
        self._credentials_cache[path] = creds

        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return

        # Attempt disk write - if it fails, we still have the cache
        # buffer_on_failure ensures data is retried periodically and saved on shutdown
        if safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=True
        ):
            lib_logger.debug(f"Saved updated Qwen OAuth credentials to '{path}'.")
        else:
            lib_logger.warning(
                "Qwen credentials cached in memory only (buffered for retry)."
            )

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_timestamp = creds.get("expiry_date", 0) / 1000
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # If cache is empty, read from file. This is safe because we hold the lock.
            if path not in self._credentials_cache:
                await self._read_creds_from_file(path)

            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing Qwen OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                lib_logger.error(f"No refresh_token found in '{Path(path).name}'")
                raise ValueError("No refresh_token found in Qwen credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None
            needs_reauth = False

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            }

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            TOKEN_ENDPOINT,
                            headers=headers,
                            data={
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                                "client_id": CLIENT_ID,
                            },
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break  # Success

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text
                        lib_logger.error(
                            f"HTTP {status_code} for '{Path(path).name}': {error_body}"
                        )

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

            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )
            creds_from_file["expiry_date"] = (
                time.time() + new_token_data["expires_in"]
            ) * 1000
            creds_from_file["resource_url"] = new_token_data.get(
                "resource_url", creds_from_file.get("resource_url")
            )

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify required fields exist after refresh
            required_fields = ["access_token", "refresh_token"]
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
                f"Successfully refreshed Qwen OAuth token for '{Path(path).name}'."
            )
            return creds_from_file

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and access token.

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

            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            base_url = creds.get("resource_url", "https://portal.qwen.ai/v1")
            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"
            access_token = creds["access_token"]
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for Qwen Code")
            base_url = "https://portal.qwen.ai/v1"
            access_token = credential_identifier

        return base_url, access_token

    async def proactively_refresh(self, credential_identifier: str):
        """
        Proactively refreshes tokens if they're close to expiry.
        Only applies to OAuth credentials (file paths or env:// paths). Direct API keys are skipped.
        """
        lib_logger.debug(f"proactively_refresh called for: {credential_identifier}")

        # Try to load credentials - this will fail for direct API keys
        # and succeed for OAuth credentials (file paths or env:// paths)
        try:
            creds = await self._load_credentials(credential_identifier)
        except IOError as e:
            # Not a valid credential path (likely a direct API key string)
            lib_logger.debug(
                f"Skipping refresh for '{credential_identifier}' - not an OAuth credential: {e}"
            )
            return

        is_expired = self._is_token_expired(creds)
        lib_logger.debug(
            f"Token expired check for '{Path(credential_identifier).name}': {is_expired}"
        )

        if is_expired:
            lib_logger.debug(
                f"Queueing refresh for '{Path(credential_identifier).name}'"
            )
            # Queue for refresh with needs_reauth=False (automated refresh)
            await self._queue_refresh(
                credential_identifier, force=False, needs_reauth=False
            )

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation (not queued/refreshing).

        [FIX PR#34] Now includes TTL-based stale entry cleanup as defense in depth.
        If a credential has been unavailable for longer than _unavailable_ttl_seconds,
        it is automatically cleaned up and considered available.
        """
        if path not in self._unavailable_credentials:
            return True

        # [FIX PR#34] Check if the entry is stale (TTL expired)
        marked_time = self._unavailable_credentials.get(path)
        if marked_time is not None:
            now = time.time()
            if now - marked_time > self._unavailable_ttl_seconds:
                # Entry is stale - clean it up and return available
                lib_logger.warning(
                    f"Credential '{Path(path).name}' was stuck in unavailable state for "
                    f"{int(now - marked_time)}s (TTL: {self._unavailable_ttl_seconds}s). "
                    f"Auto-cleaning stale entry."
                )
                self._unavailable_credentials.pop(path, None)
                return True

        return False

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
                # [FIX PR#34] Store timestamp when marking unavailable (for TTL cleanup)
                self._unavailable_credentials[path] = time.time()
                lib_logger.debug(
                    f"Marked '{Path(path).name}' as unavailable. "
                    f"Total unavailable: {len(self._unavailable_credentials)}"
                )
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
                    # [FIX PR#34] Clean up any stale unavailable entries before exiting
                    # If we're idle for 60s, no refreshes are in progress
                    async with self._queue_tracking_lock:
                        if self._unavailable_credentials:
                            stale_count = len(self._unavailable_credentials)
                            lib_logger.warning(
                                f"Queue processor idle timeout. Cleaning {stale_count} "
                                f"stale unavailable credentials: {list(self._unavailable_credentials.keys())}"
                            )
                            self._unavailable_credentials.clear()
                        # [FIX BUG#6] Also clear queued credentials to prevent stuck state
                        if self._queued_credentials:
                            lib_logger.debug(
                                f"Clearing {len(self._queued_credentials)} queued credentials on timeout"
                            )
                            self._queued_credentials.clear()
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
                                self._unavailable_credentials.pop(path, None)
                                lib_logger.debug(
                                    f"Credential '{Path(path).name}' no longer expired, marked available. "
                                    f"Remaining unavailable: {len(self._unavailable_credentials)}"
                                )
                            continue

                        # Perform refresh
                        if not creds:
                            creds = await self._load_credentials(path)
                        await self._refresh_token(path, force=force)

                        # SUCCESS: Mark as available again
                        async with self._queue_tracking_lock:
                            self._unavailable_credentials.pop(path, None)
                            lib_logger.debug(
                                f"Refresh SUCCESS for '{Path(path).name}', marked available. "
                                f"Remaining unavailable: {len(self._unavailable_credentials)}"
                            )

                finally:
                    # [FIX PR#34] Remove from BOTH queued set AND unavailable credentials
                    # This ensures cleanup happens in ALL exit paths (success, exception, etc.)
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        # [FIX PR#34] Always clean up unavailable credentials in finally block
                        self._unavailable_credentials.pop(path, None)
                        lib_logger.debug(
                            f"Finally cleanup for '{Path(path).name}'. "
                            f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        )
                    self._refresh_queue.task_done()
            except asyncio.CancelledError:
                # [FIX PR#34] Clean up the current credential before breaking
                if path:
                    async with self._queue_tracking_lock:
                        self._unavailable_credentials.pop(path, None)
                        lib_logger.debug(
                            f"CancelledError cleanup for '{Path(path).name}'. "
                            f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        )
                break
            except Exception as e:
                lib_logger.error(f"Error in queue processor: {e}")
                # Even on error, mark as available (backoff will prevent immediate retry)
                if path:
                    async with self._queue_tracking_lock:
                        self._unavailable_credentials.pop(path, None)
                        lib_logger.debug(
                            f"Error cleanup for '{Path(path).name}': {e}. "
                            f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        )

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth device flow (browser-based authentication).

        This method is called via the global ReauthCoordinator to ensure
        only one interactive OAuth flow runs at a time across all providers.

        Args:
            path: Credential file path
            creds: Current credentials dict (will be updated)
            display_name: Display name for logging/UI

        Returns:
            Updated credentials dict with new tokens
        """
        # [HEADLESS DETECTION] Check if running in headless environment
        is_headless = is_headless_environment()

        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("utf-8")).digest()
            )
            .decode("utf-8")
            .rstrip("=")
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient() as client:
            request_data = {
                "client_id": CLIENT_ID,
                "scope": SCOPE,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
            lib_logger.debug(f"Qwen device code request data: {request_data}")
            try:
                dev_response = await client.post(
                    "https://chat.qwen.ai/api/v1/oauth2/device/code",
                    headers=headers,
                    data=request_data,
                )
                dev_response.raise_for_status()
                dev_data = dev_response.json()
                lib_logger.debug(f"Qwen device auth response: {dev_data}")
            except httpx.HTTPStatusError as e:
                lib_logger.error(
                    f"Qwen device code request failed with status {e.response.status_code}: {e.response.text}"
                )
                raise e

            # [HEADLESS SUPPORT] Display appropriate instructions
            if is_headless:
                auth_panel_text = Text.from_markup(
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:\n"
                    "1. Visit the URL below to sign in.\n"
                    "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                    "3. You will be prompted to enter your identifier after authorization."
                )
            else:
                auth_panel_text = Text.from_markup(
                    "1. Visit the URL below to sign in.\n"
                    "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                    "3. You will be prompted to enter your identifier after authorization."
                )

            console.print(
                Panel(
                    auth_panel_text,
                    title=f"Qwen OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                    style="bold blue",
                )
            )
            verification_url = dev_data["verification_uri_complete"]
            escaped_url = rich_escape(verification_url)
            console.print(
                f"[bold]URL:[/bold] [link={verification_url}]{escaped_url}[/link]\n"
            )

            # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
            if not is_headless:
                try:
                    webbrowser.open(dev_data["verification_uri_complete"])
                    lib_logger.info("Browser opened successfully for Qwen OAuth flow")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. Please open the URL manually."
                    )

            token_data = None
            start_time = time.time()
            interval = dev_data.get("interval", 5)

            with console.status(
                "[bold green]Polling for token, please complete authentication in the browser...[/bold green]",
                spinner="dots",
            ) as status:
                while time.time() - start_time < dev_data["expires_in"]:
                    poll_response = await client.post(
                        TOKEN_ENDPOINT,
                        headers=headers,
                        data={
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "device_code": dev_data["device_code"],
                            "client_id": CLIENT_ID,
                            "code_verifier": code_verifier,
                        },
                    )
                    if poll_response.status_code == 200:
                        token_data = poll_response.json()
                        lib_logger.info("Successfully received token.")
                        break
                    elif poll_response.status_code == 400:
                        poll_data = poll_response.json()
                        error_type = poll_data.get("error")
                        if error_type == "authorization_pending":
                            lib_logger.debug(
                                f"Polling status: {error_type}, waiting {interval}s"
                            )
                        elif error_type == "slow_down":
                            interval = int(interval * 1.5)
                            if interval > 10:
                                interval = 10
                            lib_logger.debug(
                                f"Polling status: {error_type}, waiting {interval}s"
                            )
                        else:
                            raise ValueError(
                                f"Token polling failed: {poll_data.get('error_description', error_type)}"
                            )
                    else:
                        poll_response.raise_for_status()

                    await asyncio.sleep(interval)

            if not token_data:
                raise TimeoutError("Qwen device flow timed out.")

            creds.update(
                {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                    "resource_url": token_data.get("resource_url"),
                }
            )

            # Prompt for user identifier and create metadata object if needed
            if not creds.get("_proxy_metadata", {}).get("email"):
                try:
                    prompt_text = Text.from_markup(
                        f"\\n[bold]Please enter your email or a unique identifier for [yellow]'{display_name}'[/yellow][/bold]"
                    )
                    email = Prompt.ask(prompt_text)
                    creds["_proxy_metadata"] = {
                        "email": email.strip(),
                        "last_check_timestamp": time.time(),
                    }
                except (EOFError, KeyboardInterrupt):
                    console.print(
                        "\\n[bold yellow]No identifier provided. Deduplication will not be possible.[/bold yellow]"
                    )
                    creds["_proxy_metadata"] = {
                        "email": None,
                        "last_check_timestamp": time.time(),
                    }

            if path:
                await self._save_credentials(path, creds)
            lib_logger.info(
                f"Qwen OAuth initialized successfully for '{display_name}'."
            )
        return creds

    async def initialize_token(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token, triggering interactive device flow if needed.

        If interactive OAuth is required (expired refresh token, missing credentials, etc.),
        the flow is coordinated globally via ReauthCoordinator to ensure only one
        interactive OAuth flow runs at a time across all providers.
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Qwen token for '{display_name}'...")
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
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"Qwen OAuth token for '{display_name}' needs setup: {reason}."
                )

                # [GLOBAL REAUTH COORDINATION] Use the global coordinator to ensure
                # only one interactive OAuth flow runs at a time across all providers
                coordinator = get_reauth_coordinator()

                # Define the interactive OAuth function to be executed by coordinator
                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(
                        path, creds, display_name
                    )

                # Execute via global coordinator (ensures only one at a time)
                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name="QWEN_CODE",
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,  # 5 minute timeout for user to complete OAuth
                )

            lib_logger.info(f"Qwen OAuth token at '{display_name}' is valid.")
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Qwen OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Retrieves user info from the _proxy_metadata in the credential file.
        """
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            # This will ensure the token is valid and metadata exists if the flow was just run
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(
                    path
                )  # Re-load after potential init

            metadata = creds.get("_proxy_metadata", {"email": None})
            email = metadata.get("email")

            if not email:
                lib_logger.warning(
                    f"No email found in _proxy_metadata for '{path or 'in-memory object'}'."
                )

            # Update timestamp on check and save if it's a file-based credential
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get Qwen user info from credentials: {e}")
            return {"email": None}
