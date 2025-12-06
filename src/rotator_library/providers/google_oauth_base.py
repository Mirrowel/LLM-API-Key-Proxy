# src/rotator_library/providers/google_oauth_base.py

import os
import webbrowser
from typing import Union, Optional
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger("rotator_library")

console = Console()


class GoogleOAuthBase:
    """
    Base class for Google OAuth2 authentication providers.

    Subclasses must override:
        - CLIENT_ID: OAuth client ID
        - CLIENT_SECRET: OAuth client secret
        - OAUTH_SCOPES: List of OAuth scopes
        - ENV_PREFIX: Prefix for environment variables (e.g., "GEMINI_CLI", "ANTIGRAVITY")

    Subclasses may optionally override:
        - CALLBACK_PORT: Local OAuth callback server port (default: 8085)
        - CALLBACK_PATH: OAuth callback path (default: "/oauth2callback")
        - REFRESH_EXPIRY_BUFFER_SECONDS: Time buffer before token expiry (default: 30 minutes)
    """

    # Subclasses MUST override these
    CLIENT_ID: str = None
    CLIENT_SECRET: str = None
    OAUTH_SCOPES: list = None
    ENV_PREFIX: str = None

    # Subclasses MAY override these
    TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    USER_INFO_URI: str = "https://www.googleapis.com/oauth2/v1/userinfo"
    CALLBACK_PORT: int = 8085
    CALLBACK_PATH: str = "/oauth2callback"
    REFRESH_EXPIRY_BUFFER_SECONDS: int = 30 * 60  # 30 minutes

    def __init__(self):
        # Validate that subclass has set required attributes
        if self.CLIENT_ID is None:
            raise NotImplementedError(f"{self.__class__.__name__} must set CLIENT_ID")
        if self.CLIENT_SECRET is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set CLIENT_SECRET"
            )
        if self.OAUTH_SCOPES is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set OAUTH_SCOPES"
            )
        if self.ENV_PREFIX is None:
            raise NotImplementedError(f"{self.__class__.__name__} must set ENV_PREFIX")

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
        - "env://provider/1" - First numbered credential (PROVIDER_1_ACCESS_TOKEN)
        - "env://provider/2" - Second numbered credential, etc.

        Returns:
            The credential index as string ("0" for legacy, "1", "2", etc. for numbered)
            or None if path is not an env:// path
        """
        if not path.startswith("env://"):
            return None

        # Parse: env://provider/index
        parts = path[6:].split("/")  # Remove "env://" prefix
        if len(parts) >= 2:
            return parts[1]  # Return the index
        return "0"  # Default to legacy format

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables for stateless deployments.

        Supports two formats:
        1. Legacy (credential_index="0" or None): PROVIDER_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): PROVIDER_1_ACCESS_TOKEN, PROVIDER_2_ACCESS_TOKEN

        Expected environment variables (for numbered format with index N):
        - {ENV_PREFIX}_{N}_ACCESS_TOKEN (required)
        - {ENV_PREFIX}_{N}_REFRESH_TOKEN (required)
        - {ENV_PREFIX}_{N}_EXPIRY_DATE (optional, defaults to 0)
        - {ENV_PREFIX}_{N}_CLIENT_ID (optional, uses default)
        - {ENV_PREFIX}_{N}_CLIENT_SECRET (optional, uses default)
        - {ENV_PREFIX}_{N}_TOKEN_URI (optional, uses default)
        - {ENV_PREFIX}_{N}_UNIVERSE_DOMAIN (optional, defaults to googleapis.com)
        - {ENV_PREFIX}_{N}_EMAIL (optional, defaults to "env-user-{N}")
        - {ENV_PREFIX}_{N}_PROJECT_ID (optional)
        - {ENV_PREFIX}_{N}_TIER (optional)

        For legacy format (index="0" or None), omit the _{N}_ part.

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            # Numbered format: PROVIDER_N_ACCESS_TOKEN
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            # Legacy format: PROVIDER_ACCESS_TOKEN
            prefix = self.ENV_PREFIX
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        # Both access and refresh tokens are required
        if not (access_token and refresh_token):
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

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
            "client_id": os.getenv(f"{prefix}_CLIENT_ID", self.CLIENT_ID),
            "client_secret": os.getenv(f"{prefix}_CLIENT_SECRET", self.CLIENT_SECRET),
            "token_uri": os.getenv(f"{prefix}_TOKEN_URI", self.TOKEN_URI),
            "universe_domain": os.getenv(f"{prefix}_UNIVERSE_DOMAIN", "googleapis.com"),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,  # Flag to indicate env-based credentials
                "env_credential_index": credential_index
                or "0",  # Track which env credential this is
            },
        }

        # Add project_id if provided
        project_id = os.getenv(f"{prefix}_PROJECT_ID")
        if project_id:
            creds["_proxy_metadata"]["project_id"] = project_id

        # Add tier if provided
        tier = os.getenv(f"{prefix}_TIER")
        if tier:
            creds["_proxy_metadata"]["tier"] = tier

        return creds

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                # Load from environment variables with specific index
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

            # For file paths, first try loading from legacy env vars (for backwards compatibility)
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info(
                    f"Using {self.ENV_PREFIX} credentials from environment variables"
                )
                # Cache env-based credentials using the path as key
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            try:
                lib_logger.debug(
                    f"Loading {self.ENV_PREFIX} credentials from file: {path}"
                )
                with open(path, "r") as f:
                    creds = json.load(f)
                # Handle gcloud-style creds file which nest tokens under "credential"
                if "credential" in creds:
                    creds = creds["credential"]
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
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials from '{path}': {e}"
                )

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
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

            # Update cache AFTER successful file write (prevents cache/file inconsistency)
            self._credentials_cache[path] = creds
            lib_logger.debug(
                f"Saved updated {self.ENV_PREFIX} OAuth credentials to '{path}' (atomic write)."
            )

        except Exception as e:
            lib_logger.error(
                f"Failed to save updated {self.ENV_PREFIX} OAuth credentials to '{path}': {e}"
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
        expiry = creds.get("token_expiry")  # gcloud format
        if not expiry:  # gemini-cli format
            expiry_timestamp = creds.get("expiry_date", 0) / 1000
        else:
            expiry_timestamp = time.mktime(time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ"))
        return expiry_timestamp < time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(
        self, path: str, creds: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        async with await self._get_lock(path):
            # Skip the expiry check if a refresh is being forced
            if not force and not self._is_token_expired(
                self._credentials_cache.get(path, creds)
            ):
                return self._credentials_cache.get(path, creds)

            lib_logger.debug(
                f"Refreshing {self.ENV_PREFIX} OAuth token for '{Path(path).name}' (forced: {force})..."
            )
            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None
            needs_reauth = False

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            self.TOKEN_URI,
                            data={
                                "client_id": creds.get("client_id", self.CLIENT_ID),
                                "client_secret": creds.get(
                                    "client_secret", self.CLIENT_SECRET
                                ),
                                "refresh_token": refresh_token,
                                "grant_type": "refresh_token",
                            },
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break  # Success, exit retry loop

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code

                        # [INVALID GRANT HANDLING] Handle 401/403 by triggering re-authentication
                        if status_code == 401 or status_code == 403:
                            lib_logger.warning(
                                f"Refresh token invalid for '{Path(path).name}' (HTTP {status_code}). "
                                f"Token may have been revoked or expired. Starting re-authentication..."
                            )
                            needs_reauth = True
                            break  # Exit retry loop to trigger re-auth

                        elif status_code == 429:
                            # Rate limit - honor Retry-After header if present
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(
                                f"Rate limited (HTTP 429), retry after {retry_after}s"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif status_code >= 500 and status_code < 600:
                            # Server error - retry with exponential backoff
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt  # 1s, 2s, 4s
                                lib_logger.warning(
                                    f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            raise  # Final attempt failed

                        else:
                            # Other errors - don't retry
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        # Network errors - retry with backoff
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
                    return new_creds
                except Exception as reauth_error:
                    lib_logger.error(
                        f"Re-authentication failed for '{Path(path).name}': {reauth_error}"
                    )
                    raise ValueError(
                        f"Refresh token invalid and re-authentication failed: {reauth_error}"
                    )

            # If we exhausted retries without success
            if new_token_data is None:
                raise last_error or Exception("Token refresh failed after all retries")

            # [FIX 1] Update OAuth token fields from response
            creds["access_token"] = new_token_data["access_token"]
            expiry_timestamp = time.time() + new_token_data["expires_in"]
            creds["expiry_date"] = expiry_timestamp * 1000  # gemini-cli format

            # [FIX 2] Update refresh_token if server provided a new one (rare but possible with Google OAuth)
            if "refresh_token" in new_token_data:
                creds["refresh_token"] = new_token_data["refresh_token"]

            # [FIX 3] Ensure all required OAuth client fields are present (restore if missing)
            if "client_id" not in creds or not creds["client_id"]:
                creds["client_id"] = self.CLIENT_ID
            if "client_secret" not in creds or not creds["client_secret"]:
                creds["client_secret"] = self.CLIENT_SECRET
            if "token_uri" not in creds or not creds["token_uri"]:
                creds["token_uri"] = self.TOKEN_URI
            if "universe_domain" not in creds or not creds["universe_domain"]:
                creds["universe_domain"] = "googleapis.com"

            # [FIX 4] Add scopes array if missing
            if "scopes" not in creds:
                creds["scopes"] = self.OAUTH_SCOPES

            # [FIX 5] Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}
            creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify refreshed credentials have all required fields
            required_fields = [
                "access_token",
                "refresh_token",
                "client_id",
                "client_secret",
                "token_uri",
            ]
            missing_fields = [
                field for field in required_fields if not creds.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Refreshed credentials missing required fields: {missing_fields}"
                )

            # [VALIDATION] Optional: Test that the refreshed token is actually usable
            try:
                async with httpx.AsyncClient() as client:
                    test_response = await client.get(
                        self.USER_INFO_URI,
                        headers={"Authorization": f"Bearer {creds['access_token']}"},
                        timeout=5.0,
                    )
                    test_response.raise_for_status()
                    lib_logger.debug(
                        f"Token validation successful for '{Path(path).name}'"
                    )
            except Exception as e:
                lib_logger.warning(
                    f"Refreshed token validation failed for '{Path(path).name}': {e}"
                )
                # Don't fail the refresh - the token might still work for other endpoints
                # But log it for debugging purposes

            await self._save_credentials(path, creds)
            lib_logger.debug(
                f"Successfully refreshed {self.ENV_PREFIX} OAuth token for '{Path(path).name}'."
            )
            return creds

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a credential by queueing it for refresh."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            # Queue for refresh with needs_reauth=False (automated refresh)
            await self._queue_refresh(credential_path, force=False, needs_reauth=False)

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        # This prevents TOCTOU bug where multiple coroutines check and create simultaneously
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
                        await self._refresh_token(path, creds, force=force)

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
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
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
            reason = ""
            if not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path, creds)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: {reason}."
                )

                # [HEADLESS DETECTION] Check if running in headless environment
                is_headless = is_headless_environment()

                auth_code_future = asyncio.get_event_loop().create_future()
                server = None

                async def handle_callback(reader, writer):
                    try:
                        request_line_bytes = await reader.readline()
                        if not request_line_bytes:
                            return
                        path_str = (
                            request_line_bytes.decode("utf-8").strip().split(" ")[1]
                        )
                        while await reader.readline() != b"\r\n":
                            pass
                        from urllib.parse import urlparse, parse_qs

                        query_params = parse_qs(urlparse(path_str).query)
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
                        )
                        if "code" in query_params:
                            if not auth_code_future.done():
                                auth_code_future.set_result(query_params["code"][0])
                            writer.write(
                                b"<html><body><h1>Authentication successful!</h1><p>You can close this window.</p></body></html>"
                            )
                        else:
                            error = query_params.get("error", ["Unknown error"])[0]
                            if not auth_code_future.done():
                                auth_code_future.set_exception(
                                    Exception(f"OAuth failed: {error}")
                                )
                            writer.write(
                                f"<html><body><h1>Authentication Failed</h1><p>Error: {error}. Please try again.</p></body></html>".encode()
                            )
                        await writer.drain()
                    except Exception as e:
                        lib_logger.error(f"Error in OAuth callback handler: {e}")
                    finally:
                        writer.close()

                try:
                    server = await asyncio.start_server(
                        handle_callback, "127.0.0.1", self.CALLBACK_PORT
                    )
                    from urllib.parse import urlencode

                    auth_url = (
                        "https://accounts.google.com/o/oauth2/v2/auth?"
                        + urlencode(
                            {
                                "client_id": self.CLIENT_ID,
                                "redirect_uri": f"http://localhost:{self.CALLBACK_PORT}{self.CALLBACK_PATH}",
                                "scope": " ".join(self.OAUTH_SCOPES),
                                "access_type": "offline",
                                "response_type": "code",
                                "prompt": "consent",
                            }
                        )
                    )

                    # [HEADLESS SUPPORT] Display appropriate instructions
                    if is_headless:
                        auth_panel_text = Text.from_markup(
                            "Running in headless environment (no GUI detected).\n"
                            "Please open the URL below in a browser on another machine to authorize:\n"
                        )
                    else:
                        auth_panel_text = Text.from_markup(
                            "1. Your browser will now open to log in and authorize the application.\n"
                            "2. If it doesn't open automatically, please open the URL below manually."
                        )

                    console.print(
                        Panel(
                            auth_panel_text,
                            title=f"{self.ENV_PREFIX} OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
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
                                "Browser opened successfully for OAuth flow"
                            )
                        except Exception as e:
                            lib_logger.warning(
                                f"Failed to open browser automatically: {e}. Please open the URL manually."
                            )

                    with console.status(
                        f"[bold green]Waiting for you to complete authentication in the browser...[/bold green]",
                        spinner="dots",
                    ):
                        auth_code = await asyncio.wait_for(
                            auth_code_future, timeout=300
                        )
                except asyncio.TimeoutError:
                    raise Exception("OAuth flow timed out. Please try again.")
                finally:
                    if server:
                        server.close()
                        await server.wait_closed()

                lib_logger.info(
                    f"Attempting to exchange authorization code for tokens..."
                )
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.TOKEN_URI,
                        data={
                            "code": auth_code.strip(),
                            "client_id": self.CLIENT_ID,
                            "client_secret": self.CLIENT_SECRET,
                            "redirect_uri": f"http://localhost:{self.CALLBACK_PORT}{self.CALLBACK_PATH}",
                            "grant_type": "authorization_code",
                        },
                    )
                    response.raise_for_status()
                    token_data = response.json()
                    # Start with the full token data from the exchange
                    creds = token_data.copy()

                    # Convert 'expires_in' to 'expiry_date' in milliseconds
                    creds["expiry_date"] = (
                        time.time() + creds.pop("expires_in")
                    ) * 1000

                    # Ensure client_id and client_secret are present
                    creds["client_id"] = self.CLIENT_ID
                    creds["client_secret"] = self.CLIENT_SECRET

                    creds["token_uri"] = self.TOKEN_URI
                    creds["universe_domain"] = "googleapis.com"

                    # Fetch user info and add metadata
                    user_info_response = await client.get(
                        self.USER_INFO_URI,
                        headers={"Authorization": f"Bearer {creds['access_token']}"},
                    )
                    user_info_response.raise_for_status()
                    user_info = user_info_response.json()
                    creds["_proxy_metadata"] = {
                        "email": user_info.get("email"),
                        "last_check_timestamp": time.time(),
                    }

                    if path:
                        await self._save_credentials(path, creds)
                    lib_logger.info(
                        f"{self.ENV_PREFIX} OAuth initialized successfully for '{display_name}'."
                    )
                return creds

            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
            )
            return creds
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            )

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path, creds)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None
        creds = await self._load_credentials(creds_or_path) if path else creds_or_path

        if path and self._is_token_expired(creds):
            creds = await self._refresh_token(path, creds)

        # Prefer locally stored metadata
        if creds.get("_proxy_metadata", {}).get("email"):
            if path:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)
            return {"email": creds["_proxy_metadata"]["email"]}

        # Fallback to API call if metadata is missing
        headers = {"Authorization": f"Bearer {creds['access_token']}"}
        async with httpx.AsyncClient() as client:
            response = await client.get(self.USER_INFO_URI, headers=headers)
            response.raise_for_status()
            user_info = response.json()

            # Save the retrieved info for future use
            creds["_proxy_metadata"] = {
                "email": user_info.get("email"),
                "last_check_timestamp": time.time(),
            }
            if path:
                await self._save_credentials(path, creds)
            return {"email": user_info.get("email")}
