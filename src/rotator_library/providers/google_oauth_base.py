# src/rotator_library/providers/google_oauth_base.py

import os
import re
import webbrowser
from dataclasses import dataclass, field
from typing import Union, Optional, List
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
from glob import glob

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json
from ..error_handler import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")

console = Console()

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================
# These are class-level defaults that can be overridden per subclass.
# Environment variable overrides are checked at runtime (see callback_port property).

# Default OAuth callback port for local redirect server
# Override per-provider: {ENV_PREFIX}_OAUTH_PORT=<port>
DEFAULT_OAUTH_CALLBACK_PORT: int = 8085

# Default OAuth callback path
DEFAULT_OAUTH_CALLBACK_PATH: str = "/oauth2callback"

# Token refresh buffer in seconds (refresh tokens this far before expiry)
# Default: 30 minutes before expiry
DEFAULT_REFRESH_EXPIRY_BUFFER: int = 30 * 60  # 1800 seconds


@dataclass
class CredentialSetupResult:
    """
    Standardized result structure for credential setup operations.

    Used by all auth classes to return consistent setup results to the credential tool.
    """

    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    project_id: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


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
    CALLBACK_PORT: int = DEFAULT_OAUTH_CALLBACK_PORT
    CALLBACK_PATH: str = DEFAULT_OAUTH_CALLBACK_PATH
    REFRESH_EXPIRY_BUFFER_SECONDS: int = DEFAULT_REFRESH_EXPIRY_BUFFER

    # =========================================================================
    # PKCE (Proof Key for Code Exchange) SUPPORT
    # =========================================================================

    def _generate_pkce(self) -> tuple:
        """
        Generate PKCE code_verifier and code_challenge.

        PKCE (Proof Key for Code Exchange) prevents authorization code interception attacks.
        Required for public OAuth clients per RFC 7636.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        import secrets
        import hashlib
        import base64

        # code_verifier: 43-128 chars, using URL-safe base64
        code_verifier = secrets.token_urlsafe(32)  # Produces 43 chars

        # code_challenge: BASE64URL(SHA256(code_verifier)) without padding
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

        return code_verifier, code_challenge

    def _encode_oauth_state(self, code_verifier: str) -> str:
        """
        Encode OAuth state parameter containing PKCE verifier.

        The state parameter provides CSRF protection and carries the PKCE verifier
        so it can be recovered after the OAuth callback.

        Args:
            code_verifier: The PKCE code verifier to encode

        Returns:
            Base64url-encoded state string
        """
        import base64

        state_data = {"v": code_verifier}  # Minimal - just verifier
        json_bytes = json.dumps(state_data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("ascii").rstrip("=")

    def _decode_oauth_state(self, state: str) -> str:
        """
        Decode OAuth state and return code_verifier.

        Args:
            state: The base64url-encoded state string from OAuth callback

        Returns:
            The decoded code_verifier

        Raises:
            ValueError: If state cannot be decoded or is missing verifier
        """
        import base64

        # Re-pad base64 string (base64url encoding strips padding)
        padded = state + "=" * (-len(state) % 4)
        try:
            state_data = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            if "v" not in state_data:
                raise ValueError("Missing verifier in state")
            return state_data["v"]
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid state parameter: {e}")

    @property
    def callback_port(self) -> int:
        """
        Get the OAuth callback port, checking environment variable first.

        Reads from {ENV_PREFIX}_OAUTH_PORT environment variable, falling back
        to the class's CALLBACK_PORT default if not set.
        """
        env_var = f"{self.ENV_PREFIX}_OAUTH_PORT"
        env_value = os.getenv(env_var)
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                lib_logger.warning(
                    f"Invalid {env_var} value: {env_value}, using default {self.CALLBACK_PORT}"
                )
        return self.CALLBACK_PORT

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

        # [QUEUE SYSTEM] Sequential refresh processing with two separate queues
        # Normal refresh queue: for proactive token refresh (old token still valid)
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Re-auth queue: for invalid refresh tokens (requires user interaction)
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Tracking sets/dicts
        self._queued_credentials: set = set()  # Track credentials in either queue
        # Only credentials in re-auth queue are marked unavailable (not normal refresh)
        # TTL cleanup is defense-in-depth for edge cases where re-auth processor crashes
        self._unavailable_credentials: Dict[
            str, float
        ] = {}  # Maps credential path -> timestamp when marked unavailable
        # TTL should exceed reauth timeout (300s) to avoid premature cleanup
        self._unavailable_ttl_seconds: int = 360  # 6 minutes TTL for stale entries
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets

        # Retry tracking for normal refresh queue
        self._queue_retry_count: Dict[
            str, int
        ] = {}  # Track retry attempts per credential

        # Configuration constants
        self._refresh_timeout_seconds: int = 15  # Max time for single refresh
        self._refresh_interval_seconds: int = 30  # Delay between queue items
        self._refresh_max_retries: int = 3  # Attempts before kicked out
        self._reauth_timeout_seconds: int = 300  # Time for user to complete OAuth

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

            # Try file-based loading first (preferred for explicit file paths)
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
                # File not found - fall back to legacy env vars for backwards compatibility
                # This handles the case where only env vars are set and file paths are placeholders
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using {self.ENV_PREFIX} credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"{self.ENV_PREFIX} OAuth credential file not found at '{path}'"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials from '{path}': {e}"
                )

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
            lib_logger.debug(
                f"Saved updated {self.ENV_PREFIX} OAuth credentials to '{path}'."
            )
        else:
            lib_logger.warning(
                f"Credentials for {self.ENV_PREFIX} cached in memory only (buffered for retry)."
            )

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
                        error_body = e.response.text

                        # [INVALID GRANT HANDLING] Handle 400/401/403 by queuing for re-auth
                        # We must NOT call initialize_token from here as we hold a lock (would deadlock)
                        if status_code == 400:
                            # Check if this is an invalid_grant error
                            if "invalid_grant" in error_body.lower():
                                lib_logger.info(
                                    f"Credential '{Path(path).name}' needs re-auth (HTTP 400: invalid_grant). "
                                    f"Queued for re-authentication, rotating to next credential."
                                )
                                asyncio.create_task(
                                    self._queue_refresh(
                                        path, force=True, needs_reauth=True
                                    )
                                )
                                raise CredentialNeedsReauthError(
                                    credential_path=path,
                                    message=f"Refresh token invalid for '{Path(path).name}'. Re-auth queued.",
                                )
                            else:
                                # Other 400 error - raise it
                                raise

                        elif status_code in (401, 403):
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                                f"Queued for re-authentication, rotating to next credential."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}). Re-auth queued.",
                            )

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
            # lib_logger.info(f"Proactive refresh triggered for '{Path(credential_path).name}'")
            await self._queue_refresh(credential_path, force=False, needs_reauth=False)

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        # This prevents TOCTOU bug where multiple coroutines check and create simultaneously
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry, not just threshold).

        This is different from _is_token_expired() which uses a buffer for proactive refresh.
        This method checks if the token is actually unusable.
        """
        expiry = creds.get("token_expiry")  # gcloud format
        if not expiry:  # gemini-cli format
            expiry_timestamp = creds.get("expiry_date", 0) / 1000
        else:
            expiry_timestamp = time.mktime(time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ"))
        return expiry_timestamp < time.time()

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation.

        Credentials are unavailable if:
        1. In re-auth queue (token is truly broken, requires user interaction)
        2. Token is TRULY expired (past actual expiry, not just threshold)

        Note: Credentials in normal refresh queue are still available because
        the old token is valid until actual expiry.

        TTL cleanup (defense-in-depth): If a credential has been in the re-auth
        queue longer than _unavailable_ttl_seconds without being processed, it's
        cleaned up. This should only happen if the re-auth processor crashes or
        is cancelled without proper cleanup.
        """
        # Check if in re-auth queue (truly unavailable)
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    # Entry is stale - clean it up and return available
                    # This is a defense-in-depth for edge cases where re-auth
                    # processor crashed or was cancelled without cleanup
                    lib_logger.warning(
                        f"Credential '{Path(path).name}' stuck in re-auth queue for "
                        f"{int(now - marked_time)}s (TTL: {self._unavailable_ttl_seconds}s). "
                        f"Re-auth processor may have crashed. Auto-cleaning stale entry."
                    )
                    # Clean up both tracking structures for consistency
                    self._unavailable_credentials.pop(path, None)
                    self._queued_credentials.discard(path)
                else:
                    return False  # Still in re-auth, not available

        # Check if token is TRULY expired (not just threshold-expired)
        creds = self._credentials_cache.get(path)
        if creds and self._is_token_truly_expired(creds):
            # Token is actually expired - should not be used
            # Queue for refresh if not already queued
            if path not in self._queued_credentials:
                # lib_logger.debug(
                #     f"Credential '{Path(path).name}' is truly expired, queueing for refresh"
                # )
                asyncio.create_task(
                    self._queue_refresh(path, force=True, needs_reauth=False)
                )
            return False

        return True

    async def _ensure_queue_processor_running(self):
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _ensure_reauth_processor_running(self):
        """Lazily starts the re-auth queue processor if not already running."""
        if self._reauth_processor_task is None or self._reauth_processor_task.done():
            self._reauth_processor_task = asyncio.create_task(
                self._process_reauth_queue()
            )

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the appropriate refresh queue if not already queued.

        Args:
            path: Credential file path
            force: Force refresh even if not expired
            needs_reauth: True if full re-authentication needed (routes to re-auth queue)

        Queue routing:
        - needs_reauth=True: Goes to re-auth queue, marks as unavailable
        - needs_reauth=False: Goes to normal refresh queue, does NOT mark unavailable
          (old token is still valid until actual expiry)
        """
        # IMPORTANT: Only check backoff for simple automated refreshes
        # Re-authentication (interactive OAuth) should BYPASS backoff since it needs user input
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
                    # Credential is in backoff for automated refresh, do not queue
                    # remaining = int(backoff_until - now)
                    # lib_logger.debug(
                    #     f"Skipping automated refresh for '{Path(path).name}' (in backoff for {remaining}s)"
                    # )
                    return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)

                if needs_reauth:
                    # Re-auth queue: mark as unavailable (token is truly broken)
                    self._unavailable_credentials[path] = time.time()
                    # lib_logger.debug(
                    #     f"Queued '{Path(path).name}' for RE-AUTH (marked unavailable). "
                    #     f"Total unavailable: {len(self._unavailable_credentials)}"
                    # )
                    await self._reauth_queue.put(path)
                    await self._ensure_reauth_processor_running()
                else:
                    # Normal refresh queue: do NOT mark unavailable (old token still valid)
                    # lib_logger.debug(
                    #     f"Queued '{Path(path).name}' for refresh (still available). "
                    #     f"Queue size: {self._refresh_queue.qsize() + 1}"
                    # )
                    await self._refresh_queue.put((path, force))
                    await self._ensure_queue_processor_running()

    async def _process_refresh_queue(self):
        """Background worker that processes normal refresh requests sequentially.

        Key behaviors:
        - 15s timeout per refresh operation
        - 30s delay between processing credentials (prevents thundering herd)
        - On failure: back of queue, max 3 retries before kicked
        - If 401/403 detected: routes to re-auth queue
        - Does NOT mark credentials unavailable (old token still valid)
        """
        # lib_logger.info("Refresh queue processor started")
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path, force = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - clean up and exit
                    async with self._queue_tracking_lock:
                        # Clear any stale retry counts
                        self._queue_retry_count.clear()
                    self._queue_processor_task = None
                    # lib_logger.debug("Refresh queue processor idle, shutting down")
                    return

                try:
                    # Quick check if still expired (optimization to avoid unnecessary refresh)
                    creds = self._credentials_cache.get(path)
                    if creds and not self._is_token_expired(creds):
                        # No longer expired, skip refresh
                        # lib_logger.debug(
                        #     f"Credential '{Path(path).name}' no longer expired, skipping refresh"
                        # )
                        # Clear retry count on skip (not a failure)
                        self._queue_retry_count.pop(path, None)
                        continue

                    # Perform refresh with timeout
                    if not creds:
                        creds = await self._load_credentials(path)

                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, creds, force=force)

                        # SUCCESS: Clear retry count
                        self._queue_retry_count.pop(path, None)
                        # lib_logger.info(f"Refresh SUCCESS for '{Path(path).name}'")

                    except asyncio.TimeoutError:
                        lib_logger.warning(
                            f"Refresh timeout ({self._refresh_timeout_seconds}s) for '{Path(path).name}'"
                        )
                        await self._handle_refresh_failure(path, force, "timeout")

                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        if status_code in (401, 403):
                            # Invalid refresh token - route to re-auth queue
                            lib_logger.warning(
                                f"Refresh token invalid for '{Path(path).name}' (HTTP {status_code}). "
                                f"Routing to re-auth queue."
                            )
                            self._queue_retry_count.pop(path, None)  # Clear retry count
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(
                                    path
                                )  # Remove from queued
                            await self._queue_refresh(
                                path, force=True, needs_reauth=True
                            )
                        else:
                            await self._handle_refresh_failure(
                                path, force, f"HTTP {status_code}"
                            )

                    except Exception as e:
                        await self._handle_refresh_failure(path, force, str(e))

                finally:
                    # Remove from queued set (unless re-queued by failure handler)
                    async with self._queue_tracking_lock:
                        # Only discard if not re-queued (check if still in queue set from retry)
                        if (
                            path in self._queued_credentials
                            and self._queue_retry_count.get(path, 0) == 0
                        ):
                            self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()

                # Wait between credentials to spread load
                await asyncio.sleep(self._refresh_interval_seconds)

            except asyncio.CancelledError:
                # lib_logger.debug("Refresh queue processor cancelled")
                break
            except Exception as e:
                lib_logger.error(f"Error in refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic.

        - Increments retry count
        - If under max retries: re-adds to END of queue
        - If at max retries: kicks credential out (retried next BackgroundRefresher cycle)
        """
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            # Kicked out until next BackgroundRefresher cycle
            lib_logger.error(
                f"Max retries ({self._refresh_max_retries}) reached for '{Path(path).name}' "
                f"(last error: {error}). Will retry next refresh cycle."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        # Re-add to END of queue for retry
        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}, back of queue."
        )
        # Keep in queued_credentials set, add back to queue
        await self._refresh_queue.put((path, force))

    async def _process_reauth_queue(self):
        """Background worker that processes re-auth requests.

        Key behaviors:
        - Credentials ARE marked unavailable (token is truly broken)
        - Uses ReauthCoordinator for interactive OAuth
        - No automatic retry (requires user action)
        - Cleans up unavailable status when done
        """
        # lib_logger.info("Re-auth queue processor started")
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - exit
                    self._reauth_processor_task = None
                    # lib_logger.debug("Re-auth queue processor idle, shutting down")
                    return

                try:
                    lib_logger.info(f"Starting re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Re-auth SUCCESS for '{Path(path).name}'")

                except Exception as e:
                    lib_logger.error(f"Re-auth FAILED for '{Path(path).name}': {e}")
                    # No automatic retry for re-auth (requires user action)

                finally:
                    # Always clean up
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                        # lib_logger.debug(
                        #     f"Re-auth cleanup for '{Path(path).name}'. "
                        #     f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        # )
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                # Clean up current credential before breaking
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                # lib_logger.debug("Re-auth queue processor cancelled")
                break
            except Exception as e:
                lib_logger.error(f"Error in re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth flow (browser-based authentication).

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

        # [PKCE] Generate PKCE code verifier and challenge for enhanced security
        code_verifier, code_challenge = self._generate_pkce()

        # [STATE] Encode state parameter with PKCE verifier for CSRF protection
        oauth_state = self._encode_oauth_state(code_verifier)

        auth_code_future = asyncio.get_event_loop().create_future()
        server = None

        async def handle_callback(reader, writer):
            try:
                request_line_bytes = await reader.readline()
                if not request_line_bytes:
                    return
                path_str = request_line_bytes.decode("utf-8").strip().split(" ")[1]
                while await reader.readline() != b"\r\n":
                    pass
                from urllib.parse import urlparse, parse_qs

                query_params = parse_qs(urlparse(path_str).query)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
                if "code" in query_params:
                    # Extract code and state from callback
                    auth_code = query_params["code"][0]
                    received_state = query_params.get("state", [None])[0]

                    if not auth_code_future.done():
                        # Return both code and state for validation
                        auth_code_future.set_result((auth_code, received_state))
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
                handle_callback, "127.0.0.1", self.callback_port
            )
            from urllib.parse import urlencode

            # [PKCE + STATE] Include code_challenge, code_challenge_method, and state in auth URL
            auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
                {
                    "client_id": self.CLIENT_ID,
                    "redirect_uri": f"http://localhost:{self.callback_port}{self.CALLBACK_PATH}",
                    "scope": " ".join(self.OAUTH_SCOPES),
                    "access_type": "offline",
                    "response_type": "code",
                    "prompt": "consent",
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                    "state": oauth_state,
                }
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
            console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

            # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
            if not is_headless:
                try:
                    webbrowser.open(auth_url)
                    lib_logger.info("Browser opened successfully for OAuth flow")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. Please open the URL manually."
                    )

            with console.status(
                f"[bold green]Waiting for you to complete authentication in the browser...[/bold green]",
                spinner="dots",
            ):
                # Note: The 300s timeout here is handled by the ReauthCoordinator
                # We use a slightly longer internal timeout to let the coordinator handle it
                auth_code, received_state = await asyncio.wait_for(
                    auth_code_future, timeout=310
                )

            # [STATE VALIDATION] Validate state parameter and extract verifier
            effective_verifier = code_verifier  # Default to original verifier
            if received_state:
                try:
                    decoded_verifier = self._decode_oauth_state(received_state)
                    if decoded_verifier != code_verifier:
                        lib_logger.warning(
                            "OAuth state verifier mismatch - possible CSRF attempt. "
                            "Using original verifier."
                        )
                    else:
                        effective_verifier = decoded_verifier
                except ValueError as e:
                    lib_logger.warning(
                        f"Failed to decode OAuth state: {e}. Using original verifier."
                    )
            else:
                lib_logger.debug(
                    "No state parameter in callback - using original verifier"
                )
        except asyncio.TimeoutError:
            raise Exception("OAuth flow timed out. Please try again.")
        finally:
            if server:
                server.close()
                await server.wait_closed()

        lib_logger.info(f"Attempting to exchange authorization code for tokens...")
        async with httpx.AsyncClient() as client:
            # [PKCE + HEADERS] Include code_verifier and explicit headers for token exchange
            # Uses GEMINI_CLI style headers for consistent fingerprinting
            response = await client.post(
                self.TOKEN_URI,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "Accept": "*/*",
                    "Accept-Encoding": "gzip, deflate, br",
                    "User-Agent": "google-api-nodejs-client/10.3.0",
                    "X-Goog-Api-Client": "gl-node/22.18.0",
                },
                data={
                    "code": auth_code.strip(),
                    "client_id": self.CLIENT_ID,
                    "client_secret": self.CLIENT_SECRET,
                    "redirect_uri": f"http://localhost:{self.callback_port}{self.CALLBACK_PATH}",
                    "grant_type": "authorization_code",
                    "code_verifier": effective_verifier,
                },
            )
            response.raise_for_status()
            token_data = response.json()
            # Start with the full token data from the exchange
            new_creds = token_data.copy()

            # Convert 'expires_in' to 'expiry_date' in milliseconds
            new_creds["expiry_date"] = (
                time.time() + new_creds.pop("expires_in")
            ) * 1000

            # Ensure client_id and client_secret are present
            new_creds["client_id"] = self.CLIENT_ID
            new_creds["client_secret"] = self.CLIENT_SECRET

            new_creds["token_uri"] = self.TOKEN_URI
            new_creds["universe_domain"] = "googleapis.com"

            # Fetch user info and add metadata
            # Uses GEMINI_CLI style headers per PR 246
            user_info_response = await client.get(
                self.USER_INFO_URI,
                headers={
                    "Authorization": f"Bearer {new_creds['access_token']}",
                    "User-Agent": "google-api-nodejs-client/10.3.0",
                    "X-Goog-Api-Client": "gl-node/22.18.0",
                },
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()
            new_creds["_proxy_metadata"] = {
                "email": user_info.get("email"),
                "last_check_timestamp": time.time(),
            }

            if path:
                await self._save_credentials(path, new_creds)
            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth initialized successfully for '{display_name}'."
            )

            # Perform post-auth discovery (tier, project, etc.) while we have a fresh token
            if path:
                try:
                    await self._post_auth_discovery(path, new_creds["access_token"])
                except Exception as e:
                    # Don't fail auth if discovery fails - it can be retried on first request
                    lib_logger.warning(
                        f"Post-auth discovery failed for '{display_name}': {e}. "
                        "Tier/project will be discovered on first request."
                    )

        return new_creds

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token, triggering interactive OAuth flow if needed.

        If interactive OAuth is required (expired refresh token, missing credentials, etc.),
        the flow is coordinated globally via ReauthCoordinator to ensure only one
        interactive OAuth flow runs at a time across all providers.

        Args:
            creds_or_path: Either a credentials dict or path to credentials file.
            force_interactive: If True, skip expiry checks and force interactive OAuth.
                               Use this when the refresh token is known to be invalid
                               (e.g., after HTTP 400 from token endpoint).
        """
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
            if force_interactive:
                reason = (
                    "re-authentication was explicitly requested (refresh token invalid)"
                )
            elif not creds.get("refresh_token"):
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
                    provider_name=self.ENV_PREFIX,
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,  # 5 minute timeout for user to complete OAuth
                )

            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
            )
            return creds
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            )

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Get auth header with graceful degradation if refresh fails."""
        try:
            creds = await self._load_credentials(credential_path)
            if self._is_token_expired(creds):
                try:
                    creds = await self._refresh_token(credential_path, creds)
                except Exception as e:
                    # Check if we have a cached token that might still work
                    cached = self._credentials_cache.get(credential_path)
                    if cached and cached.get("access_token"):
                        lib_logger.warning(
                            f"Token refresh failed for {Path(credential_path).name}: {e}. "
                            "Using cached token (may be expired)."
                        )
                        creds = cached
                    else:
                        raise
            return {"Authorization": f"Bearer {creds['access_token']}"}
        except Exception as e:
            # Check if any cached credential exists as last resort
            cached = self._credentials_cache.get(credential_path)
            if cached and cached.get("access_token"):
                lib_logger.error(
                    f"Credential load failed for {credential_path}: {e}. "
                    "Using stale cached token as last resort."
                )
                return {"Authorization": f"Bearer {cached['access_token']}"}
            raise

    async def _post_auth_discovery(
        self, credential_path: str, access_token: str
    ) -> None:
        """
        Hook for subclasses to perform post-authentication discovery.

        Called after successful OAuth authentication (both initial and re-auth).
        Subclasses can override this to discover and cache tier/project information
        during the authentication flow rather than waiting for the first API request.

        Args:
            credential_path: Path to the credential file
            access_token: The newly obtained access token
        """
        # Default implementation does nothing - subclasses can override
        pass

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

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """
        Get the file name prefix for this provider's credential files.

        Override in subclasses if the prefix differs from ENV_PREFIX.
        Default: lowercase ENV_PREFIX with underscores (e.g., "gemini_cli")
        """
        return self.ENV_PREFIX.lower()

    def _get_oauth_base_dir(self) -> Path:
        """
        Get the base directory for OAuth credential files.

        Can be overridden to customize credential storage location.
        """
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_email(
        self, email: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Find an existing credential file for the given email.

        Args:
            email: Email address to search for
            base_dir: Directory to search in (defaults to oauth_creds)

        Returns:
            Path to existing credential file, or None if not found
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_email = creds.get("_proxy_metadata", {}).get("email")
                if existing_email == email:
                    return Path(cred_file)
            except (json.JSONDecodeError, IOError) as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
                continue

        return None

    def _get_next_credential_number(self, base_dir: Optional[Path] = None) -> int:
        """
        Get the next available credential number for new credential files.

        Args:
            base_dir: Directory to scan (defaults to oauth_creds)

        Returns:
            Next available credential number (1, 2, 3, etc.)
        """
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
        """
        Build a path for a new credential file.

        Args:
            base_dir: Directory for credential files (defaults to oauth_creds)
            number: Credential number (auto-determined if None)

        Returns:
            Path for the new credential file
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        filename = f"{prefix}_oauth_{number}.json"
        return base_dir / filename

    async def setup_credential(
        self, base_dir: Optional[Path] = None
    ) -> CredentialSetupResult:
        """
        Complete credential setup flow: OAuth -> save -> discovery.

        This is the main entry point for setting up new credentials.
        Handles the entire lifecycle:
        1. Perform OAuth authentication
        2. Get user info (email) for deduplication
        3. Find existing credential or create new file path
        4. Save credentials to file
        5. Perform post-auth discovery (tier/project for Google OAuth)

        Args:
            base_dir: Directory for credential files (defaults to oauth_creds)

        Returns:
            CredentialSetupResult with status and details
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        # Ensure directory exists
        base_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Perform OAuth authentication (returns credentials dict)
            temp_creds = {
                "_proxy_metadata": {"display_name": f"new {self.ENV_PREFIX} credential"}
            }
            new_creds = await self.initialize_token(temp_creds)

            # Step 2: Get user info for deduplication
            user_info = await self.get_user_info(new_creds)
            email = user_info.get("email")

            if not email:
                return CredentialSetupResult(
                    success=False, error="Could not retrieve email from OAuth response"
                )

            # Step 3: Check for existing credential with same email
            existing_path = self._find_existing_credential_by_email(email, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
                lib_logger.info(
                    f"Found existing credential for {email}, updating {file_path.name}"
                )
            else:
                file_path = self._build_credential_path(base_dir)
                lib_logger.info(
                    f"Creating new credential for {email} at {file_path.name}"
                )

            # Step 4: Save credentials to file
            await self._save_credentials(str(file_path), new_creds)

            # Step 5: Perform post-auth discovery (tier, project_id)
            # This is already called in _perform_interactive_oauth, but we call it again
            # in case credentials were loaded from existing token
            tier = None
            project_id = None
            try:
                await self._post_auth_discovery(
                    str(file_path), new_creds["access_token"]
                )
                # Reload credentials to get discovered metadata
                with open(file_path, "r") as f:
                    updated_creds = json.load(f)
                tier = updated_creds.get("_proxy_metadata", {}).get("tier")
                project_id = updated_creds.get("_proxy_metadata", {}).get("project_id")
                new_creds = updated_creds
            except Exception as e:
                lib_logger.warning(
                    f"Post-auth discovery failed: {e}. Tier/project will be discovered on first request."
                )

            return CredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email,
                tier=tier,
                project_id=project_id,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Credential setup failed: {e}")
            return CredentialSetupResult(success=False, error=str(e))

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """
        Generate .env file lines for a credential.

        Subclasses should override to include provider-specific fields
        (e.g., tier, project_id for Google OAuth providers).

        Args:
            creds: Credential dictionary loaded from JSON
            cred_number: Credential number (1, 2, 3, etc.)

        Returns:
            List of .env file lines
        """
        email = creds.get("_proxy_metadata", {}).get("email", "unknown")
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        lines = [
            f"# {self.ENV_PREFIX} Credential #{cred_number} for: {email}",
            f"# Exported from: {self._get_provider_file_prefix()}_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_SCOPE={creds.get('scope', '')}",
            f"{prefix}_TOKEN_TYPE={creds.get('token_type', 'Bearer')}",
            f"{prefix}_ID_TOKEN={creds.get('id_token', '')}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', 0)}",
            f"{prefix}_CLIENT_ID={creds.get('client_id', '')}",
            f"{prefix}_CLIENT_SECRET={creds.get('client_secret', '')}",
            f"{prefix}_TOKEN_URI={creds.get('token_uri', 'https://oauth2.googleapis.com/token')}",
            f"{prefix}_UNIVERSE_DOMAIN={creds.get('universe_domain', 'googleapis.com')}",
            f"{prefix}_EMAIL={email}",
        ]

        return lines

    def export_credential_to_env(
        self, credential_path: str, output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Export a credential file to .env format.

        Args:
            credential_path: Path to the credential JSON file
            output_dir: Directory for output .env file (defaults to same as credential)

        Returns:
            Path to the exported .env file, or None on error
        """
        try:
            cred_path = Path(credential_path)

            # Load credential
            with open(cred_path, "r") as f:
                creds = json.load(f)

            # Extract metadata
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")

            # Get credential number from filename
            match = re.search(r"_oauth_(\d+)\.json$", cred_path.name)
            cred_number = int(match.group(1)) if match else 1

            # Build output path
            if output_dir is None:
                output_dir = cred_path.parent

            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = self._get_provider_file_prefix()
            env_filename = f"{prefix}_{cred_number}_{safe_email}.env"
            env_path = output_dir / env_filename

            # Build and write content
            env_lines = self.build_env_lines(creds, cred_number)
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))

            lib_logger.info(f"Exported credential to {env_path}")
            return str(env_path)

        except Exception as e:
            lib_logger.error(f"Failed to export credential: {e}")
            return None

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all credential files for this provider.

        Args:
            base_dir: Directory to search (defaults to oauth_creds)

        Returns:
            List of dicts with credential info:
            - file_path: Path to credential file
            - email: User email
            - tier: Tier info (if available)
            - project_id: Project ID (if available)
            - number: Credential number
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

                # Extract number from filename
                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append(
                    {
                        "file_path": cred_file,
                        "email": metadata.get("email", "unknown"),
                        "tier": metadata.get("tier"),
                        "project_id": metadata.get("project_id"),
                        "number": number,
                    }
                )
            except Exception as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
                continue

        return credentials

    def delete_credential(self, credential_path: str) -> bool:
        """
        Delete a credential file.

        Args:
            credential_path: Path to the credential file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            cred_path = Path(credential_path)

            # Validate that it's one of our credential files
            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be a {self.ENV_PREFIX} credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            # Remove from cache if present
            self._credentials_cache.pop(credential_path, None)

            # Delete the file
            cred_path.unlink()
            lib_logger.info(f"Deleted credential file: {credential_path}")
            return True

        except Exception as e:
            lib_logger.error(f"Failed to delete credential: {e}")
            return False
