# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/anthropic_auth_base.py

import base64
import hashlib
import json
import os
import secrets
import time
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..utils.resilient_io import safe_write_json
from ..error_handler import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")

# Anthropic OAuth constants — matches Claude Code's flow exactly
ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_TOKEN_ENDPOINT = "https://console.anthropic.com/v1/oauth/token"
ANTHROPIC_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
ANTHROPIC_SCOPES = "org:create_api_key user:profile user:inference"

# Refresh 5 minutes before expiry
REFRESH_EXPIRY_BUFFER_SECONDS = 5 * 60

console = Console()


@dataclass
class AnthropicCredentialSetupResult:
    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _build_authorize_url(verifier: str, challenge: str) -> str:
    """Build the authorization URL for Claude Pro/Max OAuth."""
    params = {
        "code": "true",
        "client_id": ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTHROPIC_REDIRECT_URI,
        "scope": ANTHROPIC_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    from urllib.parse import urlencode
    return f"https://claude.ai/oauth/authorize?{urlencode(params)}"


class AnthropicAuthBase:
    """
    Anthropic OAuth authentication base class.
    Implements PKCE authorization code flow matching Claude Code's OAuth exactly.
    Uses manual code-paste (no local callback server).
    """

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}
        self._permanently_expired_credentials: set = set()

    # =========================================================================
    # CREDENTIAL LOADING / SAVING
    # =========================================================================

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        if not path.startswith("env://"):
            return None
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if credential_index and credential_index != "0":
            prefix = f"ANTHROPIC_OAUTH_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "ANTHROPIC_OAUTH"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not (access_token and refresh_token):
            return None

        lib_logger.debug(
            f"Loading Anthropic OAuth credentials from env vars (prefix: {prefix})"
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": os.getenv(f"{prefix}_EXPIRY_DATE", ""),
            "email": os.getenv(f"{prefix}_EMAIL", default_email),
            "token_type": "Bearer",
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
                "credential_type": "oauth",
            },
        }

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"Anthropic OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(
                f"Failed to load Anthropic OAuth credentials from '{path}': {e}"
            )

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Env vars for Anthropic credential index {credential_index} not found"
                    )

            try:
                return await self._read_creds_from_file(path)
            except IOError:
                env_creds = self._load_from_env()
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise

    async def _save_credentials(self, path: str, creds: Dict[str, Any]) -> bool:
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            self._credentials_cache[path] = creds
            return True

        if not safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=False
        ):
            lib_logger.error(
                f"Failed to write Anthropic credentials to disk for '{Path(path).name}'."
            )
            return False

        self._credentials_cache[path] = creds
        return True

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True
        try:
            from datetime import datetime

            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            expiry_timestamp = expiry_dt.timestamp()
        except (ValueError, AttributeError):
            try:
                expiry_timestamp = float(expiry_str)
            except (ValueError, TypeError):
                return True

        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _get_lock(self, path: str) -> asyncio.Lock:
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def _mark_credential_expired(self, path: str, reason: str) -> None:
        self._permanently_expired_credentials.add(path)

        display_name = path if path.startswith("env://") else Path(path).name

        console.print(
            Panel(
                f"[bold red]Credential:[/bold red] {display_name}\n"
                f"[bold red]Reason:[/bold red] {reason}\n\n"
                f"[yellow]This credential has been removed from rotation.[/yellow]\n"
                f"[yellow]To fix: Run 'python credential_tool.py' to re-authenticate,[/yellow]\n"
                f"[yellow]then restart the proxy.[/yellow]",
                title="[bold red]CREDENTIAL EXPIRED - REMOVED FROM ROTATION[/bold red]",
                border_style="red",
            )
        )
        lib_logger.error(
            f"CREDENTIAL EXPIRED | Credential: {display_name} | Reason: {reason}"
        )

    # =========================================================================
    # TOKEN EXCHANGE & REFRESH
    # =========================================================================

    async def _exchange_code(self, auth_code: str, verifier: str) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.
        The code from Anthropic is formatted as `code#state`.
        """
        splits = auth_code.split("#")
        code_part = splits[0]
        state_part = splits[1] if len(splits) > 1 else ""

        payload = {
            "code": code_part,
            "state": state_part,
            "grant_type": "authorization_code",
            "client_id": ANTHROPIC_CLIENT_ID,
            "redirect_uri": ANTHROPIC_REDIRECT_URI,
            "code_verifier": verifier,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ANTHROPIC_TOKEN_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            if not response.is_success:
                raise ValueError(
                    f"Token exchange failed: {response.status_code} {response.text}"
                )
            data = response.json()

        access_token = data.get("access_token")
        if not access_token:
            raise ValueError("Missing access_token in token response")

        refresh_token = data.get("refresh_token", "")
        expires_in = data.get("expires_in", 3600)

        from datetime import datetime, timedelta

        expiry_date = (
            datetime.utcnow() + timedelta(seconds=expires_in)
        ).isoformat() + "Z"

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "token_type": "Bearer",
        }

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            if not path.startswith("env://"):
                await self._read_creds_from_file(path)
            creds = self._credentials_cache[path]

            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token in Anthropic credentials.")

            lib_logger.debug(
                f"Refreshing Anthropic OAuth token for '{Path(path).name if not path.startswith('env://') else path}'..."
            )

            max_retries = 3
            new_token_data = None
            last_error = None

            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            ANTHROPIC_TOKEN_ENDPOINT,
                            headers={"Content-Type": "application/json"},
                            json={
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                                "client_id": ANTHROPIC_CLIENT_ID,
                            },
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code

                        if status_code in (400, 401, 403):
                            error_body = e.response.text
                            self._mark_credential_expired(
                                path,
                                f"Refresh token invalid (HTTP {status_code}: {error_body})",
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Anthropic refresh token invalid. Credential removed from rotation.",
                            )

                        if status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        if 500 <= status_code < 600:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            raise

                        raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise

            if new_token_data is None:
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                raise last_error or Exception(
                    "Anthropic token refresh failed after all retries"
                )

            access_token = new_token_data.get("access_token")
            if not access_token:
                raise ValueError("Missing access_token in Anthropic refresh response")

            creds["access_token"] = access_token
            creds["refresh_token"] = new_token_data.get(
                "refresh_token", creds["refresh_token"]
            )

            expires_in = new_token_data.get("expires_in", 3600)
            from datetime import datetime, timedelta

            creds["expiry_date"] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat() + "Z"

            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}
            creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            if not await self._save_credentials(path, creds):
                raise IOError(
                    f"Failed to persist refreshed Anthropic credentials for '{Path(path).name}'."
                )

            lib_logger.debug("Successfully refreshed Anthropic OAuth token.")
            return self._credentials_cache[path]

    # =========================================================================
    # PUBLIC INTERFACE (called by ProviderInterface / executor)
    # =========================================================================

    async def get_access_token(self, credential_identifier: str) -> str:
        """Get a valid access token, refreshing if needed."""
        if os.path.isfile(credential_identifier) or credential_identifier.startswith(
            "env://"
        ):
            creds = await self._load_credentials(credential_identifier)
            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)
            return creds["access_token"]
        return credential_identifier

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Returns Bearer auth header with the OAuth access token."""
        token = await self.get_access_token(credential_identifier)
        return {"Authorization": f"Bearer {token}"}

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh tokens if close to expiry."""
        try:
            creds = await self._load_credentials(credential_path)
        except IOError:
            return

        if self._is_token_expired(creds):
            try:
                await self._refresh_token(credential_path)
            except Exception as e:
                lib_logger.warning(f"Proactive Anthropic token refresh failed: {e}")

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token — load from disk and refresh if expired.
        Compatible with the proxy's startup credential processing flow.
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Anthropic token for '{display_name}'...")

        creds = await self._load_credentials(path or "")
        if self._is_token_expired(creds):
            creds = await self._refresh_token(path or "")

        lib_logger.info(f"Anthropic credential initialized: {display_name}")
        return creds

    async def initialize_credentials(self, credential_paths):
        """Initialize all credentials at startup."""
        for path in credential_paths:
            try:
                await self.initialize_token(path)
            except Exception as e:
                lib_logger.error(
                    f"Failed to initialize Anthropic credential {path}: {e}"
                )

    # =========================================================================
    # INTERACTIVE SETUP
    # =========================================================================

    async def setup_credential(self, base_dir: str) -> AnthropicCredentialSetupResult:
        """
        Interactive OAuth setup: prints URL, user pastes code.
        """
        from ..utils.paths import get_oauth_dir

        oauth_dir = Path(base_dir) if base_dir else get_oauth_dir()
        oauth_dir.mkdir(parents=True, exist_ok=True)

        verifier, challenge = _generate_pkce()
        auth_url = _build_authorize_url(verifier, challenge)

        console.print()
        console.print(
            Panel(
                "[bold cyan]Anthropic OAuth Setup (Claude Pro/Max)[/bold cyan]\n\n"
                "1. Open the URL below in your browser\n"
                "2. Authorize the application\n"
                "3. Copy the authorization code shown\n"
                "4. Paste it here\n\n"
                f"[link={auth_url}]{auth_url}[/link]",
                title="[bold]Anthropic OAuth[/bold]",
                border_style="cyan",
            )
        )

        try:
            import webbrowser

            webbrowser.open(auth_url)
            console.print("[dim]Browser opened automatically.[/dim]")
        except Exception:
            console.print(
                "[dim]Could not open browser. Please copy the URL above.[/dim]"
            )

        auth_code = Prompt.ask("\nPaste authorization code")
        if not auth_code or not auth_code.strip():
            return AnthropicCredentialSetupResult(
                success=False, error="No authorization code provided"
            )

        try:
            tokens = await self._exchange_code(auth_code.strip(), verifier)
        except Exception as e:
            return AnthropicCredentialSetupResult(
                success=False, error=f"Token exchange failed: {e}"
            )

        creds = {
            **tokens,
            "email": "anthropic-oauth-user",
            "_proxy_metadata": {
                "email": "anthropic-oauth-user",
                "last_check_timestamp": time.time(),
                "credential_type": "oauth",
            },
        }

        # Find next available file number
        existing = sorted(oauth_dir.glob("anthropic_oauth_*.json"))
        next_num = len(existing) + 1

        # Check for duplicate by access token prefix
        is_update = False
        file_path = None
        new_prefix = tokens["access_token"][:20]
        for existing_file in existing:
            try:
                with open(existing_file) as f:
                    existing_creds = json.load(f)
                if existing_creds.get("access_token", "")[:20] == new_prefix:
                    file_path = str(existing_file)
                    is_update = True
                    break
            except Exception:
                continue

        if not file_path:
            file_path = str(oauth_dir / f"anthropic_oauth_{next_num}.json")

        if not safe_write_json(file_path, creds, lib_logger, secure_permissions=True):
            return AnthropicCredentialSetupResult(
                success=False, error="Failed to save credentials"
            )

        action = "Updated" if is_update else "Created"
        console.print(f"\n[green]{action} credential at {Path(file_path).name}[/green]")

        return AnthropicCredentialSetupResult(
            success=True,
            file_path=file_path,
            email="anthropic-oauth-user",
            is_update=is_update,
            credentials=creds,
        )
