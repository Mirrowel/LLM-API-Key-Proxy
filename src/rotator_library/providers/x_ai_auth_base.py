# SPDX-License-Identifier: LGPL-3.0-only

# src/rotator_library/providers/xai_auth_base.py
"""
xAI Grok OAuth Base Class

Base class for xAI OAuth2 authentication (SuperGrok / X Premium+ subscribers).
Supports two flows:
  1. PKCE Authorization Code Flow (loopback redirect on 127.0.0.1:56121/callback)
  2. Device Code Flow (headless environments)

OAuth Configuration (from https://auth.x.ai/.well-known/openid-configuration):
- Client ID: b1a00492-073a-47ea-816f-4c329264a828 (pre-registered public client)
- Authorization URL: https://auth.x.ai/oauth2/authorize
- Token URL: https://auth.x.ai/oauth2/token
- Device Code URL: https://auth.x.ai/oauth2/device/code
- Redirect URI: http://127.0.0.1:56121/callback
- Scopes: openid email profile offline_access api:access grok-cli:access
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from .openai_oauth_base import (
    OpenAIOAuthBase,
    _generate_pkce,
    _parse_jwt_claims,
)

lib_logger = logging.getLogger("rotator_library")
console = Console()

# =============================================================================
# XAI OAUTH CONFIGURATION
# =============================================================================

XAI_AUTH_URL = "https://auth.x.ai/oauth2/authorize"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"

# Pre-registered public client — no client_secret required.
XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"

XAI_OAUTH_SCOPES = ["openid", "email", "profile", "offline_access", "api:access", "grok-cli:access"]

# Userinfo endpoint for fallback email discovery
XAI_USERINFO_URL = "https://auth.x.ai/oauth2/userinfo"

# Loopback redirect callback
XAI_CALLBACK_PORT = 56121
XAI_CALLBACK_PATH = "/callback"


class XAiAuthBase(OpenAIOAuthBase):
    """
    xAI Grok OAuth2 authentication base class.

    Inherits the PKCE flow, token refresh, credential loading/saving, and
    background refresh queue from OpenAIOAuthBase.  Overrides the interactive
    flow to offer Device Code authorization as an alternative for headless
    environments.
    """

    # Override OpenAIOAuthBase class-level constants
    CLIENT_ID: str = XAI_CLIENT_ID
    OAUTH_SCOPES: List[str] = XAI_OAUTH_SCOPES
    ENV_PREFIX: str = "X_AI_OAUTH"

    AUTH_URL: str = XAI_AUTH_URL
    TOKEN_URL: str = XAI_TOKEN_URL
    CALLBACK_PORT: int = XAI_CALLBACK_PORT
    CALLBACK_PATH: str = XAI_CALLBACK_PATH

    # xAI tokens typically last 1 hour; refresh 5 minutes before expiry
    REFRESH_EXPIRY_BUFFER_SECONDS: int = 5 * 60

    # =================================================================
    # DEVICE CODE FLOW
    # =================================================================

    async def _perform_device_code_flow(
        self, path: Optional[str], creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform xAI Device Code authorization flow.

        Steps:
        1. POST to /oauth2/device/code to get user_code and verification_uri
        2. Display the code and URI for the user
        3. Poll /oauth2/token until the user completes authorization
        """
        proxy_kwargs = self._build_proxy_client_kwargs(path) if path else {}
        async with httpx.AsyncClient(**proxy_kwargs) as client:
            # Step 1: Request device code
            device_response = await client.post(
                XAI_DEVICE_CODE_URL,
                data={
                    "client_id": self.CLIENT_ID,
                    "scope": " ".join(self.OAUTH_SCOPES),
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )

            if not device_response.is_success:
                raise Exception(
                    f"Failed to initiate xAI device authorization: "
                    f"{device_response.text}"
                )

            device_data = device_response.json()
            user_code = device_data.get("user_code", "")
            verification_uri = device_data.get("verification_uri", "")
            verification_uri_complete = device_data.get(
                "verification_uri_complete", verification_uri
            )
            device_code = device_data.get("device_code", "")
            interval = device_data.get("interval", 5)
            expires_in = device_data.get("expires_in", 600)

            # Step 2: Display instructions
            is_headless = is_headless_environment()

            if is_headless:
                console.print(
                    Panel(
                        Text.from_markup(
                            "Running in headless environment (no GUI detected).\n"
                            "Open the URL below in a browser on another machine.\n"
                        ),
                        title=f"xAI Device Authorization for [bold yellow]{display_name}[/bold yellow]",
                        style="bold blue",
                    )
                )
            else:
                console.print(
                    Panel(
                        Text.from_markup(
                            "Please visit the URL below and enter the code to authorize.\n"
                        ),
                        title=f"xAI Device Authorization for [bold yellow]{display_name}[/bold yellow]",
                        style="bold blue",
                    )
                )

            console.print(f"  [bold]URL:[/bold]  {rich_escape(verification_uri_complete)}")
            console.print(f"  [bold]Code:[/bold] [bold green]{user_code}[/bold green]\n")

            if not is_headless:
                try:
                    import webbrowser
                    webbrowser.open(verification_uri_complete)
                    lib_logger.info("Browser opened for xAI device authorization")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. "
                        "Please open the URL manually."
                    )

            # Step 3: Poll for authorization
            max_polls = expires_in // interval
            with console.status(
                "[bold green]Waiting for you to authorize in the browser...[/bold green]",
                spinner="dots",
            ):
                for _ in range(max_polls):
                    await asyncio.sleep(interval)

                    token_response = await client.post(
                        self.TOKEN_URL,
                        data={
                            "client_id": self.CLIENT_ID,
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        },
                        headers={
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Accept": "application/json",
                        },
                        timeout=30.0,
                    )

                    if not token_response.is_success:
                        # Check for expected polling responses
                        try:
                            err_data = token_response.json()
                            error = err_data.get("error", "")
                        except Exception:
                            error = ""

                        if error == "authorization_pending":
                            continue
                        elif error == "slow_down":
                            interval = min(interval + 5, 30)
                            continue
                        elif error == "expired_token":
                            raise Exception(
                                "Device authorization expired. Please try again."
                            )
                        elif error == "access_denied":
                            raise Exception(
                                "Authorization was denied by the user."
                            )
                        else:
                            # Unexpected error — keep polling
                            continue

                    token_data = token_response.json()

                    if "access_token" in token_data:
                        # Success!
                        return await self._build_credentials_from_token_data(
                            token_data, path, display_name
                        )

            raise Exception(
                "Device authorization timed out. Please try again."
            )

    # =================================================================
    # CREDENTIAL BUILDER (shared between flows)
    # =================================================================

    async def _build_credentials_from_token_data(
        self,
        token_data: Dict[str, Any],
        path: Optional[str],
        display_name: str,
    ) -> Dict[str, Any]:
        """
        Build a credential dict from a successful token response.
        Shared between PKCE and Device Code flows.

        Email discovery order:
        1. Parse the ID token JWT for an 'email' claim
        2. Call the userinfo endpoint with the access token
        3. Fall back to 'sub' (subject identifier) so setup_credential() doesn't reject
        """
        access_token = token_data.get("access_token", "")

        new_creds: Dict[str, Any] = {
            "access_token": access_token,
            "refresh_token": token_data.get("refresh_token"),
            "id_token": token_data.get("id_token"),
            "expiry_date": time.time() + token_data.get("expires_in", 3600),
        }

        # Parse ID token for user info
        id_token_claims = _parse_jwt_claims(
            token_data.get("id_token", "")
        ) or {}

        email = id_token_claims.get("email", "")
        name = id_token_claims.get("name", "")
        sub = id_token_claims.get("sub", "")

        # Fallback: fetch from userinfo endpoint if ID token lacks email
        if not email and access_token:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        XAI_USERINFO_URL,
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=10.0,
                    )
                    if resp.is_success:
                        userinfo = resp.json()
                        email = userinfo.get("email", "")
                        name = name or userinfo.get("name", "")
                        sub = sub or userinfo.get("sub", "")
                        lib_logger.info(
                            f"xAI userinfo resolved email: {email or '(none)'}"
                        )
                    else:
                        lib_logger.warning(
                            f"xAI userinfo request failed ({resp.status_code}): {resp.text[:200]}"
                        )
            except Exception as e:
                lib_logger.warning(f"xAI userinfo fallback failed: {e}")

        # Last resort: use sub as the identifier if email is still empty.
        # The parent setup_credential() requires a non-empty email field.
        if not email:
            email = sub or f"xai-user-{int(time.time())}"
            lib_logger.warning(
                f"xAI OAuth: no email found in ID token or userinfo. "
                f"Using identifier: {email}"
            )

        # Use sub (subject) as account_id for xAI
        account_id = sub or email

        new_creds["account_id"] = account_id
        new_creds["_proxy_metadata"] = {
            "email": email,
            "name": name,
            "account_id": account_id,
            "last_check_timestamp": time.time(),
        }

        lib_logger.info(
            f"xAI OAuth initialized successfully for '{display_name}' "
            f"(email: {email}, name: {name or 'unknown'})."
        )

        return new_creds

    # =================================================================
    # OVERRIDE: Interactive OAuth to support Device Code
    # =================================================================

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth flow for xAI.

        In headless environments, uses Device Code flow automatically.
        In interactive environments, offers the user a choice between
        browser-based PKCE and Device Code flows.
        """
        is_headless = is_headless_environment()

        if is_headless:
            # Headless: always use Device Code flow
            lib_logger.info(
                "Headless environment detected — using xAI Device Code flow."
            )
            new_creds = await self._perform_device_code_flow(
                path, creds, display_name
            )
        else:
            # Interactive: offer choice
            console.print(
                Panel(
                    Text.from_markup(
                        "Choose an authentication method:\n\n"
                        "  [bold cyan]1[/bold cyan] — Browser login (PKCE — recommended)\n"
                        "  [bold cyan]2[/bold cyan] — Device code (paste a code in your browser)\n"
                    ),
                    title="xAI Authentication",
                    style="bold blue",
                )
            )

            try:
                choice = input("Enter choice [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                choice = "1"

            if choice == "2":
                new_creds = await self._perform_device_code_flow(
                    path, creds, display_name
                )
            else:
                # Use the parent class PKCE flow
                new_creds = await self._perform_pkce_flow(
                    path, creds, display_name
                )

        if path:
            await self._save_credentials(path, new_creds)

        return new_creds

    async def _perform_pkce_flow(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform PKCE Authorization Code flow via loopback redirect.

        Uses the same mechanism as OpenAIOAuthBase._perform_interactive_oauth
        but with xAI-specific endpoints and without the OpenAI-specific
        auth params (codex_cli_simplified_flow, id_token_add_organizations).
        """
        is_headless = is_headless_environment()

        # Generate PKCE codes
        code_verifier, code_challenge = _generate_pkce()
        state = secrets.token_hex(32)

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
                    received_state = query_params.get("state", [None])[0]
                    if received_state != state:
                        if not auth_code_future.done():
                            auth_code_future.set_exception(
                                Exception("OAuth state mismatch")
                            )
                        writer.write(
                            b"<html><body><h1>State Mismatch</h1>"
                            b"<p>Security error. Please try again.</p></body></html>"
                        )
                    elif not auth_code_future.done():
                        auth_code_future.set_result(query_params["code"][0])
                        writer.write(
                            b"<html><body><h1>Authentication successful!</h1>"
                            b"<p>You can close this window.</p></body></html>"
                        )
                else:
                    error = query_params.get("error", ["Unknown error"])[0]
                    if not auth_code_future.done():
                        auth_code_future.set_exception(
                            Exception(f"OAuth failed: {error}")
                        )
                    writer.write(
                        f"<html><body><h1>Authentication Failed</h1>"
                        f"<p>Error: {error}</p></body></html>".encode()
                    )

                await writer.drain()
            except Exception as e:
                lib_logger.error(f"Error in xAI OAuth callback handler: {e}")
            finally:
                writer.close()

        try:
            server = await asyncio.start_server(
                handle_callback, "127.0.0.1", self.callback_port
            )

            from urllib.parse import urlencode

            redirect_uri = (
                f"http://127.0.0.1:{self.callback_port}{self.CALLBACK_PATH}"
            )

            auth_params = {
                "response_type": "code",
                "client_id": self.CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": " ".join(self.OAUTH_SCOPES),
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "state": state,
            }

            auth_url = f"{self.AUTH_URL}?" + urlencode(auth_params)

            if is_headless:
                auth_panel_text = Text.from_markup(
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:\n"
                )
            else:
                auth_panel_text = Text.from_markup(
                    "1. Your browser will now open to log in and authorize.\n"
                    "2. If it doesn't open automatically, please open the URL below manually."
                )

            console.print(
                Panel(
                    auth_panel_text,
                    title=f"xAI OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                    style="bold blue",
                )
            )

            escaped_url = rich_escape(auth_url)
            console.print(
                f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n"
            )

            if not is_headless:
                try:
                    import webbrowser
                    webbrowser.open(auth_url)
                    lib_logger.info(
                        "Browser opened successfully for xAI OAuth flow"
                    )
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. "
                        "Please open the URL manually."
                    )

            with console.status(
                "[bold green]Waiting for you to complete authentication in the browser...[/bold green]",
                spinner="dots",
            ):
                auth_code = await asyncio.wait_for(auth_code_future, timeout=310)

        except asyncio.TimeoutError:
            raise Exception("xAI OAuth flow timed out. Please try again.")
        finally:
            if server:
                server.close()
                await server.wait_closed()

        lib_logger.info("Exchanging xAI authorization code for tokens...")

        # Exchange authorization code for tokens
        proxy_kwargs = self._build_proxy_client_kwargs(path) if path else {}
        async with httpx.AsyncClient(**proxy_kwargs) as client:
            redirect_uri = (
                f"http://127.0.0.1:{self.callback_port}{self.CALLBACK_PATH}"
            )

            response = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code.strip(),
                    "client_id": self.CLIENT_ID,
                    "code_verifier": code_verifier,
                    "redirect_uri": redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()

            new_creds = await self._build_credentials_from_token_data(
                token_data, path, display_name
            )

            return new_creds

    # =================================================================
    # OVERRIDE: get_auth_header (simplified — no API key exchange)
    # =================================================================

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Get auth header for xAI API calls.

        xAI does not support API key exchange; always uses access_token.
        """
        try:
            creds = await self._load_credentials(credential_path)

            if self._is_token_expired(creds):
                try:
                    creds = await self._refresh_token(credential_path, creds)
                except Exception as e:
                    cached = self._credentials_cache.get(credential_path)
                    if cached and cached.get("access_token"):
                        lib_logger.warning(
                            f"Token refresh failed for xAI credential: {e}. "
                            "Using cached token."
                        )
                        self._record_refresh_error(
                            credential_path,
                            "TokenRefreshFailed",
                            f"Token refresh failed: {e}",
                            status_code=getattr(e, "status_code", None),
                        )
                        creds = cached
                    else:
                        raise

            token = creds.get("access_token")
            if not token:
                raise ValueError(
                    "No access_token found in xAI credentials"
                )
            return {"Authorization": f"Bearer {token}"}

        except Exception as e:
            cached = self._credentials_cache.get(credential_path)
            if cached and cached.get("access_token"):
                lib_logger.error(
                    f"Credential load failed for xAI {credential_path}: {e}. "
                    "Using stale cached token."
                )
                return {"Authorization": f"Bearer {cached['access_token']}"}
            raise
