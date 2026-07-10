# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Player2 authentication helpers.

Player2 (https://player2.game) issues a single bearer token ("p2Key") that is
used to call its OpenAI-compatible `/chat/completions` endpoint. Unlike most
OAuth providers in this project, there is no separate access/refresh token
pair to manage: once a p2Key is obtained it is used exactly like a normal API
key (see PLAYER2_API_KEY / PLAYER2_API_KEY_N in the .env file).

This module only handles *obtaining* that key interactively, via two paths:

1. Local app detection (fast path): if the user already has the Player2
   desktop app open and logged in, we can get a key instantly by hitting the
   app's local HTTP server.
2. Device Authorization Flow (fallback): works anywhere (headless servers,
   Docker, CI), no local app or browser redirect required. The user is given
   a short code / URL to open on any device to approve access, while we poll
   for the resulting key.

Both paths require a Player2 "Game Client ID", obtained from the Player2
Developer Dashboard (https://player2.game) by registering an app/game. This
project treats that ID as a normal configuration value: `PLAYER2_CLIENT_ID`.

Reference: https://player2.game/api/api.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import httpx

lib_logger = logging.getLogger("rotator_library")

# Base URL for the Player2 desktop app's local HTTP server.
PLAYER2_LOCAL_APP_BASE = "http://127.0.0.1:4315/v1"

# Base URL for the hosted Player2 Web API.
PLAYER2_API_BASE = "https://api.player2.game/v1"

# Game Client ID used to authenticate against Player2's Web API.
#
# ⚠️ PROVISIONAL: this ID was created under a personal/contributor Player2
# account for development purposes, purely so this PR could be tested
# end-to-end. Before merging, the maintainer of LLM-API-Key-Proxy should
# register their own "game" in the Player2 Developer Dashboard
# (https://player2.game) and replace this value with that Client ID, so the
# project (not a contributor's personal account) owns and controls it going
# forward. Forks/self-hosters can also override it via the PLAYER2_CLIENT_ID
# environment variable without touching this file.
DEFAULT_CLIENT_ID = "019f426e-760b-7ba6-9009-fe85034c9057"

DEFAULT_LOCAL_APP_TIMEOUT = 2.0
DEFAULT_POLL_TIMEOUT = 300.0  # 5 minutes, generous upper bound
MIN_POLL_INTERVAL = 2.0


class Player2AuthError(Exception):
    """Raised when a Player2 login flow cannot complete."""


async def detect_local_app(
    client_id: str, timeout: float = DEFAULT_LOCAL_APP_TIMEOUT
) -> Optional[str]:
    """
    Attempts to obtain a p2Key from a Player2 desktop app running on the same
    machine. This is the fast path: no browser, no waiting, no extra clicks,
    as long as the user already has the app open and is logged in.

    Args:
        client_id: The Game Client ID from the Player2 Developer Dashboard.
        timeout: How long to wait for the local app to respond before giving
            up and falling back to the device flow. Kept short since a
            missing app should fail fast (connection refused), not hang.

    Returns:
        The p2Key string if the app is running and logged in, otherwise None.
        This function never raises for "app not running" - that is treated
        as a normal, expected outcome, not an error.
    """
    url = f"{PLAYER2_LOCAL_APP_BASE}/login/web/{client_id}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url)
            if response.status_code == 200:
                data = response.json()
                p2_key = data.get("p2Key")
                if p2_key:
                    lib_logger.info("Obtained Player2 key via local desktop app.")
                    return p2_key
            else:
                lib_logger.debug(
                    f"Player2 local app responded with {response.status_code}, "
                    "falling back to device flow."
                )
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
        lib_logger.debug(f"Player2 desktop app not detected locally: {e}")
    except Exception as e:
        lib_logger.debug(f"Unexpected error probing Player2 local app: {e}")
    return None


async def device_code_login(
    client_id: str,
    on_verification: Callable[[str, str], None],
    poll_timeout: float = DEFAULT_POLL_TIMEOUT,
) -> str:
    """
    Runs the OAuth 2.0 Device Authorization flow against the Player2 API.

    Args:
        client_id: The Game Client ID from the Player2 Developer Dashboard.
        on_verification: Callback invoked once as
            `on_verification(verification_uri, user_code)`, so the caller can
            display the URL/code and optionally open a browser. Called
            exactly once, before polling begins.
        poll_timeout: Hard upper bound (seconds) on how long to poll, in
            addition to whatever `expiresIn` the API returns.

    Returns:
        The p2Key string once the user approves access.

    Raises:
        Player2AuthError: if the flow times out, is denied, or the API
            returns an unexpected/terminal error.

    Note:
        Player2's documented responses for `/login/device/token` only list
        200 (success) and 500 (server error) - the exact error format used
        while waiting for approval isn't spelled out in the public API
        reference. This implementation assumes the standard RFC 8628 error
        body (`{"error": "authorization_pending"}`, `"slow_down"`,
        `"expired_token"`, `"access_denied"`, etc.) on non-200/500 responses:
        only "authorization_pending" and "slow_down" are treated as
        "keep waiting"; anything else fails immediately with Player2's own
        error message instead of silently polling until timeout.
    """
    # RFC 8628 §3.5 - errors that mean "keep polling", not "give up".
    PENDING_ERRORS = {"authorization_pending"}
    SLOW_DOWN_ERRORS = {"slow_down"}

    async with httpx.AsyncClient(timeout=15.0, base_url=PLAYER2_API_BASE) as client:
        new_resp = await client.post("/login/device/new", json={"client_id": client_id})
        new_resp.raise_for_status()
        data = new_resp.json()

        device_code = data["deviceCode"]
        user_code = data["userCode"]
        verification_uri = data.get("verificationUriComplete") or data.get("verificationUri")
        if not verification_uri:
            raise Player2AuthError(
                "Player2 did not return a verification URI for the device login."
            )
        interval = max(float(data.get("interval", 5)), MIN_POLL_INTERVAL)
        expires_in = float(data.get("expiresIn", poll_timeout))

        on_verification(verification_uri, user_code)

        deadline = time.monotonic() + min(expires_in, poll_timeout)

        while time.monotonic() < deadline:
            await asyncio.sleep(interval)

            token_resp = await client.post(
                "/login/device/token",
                json={
                    "client_id": client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )

            if token_resp.status_code == 200:
                p2_key = token_resp.json().get("p2Key")
                if p2_key:
                    lib_logger.info("Obtained Player2 key via device authorization flow.")
                    return p2_key
                # A 200 with no key is not a valid "success" response - fail
                # rather than silently keep polling on something unexpected.
                raise Player2AuthError(
                    "Player2 returned a successful response with no p2Key."
                )

            if token_resp.status_code == 500:
                raise Player2AuthError(
                    f"Player2 device login failed with a server error: {token_resp.text}"
                )

            # Try to read a standard RFC 8628 error body to tell "still
            # waiting" apart from a real, terminal failure.
            try:
                error_code = token_resp.json().get("error")
            except (json.JSONDecodeError, ValueError):
                error_code = None

            if error_code in PENDING_ERRORS:
                lib_logger.debug("Player2 device login still pending.")
                continue

            if error_code in SLOW_DOWN_ERRORS:
                interval += 5.0
                lib_logger.debug(
                    f"Player2 asked us to slow down; polling interval is now {interval}s."
                )
                continue

            # Anything else (access_denied, expired_token, invalid client_id,
            # or an unrecognized/undocumented error) is a real, terminal
            # failure - fail fast instead of polling until timeout.
            raise Player2AuthError(
                f"Player2 device login failed (status {token_resp.status_code}): "
                f"{token_resp.text}"
            )

        raise Player2AuthError(
            "Timed out waiting for Player2 login approval. Please try again."
        )

async def get_p2_key(
    client_id: str,
    on_verification: Callable[[str, str], None],
    local_app_timeout: float = DEFAULT_LOCAL_APP_TIMEOUT,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT,
) -> str:
    """
    Convenience wrapper: tries the local desktop app first, then falls back
    to the device authorization flow.

    Args:
        client_id: The Game Client ID from the Player2 Developer Dashboard.
        on_verification: Called only if the fallback device flow is used.
        local_app_timeout: Timeout for the local app probe.
        poll_timeout: Timeout for the device flow poll loop.

    Returns:
        The p2Key string.

    Raises:
        Player2AuthError: if both paths fail.
    """
    p2_key = await detect_local_app(client_id, timeout=local_app_timeout)
    if p2_key:
        return p2_key

    return await device_code_login(
        client_id, on_verification=on_verification, poll_timeout=poll_timeout
    )