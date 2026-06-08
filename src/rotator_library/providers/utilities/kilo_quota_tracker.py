"""
Kilo (KiloCode) Quota Tracking Utility

Fetches credit balance from the Kilo web dashboard API using a NextAuth
session token.  The session token is obtained from the browser cookie
`__Secure-next-auth.session-token` and passed via the environment variable
KILO_SESSION_TOKEN.

API endpoint: GET https://app.kilo.ai/api/user
Relevant fields:
    microdollars_used           – cumulative spend in microdollars (1e-6 USD)
    total_microdollars_acquired – cumulative credits in microdollars
    next_credit_expiration_at   – ISO timestamp of the next credit expiry

Session refresh: GET https://app.kilo.ai/api/auth/session returns a fresh
session token in the Set-Cookie header.  The token has a ~30-day TTL and is
refreshed on each call to /api/auth/session.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage import UsageManager

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

KILO_USER_URL = "https://app.kilo.ai/api/user"
KILO_SESSION_URL = "https://app.kilo.ai/api/auth/session"
SESSION_COOKIE_NAME = "__Secure-next-auth.session-token"

# Default refresh interval (10 minutes — Kilo is a paid balance, not a
# fast-moving rate-limit, so we don't need to poll aggressively)
DEFAULT_QUOTA_REFRESH_INTERVAL = 600

# Stale after 20 minutes
QUOTA_STALE_THRESHOLD_SECONDS = 1200


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class KiloQuotaSnapshot:
    """Credit balance snapshot for a Kilo account."""

    email: str = ""
    username: str = ""

    microdollars_used: int = 0
    total_microdollars_acquired: int = 0
    next_credit_expiration_at: Optional[str] = None

    fetched_at: float = field(default_factory=time.time)
    status: str = "success"
    error: Optional[str] = None

    # If the session was refreshed during this fetch, the new token value
    refreshed_token: Optional[str] = None

    @property
    def remaining_microdollars(self) -> int:
        return max(0, self.total_microdollars_acquired - self.microdollars_used)

    @property
    def remaining_dollars(self) -> float:
        return self.remaining_microdollars / 1_000_000

    @property
    def remaining_cents(self) -> int:
        """Balance in whole cents — used as the quota unit in UsageManager."""
        return self.remaining_microdollars // 10_000

    @property
    def total_cents(self) -> int:
        return self.total_microdollars_acquired // 10_000

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS

    @property
    def expiration_ts(self) -> Optional[float]:
        """Parse next_credit_expiration_at to a Unix timestamp."""
        if not self.next_credit_expiration_at:
            return None
        try:
            dt = datetime.fromisoformat(
                self.next_credit_expiration_at.replace("Z", "+00:00")
            )
            return dt.timestamp()
        except (ValueError, TypeError):
            return None


# =============================================================================
# TRACKER
# =============================================================================


class KiloQuotaTracker:
    """
    Fetches Kilo credit balance via the web API and pushes it to
    UsageManager so it appears in the TUI quota display.

    This is a standalone utility (not a mixin) — the KiloProvider
    delegates to an instance of this class.
    """

    def __init__(self, session_token: str, refresh_interval: int | None = None):
        self._session_token = session_token
        self._refresh_interval = refresh_interval or DEFAULT_QUOTA_REFRESH_INTERVAL
        self._snapshot: Optional[KiloQuotaSnapshot] = None

    @property
    def session_token(self) -> str:
        return self._session_token

    # -----------------------------------------------------------------
    # API FETCH
    # -----------------------------------------------------------------

    async def fetch_balance(self) -> KiloQuotaSnapshot:
        """Fetch credit balance from the Kilo /api/user endpoint."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    KILO_USER_URL,
                    cookies={SESSION_COOKIE_NAME: self._session_token},
                )

            if response.status_code == 401:
                snap = KiloQuotaSnapshot(
                    status="error",
                    error="session_expired",
                )
                self._snapshot = snap
                return snap

            response.raise_for_status()
            data = response.json()

            snap = KiloQuotaSnapshot(
                email=data.get("google_user_email", ""),
                username=data.get("google_user_name", ""),
                microdollars_used=data.get("microdollars_used", 0),
                total_microdollars_acquired=data.get(
                    "total_microdollars_acquired", 0
                ),
                next_credit_expiration_at=data.get("next_credit_expiration_at"),
            )
            self._snapshot = snap

            lib_logger.debug(
                f"Kilo balance for {snap.email}: "
                f"${snap.remaining_dollars:.2f} remaining "
                f"(${snap.total_microdollars_acquired / 1_000_000:.2f} total, "
                f"${snap.microdollars_used / 1_000_000:.2f} used)"
            )
            return snap

        except httpx.HTTPStatusError as exc:
            error_msg = f"HTTP {exc.response.status_code}"
            lib_logger.warning(f"Kilo balance fetch failed: {error_msg}")
            snap = KiloQuotaSnapshot(status="error", error=error_msg)
            self._snapshot = snap
            return snap

        except Exception as exc:
            lib_logger.warning(f"Kilo balance fetch failed: {exc}")
            snap = KiloQuotaSnapshot(status="error", error=str(exc))
            self._snapshot = snap
            return snap

    async def refresh_session_token(self) -> Optional[str]:
        """
        Hit /api/auth/session to get a fresh session token from Set-Cookie.

        Returns the new token string, or None if refresh failed.
        """
        try:
            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=False
            ) as client:
                response = await client.get(
                    KILO_SESSION_URL,
                    cookies={SESSION_COOKIE_NAME: self._session_token},
                )

            # Extract refreshed token from Set-Cookie headers
            for cookie_header in response.headers.get_list("set-cookie"):
                if SESSION_COOKIE_NAME in cookie_header:
                    # Parse "name=value; Path=...; ..."
                    parts = cookie_header.split(";")
                    for part in parts:
                        part = part.strip()
                        if part.startswith(f"{SESSION_COOKIE_NAME}="):
                            new_token = part[len(SESSION_COOKIE_NAME) + 1:]
                            self._session_token = new_token
                            lib_logger.info(
                                "Kilo session token refreshed successfully"
                            )
                            return new_token

        except Exception as exc:
            lib_logger.warning(f"Kilo session refresh failed: {exc}")

        return None

    # -----------------------------------------------------------------
    # USAGE MANAGER INTEGRATION
    # -----------------------------------------------------------------

    async def push_to_usage_manager(
        self,
        usage_manager: "UsageManager",
        credential_key: str,
        snapshot: KiloQuotaSnapshot,
    ) -> None:
        """Push the balance snapshot into UsageManager as a quota baseline."""
        if snapshot.status != "success":
            return

        remaining_cents = snapshot.remaining_cents

        await usage_manager.update_quota_baseline(
            accessor=credential_key,
            model="kilo/_balance",
            quota_max_requests=remaining_cents,
            quota_used=0,
            quota_reset_ts=snapshot.expiration_ts,
            quota_group="credits($)",
            force=True,
            apply_exhaustion=(remaining_cents <= 0),
        )

    # -----------------------------------------------------------------
    # CACHE
    # -----------------------------------------------------------------

    @property
    def cached_snapshot(self) -> Optional[KiloQuotaSnapshot]:
        return self._snapshot
