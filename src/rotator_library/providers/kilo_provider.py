"""
Kilo (KiloCode) Provider with Credit-Based Quota Tracking

Extends the standard OpenAI-compatible provider with background quota
tracking via the Kilo web dashboard API.  Credit balance is fetched using
a NextAuth session token (from browser cookie) and surfaced in the TUI
quota viewer as a dollar-denominated balance.

Environment variables:
    KILO_API_BASE               – OpenRouter-compatible endpoint (required)
    KILO_API_KEY_*              – API key(s) for request auth
    KILO_SESSION_TOKEN          – __Secure-next-auth.session-token cookie value
    KILO_QUOTA_REFRESH_INTERVAL – Refresh interval in seconds (default: 600)
"""

import os
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .openai_compatible_provider import OpenAICompatibleProvider
from .utilities.kilo_quota_tracker import KiloQuotaTracker

if TYPE_CHECKING:
    from ..usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


class KiloProvider(OpenAICompatibleProvider):
    """
    KiloCode provider with optional credit balance monitoring.

    When KILO_SESSION_TOKEN is set, a background job periodically fetches the
    account balance from app.kilo.ai and pushes it to UsageManager so it
    appears in the TUI and /v1/quota-stats.

    If the token is absent or expired, the provider still works normally for
    routing requests — quota simply shows as unknown.
    """

    # All models share one credential-level credit pool
    model_quota_groups = {
        "credits($)": ["kilo/_balance"],
    }

    skip_cost_calculation: bool = True

    def __init__(self):
        super().__init__("kilo")

        self._tracker: Optional[KiloQuotaTracker] = None

        session_token = os.environ.get("KILO_SESSION_TOKEN", "").strip()
        if session_token:
            try:
                interval = int(
                    os.environ.get("KILO_QUOTA_REFRESH_INTERVAL", "600")
                )
            except ValueError:
                interval = 600

            self._tracker = KiloQuotaTracker(session_token, interval)
            lib_logger.info(
                "Kilo quota tracking enabled "
                f"(refresh every {interval}s)"
            )
        else:
            lib_logger.info(
                "Kilo quota tracking disabled — "
                "set KILO_SESSION_TOKEN to enable"
            )

    # -----------------------------------------------------------------
    # QUOTA GROUP WIRING
    # -----------------------------------------------------------------

    def get_model_quota_group(self, model: str) -> Optional[str]:
        return "credits($)"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        if group == "credits($)":
            return ["kilo/_balance"]
        return []

    # -----------------------------------------------------------------
    # BACKGROUND JOB
    # -----------------------------------------------------------------

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        if not self._tracker:
            return None
        return {
            "interval": self._tracker._refresh_interval,
            "name": "kilo_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        if not self._tracker:
            return

        snapshot = await self._tracker.fetch_balance()

        if snapshot.status == "error" and snapshot.error == "session_expired":
            lib_logger.warning(
                "Kilo session token expired — quota will show as unknown. "
                "Update KILO_SESSION_TOKEN to restore tracking."
            )
            return

        if snapshot.status != "success":
            return

        # Push balance for every credential (they share the same account)
        for cred_key in credentials:
            await self._tracker.push_to_usage_manager(
                usage_manager, cred_key, snapshot
            )

        lib_logger.debug(
            f"Kilo quota refresh: ${snapshot.remaining_dollars:.2f} remaining"
        )

        # Periodically refresh the session token to keep it alive.
        # We do this on every background run (default 10 min) — the
        # /api/auth/session call is lightweight and extends the 30-day TTL.
        new_token = await self._tracker.refresh_session_token()
        if new_token:
            os.environ["KILO_SESSION_TOKEN"] = new_token
