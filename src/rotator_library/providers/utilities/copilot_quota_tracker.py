# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Copilot Quota Tracking Mixin

Fetches quota data from GitHub's /copilot_internal/user API endpoint.

The endpoint returns quota snapshots per bucket:
  - premium_interactions: Limited (e.g., 300/month on student plan)
  - chat: Unlimited (for now)
  - completions: Unlimited (for now)

Each snapshot includes:
  - remaining / entitlement: Current vs max counts
  - percent_remaining: 0-100
  - unlimited: Whether the bucket has no cap
  - quota_reset_at: Timestamp for reset (0 = same as monthly reset)

Authentication: Uses the GitHub OAuth token (the long-lived refresh_token),
NOT the short-lived Copilot API token.

Source: https://api.github.com/copilot_internal/user
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# API CONFIGURATION
# =============================================================================

COPILOT_USER_URL = "https://api.github.com/copilot_internal/user"

# Headers required by the Copilot API (same as token refresh)
COPILOT_API_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# Default quota refresh interval (5 minutes)
DEFAULT_QUOTA_REFRESH_INTERVAL = 300

# Quota bucket names from the API
BUCKET_PREMIUM = "premium_interactions"
BUCKET_CHAT = "chat"
BUCKET_COMPLETIONS = "completions"

# Buckets that have actual limits (non-unlimited) — the ones we track
TRACKED_BUCKETS = [BUCKET_PREMIUM]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CopilotQuotaBucket:
    """Quota snapshot for a single bucket (e.g., premium_interactions)."""

    quota_id: str
    remaining: int
    entitlement: int
    percent_remaining: float
    unlimited: bool
    overage_count: int = 0
    overage_permitted: bool = False
    has_quota: bool = False
    quota_reset_at: int = 0
    timestamp_utc: str = ""

    @property
    def is_exhausted(self) -> bool:
        """Check if this bucket's quota is exhausted."""
        if self.unlimited:
            return False
        return self.remaining <= 0

    @property
    def is_limited(self) -> bool:
        """Whether this bucket has actual limits (not unlimited)."""
        return not self.unlimited and self.entitlement > 0


@dataclass
class CopilotQuotaSnapshot:
    """Complete quota snapshot for a Copilot credential."""

    credential_path: str
    identifier: str
    login: str
    sku: str
    copilot_plan: str
    buckets: Dict[str, CopilotQuotaBucket] = field(default_factory=dict)
    quota_reset_date: str = ""
    quota_reset_date_utc: str = ""
    fetched_at: float = 0.0
    status: str = "success"  # "success" or "error"
    error: Optional[str] = None

    @property
    def primary_bucket(self) -> Optional[CopilotQuotaBucket]:
        """Get the primary limited bucket (premium_interactions)."""
        return self.buckets.get(BUCKET_PREMIUM)

    @property
    def is_stale(self) -> bool:
        """Check if snapshot is older than 15 minutes."""
        return time.time() - self.fetched_at > 900


# =============================================================================
# QUOTA TRACKER MIXIN
# =============================================================================


class CopilotQuotaTracker:
    """
    Mixin class providing quota tracking for the Copilot provider.

    Fetches quota data from GitHub's /copilot_internal/user endpoint
    using the GitHub OAuth token (refresh_token in credentials).

    Usage:
        class CopilotProvider(CopilotAuthBase, CopilotQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize in __init__:
        self._quota_cache: Dict[str, CopilotQuotaSnapshot] = {}
        self._quota_refresh_interval: int = 300
        self._usage_manager: Optional[UsageManager] = None
        self._initial_baselines_fetched: bool = False
    """

    # Type hints for attributes from provider
    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, CopilotQuotaSnapshot]
    _quota_refresh_interval: int
    _usage_manager: Optional["UsageManager"]
    _initial_baselines_fetched: bool

    def _init_quota_tracker(self):
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, CopilotQuotaSnapshot] = {}
        self._quota_refresh_interval: int = DEFAULT_QUOTA_REFRESH_INTERVAL
        self._usage_manager: Optional["UsageManager"] = None
        self._initial_baselines_fetched: bool = False

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        """Set the UsageManager reference for pushing quota updates."""
        self._usage_manager = usage_manager

    # =========================================================================
    # QUOTA API FETCHING
    # =========================================================================

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> CopilotQuotaSnapshot:
        """
        Fetch quota information from /copilot_internal/user.

        Uses the GitHub OAuth token (refresh_token) for authentication.
        The short-lived Copilot API token does NOT work with this endpoint.

        Args:
            credential_path: Path to credential file or env:// URI

        Returns:
            CopilotQuotaSnapshot with quota bucket data
        """
        identifier = (
            Path(credential_path).name
            if not credential_path.startswith("env://")
            else credential_path
        )

        try:
            # Load credentials to get the GitHub OAuth token
            creds = await self._load_credentials(credential_path)
            github_token = creds.get("refresh_token", "")

            if not github_token:
                raise ValueError("No GitHub OAuth token found in credentials")

            headers = {
                **COPILOT_API_HEADERS,
                "Authorization": f"Bearer {github_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(COPILOT_USER_URL, headers=headers)
                response.raise_for_status()
                data = response.json()

            # Parse the response
            login = data.get("login", "unknown")
            sku = data.get("access_type_sku", "")
            copilot_plan = data.get("copilot_plan", "unknown")
            quota_reset_date = data.get("quota_reset_date", "")
            quota_reset_date_utc = data.get("quota_reset_date_utc", "")

            # Parse quota buckets
            buckets = {}
            for bucket_id, bucket_data in data.get("quota_snapshots", {}).items():
                buckets[bucket_id] = CopilotQuotaBucket(
                    quota_id=bucket_data.get("quota_id", bucket_id),
                    remaining=bucket_data.get("remaining", 0),
                    entitlement=bucket_data.get("entitlement", 0),
                    percent_remaining=bucket_data.get("percent_remaining", 100.0),
                    unlimited=bucket_data.get("unlimited", False),
                    overage_count=bucket_data.get("overage_count", 0),
                    overage_permitted=bucket_data.get("overage_permitted", False),
                    has_quota=bucket_data.get("has_quota", False),
                    quota_reset_at=bucket_data.get("quota_reset_at", 0),
                    timestamp_utc=bucket_data.get("timestamp_utc", ""),
                )

            snapshot = CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login=login,
                sku=sku,
                copilot_plan=copilot_plan,
                buckets=buckets,
                quota_reset_date=quota_reset_date,
                quota_reset_date_utc=quota_reset_date_utc,
                fetched_at=time.time(),
                status="success",
                error=None,
            )

            # Cache the snapshot
            self._quota_cache[credential_path] = snapshot

            # Log the key bucket info
            premium = buckets.get(BUCKET_PREMIUM)
            if premium and premium.is_limited:
                lib_logger.debug(
                    f"Copilot quota for {login}: "
                    f"premium={premium.remaining}/{premium.entitlement} "
                    f"({premium.percent_remaining:.0f}%)"
                )
            else:
                lib_logger.debug(
                    f"Copilot quota for {login}: all buckets unlimited"
                )

            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(f"Failed to fetch Copilot quota for {identifier}: {error_msg}")
            return CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login="",
                sku="",
                copilot_plan="",
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(f"Failed to fetch Copilot quota for {identifier}: {error_msg}")
            return CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login="",
                sku="",
                copilot_plan="",
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

    # =========================================================================
    # USAGE MANAGER INTEGRATION
    # =========================================================================

    async def _push_quota_to_usage_manager(
        self,
        credential_path: str,
        snapshot: CopilotQuotaSnapshot,
    ) -> int:
        """
        Push quota snapshot data to the UsageManager as baselines.

        This makes the data visible in the TUI quota-stats display.

        Returns:
            Number of baselines stored
        """
        if not self._usage_manager:
            return 0

        stored = 0
        provider_prefix = "copilot"

        for bucket_id, bucket in snapshot.buckets.items():
            # Only push limited buckets (skip unlimited ones like chat/completions)
            if bucket.unlimited or bucket.entitlement <= 0:
                continue

            # Calculate used from remaining/entitlement
            quota_used = bucket.entitlement - bucket.remaining
            is_exhausted = bucket.is_exhausted

            # Determine reset timestamp
            # quota_reset_at is 0 for monthly reset — use quota_reset_date_utc instead
            reset_ts = None
            if bucket.quota_reset_at and bucket.quota_reset_at > 0:
                reset_ts = bucket.quota_reset_at
            elif snapshot.quota_reset_date_utc:
                # Parse ISO date like "2026-05-01T00:00:00.000Z"
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(
                        snapshot.quota_reset_date_utc.replace("Z", "+00:00")
                    )
                    reset_ts = int(dt.timestamp())
                except Exception:
                    pass

            try:
                await self._usage_manager.update_quota_baseline(
                    accessor=credential_path,
                    model=f"{provider_prefix}/_{bucket_id}",
                    quota_max_requests=bucket.entitlement,
                    quota_reset_ts=reset_ts,
                    quota_used=quota_used,
                    quota_group=bucket_id,
                    force=True,
                    apply_exhaustion=is_exhausted,
                )
                stored += 1
            except Exception as e:
                lib_logger.debug(
                    f"Failed to push Copilot quota baseline for "
                    f"{bucket_id}/{snapshot.login}: {e}"
                )

        return stored

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Return configuration for quota refresh background job."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "copilot_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Execute periodic quota refresh for Copilot credentials.

        Called by BackgroundRefresher at the configured interval.
        On first run, fetches baselines for ALL credentials and applies
        exhaustion cooldowns.

        Args:
            usage_manager: UsageManager instance for pushing baselines
            credentials: List of credential paths for this provider
        """
        if not credentials:
            return

        self._usage_manager = usage_manager

        # On first run, fetch baselines for ALL credentials
        if not self._initial_baselines_fetched:
            self._initial_baselines_fetched = True
            await self._fetch_all_baselines(credentials, usage_manager)
            return

        # Subsequent runs: refresh all credentials (quota can change anytime)
        await self._fetch_all_baselines(credentials, usage_manager)

    async def _fetch_all_baselines(
        self,
        credentials: List[str],
        usage_manager: "UsageManager",
    ) -> None:
        """Fetch quotas for all credentials and push to UsageManager."""
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self.fetch_quota_from_api(cred_path)

        tasks = [fetch_with_semaphore(cred) for cred in credentials]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_stored = 0
        exhausted_log = []

        for result in results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Copilot quota fetch error: {result}")
                continue

            cred_path, snapshot = result

            if snapshot.status != "success":
                continue

            # Push to UsageManager
            stored = await self._push_quota_to_usage_manager(cred_path, snapshot)
            total_stored += stored

            # Check for exhaustion
            premium = snapshot.primary_bucket
            if premium and premium.is_exhausted:
                exhausted_log.append(
                    f"{snapshot.login} "
                    f"(0/{premium.entitlement} premium interactions)"
                )

        if exhausted_log:
            lib_logger.warning(
                f"Copilot quota: {len(exhausted_log)} exhausted credential(s): "
                f"{', '.join(exhausted_log)}"
            )
        else:
            lib_logger.debug(
                f"Copilot quota refresh: {total_stored} baselines stored "
                f"for {len(credentials)} credentials"
            )

    # =========================================================================
    # QUOTA INFO AGGREGATION
    # =========================================================================

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        Args:
            credential_paths: List of credential paths to query
            force_refresh: If True, fetch fresh data; if False, use cache

        Returns:
            Dict with per-credential quota info and summary
        """
        results = {}
        exhausted_count = 0

        for cred_path in credential_paths:
            identifier = (
                Path(cred_path).name
                if not cred_path.startswith("env://")
                else cred_path
            )

            # Check cache unless force_refresh
            cached = self._quota_cache.get(cred_path)
            if not force_refresh and cached and not cached.is_stale:
                snapshot = cached
                status = "cached"
            else:
                snapshot = await self.fetch_quota_from_api(cred_path)
                status = snapshot.status

            # Build result entry
            entry = {
                "identifier": identifier,
                "login": snapshot.login,
                "sku": snapshot.sku,
                "copilot_plan": snapshot.copilot_plan,
                "quota_reset_date": snapshot.quota_reset_date,
                "status": status,
                "error": snapshot.error,
                "fetched_at": snapshot.fetched_at,
                "is_stale": snapshot.is_stale,
                "buckets": {},
            }

            for bucket_id, bucket in snapshot.buckets.items():
                entry["buckets"][bucket_id] = {
                    "remaining": bucket.remaining,
                    "entitlement": bucket.entitlement,
                    "percent_remaining": bucket.percent_remaining,
                    "unlimited": bucket.unlimited,
                    "is_exhausted": bucket.is_exhausted,
                    "overage_count": bucket.overage_count,
                }
                if bucket.is_exhausted:
                    exhausted_count += 1

            results[identifier] = entry

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "exhausted_count": exhausted_count,
            },
            "timestamp": time.time(),
        }
