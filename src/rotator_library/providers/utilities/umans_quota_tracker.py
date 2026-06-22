# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Umans Request-Based Quota Tracking Mixin

Fetches request-based quota from GET {UMANS_API_BASE}/v1/usage using the
Umans API key (Authorization: Bearer).

Quota model:
- Sliding 5-hour window with a 200 soft limit / 400 hard cap requests
  (code_pro plan only; max plan has no request limit)
- Weighted request accounting (some models cost >1 unit)
- Concurrency limit (3 sessions for code_pro, 4 for max)
- Server-authoritative — always force=True to UsageManager
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

# Default Umans API base for the LLM endpoint
UMANS_API_BASE_DEFAULT = "https://api.code.umans.ai"

# Default quota refresh interval in seconds
UMANS_QUOTA_REFRESH_INTERVAL_DEFAULT = 300

# Concurrency limit for parallel /v1/usage fetches
USAGE_FETCH_CONCURRENCY = 5


@dataclass
class UmansQuotaSnapshot:
    """Server-reported quota state for a single Umans API key."""

    credential_path: str  # raw API key or env://umans/N path
    identifier: str  # short display identifier (masked for raw keys)
    plan: Optional[str]  # "code_pro" | "max" | None (unknown)
    has_request_limit: bool  # True for code_pro, False for max
    requests_limit: int  # effective limit after UMANS_QUOTA_LIMIT override
    requests_hard_cap: int  # 400 for code_pro, 0 for max (display-only)
    requests_used: int  # requests_in_window
    requests_remaining: int  # remaining_requests
    weighted_used: int  # weighted_requests_in_window
    weighted_remaining: int  # weighted_remaining_requests
    concurrency_limit: int  # 3 (code_pro) or 4 (max)
    concurrency_hard_cap: int  # 6 (code_pro) or 4 (max)
    concurrent_sessions: int  # current concurrent_sessions
    window_seconds: int  # 18000 (5h), 0 if no request limit
    window_started_at: Optional[str]  # ISO timestamp
    window_resets_at: Optional[str]  # ISO timestamp
    window_reset_ts: Optional[float]  # Unix timestamp (parsed)
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    throttled: bool
    fetched_at: float
    status: str  # "success" | "error"
    error: Optional[str]


def _get_credential_identifier(credential: str) -> str:
    """Return a short, log-safe identifier for a credential."""
    if credential.startswith("env://"):
        return credential
    if len(credential) <= 8:
        return credential
    return f"{credential[:4]}...{credential[-4:]}"


def _parse_iso_to_unix(ts_str: Optional[str]) -> Optional[float]:
    """Parse an ISO 8601 timestamp to a Unix timestamp."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError) as e:
        lib_logger.warning(f"Failed to parse Umans ISO timestamp '{ts_str}': {e}")
        return None


def _detect_plan(data: dict) -> Tuple[Optional[str], bool, int]:
    """
    Detect plan from /v1/usage response.

    Returns (plan_name, has_request_limit, concurrency_limit).
    """
    limits = data.get("limits", {})
    req_limits = limits.get("requests", {})
    conc_limits = limits.get("concurrency", {})

    req_limit = int(req_limits.get("limit", 0) or 0)
    conc_limit = int(conc_limits.get("limit", 0) or 0)

    # Explicit plan field if present
    plan = data.get("plan")

    if plan is None:
        # Infer from limits
        plan = "code_pro" if req_limit > 0 else "max"

    has_request_limit = plan == "code_pro" and req_limit > 0
    return plan, has_request_limit, conc_limit


def _resolve_request_limit(
    api_limit: int, plan: Optional[str]
) -> Tuple[int, bool]:
    """
    Resolve the effective request limit for a credential.

    UMANS_QUOTA_LIMIT only applies to code_pro plan credentials.
    max-plan credentials never track request quota through this proxy.

    Returns (effective_limit, should_track).
    """
    if plan != "code_pro":
        return 0, False

    env_override = int(os.getenv("UMANS_QUOTA_LIMIT", "0") or "0")
    if env_override > 0:
        return env_override, True

    api_limit_int = int(api_limit or 0)
    if api_limit_int > 0:
        return api_limit_int, True

    # code_pro plan without a reported limit is unexpected; treat as untracked
    return 0, False


def _parse_usage_response(
    data: dict, credential_path: str, identifier: str
) -> UmansQuotaSnapshot:
    """Parse the /v1/usage JSON response into a snapshot."""
    limits = data.get("limits", {})
    req_limits = limits.get("requests", {})
    conc_limits = limits.get("concurrency", {})
    usage = data.get("usage", {})
    window = data.get("window", {})

    plan, _, conc_limit = _detect_plan(data)

    # Resolve effective request limit (env override applies only on code_pro)
    api_limit = req_limits.get("limit", 0)
    effective_limit, should_track = _resolve_request_limit(api_limit, plan)
    has_request_limit = should_track

    reset_ts = _parse_iso_to_unix(window.get("resets_at"))

    return UmansQuotaSnapshot(
        credential_path=credential_path,
        identifier=identifier,
        plan=plan,
        has_request_limit=has_request_limit,
        requests_limit=effective_limit,
        requests_hard_cap=req_limits.get("hard_cap", 0),
        requests_used=usage.get("requests_in_window", 0),
        requests_remaining=usage.get("remaining_requests", 0),
        weighted_used=usage.get("weighted_requests_in_window", 0),
        weighted_remaining=usage.get("weighted_remaining_requests", 0),
        concurrency_limit=conc_limit,
        concurrency_hard_cap=conc_limits.get("hard_cap", 0),
        concurrent_sessions=usage.get("concurrent_sessions", 0),
        window_seconds=req_limits.get("window_seconds", 0),
        window_started_at=window.get("started_at"),
        window_resets_at=window.get("resets_at"),
        window_reset_ts=reset_ts,
        tokens_in=usage.get("tokens_in", 0),
        tokens_out=usage.get("tokens_out", 0),
        tokens_cached=usage.get("tokens_cached", 0),
        throttled=data.get("throttled", False),
        fetched_at=time.time(),
        status="success",
        error=None,
    )


def _error_snapshot(
    credential_path: str, identifier: str, error_msg: str
) -> UmansQuotaSnapshot:
    """Return a snapshot representing a failed fetch."""
    return UmansQuotaSnapshot(
        credential_path=credential_path,
        identifier=identifier,
        plan=None,
        has_request_limit=False,
        requests_limit=0,
        requests_hard_cap=0,
        requests_used=0,
        requests_remaining=0,
        weighted_used=0,
        weighted_remaining=0,
        concurrency_limit=0,
        concurrency_hard_cap=0,
        concurrent_sessions=0,
        window_seconds=0,
        window_started_at=None,
        window_resets_at=None,
        window_reset_ts=None,
        tokens_in=0,
        tokens_out=0,
        tokens_cached=0,
        throttled=False,
        fetched_at=time.time(),
        status="error",
        error=error_msg,
    )


class UmansQuotaTracker:
    """
    Mixin class providing request-based quota tracking for the Umans provider.

    Usage:
        class UmansProvider(UmansQuotaTracker, ProviderInterface):
            ...
    """

    # Type hints for attributes initialized by _init_quota_tracker()
    _quota_cache: Dict[str, UmansQuotaSnapshot]
    _quota_refresh_interval: int
    _usage_manager: Optional["UsageManager"]
    _initial_baselines_fetched: bool

    def _init_quota_tracker(self) -> None:
        self._quota_cache = {}
        self._quota_refresh_interval = int(
            os.getenv(
                "UMANS_QUOTA_REFRESH_INTERVAL",
                str(UMANS_QUOTA_REFRESH_INTERVAL_DEFAULT),
            )
        )
        self._usage_manager = None
        self._initial_baselines_fetched = False

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        """Store a reference to the UsageManager (optional, used by some callers)."""
        self._usage_manager = usage_manager

    def _resolve_api_base(self) -> str:
        return os.getenv("UMANS_API_BASE", UMANS_API_BASE_DEFAULT).rstrip("/")

    async def _fetch_usage_for_credential(
        self, credential_path: str
    ) -> UmansQuotaSnapshot:
        """
        Fetch quota from GET {UMANS_API_BASE}/v1/usage for a single credential.

        Args:
            credential_path: Raw API key or env://umans/N virtual path.

        Returns:
            UmansQuotaSnapshot with status "success" or "error".
        """
        identifier = _get_credential_identifier(credential_path)
        try:
            headers = {
                "Authorization": f"Bearer {credential_path}",
                "Accept": "application/json",
            }
            base = self._resolve_api_base()
            url = f"{base}/v1/usage"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

            snapshot = _parse_usage_response(data, credential_path, identifier)
            self._quota_cache[credential_path] = snapshot
            lib_logger.debug(
                f"Umans quota fetched for {identifier}: plan={snapshot.plan}, "
                f"used={snapshot.requests_used}/{snapshot.requests_limit}, "
                f"concurrent={snapshot.concurrent_sessions}/{snapshot.concurrency_limit}"
            )
            return snapshot
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(
                f"Umans quota fetch failed for {identifier}: {error_msg}"
            )
            return _error_snapshot(credential_path, identifier, error_msg)
        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(
                f"Umans quota fetch failed for {identifier}: {error_msg}"
            )
            return _error_snapshot(credential_path, identifier, error_msg)

    async def fetch_initial_baselines(
        self, credential_paths: List[str]
    ) -> Dict[str, UmansQuotaSnapshot]:
        """
        Batch fetch quota baselines for all credentials.

        Args:
            credential_paths: List of raw API keys or env://umans/N paths.

        Returns:
            Dict mapping credential_path -> UmansQuotaSnapshot.
        """
        results: Dict[str, UmansQuotaSnapshot] = {}
        if not credential_paths:
            return results

        semaphore = asyncio.Semaphore(USAGE_FETCH_CONCURRENCY)

        async def fetch_one(cred_path: str) -> Tuple[str, UmansQuotaSnapshot]:
            async with semaphore:
                snapshot = await self._fetch_usage_for_credential(cred_path)
                return cred_path, snapshot

        tasks = [fetch_one(c) for c in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in fetch_results:
            if isinstance(item, BaseException):
                lib_logger.warning(f"Umans baseline fetch error: {item}")
                continue
            assert isinstance(item, tuple)
            cred_path, snapshot = item
            results[cred_path] = snapshot

        success_count = sum(1 for s in results.values() if s.status == "success")
        lib_logger.info(
            f"Umans: fetched {success_count}/{len(credential_paths)} quota baselines"
        )
        return results

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, UmansQuotaSnapshot],
        usage_manager: "UsageManager",
        force: bool = True,
        is_initial_fetch: bool = False,
    ) -> int:
        """
        Push quota snapshots to the UsageManager.

        Request quota is display-only (apply_exhaustion=False) until the burst
        ceiling behavior is confirmed. Concurrency is always display-only.

        Args:
            quota_results: Mapping of credential_path -> snapshot.
            usage_manager: UsageManager instance.
            force: Whether to overwrite local counts with API values.
            is_initial_fetch: Unused placeholder for interface parity.

        Returns:
            Number of baselines stored.
        """
        stored_count = 0
        provider_prefix = getattr(self, "provider_env_name", "umans")

        for cred_path, snapshot in quota_results.items():
            if snapshot.status != "success":
                continue

            # Request quota — ONLY for credentials with a request limit.
            # apply_exhaustion=False: display-only until burst ceiling is understood.
            if snapshot.has_request_limit and snapshot.requests_limit > 0:
                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_requests_5h",
                        quota_max_requests=snapshot.requests_limit,
                        quota_reset_ts=snapshot.window_reset_ts,
                        quota_used=snapshot.requests_used,
                        quota_group="5h-requests",
                        force=force,
                        apply_exhaustion=False,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Umans request baseline for {snapshot.identifier}: {e}"
                    )

            # Concurrency — display-only for all plans
            if snapshot.concurrency_limit > 0:
                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_concurrent",
                        quota_max_requests=snapshot.concurrency_limit,
                        quota_reset_ts=None,
                        quota_used=snapshot.concurrent_sessions,
                        quota_group="concurrency",
                        force=force,
                        apply_exhaustion=False,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Umans concurrency baseline for {snapshot.identifier}: {e}"
                    )

        return stored_count

    def get_cached_quota(
        self, credential_path: str
    ) -> Optional[UmansQuotaSnapshot]:
        """Return the most recently fetched snapshot for a credential."""
        return self._quota_cache.get(credential_path)

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Get aggregated quota info for all credentials.

        Args:
            credential_paths: List of credential paths/keys.
            force_refresh: If True, refetch from API before returning.

        Returns:
            Aggregated dict with credentials, summary, and timestamp.
        """
        if force_refresh:
            results = await self.fetch_initial_baselines(credential_paths)
        else:
            results = {
                p: self._quota_cache[p]
                for p in credential_paths
                if p in self._quota_cache
            }

        credentials_out: Dict[str, Any] = {}
        for path, snapshot in results.items():
            credentials_out[path] = {
                "identifier": snapshot.identifier,
                "status": snapshot.status,
                "plan": snapshot.plan,
                "requests_used": snapshot.requests_used,
                "requests_limit": snapshot.requests_limit,
                "requests_remaining": snapshot.requests_remaining,
                "concurrent_sessions": snapshot.concurrent_sessions,
                "concurrency_limit": snapshot.concurrency_limit,
                "window_resets_at": snapshot.window_resets_at,
                "fetched_at": snapshot.fetched_at,
                "error": snapshot.error,
            }

        return {
            "credentials": credentials_out,
            "summary": {
                "total_credentials": len(credential_paths),
                "fetched": sum(
                    1 for s in results.values() if s.status == "success"
                ),
            },
            "timestamp": time.time(),
        }

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure periodic quota refresh."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "umans_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """Periodic refresh cycle: fetch → push."""
        self._usage_manager = usage_manager
        quota_results = await self.fetch_initial_baselines(credentials)
        is_initial = not self._initial_baselines_fetched
        stored = await self._store_baselines_to_usage_manager(
            quota_results,
            usage_manager,
            force=True,
            is_initial_fetch=is_initial,
        )
        if stored > 0:
            self._initial_baselines_fetched = True
        elif any(s.status == "success" for s in quota_results.values()):
            lib_logger.warning(
                "Umans quota fetch succeeded but no quota baselines were stored "
                "(check plan detection / limit parsing)"
            )
