# src/rotator_library/providers/utilities/codex_quota_tracker.py
"""
Codex Quota Tracking Mixin

Provides quota tracking functionality for the Codex provider by:
1. Fetching rate limit status from the /usage endpoint
2. Parsing rate limit headers from API responses
3. Storing quota baselines in UsageManager

Rate Limit Structure (from Codex API):
- Primary window: Short-term rate limit (e.g., 5 hours)
- Secondary window: Long-term rate limit (e.g., weekly/monthly)
- Credits: Account credit balance info

Required from provider:
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self.get_account_id(credential_path) -> Optional[str]
    - self._credentials_cache: Dict[str, Dict[str, Any]]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_credential_identifier(credential_path: str) -> str:
    """Extract a short identifier from a credential path."""
    if credential_path.startswith("env://"):
        return credential_path
    return Path(credential_path).name


def _seconds_to_minutes(seconds: Optional[int]) -> Optional[int]:
    """Convert seconds to minutes, or None if input is None."""
    if seconds is None:
        return None
    return seconds // 60


# =============================================================================
# CONFIGURATION
# =============================================================================

# Codex usage API endpoint
# The Codex CLI uses different paths based on PathStyle:
# - If base contains /backend-api: use /wham/usage (ChatGptApi style)
# - Otherwise: use /api/codex/usage (CodexApi style)
# Since we use chatgpt.com/backend-api, we need /wham/usage
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

# Rate limit header names (from Codex API)
HEADER_PRIMARY_USED_PERCENT = "x-codex-primary-used-percent"
HEADER_PRIMARY_WINDOW_MINUTES = "x-codex-primary-window-minutes"
HEADER_PRIMARY_RESET_AT = "x-codex-primary-reset-at"
HEADER_SECONDARY_USED_PERCENT = "x-codex-secondary-used-percent"
HEADER_SECONDARY_WINDOW_MINUTES = "x-codex-secondary-window-minutes"
HEADER_SECONDARY_RESET_AT = "x-codex-secondary-reset-at"
HEADER_CREDITS_HAS_CREDITS = "x-codex-credits-has-credits"
HEADER_CREDITS_UNLIMITED = "x-codex-credits-unlimited"
HEADER_CREDITS_BALANCE = "x-codex-credits-balance"

# Default quota refresh interval (5 minutes)
DEFAULT_QUOTA_REFRESH_INTERVAL = 300

# Stale threshold - quota data older than this is considered stale (15 minutes)
QUOTA_STALE_THRESHOLD_SECONDS = 900


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class RateLimitWindow:
    """Rate limit window info from Codex API."""

    used_percent: float  # 0-100
    remaining_percent: float  # 100 - used_percent
    window_minutes: Optional[int]
    reset_at: Optional[int]  # Unix timestamp

    @property
    def remaining_fraction(self) -> float:
        """Get remaining quota as a fraction (0.0 to 1.0)."""
        return max(0.0, min(1.0, (100 - self.used_percent) / 100))

    @property
    def is_exhausted(self) -> bool:
        """Check if this window's quota is exhausted."""
        return self.used_percent >= 100

    def seconds_until_reset(self) -> Optional[float]:
        """Calculate seconds until reset, or None if unknown."""
        if self.reset_at is None:
            return None
        return max(0, self.reset_at - time.time())


@dataclass
class CreditsInfo:
    """Credits info from Codex API."""

    has_credits: bool
    unlimited: bool
    balance: Optional[str]  # Could be numeric string or "unlimited"


@dataclass
class CodexQuotaSnapshot:
    """Complete quota snapshot for a Codex credential."""

    credential_path: str
    identifier: str
    plan_type: Optional[str]
    primary: Optional[RateLimitWindow]
    secondary: Optional[RateLimitWindow]
    credits: Optional[CreditsInfo]
    fetched_at: float
    status: str  # "success" or "error"
    error: Optional[str]

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS


def _window_to_dict(window: RateLimitWindow) -> Dict[str, Any]:
    """Convert RateLimitWindow to dict for JSON serialization."""
    return {
        "remaining_percent": window.remaining_percent,
        "remaining_fraction": window.remaining_fraction,
        "used_percent": window.used_percent,
        "window_minutes": window.window_minutes,
        "reset_at": window.reset_at,
        "reset_in_seconds": window.seconds_until_reset(),
        "is_exhausted": window.is_exhausted,
    }


def _credits_to_dict(credits: CreditsInfo) -> Dict[str, Any]:
    """Convert CreditsInfo to dict for JSON serialization."""
    return {
        "has_credits": credits.has_credits,
        "unlimited": credits.unlimited,
        "balance": credits.balance,
    }


# =============================================================================
# HEADER PARSING
# =============================================================================


def parse_rate_limit_headers(headers: Dict[str, str]) -> CodexQuotaSnapshot:
    """
    Parse rate limit information from Codex API response headers.

    Args:
        headers: Response headers dict

    Returns:
        CodexQuotaSnapshot with parsed rate limit data
    """
    primary = _parse_window_from_headers(
        headers,
        HEADER_PRIMARY_USED_PERCENT,
        HEADER_PRIMARY_WINDOW_MINUTES,
        HEADER_PRIMARY_RESET_AT,
    )

    secondary = _parse_window_from_headers(
        headers,
        HEADER_SECONDARY_USED_PERCENT,
        HEADER_SECONDARY_WINDOW_MINUTES,
        HEADER_SECONDARY_RESET_AT,
    )

    credits = _parse_credits_from_headers(headers)

    return CodexQuotaSnapshot(
        credential_path="",
        identifier="",
        plan_type=None,
        primary=primary,
        secondary=secondary,
        credits=credits,
        fetched_at=time.time(),
        status="success" if (primary or secondary or credits) else "no_data",
        error=None,
    )


def _parse_window_from_headers(
    headers: Dict[str, str],
    used_percent_header: str,
    window_minutes_header: str,
    reset_at_header: str,
) -> Optional[RateLimitWindow]:
    """Parse a single rate limit window from headers."""
    used_percent_str = headers.get(used_percent_header)
    if not used_percent_str:
        return None

    try:
        used_percent = float(used_percent_str)
    except (ValueError, TypeError):
        return None

    # Parse optional fields
    window_minutes = None
    window_minutes_str = headers.get(window_minutes_header)
    if window_minutes_str:
        try:
            window_minutes = int(window_minutes_str)
        except (ValueError, TypeError):
            pass

    reset_at = None
    reset_at_str = headers.get(reset_at_header)
    if reset_at_str:
        try:
            reset_at = int(reset_at_str)
        except (ValueError, TypeError):
            pass

    return RateLimitWindow(
        used_percent=used_percent,
        remaining_percent=100 - used_percent,
        window_minutes=window_minutes,
        reset_at=reset_at,
    )


def _parse_credits_from_headers(headers: Dict[str, str]) -> Optional[CreditsInfo]:
    """Parse credits info from headers."""
    has_credits_str = headers.get(HEADER_CREDITS_HAS_CREDITS)
    if has_credits_str is None:
        return None

    has_credits = has_credits_str.lower() in ("true", "1")
    unlimited_str = headers.get(HEADER_CREDITS_UNLIMITED, "false")
    unlimited = unlimited_str.lower() in ("true", "1")
    balance = headers.get(HEADER_CREDITS_BALANCE)

    return CreditsInfo(
        has_credits=has_credits,
        unlimited=unlimited,
        balance=balance,
    )


# =============================================================================
# QUOTA TRACKER MIXIN
# =============================================================================


class CodexQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Codex provider.

    This mixin adds the following capabilities:
    - Fetch rate limit status from the Codex /usage API endpoint
    - Parse rate limit headers from streaming responses
    - Store quota baselines in UsageManager
    - Get structured quota info for all credentials

    Usage:
        class CodexProvider(OpenAIOAuthBase, CodexQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_cache: Dict[str, CodexQuotaSnapshot] = {}
        self._quota_refresh_interval: int = 300
    """

    # Type hints for attributes from provider
    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, CodexQuotaSnapshot]
    _quota_refresh_interval: int

    def _init_quota_tracker(self):
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, CodexQuotaSnapshot] = {}
        self._quota_refresh_interval: int = DEFAULT_QUOTA_REFRESH_INTERVAL

    # =========================================================================
    # QUOTA API FETCHING
    # =========================================================================

    async def fetch_quota_from_api(
        self,
        credential_path: str,
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> CodexQuotaSnapshot:
        """
        Fetch quota information from the Codex /usage API endpoint.

        Args:
            credential_path: Path to credential file or env:// URI
            api_base: Base URL for the Codex API

        Returns:
            CodexQuotaSnapshot with rate limit and credits info
        """
        identifier = _get_credential_identifier(credential_path)

        try:
            # Get auth headers
            auth_headers = await self.get_auth_header(credential_path)
            account_id = await self.get_account_id(credential_path)

            headers = {
                **auth_headers,
                "Content-Type": "application/json",
                "User-Agent": "codex-cli",  # Required by Codex API
            }
            if account_id:
                headers["ChatGPT-Account-Id"] = account_id  # Exact capitalization from Codex CLI

            # Use the correct Codex API URL
            url = CODEX_USAGE_URL

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()

            # Parse response
            plan_type = data.get("plan_type")

            # Parse rate_limit section
            rate_limit = data.get("rate_limit")
            primary = None
            secondary = None

            if rate_limit:
                primary_data = rate_limit.get("primary_window")
                if primary_data:
                    primary = RateLimitWindow(
                        used_percent=float(primary_data.get("used_percent", 0)),
                        remaining_percent=100 - float(primary_data.get("used_percent", 0)),
                        window_minutes=_seconds_to_minutes(
                            primary_data.get("limit_window_seconds")
                        ),
                        reset_at=primary_data.get("reset_at"),
                    )

                secondary_data = rate_limit.get("secondary_window")
                if secondary_data:
                    secondary = RateLimitWindow(
                        used_percent=float(secondary_data.get("used_percent", 0)),
                        remaining_percent=100 - float(secondary_data.get("used_percent", 0)),
                        window_minutes=_seconds_to_minutes(
                            secondary_data.get("limit_window_seconds")
                        ),
                        reset_at=secondary_data.get("reset_at"),
                    )

            # Parse credits section
            credits_data = data.get("credits")
            credits = None
            if credits_data:
                credits = CreditsInfo(
                    has_credits=credits_data.get("has_credits", False),
                    unlimited=credits_data.get("unlimited", False),
                    balance=credits_data.get("balance"),
                )

            snapshot = CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=plan_type,
                primary=primary,
                secondary=secondary,
                credits=credits,
                fetched_at=time.time(),
                status="success",
                error=None,
            )

            # Cache the snapshot
            self._quota_cache[credential_path] = snapshot

            lib_logger.debug(
                f"Fetched Codex quota for {identifier}: "
                f"primary={primary.remaining_percent:.1f}% remaining"
                if primary
                else f"Fetched Codex quota for {identifier}: no rate limit data"
            )

            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(f"Failed to fetch Codex quota for {identifier}: {error_msg}")
            return CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=None,
                primary=None,
                secondary=None,
                credits=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(f"Failed to fetch Codex quota for {identifier}: {error_msg}")
            return CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=None,
                primary=None,
                secondary=None,
                credits=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

    def update_quota_from_headers(
        self,
        credential_path: str,
        headers: Dict[str, str],
    ) -> Optional[CodexQuotaSnapshot]:
        """
        Update cached quota info from response headers.

        Call this after each API response to keep quota cache up-to-date.

        Args:
            credential_path: Credential that made the request
            headers: Response headers dict

        Returns:
            Updated CodexQuotaSnapshot or None if no quota headers present
        """
        snapshot = parse_rate_limit_headers(headers)

        if snapshot.status == "no_data":
            return None

        # Preserve existing metadata
        existing = self._quota_cache.get(credential_path)
        if existing:
            snapshot.plan_type = existing.plan_type

        snapshot.credential_path = credential_path
        snapshot.identifier = _get_credential_identifier(credential_path)

        self._quota_cache[credential_path] = snapshot

        # Log quota info when captured from headers
        if snapshot.primary:
            remaining = snapshot.primary.remaining_percent
            reset_secs = snapshot.primary.seconds_until_reset()
            if reset_secs is not None:
                reset_str = f"{int(reset_secs // 60)}m"
            else:
                reset_str = "?"
            lib_logger.debug(
                f"Codex quota from headers ({snapshot.identifier}): "
                f"{remaining:.0f}% remaining, resets in {reset_str}"
            )

        return snapshot

    def get_cached_quota(
        self,
        credential_path: str,
    ) -> Optional[CodexQuotaSnapshot]:
        """
        Get cached quota snapshot for a credential.

        Args:
            credential_path: Credential to look up

        Returns:
            Cached CodexQuotaSnapshot or None if not cached
        """
        return self._quota_cache.get(credential_path)

    # =========================================================================
    # QUOTA INFO AGGREGATION
    # =========================================================================

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        Args:
            credential_paths: List of credential paths to query
            force_refresh: If True, fetch fresh data; if False, use cache if available
            api_base: Base URL for the Codex API

        Returns:
            {
                "credentials": {
                    "identifier": {
                        "identifier": str,
                        "file_path": str | None,
                        "plan_type": str | None,
                        "status": "success" | "error" | "cached",
                        "error": str | None,
                        "primary": {
                            "remaining_percent": float,
                            "remaining_fraction": float,
                            "used_percent": float,
                            "window_minutes": int | None,
                            "reset_at": int | None,
                            "reset_in_seconds": float | None,
                            "is_exhausted": bool,
                        } | None,
                        "secondary": {...} | None,
                        "credits": {
                            "has_credits": bool,
                            "unlimited": bool,
                            "balance": str | None,
                        } | None,
                        "fetched_at": float,
                        "is_stale": bool,
                    }
                },
                "summary": {
                    "total_credentials": int,
                    "by_plan_type": Dict[str, int],
                    "exhausted_count": int,
                },
                "timestamp": float,
            }
        """
        results = {}
        plan_type_counts: Dict[str, int] = {}
        exhausted_count = 0

        for cred_path in credential_paths:
            identifier = _get_credential_identifier(cred_path)

            # Check cache first unless force_refresh
            cached = self._quota_cache.get(cred_path)
            if not force_refresh and cached and not cached.is_stale:
                snapshot = cached
                status = "cached"
            else:
                snapshot = await self.fetch_quota_from_api(cred_path, api_base)
                status = snapshot.status

            # Count plan types
            if snapshot.plan_type:
                plan_type_counts[snapshot.plan_type] = (
                    plan_type_counts.get(snapshot.plan_type, 0) + 1
                )

            # Check if exhausted
            if snapshot.primary and snapshot.primary.is_exhausted:
                exhausted_count += 1

            # Build result entry
            entry = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "plan_type": snapshot.plan_type,
                "status": status,
                "error": snapshot.error,
                "primary": _window_to_dict(snapshot.primary) if snapshot.primary else None,
                "secondary": _window_to_dict(snapshot.secondary) if snapshot.secondary else None,
                "credits": _credits_to_dict(snapshot.credits) if snapshot.credits else None,
                "fetched_at": snapshot.fetched_at,
                "is_stale": snapshot.is_stale,
            }

            results[identifier] = entry

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "by_plan_type": plan_type_counts,
                "exhausted_count": exhausted_count,
            },
            "timestamp": time.time(),
        }

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return configuration for quota refresh background job.

        Returns:
            Background job config dict
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "codex_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Execute periodic quota refresh for active credentials.

        Called by BackgroundRefresher at the configured interval.

        Args:
            usage_manager: UsageManager instance (for future baseline storage)
            credentials: List of credential paths for this provider
        """
        if not credentials:
            return

        # Only refresh credentials that have been used recently
        now = time.time()
        active_credentials = []

        for cred_path in credentials:
            cached = self._quota_cache.get(cred_path)
            # Refresh if cached and was fetched within the last hour
            if cached and (now - cached.fetched_at) < 3600:
                active_credentials.append(cred_path)

        if not active_credentials:
            lib_logger.debug("No active Codex credentials to refresh quota for")
            return

        lib_logger.debug(
            f"Refreshing Codex quota for {len(active_credentials)} active credentials"
        )

        # Fetch quotas with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return await self.fetch_quota_from_api(cred_path)

        tasks = [fetch_with_semaphore(cred) for cred in active_credentials]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(
            1
            for r in results
            if isinstance(r, CodexQuotaSnapshot) and r.status == "success"
        )

        lib_logger.debug(
            f"Codex quota refresh complete: {success_count}/{len(active_credentials)} successful"
        )

    # =========================================================================
    # USAGE MANAGER INTEGRATION
    # =========================================================================

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
    ) -> int:
        """
        Store Codex quota baselines into UsageManager.

        Codex has a global rate limit (primary/secondary window) that applies
        to all models. This method stores the same baseline for all models
        so the quota display works correctly.

        Args:
            quota_results: Dict from fetch_initial_baselines mapping cred_path -> quota data
            usage_manager: UsageManager instance to store baselines in

        Returns:
            Number of baselines successfully stored
        """
        stored_count = 0

        # Get available models from the provider (will be set by CodexProvider)
        models = getattr(self, "_available_models_for_quota", [])
        provider_prefix = getattr(self, "provider_env_name", "codex")

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            # Get remaining fraction from primary and secondary windows
            primary = quota_data.get("primary")
            secondary = quota_data.get("secondary")

            # Short credential name for logging
            if cred_path.startswith("env://"):
                short_cred = cred_path.split("/")[-1]
            else:
                short_cred = Path(cred_path).stem

            # Store primary window (5h limit) under virtual model "_5h_window"
            if primary:
                primary_remaining = primary.get("remaining_fraction", 1.0)
                primary_reset = primary.get("reset_at")
                try:
                    await usage_manager.update_quota_baseline(
                        cred_path,
                        f"{provider_prefix}/_5h_window",
                        primary_remaining,
                        max_requests=100,  # Percentage scale
                        reset_timestamp=primary_reset,
                    )
                    stored_count += 1
                    lib_logger.debug(
                        f"Stored Codex 5h baseline for {short_cred}: "
                        f"{primary_remaining * 100:.1f}% remaining"
                    )
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Codex 5h baseline for {short_cred}: {e}"
                    )

            # Store secondary window (weekly limit) under virtual model "_weekly_window"
            if secondary:
                secondary_remaining = secondary.get("remaining_fraction", 1.0)
                secondary_reset = secondary.get("reset_at")
                try:
                    await usage_manager.update_quota_baseline(
                        cred_path,
                        f"{provider_prefix}/_weekly_window",
                        secondary_remaining,
                        max_requests=100,  # Percentage scale
                        reset_timestamp=secondary_reset,
                    )
                    stored_count += 1
                    lib_logger.debug(
                        f"Stored Codex weekly baseline for {short_cred}: "
                        f"{secondary_remaining * 100:.1f}% remaining"
                    )
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Codex weekly baseline for {short_cred}: {e}"
                    )

        return stored_count

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        This matches the interface expected by RotatingClient for quota tracking.

        Args:
            credential_paths: All credential paths to fetch baselines for
            api_base: Base URL for the Codex API

        Returns:
            Dict mapping credential_path -> quota data in format:
            {
                "status": "success" | "error",
                "error": str | None,
                "primary": {
                    "remaining_fraction": float,
                    "remaining_percent": float,
                    "used_percent": float,
                    "reset_at": int | None,
                    ...
                },
                "secondary": {...} | None,
                "plan_type": str | None,
            }
        """
        if not credential_paths:
            return {}

        lib_logger.info(
            f"codex: Fetching initial quota baselines for {len(credential_paths)} credentials..."
        )

        results: Dict[str, Dict[str, Any]] = {}

        # Fetch quotas concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                snapshot = await self.fetch_quota_from_api(cred_path, api_base)
                return cred_path, snapshot

        tasks = [fetch_with_semaphore(cred) for cred in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Codex quota fetch error: {result}")
                continue

            cred_path, snapshot = result

            # Convert snapshot to dict format expected by client.py
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "plan_type": snapshot.plan_type,
                    "primary": {
                        "remaining_fraction": snapshot.primary.remaining_fraction if snapshot.primary else 0,
                        "remaining_percent": snapshot.primary.remaining_percent if snapshot.primary else 0,
                        "used_percent": snapshot.primary.used_percent if snapshot.primary else 100,
                        "reset_at": snapshot.primary.reset_at if snapshot.primary else None,
                        "window_minutes": snapshot.primary.window_minutes if snapshot.primary else None,
                        "is_exhausted": snapshot.primary.is_exhausted if snapshot.primary else True,
                    } if snapshot.primary else None,
                    "secondary": {
                        "remaining_fraction": snapshot.secondary.remaining_fraction,
                        "remaining_percent": snapshot.secondary.remaining_percent,
                        "used_percent": snapshot.secondary.used_percent,
                        "reset_at": snapshot.secondary.reset_at,
                        "window_minutes": snapshot.secondary.window_minutes,
                        "is_exhausted": snapshot.secondary.is_exhausted,
                    } if snapshot.secondary else None,
                    "credits": {
                        "has_credits": snapshot.credits.has_credits,
                        "unlimited": snapshot.credits.unlimited,
                        "balance": snapshot.credits.balance,
                    } if snapshot.credits else None,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        success_count = sum(1 for v in results.values() if v.get("status") == "success")
        lib_logger.info(
            f"codex: Fetched {success_count}/{len(credential_paths)} quota baselines"
        )

        return results
