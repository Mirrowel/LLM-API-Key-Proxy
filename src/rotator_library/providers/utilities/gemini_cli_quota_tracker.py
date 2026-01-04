"""
Gemini CLI Quota Tracking Mixin

Provides quota tracking and retrieval methods for the Gemini CLI provider.
Uses the Google Code Assist retrieveUserQuota API to fetch actual quota data.

This mirrors the AntigravityQuotaTracker pattern but uses the Gemini CLI-specific
quota API endpoint discovered from the official gemini-cli source code.

API Details (from google-gemini/gemini-cli):
- Endpoint: https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota
- Request: { project: string, userAgent?: string }
- Response: { buckets?: BucketInfo[] }
- BucketInfo: { remainingAmount?, remainingFraction?, resetTime?, tokenType?, modelId? }

Required from provider:
    - self.project_id_cache: Dict[str, str]
    - self.project_tier_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, params) -> str
    - self._load_tier_from_file(cred_path) -> Optional[str]
    - self.list_credentials(base_dir) -> List[Dict[str, Any]]
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

from ...utils.paths import get_cache_dir

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# Gemini CLI Code Assist endpoint
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

# Models exposed by Gemini CLI
GEMINI_CLI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]

# Model ID mappings for quota buckets
# The quota API may return different model IDs than what we use
_API_TO_USER_MODEL_MAP: Dict[str, str] = {
    # Map API model IDs to user-facing names if they differ
    "gemini-2.5-pro-preview": "gemini-2.5-pro",
    "gemini-2.5-flash-preview": "gemini-2.5-flash",
}

_USER_TO_API_MODEL_MAP: Dict[str, str] = {
    v: k for k, v in _API_TO_USER_MODEL_MAP.items()
}

# =============================================================================
# QUOTA COST CONSTANTS (in PERCENTAGE format)
# =============================================================================
# Quota costs per request as PERCENTAGE of 100% quota.
# E.g., 0.1 means 0.1% per request = 1000 requests total (100 / 0.1 = 1000)
# These are initial estimates based on observed quota bucket behavior.
# Learned costs override these if available.

DEFAULT_QUOTA_COSTS: Dict[str, Dict[str, float]] = {
    "standard-tier": {
        # Standard tier has higher daily limits (~1500-2000 requests/day)
        "gemini-2.0-flash": 0.05,  # ~2000 requests
        "gemini-2.5-pro": 0.1,  # ~1000 requests
        "gemini-2.5-flash": 0.05,  # ~2000 requests
        "gemini-2.5-flash-lite": 0.05,  # ~2000 requests
        "gemini-3-pro-preview": 0.1,  # ~1000 requests
        "gemini-3-flash-preview": 0.1,  # ~1000 requests
    },
    "free-tier": {
        # Free tier has lower daily limits (~1000 requests/day)
        "gemini-2.0-flash": 0.1,  # ~1000 requests
        "gemini-2.5-pro": 0.2,  # ~500 requests
        "gemini-2.5-flash": 0.1,  # ~1000 requests
        "gemini-2.5-flash-lite": 0.1,  # ~1000 requests
        "gemini-3-pro-preview": 0.2,  # ~500 requests
        "gemini-3-flash-preview": 0.2,  # ~500 requests
    },
}

# Default quota cost for unknown models (0.1% = 1000 requests max)
DEFAULT_QUOTA_COST_UNKNOWN = 0.1

# Delay before fetching quota after a request (API needs time to update)
# Used for manual cost discovery
QUOTA_DISCOVERY_DELAY_SECONDS = 3.0


def _get_gemini_cli_cache_dir() -> Path:
    """Get the cache directory for Gemini CLI files."""
    return get_cache_dir(subdir="gemini_cli")


class GeminiCliQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Gemini CLI provider.

    This mixin adds the following capabilities:
    - Fetch real-time quota info from the Gemini CLI retrieveUserQuota API
    - Discover all credentials (file-based and env-based)
    - Get structured quota info for all credentials

    Usage:
        class GeminiCliProvider(GeminiAuthBase, GeminiCliQuotaTracker):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes that must exist on the provider
    _quota_refresh_interval: int
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    # Learned costs storage (instance variables initialized lazily)
    _learned_costs: Dict[str, Dict[str, float]]
    _learned_costs_loaded: bool

    # =========================================================================
    # QUOTA COST METHODS
    # =========================================================================

    def _get_learned_costs_file(self) -> Path:
        """Get the file path for storing learned quota costs."""
        return _get_gemini_cli_cache_dir() / "learned_quota_costs.json"

    def _load_learned_costs(self) -> None:
        """
        Load learned quota costs from cache file.

        Learned costs override the default estimates when available.
        They are populated through manual cost discovery or observation.
        """
        # Initialize if not present
        if not hasattr(self, "_learned_costs"):
            self._learned_costs = {}
        if not hasattr(self, "_learned_costs_loaded"):
            self._learned_costs_loaded = False

        if self._learned_costs_loaded:
            return

        costs_file = self._get_learned_costs_file()
        if costs_file.exists():
            try:
                with open(costs_file, "r") as f:
                    data = json.load(f)
                    # Validate schema
                    if data.get("schema_version") == 1:
                        self._learned_costs = data.get("costs", {})
                        lib_logger.debug(
                            f"Loaded {sum(len(v) for v in self._learned_costs.values())} "
                            f"learned Gemini CLI quota costs"
                        )
            except Exception as e:
                # Failed to load learned costs; use defaults
                lib_logger.warning(f"Failed to load learned quota costs: {e}")

        self._learned_costs_loaded = True

    def _save_learned_costs(self) -> None:
        """Save learned quota costs to cache file."""
        if not hasattr(self, "_learned_costs") or not self._learned_costs:
            return

        costs_file = self._get_learned_costs_file()
        try:
            costs_file.parent.mkdir(parents=True, exist_ok=True)
            with open(costs_file, "w") as f:
                json.dump(
                    {
                        "schema_version": 1,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "costs": self._learned_costs,
                    },
                    f,
                    indent=2,
                )
            lib_logger.debug(f"Saved learned Gemini CLI quota costs to {costs_file}")
        except Exception as e:
            lib_logger.warning(f"Failed to save learned quota costs: {e}")

    def get_quota_cost(self, model: str, tier: str) -> float:
        """
        Get quota cost per request for a model/tier combination.

        Cost is expressed as a PERCENTAGE (0-100 scale).
        E.g., 0.1 means each request uses 0.1% of quota = 1000 max requests.

        Priority: learned costs > default costs > unknown fallback

        Args:
            model: Model name (without provider prefix)
            tier: Tier name (e.g., "standard-tier", "free-tier")

        Returns:
            Cost per request as percentage (0.1 = 0.1% per request)
        """
        self._load_learned_costs()

        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model

        # Check learned costs first
        if tier in self._learned_costs and clean_model in self._learned_costs[tier]:
            return self._learned_costs[tier][clean_model]

        # Fall back to defaults
        tier_costs = DEFAULT_QUOTA_COSTS.get(
            tier, DEFAULT_QUOTA_COSTS.get("standard-tier", {})
        )
        return tier_costs.get(clean_model, DEFAULT_QUOTA_COST_UNKNOWN)

    def get_max_requests_for_model(self, model: str, tier: str) -> int:
        """
        Calculate the maximum number of requests for a model/tier.

        Based on quota cost: max_requests = 100 / cost_percentage

        Args:
            model: Model name (without provider prefix)
            tier: Tier name

        Returns:
            Maximum number of requests (e.g., 1000 for 0.1% cost)
        """
        cost = self.get_quota_cost(model, tier)
        if cost <= 0:
            return 0
        return int(100 / cost)

    def update_learned_cost(self, model: str, tier: str, cost: float) -> None:
        """
        Update a learned cost for a model/tier combination.

        This can be called after observing actual quota consumption to
        refine the cost estimates over time.

        Args:
            model: Model name (without provider prefix)
            tier: Tier name
            cost: New cost value (percentage per request)
        """
        self._load_learned_costs()

        clean_model = model.split("/")[-1] if "/" in model else model

        if tier not in self._learned_costs:
            self._learned_costs[tier] = {}

        if cost <= 0:
            lib_logger.warning(
                f"Invalid quota cost {cost} for {tier}/{clean_model}; cost must be > 0"
            )
            return

        self._learned_costs[tier][clean_model] = cost
        self._save_learned_costs()

        lib_logger.info(
            f"Updated learned quota cost: {tier}/{clean_model} = {cost}% "
            f"(~{int(100 / cost)} requests)"
        )

    def _user_to_api_model(self, model: str) -> str:
        """
        Convert user-facing model name to API model name for quota lookup.

        Args:
            model: User-facing model name (without provider prefix)

        Returns:
            API model name to look up in retrieveUserQuota response
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        return _USER_TO_API_MODEL_MAP.get(clean_model, clean_model)

    def _api_to_user_model(self, model: str) -> str:
        """
        Convert API model name to user-facing model name.

        Args:
            model: API model name from retrieveUserQuota response

        Returns:
            User-facing model name
        """
        return _API_TO_USER_MODEL_MAP.get(model, model)

    def _get_gemini_cli_headers(self) -> Dict[str, str]:
        """Get standard headers for Gemini CLI API requests."""
        return {
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def retrieve_user_quota(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Gemini CLI retrieveUserQuota API.

        This is the primary quota API for Gemini CLI, discovered from the
        official google-gemini/gemini-cli source code.

        Args:
            credential_path: Path to credential file or "env://gemini_cli/N"

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "identifier": str,
                "tier": str | None,
                "project_id": str | None,
                "buckets": [
                    {
                        "model_id": str | None,
                        "remaining_fraction": float,  # 0.0 to 1.0
                        "remaining_amount": str | None,
                        "reset_time_iso": str | None,
                        "reset_timestamp": float | None,
                        "token_type": str | None,
                        "is_exhausted": bool,
                    }
                ],
                "fetched_at": float,
            }
        """
        identifier = (
            Path(credential_path).name
            if not credential_path.startswith("env://")
            else credential_path
        )

        try:
            # Get auth header and project_id
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Get or discover project_id
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(
                    credential_path, access_token, {}
                )

            tier = self.project_tier_cache.get(credential_path)

            # Make API request to retrieveUserQuota
            url = f"{CODE_ASSIST_ENDPOINT}:retrieveUserQuota"
            headers = {
                "Authorization": f"Bearer {access_token}",
                **self._get_gemini_cli_headers(),
            }
            payload = {"project": project_id} if project_id else {}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=30
                )
                response.raise_for_status()
                data = response.json()

            # Parse buckets from response
            buckets_data = []
            for bucket in data.get("buckets", []):
                # Parse remaining fraction (0.0 to 1.0)
                remaining = bucket.get("remainingFraction")
                if remaining is None:
                    # NULL means exhausted
                    remaining = 0.0
                    is_exhausted = True
                else:
                    is_exhausted = remaining <= 0

                # Parse reset time
                reset_time_iso = bucket.get("resetTime")
                reset_timestamp = None
                if reset_time_iso:
                    try:
                        reset_dt = datetime.fromisoformat(
                            reset_time_iso.replace("Z", "+00:00")
                        )
                        reset_timestamp = reset_dt.timestamp()
                    except (ValueError, AttributeError):
                        # Reset time parsing failed; leave reset_timestamp as None
                        pass

                buckets_data.append(
                    {
                        "model_id": bucket.get("modelId"),
                        "remaining_fraction": remaining,
                        "remaining_amount": bucket.get("remainingAmount"),
                        "reset_time_iso": reset_time_iso,
                        "reset_timestamp": reset_timestamp,
                        "token_type": bucket.get("tokenType"),
                        "is_exhausted": is_exhausted,
                    }
                )

            return {
                "status": "success",
                "error": None,
                "identifier": identifier,
                "tier": tier,
                "project_id": project_id,
                "buckets": buckets_data,
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.text
                if error_body:
                    error_msg = f"{error_msg}: {error_body[:200]}"
            except Exception:
                # Best-effort extraction of HTTP error body; fall back to status-only message
                pass
            lib_logger.warning(f"Failed to fetch quota for {identifier}: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "identifier": identifier,
                "tier": self.project_tier_cache.get(credential_path),
                "project_id": self.project_id_cache.get(credential_path),
                "buckets": [],
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch quota for {identifier}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "identifier": identifier,
                "tier": self.project_tier_cache.get(credential_path),
                "project_id": self.project_id_cache.get(credential_path),
                "buckets": [],
                "fetched_at": time.time(),
            }

    def discover_all_credentials(
        self,
        oauth_base_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Discover all Gemini CLI credentials (file-based and env-based).

        Args:
            oauth_base_dir: Directory for file-based credentials (default: oauth_creds)

        Returns:
            List of credential identifiers (file paths or env:// URIs)
        """
        credentials = []

        # 1. File-based credentials
        file_creds = self.list_credentials(oauth_base_dir)
        credentials.extend([c["file_path"] for c in file_creds])

        # 2. Env-based credentials
        # Check for GEMINI_CLI_1_ACCESS_TOKEN, GEMINI_CLI_2_ACCESS_TOKEN, etc.
        for i in range(1, 100):  # Reasonable upper limit
            if os.getenv(f"GEMINI_CLI_{i}_ACCESS_TOKEN"):
                credentials.append(f"env://gemini_cli/{i}")
            else:
                break  # Stop at first gap

        # Also check legacy single credential (if no numbered ones found)
        if not credentials and os.getenv("GEMINI_CLI_ACCESS_TOKEN"):
            credentials.append("env://gemini_cli/0")

        return credentials

    async def get_all_quota_info(
        self,
        credential_paths: Optional[List[str]] = None,
        oauth_base_dir: Optional[Path] = None,
        usage_data: Optional[Dict[str, Any]] = None,
        include_estimates: bool = True,
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        Args:
            credential_paths: Specific paths to fetch (None = discover all)
            oauth_base_dir: Directory for file-based credential discovery
            usage_data: Usage data from UsageManager (for estimates)
            include_estimates: If True, include local estimates

        Returns:
            {
                "credentials": {
                    "identifier": {
                        "identifier": str,
                        "file_path": str | None,
                        "email": str | None,
                        "tier": str | None,
                        "project_id": str | None,
                        "status": "success" | "error",
                        "error": str | None,
                        "model_quotas": {
                            "model_id": {
                                "remaining_fraction": float,
                                "remaining_percent": str,
                                "is_exhausted": bool,
                                "reset_time_iso": str | None,
                                "token_type": str | None,
                            }
                        }
                    }
                },
                "summary": {
                    "total_credentials": int,
                    "by_tier": Dict[str, int],
                },
                "timestamp": float,
            }
        """
        if credential_paths is None:
            credential_paths = self.discover_all_credentials(oauth_base_dir)

        results = {}
        tier_counts: Dict[str, int] = {}

        # Fetch quota for all credentials in parallel with limited concurrency
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self.retrieve_user_quota(cred_path)

        tasks = [fetch_with_semaphore(cred) for cred in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Quota fetch failed: {result}")
                continue

            cred_path, quota_data = result
            identifier = quota_data["identifier"]

            # Count tiers
            tier = quota_data.get("tier") or "unknown"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            # Get email from credential file
            email = None
            if not cred_path.startswith("env://"):
                try:
                    with open(cred_path, "r") as f:
                        creds = json.load(f)
                    email = creds.get("_proxy_metadata", {}).get("email")
                except (IOError, json.JSONDecodeError):
                    # Failed to read credential metadata; email will remain None
                    lib_logger.debug(
                        f"Could not read email from credential file: {cred_path}"
                    )

            # Build model quotas from buckets
            model_quotas = {}
            for bucket in quota_data.get("buckets", []):
                model_id = bucket.get("model_id")
                if not model_id:
                    continue

                # Convert to user-facing model name
                user_model = self._api_to_user_model(model_id)

                remaining = bucket.get("remaining_fraction", 0.0)
                model_quotas[user_model] = {
                    "remaining_fraction": remaining,
                    "remaining_percent": f"{int(remaining * 100)}%",
                    "is_exhausted": bucket.get("is_exhausted", False),
                    "reset_time_iso": bucket.get("reset_time_iso"),
                    "token_type": bucket.get("token_type"),
                }

            results[identifier] = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "email": email,
                "tier": tier,
                "project_id": quota_data.get("project_id"),
                "status": quota_data.get("status", "error"),
                "error": quota_data.get("error"),
                "model_quotas": model_quotas,
                "fetched_at": quota_data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "by_tier": tier_counts,
            },
            "timestamp": time.time(),
        }

    async def refresh_active_quota_baselines(
        self,
        credential_paths: List[str],
        usage_data: Dict[str, Any],
        interval_seconds: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Refresh quota baselines for credentials with recent activity.

        Only refreshes credentials that were used within the interval.

        Args:
            credential_paths: All credential paths to consider
            usage_data: Usage data from UsageManager
            interval_seconds: Consider "active" if used within this time (default: _quota_refresh_interval)

        Returns:
            Dict mapping credential_path -> fetched quota data (for updating baselines)
        """
        if interval_seconds is None:
            interval_seconds = self._quota_refresh_interval

        now = time.time()
        active_credentials = []

        for cred_path in credential_paths:
            cred_usage = usage_data.get(cred_path, {})
            last_used = cred_usage.get("last_used_ts", 0)

            if now - last_used < interval_seconds:
                active_credentials.append(cred_path)

        if not active_credentials:
            lib_logger.debug(
                "No recently active credentials to refresh quota baselines"
            )
            return {}

        lib_logger.debug(
            f"Refreshing Gemini CLI quota baselines for {len(active_credentials)} "
            f"recently active credentials"
        )

        results = {}
        for cred_path in active_credentials:
            quota_data = await self.retrieve_user_quota(cred_path)
            results[cred_path] = quota_data

        return results

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        Fetches quota data from the Gemini CLI API for all provided credentials
        with limited concurrency to avoid rate limiting.

        Args:
            credential_paths: All credential paths to fetch baselines for

        Returns:
            Dict mapping credential_path -> fetched quota data
        """
        if not credential_paths:
            return {}

        lib_logger.debug(
            f"Fetching Gemini CLI quota baselines for {len(credential_paths)} credentials..."
        )

        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self.retrieve_user_quota(cred_path)

        # Fetch all in parallel with limited concurrency
        tasks = [fetch_with_semaphore(cred) for cred in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Baseline fetch failed: {result}")
                continue

            cred_path, quota_data = result
            if quota_data["status"] == "success":
                success_count += 1
            results[cred_path] = quota_data

        lib_logger.debug(
            f"Gemini CLI baseline fetch complete: {success_count}/{len(credential_paths)} successful"
        )

        return results

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
    ) -> int:
        """
        Store fetched quota baselines into UsageManager.

        Args:
            quota_results: Dict from retrieve_user_quota or fetch_initial_baselines
            usage_manager: UsageManager instance to store baselines in

        Returns:
            Number of baselines successfully stored
        """
        stored_count = 0

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            for bucket in quota_data.get("buckets", []):
                model_id = bucket.get("model_id")
                if not model_id:
                    continue

                remaining = bucket.get("remaining_fraction")
                if remaining is None:
                    continue

                # Convert to user-facing model name and add provider prefix
                user_model = self._api_to_user_model(model_id)
                prefixed_model = f"gemini_cli/{user_model}"

                # Get tier for this credential (handles both path and env://)
                tier = self.project_tier_cache.get(cred_path, "standard-tier")

                # Calculate max_requests from tier-based cost
                max_requests = self.get_max_requests_for_model(user_model, tier)

                # Store baseline with calculated max_requests
                await usage_manager.update_quota_baseline(
                    cred_path, prefixed_model, remaining, max_requests=max_requests
                )
                stored_count += 1

        return stored_count

    def _get_effective_quota_groups(self) -> Dict[str, List[str]]:
        """
        Get quota groups for Gemini CLI models.

        Each model has its own separate quota bucket from the API,
        so we show each as its own line in the quota display.

        Returns:
            Dict mapping group name -> list of models in that group
        """
        return {
            # Each model is its own quota bucket (no grouping)
            "gemini-2.0-flash": ["gemini-2.0-flash"],
            "gemini-2.5-pro": ["gemini-2.5-pro"],
            "gemini-2.5-flash": ["gemini-2.5-flash"],
            "gemini-2.5-flash-lite": ["gemini-2.5-flash-lite"],
            "gemini-3-pro-preview": ["gemini-3-pro-preview"],
            "gemini-3-flash-preview": ["gemini-3-flash-preview"],
        }

    def _resolve_tier_priority(self, tier: str) -> int:
        """
        Get priority value for a tier (lower = higher priority).

        Used by the quota stats display to sort credentials.

        Args:
            tier: Tier string (e.g., 'standard-tier', 'free-tier')

        Returns:
            Priority value (lower = better)
        """
        tier_priorities = {
            "standard-tier": 1,
            "legacy-tier": 2,
            "free-tier": 3,
        }
        return tier_priorities.get(tier, 10)
