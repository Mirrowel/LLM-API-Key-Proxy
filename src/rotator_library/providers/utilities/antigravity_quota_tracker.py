"""
Antigravity Quota Tracking Mixin

Provides quota tracking, estimation, and verification methods for the
Antigravity provider. This is a mixin class that assumes the provider
has certain methods and attributes available.

Required from provider:
    - self._get_effective_quota_groups() -> Dict[str, List[str]]
    - self._get_available_models() -> List[str]  # User-facing model names
    - self._get_antigravity_headers() -> Dict[str, str]  # API headers for requests
    - self.list_credentials(base_dir) -> List[Dict[str, Any]]
    - self.project_tier_cache: Dict[str, str]
    - self.project_id_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, headers) -> str
    - self._get_base_url() -> str
    - self._load_tier_from_file(cred_path) -> Optional[str]
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


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


# =============================================================================
# QUOTA LIMITS (max requests per 100% quota)
# =============================================================================
# Max requests per quota period. This is the SOURCE OF TRUTH.
# Cost percentage is derived as: 100 / max_requests
# Using integers avoids floating-point precision issues (e.g., 149 vs 150).
#
# Verified empirically 2026-01-07 - see tests/quota_verification/QUOTA_TESTING_GUIDE.md
# Learned values (from file) override these defaults if available.

DEFAULT_MAX_REQUESTS: Dict[str, Dict[str, int]] = {
    "standard-tier": {
        # Claude/GPT-OSS group (verified: 0.6667% per request = 150 requests)
        "claude-sonnet-4-5": 150,
        "claude-sonnet-4-5-thinking": 150,
        "claude-opus-4-5": 150,
        "claude-opus-4-5-thinking": 150,
        "gpt-oss-120b-medium": 150,
        # Gemini 3 Pro group (verified: 0.3125% per request = 320 requests)
        "gemini-3-pro-high": 320,
        "gemini-3-pro-low": 320,
        "gemini-3-pro-preview": 320,
        # Gemini 3 Flash (verified: 0.25% per request = 400 requests)
        "gemini-3-flash": 400,
        # Gemini 2.5 Flash group (verified: 0.0333% per request = 3000 requests)
        "gemini-2.5-flash": 3000,
        "gemini-2.5-flash-thinking": 3000,
        # Gemini 2.5 Flash Lite - SEPARATE pool (verified: 0.02% per request = 5000 requests)
        "gemini-2.5-flash-lite": 5000,
        # Gemini 2.5 Pro - UNVERIFIED/UNUSED (assumed 0.1% = 1000 requests)
        "gemini-2.5-pro": 1000,
    },
    "free-tier": {
        # Claude/GPT-OSS group (verified: 2.0% per request = 50 requests)
        "claude-sonnet-4-5": 50,
        "claude-sonnet-4-5-thinking": 50,
        "claude-opus-4-5": 50,
        "claude-opus-4-5-thinking": 50,
        "gpt-oss-120b-medium": 50,
        # Gemini 3 Pro group (verified: 0.6667% per request = 150 requests)
        "gemini-3-pro-high": 150,
        "gemini-3-pro-low": 150,
        "gemini-3-pro-preview": 150,
        # Gemini 3 Flash (verified: 0.2% per request = 500 requests)
        "gemini-3-flash": 500,
        # Gemini 2.5 Flash group (verified: 0.0333% per request = 3000 requests)
        "gemini-2.5-flash": 3000,
        "gemini-2.5-flash-thinking": 3000,
        # Gemini 2.5 Flash Lite - SEPARATE pool (verified: 0.02% per request = 5000 requests)
        "gemini-2.5-flash-lite": 5000,
        # Gemini 2.5 Pro - UNVERIFIED/UNUSED (assumed 0.1% = 1000 requests)
        "gemini-2.5-pro": 1000,
    },
}

# Default max requests for unknown models (1% = 100 requests)
DEFAULT_MAX_REQUESTS_UNKNOWN = 100

# Delay before fetching quota after a request (API needs time to update)
# Used by discover_quota_costs() for manual cost discovery
QUOTA_DISCOVERY_DELAY_SECONDS = 3.0

# =============================================================================
# MODEL NAME MAPPINGS
# =============================================================================
# Some user-facing model names don't exist in the API response.
# These mappings convert between user-facing names and API names.

# User-facing name → API name (for looking up quota in fetchAvailableModels response)
_USER_TO_API_MODEL_MAP: Dict[str, str] = {
    "claude-opus-4-5": "claude-opus-4-5-thinking",  # Opus only exists as -thinking in API
    "gemini-3-pro-preview": "gemini-3-pro-high",  # Preview maps to high by default
}

# API name → User-facing name (for consistency when processing API responses)
_API_TO_USER_MODEL_MAP: Dict[str, str] = {
    "claude-opus-4-5-thinking": "claude-opus-4-5",  # Normalize to user-facing name
    "claude-sonnet-4-5-thinking": "claude-sonnet-4-5",  # Normalize to user-facing name
    "gemini-3-pro-high": "gemini-3-pro-preview",  # Could map to preview (but high is valid too)
    "gemini-3-pro-low": "gemini-3-pro-preview",  # Could map to preview (but low is valid too)
    "gemini-2.5-flash-thinking": "gemini-2.5-flash",  # Normalize to user-facing name
}


def _get_antigravity_cache_dir() -> Path:
    """Get the cache directory for Antigravity files."""
    return get_cache_dir(subdir="antigravity")


def _get_learned_costs_file() -> Path:
    """Get path to the learned quota costs JSON file."""
    return _get_antigravity_cache_dir() / "learned_quota_costs.json"


class AntigravityQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Antigravity provider.

    This mixin adds the following capabilities:
    - Fetch quota info from the Antigravity fetchAvailableModels API
    - Track requests locally to estimate remaining quota
    - Verify and learn quota costs adaptively
    - Discover all credentials (file-based and env-based)
    - Get structured quota info for all credentials

    Usage:
        class AntigravityProvider(GoogleOAuthBase, AntigravityQuotaTracker):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._learned_costs: Dict[str, Dict[str, int]] = {}
        self._learned_costs_loaded: bool = False
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes that must exist on the provider
    _learned_costs: Dict[str, Dict[str, int]]
    _learned_costs_loaded: bool
    _quota_refresh_interval: int
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    def _load_learned_costs(self) -> None:
        """Load learned max_requests values from persistent file."""
        if self._learned_costs_loaded:
            return

        costs_file = _get_learned_costs_file()
        if not costs_file.exists():
            self._learned_costs_loaded = True
            return

        try:
            with open(costs_file, "r") as f:
                data = json.load(f)

            # Support both old format (float costs) and new format (int max_requests)
            raw_costs = data.get("max_requests", data.get("costs", {}))

            # Convert to int if loading old float format
            self._learned_costs = {}
            for tier, models in raw_costs.items():
                self._learned_costs[tier] = {}
                for model, value in models.items():
                    if isinstance(value, float) and value < 10:
                        # Old format: cost percentage -> convert to max_requests
                        self._learned_costs[tier][model] = (
                            int(100.0 / value) if value > 0 else 100
                        )
                    else:
                        # New format: already max_requests
                        self._learned_costs[tier][model] = int(value)

            lib_logger.debug(
                f"Loaded learned quota limits from {costs_file.name}: "
                f"{sum(len(m) for m in self._learned_costs.values())} model entries"
            )
        except (json.JSONDecodeError, IOError) as e:
            lib_logger.warning(f"Failed to load learned costs: {e}")
            self._learned_costs = {}

        self._learned_costs_loaded = True

    def _save_learned_costs(self) -> None:
        """Persist learned max_requests values to file."""
        costs_file = _get_learned_costs_file()
        costs_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "schema_version": 2,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "max_requests": self._learned_costs,
        }

        try:
            with open(costs_file, "w") as f:
                json.dump(data, f, indent=2)
            lib_logger.debug(f"Saved learned quota limits to {costs_file.name}")
        except IOError as e:
            lib_logger.warning(f"Failed to save learned costs: {e}")

    def get_quota_cost(self, model: str, tier: str) -> float:
        """
        Get quota cost per request for a model/tier combination.

        Cost is DERIVED from max_requests: cost = 100 / max_requests
        This ensures exact integer results when calculating max_requests back.

        Args:
            model: Model name (without provider prefix)
            tier: Account tier ("standard-tier" or "free-tier")

        Returns:
            Cost as percentage (e.g., 0.6667 for 0.6667% per request)
        """
        max_requests = self.get_max_requests_for_model(model, tier)
        if max_requests <= 0:
            return 100.0  # Fallback: 1 request max
        return 100.0 / max_requests

    def get_max_requests_for_model(self, model: str, tier: str) -> int:
        """
        Get maximum requests per 100% quota for a model/tier.

        This is a direct lookup from DEFAULT_MAX_REQUESTS (source of truth).
        Learned values override defaults if available.
        Using integers avoids floating-point precision issues.

        Args:
            model: Model name
            tier: Account tier

        Returns:
            Max requests (e.g., 150 for Claude on standard-tier)
        """
        # Ensure learned values are loaded
        self._load_learned_costs()

        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model

        # Check learned values first (stored as max_requests integers)
        if tier in self._learned_costs:
            if clean_model in self._learned_costs[tier]:
                return self._learned_costs[tier][clean_model]

        # Fall back to defaults
        if tier in DEFAULT_MAX_REQUESTS:
            if clean_model in DEFAULT_MAX_REQUESTS[tier]:
                return DEFAULT_MAX_REQUESTS[tier][clean_model]

        # Unknown model - use conservative default
        lib_logger.debug(
            f"Unknown max requests for model={clean_model}, tier={tier}. "
            f"Using default {DEFAULT_MAX_REQUESTS_UNKNOWN}"
        )
        return DEFAULT_MAX_REQUESTS_UNKNOWN

    def _get_quota_group_for_model(self, model: str) -> Optional[str]:
        """Get the quota group name for a model."""
        clean_model = model.split("/")[-1] if "/" in model else model
        groups = self._get_effective_quota_groups()
        for group_name, models in groups.items():
            if clean_model in models:
                return group_name
        return None

    def _user_to_api_model(self, model: str) -> str:
        """
        Convert user-facing model name to API model name for quota lookup.

        Some models the user requests don't exist in the API response:
        - claude-opus-4-5 → claude-opus-4-5-thinking (opus only has thinking variant)
        - gemini-3-pro-preview → gemini-3-pro-high (preview maps to high by default)

        Args:
            model: User-facing model name (without provider prefix)

        Returns:
            API model name to look up in fetchAvailableModels response
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        return _USER_TO_API_MODEL_MAP.get(clean_model, clean_model)

    def _api_to_user_model(self, model: str) -> str:
        """
        Convert API model name to user-facing model name.

        Normalizes API-specific names (like -thinking variants) to user-facing names
        for consistent storage and display.

        Args:
            model: API model name from fetchAvailableModels response

        Returns:
            User-facing model name
        """
        return _API_TO_USER_MODEL_MAP.get(model, model)

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Antigravity fetchAvailableModels API.

        Args:
            credential_path: Path to credential file or "env://antigravity/N"

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "identifier": str,
                "tier": str | None,
                "project_id": str | None,
                "models": {
                    "model_name": {
                        "remaining_fraction": 0.95,  # None from API = 0.0 (EXHAUSTED)
                        "is_exhausted": bool,
                        "reset_time_iso": "2025-12-16T10:31:36Z" | None,
                        "reset_timestamp": float | None,
                        "display_name": str | None,
                    }
                },
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

            # Make API request
            url = f"{self._get_base_url()}:fetchAvailableModels"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **self._get_antigravity_headers(),
            }
            payload = {"project": project_id} if project_id else {}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=30
                )
                response.raise_for_status()
                data = response.json()

            # Parse models
            models_data = {}
            for model_name, model_info in data.get("models", {}).items():
                quota_info = model_info.get("quotaInfo", {})

                # CRITICAL: NULL remainingFraction means EXHAUSTED (0.0)
                remaining = quota_info.get("remainingFraction")
                if remaining is None:
                    remaining = 0.0
                    is_exhausted = True
                else:
                    is_exhausted = remaining <= 0

                reset_time_iso = quota_info.get("resetTime")
                reset_timestamp = None
                if reset_time_iso:
                    try:
                        reset_dt = datetime.fromisoformat(
                            reset_time_iso.replace("Z", "+00:00")
                        )
                        reset_timestamp = reset_dt.timestamp()
                    except (ValueError, AttributeError):
                        pass

                models_data[model_name] = {
                    "remaining_fraction": remaining,
                    "is_exhausted": is_exhausted,
                    "reset_time_iso": reset_time_iso,
                    "reset_timestamp": reset_timestamp,
                    "display_name": model_info.get("displayName"),
                }

            return {
                "status": "success",
                "error": None,
                "identifier": identifier,
                "tier": tier,
                "project_id": project_id,
                "models": models_data,
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
                "models": {},
                "fetched_at": time.time(),
            }

    def estimate_remaining_quota(
        self,
        credential_path: str,
        model: str,
        model_data: Dict[str, Any],
        tier: str,
    ) -> Dict[str, Any]:
        """
        Estimate remaining quota based on baseline + request tracking.

        Args:
            credential_path: Credential identifier
            model: Model name (with or without provider prefix)
            model_data: The model's usage data from UsageManager (per-model structure)
            tier: Account tier ("standard-tier" or "free-tier")

        Returns:
            {
                "remaining_fraction": 0.85,
                "remaining_percent": "85%",
                "is_exhausted": False,
                "is_estimated": True,
                "requests_used": 25,
                "requests_total": 250,
                "display": "25/250",
                "confidence": "high" | "medium" | "low",
                "baseline_age_seconds": 120,
            }
        """
        clean_model = model.split("/")[-1] if "/" in model else model

        baseline_remaining = model_data.get("baseline_remaining_fraction")
        baseline_fetched_at = model_data.get("baseline_fetched_at")
        requests_at_baseline = model_data.get("requests_at_baseline", 0)
        current_request_count = model_data.get("request_count", 0)

        # Calculate requests since baseline
        requests_since_baseline = current_request_count - (requests_at_baseline or 0)

        # Get cost per request (in percentage format, e.g., 0.4 = 0.4%)
        cost_per_request_percent = self.get_quota_cost(clean_model, tier)
        # Convert to fraction for calculation with baseline_remaining (0.0 to 1.0)
        cost_per_request_fraction = cost_per_request_percent / 100.0
        max_requests = self.get_max_requests_for_model(clean_model, tier)

        # Calculate estimated remaining
        if baseline_remaining is not None:
            estimated_remaining = baseline_remaining - (
                requests_since_baseline * cost_per_request_fraction
            )
            estimated_remaining = max(0.0, min(1.0, estimated_remaining))
            is_estimated = True
            baseline_age = (
                time.time() - baseline_fetched_at
                if baseline_fetched_at
                else float("inf")
            )
        else:
            # No baseline - can't estimate, assume full quota
            estimated_remaining = 1.0
            is_estimated = False
            baseline_age = float("inf")

        # Determine confidence
        if baseline_age < 300:  # 5 minutes
            confidence = "high"
        elif baseline_age < 1800:  # 30 minutes
            confidence = "medium"
        else:
            confidence = "low"

        # Calculate display values
        is_exhausted = estimated_remaining <= 0
        remaining_percent = f"{int(estimated_remaining * 100)}%"
        requests_used = current_request_count
        display = (
            f"{requests_used}/{max_requests}"
            if max_requests > 0
            else f"{requests_used}/?"
        )

        return {
            "remaining_fraction": estimated_remaining,
            "remaining_percent": remaining_percent,
            "is_exhausted": is_exhausted,
            "is_estimated": is_estimated,
            "requests_used": requests_used,
            "requests_total": max_requests,
            "display": display,
            "confidence": confidence,
            "baseline_age_seconds": baseline_age
            if baseline_age != float("inf")
            else None,
        }

    def discover_all_credentials(
        self,
        oauth_base_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Discover all Antigravity credentials (file-based and env-based).

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
        # Check for ANTIGRAVITY_1_ACCESS_TOKEN, ANTIGRAVITY_2_ACCESS_TOKEN, etc.
        for i in range(1, 100):  # Reasonable upper limit
            if os.getenv(f"ANTIGRAVITY_{i}_ACCESS_TOKEN"):
                credentials.append(f"env://antigravity/{i}")
            else:
                break  # Stop at first gap

        # Also check legacy single credential (if no numbered ones found)
        if not credentials and os.getenv("ANTIGRAVITY_ACCESS_TOKEN"):
            credentials.append("env://antigravity/0")

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
                        "model_groups": {
                            "group_name": {
                                "remaining_fraction": float,
                                "remaining_percent": str,
                                "is_estimated": bool,
                                "is_exhausted": bool,
                                "requests_used": int,
                                "requests_total": int,
                                "display": str,
                                "reset_time_iso": str | None,
                                "models": List[str],
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

        for cred_path in credential_paths:
            identifier = (
                Path(cred_path).name
                if not cred_path.startswith("env://")
                else cred_path
            )

            try:
                # Get tier
                tier = self.project_tier_cache.get(cred_path)
                if not tier:
                    tier = self._load_tier_from_file(cred_path)
                tier = tier or "unknown"

                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Get email from credential
                email = None
                if not cred_path.startswith("env://"):
                    try:
                        with open(cred_path, "r") as f:
                            creds = json.load(f)
                        email = creds.get("_proxy_metadata", {}).get("email")
                    except (IOError, json.JSONDecodeError):
                        pass

                project_id = self.project_id_cache.get(cred_path)

                # Build model groups from quota groups
                groups = self._get_effective_quota_groups()
                model_groups = {}

                for group_name, group_models in groups.items():
                    # Get usage data for this group if available
                    group_info = {
                        "remaining_fraction": 1.0,
                        "remaining_percent": "100%",
                        "is_estimated": False,
                        "is_exhausted": False,
                        "requests_used": 0,
                        "requests_total": self.get_max_requests_for_model(
                            group_models[0], tier
                        ),
                        "display": f"0/{self.get_max_requests_for_model(group_models[0], tier)}",
                        "reset_time_iso": None,
                        "models": group_models,
                        "confidence": "low",
                    }

                    # If usage data provided, calculate estimates
                    if usage_data and include_estimates and cred_path in usage_data:
                        cred_usage = usage_data[cred_path]
                        models_usage = cred_usage.get("models", {})

                        # Get request_count from representative model (synced across group)
                        # Try with and without provider prefix for first model in group
                        representative_model = group_models[0]
                        prefixed_model = f"antigravity/{representative_model}"
                        model_usage = models_usage.get(
                            prefixed_model
                        ) or models_usage.get(representative_model, {})

                        total_requests = model_usage.get("request_count", 0)
                        baseline_remaining = model_usage.get(
                            "baseline_remaining_fraction"
                        )
                        baseline_fetched_at = model_usage.get("baseline_fetched_at")
                        max_requests = model_usage.get("quota_max_requests")

                        # Get reset time from any model in group (also synced)
                        reset_time_iso = None
                        if model_usage.get("quota_reset_ts"):
                            ts = model_usage["quota_reset_ts"]
                            try:
                                reset_time_iso = datetime.fromtimestamp(
                                    ts, tz=timezone.utc
                                ).isoformat()
                            except (ValueError, OSError):
                                pass

                        # Calculate estimate
                        # cost_per_request is in percentage (0.4 = 0.4%), convert to fraction
                        cost_per_request_percent = self.get_quota_cost(
                            group_models[0], tier
                        )
                        cost_per_request_fraction = cost_per_request_percent / 100.0
                        # Use max_requests from usage data if available, otherwise calculate
                        if max_requests is None:
                            max_requests = self.get_max_requests_for_model(
                                group_models[0], tier
                            )

                        if baseline_remaining is not None:
                            estimated_remaining = baseline_remaining - (
                                total_requests * cost_per_request_fraction
                            )
                            estimated_remaining = max(
                                0.0, min(1.0, estimated_remaining)
                            )
                            is_estimated = True

                            baseline_age = (
                                time.time() - baseline_fetched_at
                                if baseline_fetched_at
                                else float("inf")
                            )
                            if baseline_age < 300:
                                confidence = "high"
                            elif baseline_age < 1800:
                                confidence = "medium"
                            else:
                                confidence = "low"
                        else:
                            estimated_remaining = 1.0
                            is_estimated = False
                            confidence = "low"

                        group_info.update(
                            {
                                "remaining_fraction": estimated_remaining,
                                "remaining_percent": f"{int(estimated_remaining * 100)}%",
                                "is_estimated": is_estimated,
                                "is_exhausted": estimated_remaining <= 0,
                                "requests_used": total_requests,
                                "requests_total": max_requests,
                                "display": f"{total_requests}/{max_requests}",
                                "reset_time_iso": reset_time_iso,
                                "confidence": confidence,
                            }
                        )

                    model_groups[group_name] = group_info

                results[identifier] = {
                    "identifier": identifier,
                    "file_path": cred_path
                    if not cred_path.startswith("env://")
                    else None,
                    "email": email,
                    "tier": tier,
                    "project_id": project_id,
                    "status": "success",
                    "error": None,
                    "model_groups": model_groups,
                }

            except Exception as e:
                lib_logger.warning(f"Failed to get quota info for {identifier}: {e}")
                results[identifier] = {
                    "identifier": identifier,
                    "file_path": cred_path
                    if not cred_path.startswith("env://")
                    else None,
                    "email": None,
                    "tier": None,
                    "project_id": None,
                    "status": "error",
                    "error": str(e),
                    "model_groups": {},
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
            f"Refreshing quota baselines for {len(active_credentials)} "
            f"recently active credentials"
        )

        results = {}
        for cred_path in active_credentials:
            quota_data = await self.fetch_quota_from_api(cred_path)
            results[cred_path] = quota_data

        return results

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        Fetches quota data from the Antigravity API for all provided credentials
        with limited concurrency to avoid rate limiting.

        Args:
            credential_paths: All credential paths to fetch baselines for

        Returns:
            Dict mapping credential_path -> fetched quota data
        """
        if not credential_paths:
            return {}

        lib_logger.debug(
            f"Fetching quota baselines for {len(credential_paths)} credentials..."
        )

        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self.fetch_quota_from_api(cred_path)

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
            f"Baseline fetch complete: {success_count}/{len(credential_paths)} successful"
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
            quota_results: Dict from fetch_quota_from_api or fetch_initial_baselines
            usage_manager: UsageManager instance to store baselines in

        Returns:
            Number of baselines successfully stored
        """
        stored_count = 0

        # Get user-facing model names we care about
        available_models = set(self._get_available_models())

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            # Get tier for this credential (needed for max_requests calculation)
            tier = self.project_tier_cache.get(cred_path, "unknown")

            models = quota_data.get("models", {})
            # Track which user-facing models we've already stored to avoid duplicates
            stored_for_cred: set = set()

            for api_model_name, model_info in models.items():
                remaining = model_info.get("remaining_fraction")
                if remaining is None:
                    continue

                # Convert API name to user-facing name
                user_model = self._api_to_user_model(api_model_name)

                # Only store if this is a model we expose to users
                if user_model not in available_models:
                    continue

                # Skip if we already stored this user-facing model
                # (e.g., claude-sonnet-4-5 and claude-sonnet-4-5-thinking both map to claude-sonnet-4-5)
                if user_model in stored_for_cred:
                    continue

                # Calculate max_requests for this model/tier
                max_requests = self.get_max_requests_for_model(user_model, tier)

                # Store with provider prefix for consistency with usage tracking
                prefixed_model = f"antigravity/{user_model}"
                await usage_manager.update_quota_baseline(
                    cred_path, prefixed_model, remaining, max_requests
                )
                stored_for_cred.add(user_model)
                stored_count += 1

        return stored_count

    async def discover_quota_costs(
        self,
        credential_path: str,
        models_to_test: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Discover quota limits by making test requests and measuring before/after.

        MANUAL USE ONLY - This makes actual API requests that consume quota.
        Use once per new tier to establish baseline limits for unknown tiers.

        The method tests one model per quota group, measures the quota consumption,
        and stores the discovered max_requests in the learned_quota_costs.json file.

        Args:
            credential_path: Credential to test with (file path or env:// URI)
            models_to_test: Specific models to test (None = one representative per quota group)

        Returns:
            {
                "status": "success" | "partial" | "error",
                "tier": str,
                "credential": str,
                "discovered_max_requests": {"model": max_requests_int, ...},
                "updated_groups": ["group1", "group2", ...],
                "errors": [...],
                "message": str,
            }
        """
        identifier = (
            Path(credential_path).name
            if not credential_path.startswith("env://")
            else credential_path
        )

        result: Dict[str, Any] = {
            "status": "error",
            "tier": "unknown",
            "credential": identifier,
            "discovered_max_requests": {},
            "updated_groups": [],
            "errors": [],
            "message": "",
        }

        # 1. Get tier for this credential
        tier = self.project_tier_cache.get(credential_path)
        if not tier:
            tier = self._load_tier_from_file(credential_path)

        if not tier or tier == "unknown":
            # Try to discover tier by making a fetch first
            try:
                quota_data = await self.fetch_quota_from_api(credential_path)
                if quota_data["status"] == "success":
                    tier = quota_data.get("tier") or self.project_tier_cache.get(
                        credential_path
                    )
            except Exception as e:
                result["errors"].append(f"Failed to discover tier: {e}")

        if not tier or tier == "unknown":
            result["errors"].append(
                "Could not determine tier for credential. "
                "Make at least one successful request first to discover the tier."
            )
            result["message"] = "Failed: unknown tier"
            return result

        result["tier"] = tier

        # 2. Determine which models to test (one per quota group)
        if models_to_test is None:
            groups = self._get_effective_quota_groups()
            models_to_test = []
            for group_name, group_models in groups.items():
                # Pick first model in each group as representative
                if group_models:
                    models_to_test.append(group_models[0])

        if not models_to_test:
            result["errors"].append("No models to test")
            result["message"] = "Failed: no models to test"
            return result

        lib_logger.info(
            f"Starting quota cost discovery for {identifier} (tier={tier}). "
            f"Testing {len(models_to_test)} models..."
        )

        # 3. Test each model
        discovered_max_requests: Dict[str, int] = {}
        updated_groups: List[str] = []

        for model in models_to_test:
            try:
                # Fetch quota before
                before_quota = await self.fetch_quota_from_api(credential_path)
                if before_quota["status"] != "success":
                    result["errors"].append(
                        f"{model}: Failed to fetch before quota: {before_quota.get('error')}"
                    )
                    continue

                # Get remaining before (map user model to API model)
                api_model = self._user_to_api_model(model)
                before_info = before_quota["models"].get(api_model, {})
                before_remaining = before_info.get("remaining_fraction")

                if before_remaining is None:
                    result["errors"].append(f"{model}: Quota exhausted (cannot test)")
                    continue

                if before_remaining <= 0.01:
                    result["errors"].append(
                        f"{model}: Quota too low to test safely ({before_remaining:.2%})"
                    )
                    continue

                # Make a minimal test request
                lib_logger.debug(f"Making test request for {model}...")
                test_result = await self._make_test_request(credential_path, model)

                if not test_result["success"]:
                    result["errors"].append(
                        f"{model}: Test request failed: {test_result.get('error')}"
                    )
                    continue

                # Wait for API to update quota
                lib_logger.debug(
                    f"Waiting {QUOTA_DISCOVERY_DELAY_SECONDS}s for API to update..."
                )
                await asyncio.sleep(QUOTA_DISCOVERY_DELAY_SECONDS)

                # Fetch quota after
                after_quota = await self.fetch_quota_from_api(credential_path)
                if after_quota["status"] != "success":
                    result["errors"].append(
                        f"{model}: Failed to fetch after quota: {after_quota.get('error')}"
                    )
                    continue

                after_info = after_quota["models"].get(api_model, {})
                after_remaining = after_info.get("remaining_fraction")

                if after_remaining is None:
                    # Quota exhausted after our request
                    after_remaining = 0.0

                # Calculate max_requests from the delta
                delta = before_remaining - after_remaining
                if delta < 0:
                    result["errors"].append(
                        f"{model}: Negative delta (quota reset during test?)"
                    )
                    continue

                cost_percent = delta * 100.0  # Convert fraction to percentage

                if cost_percent < 0.001:
                    result["errors"].append(
                        f"{model}: Cost too small ({cost_percent}%) - API may not have updated yet"
                    )
                    continue

                # Calculate max_requests as integer (source of truth)
                max_requests = int(round(100.0 / cost_percent))

                discovered_max_requests[model] = max_requests
                lib_logger.info(
                    f"Discovered max requests for {model}: {max_requests} "
                    f"({cost_percent:.4f}% per request)"
                )

                # Update all models in the same group
                quota_group = self._get_quota_group_for_model(model)
                if quota_group:
                    groups = self._get_effective_quota_groups()
                    for group_model in groups.get(quota_group, []):
                        discovered_max_requests[group_model] = max_requests
                    updated_groups.append(quota_group)

            except Exception as e:
                result["errors"].append(f"{model}: Exception: {e}")
                lib_logger.warning(f"Error testing {model}: {e}")

        # 4. Save discovered max_requests to file
        if discovered_max_requests:
            self._load_learned_costs()
            if tier not in self._learned_costs:
                self._learned_costs[tier] = {}
            self._learned_costs[tier].update(discovered_max_requests)
            self._save_learned_costs()

            result["status"] = "success" if not result["errors"] else "partial"
            result["discovered_max_requests"] = discovered_max_requests
            result["updated_groups"] = updated_groups
            result["message"] = (
                f"Discovered max requests for {len(discovered_max_requests)} models in tier '{tier}'. "
                f"Saved to learned_quota_costs.json"
            )
            lib_logger.info(result["message"])
        else:
            result["message"] = "No max requests discovered"

        return result

    async def _make_test_request(
        self,
        credential_path: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Make a minimal test request to consume quota.

        Args:
            credential_path: Credential to use
            model: Model to test

        Returns:
            {"success": bool, "error": str | None}
        """
        try:
            # Get auth header
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Get project_id
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(
                    credential_path, access_token, {}
                )

            # Map user model to internal model name
            internal_model = self._user_to_api_model(model)

            # Build minimal request payload
            url = f"{self._get_base_url()}:generateContent"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **self._get_antigravity_headers(),
            }

            payload = {
                "project": project_id,
                "model": internal_model,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "Say 'test'"}]}],
                    "generationConfig": {"maxOutputTokens": 10},
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=60
                )

                if response.status_code == 200:
                    return {"success": True, "error": None}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}
