"""
Gemini CLI Quota Tracking Mixin

Provides quota tracking and retrieval methods for the Gemini CLI provider.
Uses the Google Code Assist retrieveUserQuota API to fetch actual quota data.

This inherits from BaseQuotaTracker for shared functionality and implements
Gemini CLI-specific quota API calls.

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

from .base_quota_tracker import BaseQuotaTracker
from .gemini_shared_utils import CODE_ASSIST_ENDPOINT

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# QUOTA LIMITS (max requests per 100% quota)
# =============================================================================
# Max requests per quota period. This is the SOURCE OF TRUTH.
# Cost percentage is derived as: 100 / max_requests
# Using integers avoids floating-point precision issues.
#
# Verified 2026-01-07 via quota verification tests (see GEMINI_CLI_QUOTA_REPORT.md)
# Learned values (from file) override these defaults if available.

DEFAULT_MAX_REQUESTS: Dict[str, Dict[str, int]] = {
    "standard-tier": {
        # Pro group (verified: 0.4% per request = 250 requests)
        "gemini-2.5-pro": 250,
        "gemini-3-pro-preview": 250,
        # Flash group - 2.5 (verified: ~0.0667% per request = 1500 requests)
        # gemini-2.0-flash shares quota with 2.5-flash models
        "gemini-2.0-flash": 1500,
        "gemini-2.5-flash": 1500,
        "gemini-2.5-flash-lite": 1500,
        # 3-Flash group (verified: ~0.0667% per request = 1500 requests)
        "gemini-3-flash-preview": 1500,
    },
    "free-tier": {
        # Pro group (verified: 1.0% per request = 100 requests)
        "gemini-2.5-pro": 100,
        "gemini-3-pro-preview": 100,
        # Flash group - 2.5 (verified: 0.1% per request = 1000 requests)
        "gemini-2.0-flash": 1000,
        "gemini-2.5-flash": 1000,
        "gemini-2.5-flash-lite": 1000,
        # 3-Flash group (verified: 0.1% per request = 1000 requests)
        "gemini-3-flash-preview": 1000,
    },
}

# Default max requests for unknown models (1% = 100 requests)
DEFAULT_MAX_REQUESTS_UNKNOWN = 1000


class GeminiCliQuotaTracker(BaseQuotaTracker):
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
        self._learned_costs: Dict[str, Dict[str, float]] = {}
        self._learned_costs_loaded: bool = False
    """

    # =========================================================================
    # CLASS ATTRIBUTES - BaseQuotaTracker configuration
    # =========================================================================

    provider_env_prefix = "GEMINI_CLI"
    cache_subdir = "gemini_cli"

    # No model name mappings needed - API names match public names
    user_to_api_model_map: Dict[str, str] = {}
    api_to_user_model_map: Dict[str, str] = {}

    # Type hints for attributes from provider
    _learned_costs: Dict[str, Dict[str, int]]
    _learned_costs_loaded: bool
    _quota_refresh_interval: int
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    # =========================================================================
    # GEMINI CLI-SPECIFIC HELPERS
    # =========================================================================

    def _get_gemini_cli_headers(self) -> Dict[str, str]:
        """Get standard headers for Gemini CLI API requests."""
        return {
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _get_provider_prefix(self) -> str:
        """Get the provider prefix for model names."""
        return "gemini_cli"

    # =========================================================================
    # LEARNED COSTS MANAGEMENT (Override for integer max_requests)
    # =========================================================================

    def _load_learned_costs(self) -> None:
        """Load learned max_requests values from persistent file."""
        if self._learned_costs_loaded:
            return

        costs_file = self._get_learned_costs_file()
        if not costs_file.exists():
            self._learned_costs = {}
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
                            int(100.0 / value) if value > 0 else 1000
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
        costs_file = self._get_learned_costs_file()
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
            Cost as percentage (e.g., 0.4 for 0.4% per request)
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
            Max requests (e.g., 250 for Pro on standard-tier)
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

    # =========================================================================
    # BaseQuotaTracker ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    async def _fetch_quota_for_credential(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Gemini CLI retrieveUserQuota API.

        This is the primary quota API for Gemini CLI, discovered from the
        official google-gemini/gemini-cli source code.
        """
        return await self.retrieve_user_quota(credential_path)

    def _extract_model_quota_from_response(
        self,
        quota_data: Dict[str, Any],
        tier: str,
    ) -> List[Tuple[str, float, Optional[int]]]:
        """
        Extract model quota information from Gemini CLI bucket response.

        Returns:
            List of tuples: (model_name, remaining_fraction, max_requests)
        """
        results = []

        for bucket in quota_data.get("buckets", []):
            model_id = bucket.get("model_id")
            if not model_id:
                continue

            remaining = bucket.get("remaining_fraction")
            if remaining is None:
                remaining = 0.0

            # Convert to user-facing model name
            user_model = self._api_to_user_model(model_id)

            # Calculate max_requests from tier-based cost
            max_requests = self.get_max_requests_for_model(user_model, tier)

            results.append((user_model, remaining, max_requests))

        return results

    async def _make_test_request(
        self,
        credential_path: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Make a minimal test request to consume quota.

        Args:
            credential_path: Credential to use
            model: Model to test (e.g., "gemini-2.5-pro")

        Returns:
            {"success": bool, "error": str | None}
        """
        try:
            # Get auth header
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Get project_id (use cache or discover with proper signature)
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(
                    credential_path, access_token, {}
                )

            # Build minimal request payload for Gemini CLI
            url = f"{CODE_ASSIST_ENDPOINT}:generateContent"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "project": project_id,
                "model": model,
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

    # =========================================================================
    # GEMINI CLI-SPECIFIC QUOTA API
    # =========================================================================

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

    # =========================================================================
    # GET ALL QUOTA INFO (uses shared infrastructure)
    # =========================================================================

    async def get_all_quota_info(
        self,
        credential_paths: Optional[List[str]] = None,
        oauth_base_dir: Optional[Path] = None,
        usage_data: Optional[Dict[str, Any]] = None,
        include_estimates: bool = True,
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        This method uses the same structure as AntigravityQuotaTracker for
        consistency in the TUI and quota stats endpoint.

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
                    lib_logger.debug(
                        f"Could not read email from credential file: {cred_path}"
                    )

            # Build a lookup of model_id -> bucket data for easy access
            bucket_by_model = {}
            for bucket in quota_data.get("buckets", []):
                model_id = bucket.get("model_id")
                if model_id:
                    user_model = self._api_to_user_model(model_id)
                    bucket_by_model[user_model] = bucket

            # Build model_groups from quota groups (same structure as Antigravity)
            groups = self._get_effective_quota_groups()
            model_groups = {}

            for group_name, group_models in groups.items():
                # Default values
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

                # Find quota data from the first model in the group that has data
                for model in group_models:
                    bucket = bucket_by_model.get(model)
                    if bucket:
                        remaining = bucket.get("remaining_fraction", 1.0)
                        is_exhausted = bucket.get("is_exhausted", False)
                        reset_time_iso = bucket.get("reset_time_iso")

                        # Calculate requests used from remaining fraction
                        max_requests = self.get_max_requests_for_model(model, tier)
                        requests_used = int((1.0 - remaining) * max_requests)

                        group_info.update(
                            {
                                "remaining_fraction": remaining,
                                "remaining_percent": f"{int(remaining * 100)}%",
                                "is_estimated": False,  # Real data from API
                                "is_exhausted": is_exhausted,
                                "requests_used": requests_used,
                                "requests_total": max_requests,
                                "display": f"{requests_used}/{max_requests}",
                                "reset_time_iso": reset_time_iso,
                                "confidence": "high",  # Real API data
                            }
                        )
                        break  # Use first model with data (they share quota)

                # Enrich with usage data if available
                if usage_data and include_estimates and cred_path in usage_data:
                    cred_usage = usage_data[cred_path]
                    models_usage = cred_usage.get("models", {})

                    # Get request_count from representative model
                    representative_model = group_models[0]
                    prefixed_model = f"gemini_cli/{representative_model}"
                    model_usage = models_usage.get(prefixed_model) or models_usage.get(
                        representative_model, {}
                    )

                    total_requests = model_usage.get("request_count", 0)
                    baseline_remaining = model_usage.get("baseline_remaining_fraction")
                    max_requests_from_usage = model_usage.get("quota_max_requests")

                    if total_requests > 0:
                        # Use tracked request count
                        max_requests = (
                            max_requests_from_usage or group_info["requests_total"]
                        )
                        group_info["requests_used"] = total_requests
                        group_info["display"] = f"{total_requests}/{max_requests}"

                model_groups[group_name] = group_info

            results[identifier] = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "email": email,
                "tier": tier,
                "project_id": quota_data.get("project_id"),
                "status": quota_data.get("status", "error"),
                "error": quota_data.get("error"),
                "model_groups": model_groups,
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

    # NOTE: The following methods are now inherited from BaseQuotaTracker:
    # - _load_learned_costs()
    # - _save_learned_costs()
    # - get_quota_cost()
    # - get_max_requests_for_model()
    # - update_learned_cost()
    # - _user_to_api_model()
    # - _api_to_user_model()
    # - discover_all_credentials()
    # - fetch_initial_baselines()
    # - refresh_active_quota_baselines()
    # - _store_baselines_to_usage_manager()
    # - discover_quota_costs()
    # - _get_quota_group_for_model()

    # NOTE: _get_effective_quota_groups() is inherited from ProviderInterface
    # The quota groups are defined on GeminiCliProvider.model_quota_groups class attribute
    # This allows .env overrides via QUOTA_GROUPS_GEMINI_CLI_{GROUP}="model1,model2"
