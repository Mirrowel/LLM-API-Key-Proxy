"""
Base Quota Tracking Mixin

Provides shared quota tracking infrastructure for providers that use OAuth
credentials with quota-based rate limiting (e.g., Antigravity, Gemini CLI).

This base class handles:
- Learned costs management (load/save/lookup)
- Credential discovery (file-based and env-based)
- Quota baseline fetching (initial and incremental)
- Baseline storage to UsageManager
- Model name mappings (user ↔ API)

Subclasses must implement:
- _fetch_quota_for_credential() - Provider-specific quota API call
- _extract_model_quota_from_response() - Parse provider-specific response format
- _get_provider_prefix() - Return provider prefix for model names (e.g., "gemini_cli")

Required from provider (via mixin inheritance):
    - self.project_id_cache: Dict[str, str]
    - self.project_tier_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, params) -> str
    - self.list_credentials(base_dir) -> List[Dict[str, Any]]
"""

import asyncio
import json
import logging
import os
import time
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...utils.paths import get_cache_dir

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Delay before fetching quota after a request (API needs time to update)
# Used for manual cost discovery
QUOTA_DISCOVERY_DELAY_SECONDS: float = 3.0

# Maximum concurrent quota fetch requests (prevents overwhelming API)
QUOTA_FETCH_CONCURRENCY: int = 5

# Upper limit for environment variable credential discovery
# Checks for {PREFIX}_1_ACCESS_TOKEN through {PREFIX}_N_ACCESS_TOKEN
ENV_CREDENTIAL_DISCOVERY_LIMIT: int = 100


class BaseQuotaTracker:
    """
    Base mixin class providing shared quota tracking functionality.

    Subclasses provide:
    - provider_env_prefix: str (e.g., "GEMINI_CLI", "ANTIGRAVITY")
    - cache_subdir: str (e.g., "gemini_cli", "antigravity")
    - default_quota_costs: Dict[str, Dict[str, float]] - tier -> model -> cost%
    - default_quota_cost_unknown: float - fallback cost for unknown models
    - user_to_api_model_map: Dict[str, str] - optional model name mappings
    - api_to_user_model_map: Dict[str, str] - optional reverse mappings

    The provider class must initialize these instance attributes in __init__:
        self._quota_refresh_interval: int = 300  # 5 min default
        self._learned_costs: Dict[str, Dict[str, float]] = {}
        self._learned_costs_loaded: bool = False
    """

    # =========================================================================
    # CLASS ATTRIBUTES - Override in subclass
    # =========================================================================

    # Environment variable prefix for credential discovery
    # e.g., "GEMINI_CLI" looks for GEMINI_CLI_1_ACCESS_TOKEN, etc.
    provider_env_prefix: str = ""

    # Cache subdirectory name for learned costs file
    cache_subdir: str = ""

    # Default quota costs per tier (tier -> model -> cost_percentage)
    # e.g., {"standard-tier": {"model-a": 0.4}, "free-tier": {"model-a": 1.0}}
    default_quota_costs: Dict[str, Dict[str, float]] = {}

    # Default cost for unknown models (as percentage)
    default_quota_cost_unknown: float = 1.0

    # Model name mappings (user-facing ↔ API names)
    user_to_api_model_map: Dict[str, str] = {}
    api_to_user_model_map: Dict[str, str] = {}

    # =========================================================================
    # TYPE HINTS for attributes from provider
    # =========================================================================

    _quota_refresh_interval: int
    _learned_costs: Dict[str, Dict[str, float]]
    _learned_costs_loaded: bool
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    # =========================================================================
    # ABSTRACT METHODS - Must implement in subclass
    # =========================================================================

    @abstractmethod
    async def _fetch_quota_for_credential(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the provider's API.

        Args:
            credential_path: Path to credential file or "env://provider/N"

        Returns:
            Provider-specific quota response dict with at minimum:
            {
                "status": "success" | "error",
                "error": str | None,
                "identifier": str,
                "tier": str | None,
                "fetched_at": float,
                ... provider-specific fields ...
            }
        """
        pass

    @abstractmethod
    def _extract_model_quota_from_response(
        self,
        quota_data: Dict[str, Any],
        tier: str,
    ) -> List[Tuple[str, float, Optional[int]]]:
        """
        Extract model quota information from a provider-specific response.

        Args:
            quota_data: Response from _fetch_quota_for_credential
            tier: Tier name for max_requests calculation

        Returns:
            List of tuples: (model_name, remaining_fraction, max_requests)
            - model_name: User-facing model name (without provider prefix)
            - remaining_fraction: 0.0 to 1.0
            - max_requests: Optional max requests for this model/tier
        """
        pass

    @abstractmethod
    def _get_provider_prefix(self) -> str:
        """
        Get the provider prefix for model names.

        Returns:
            Provider prefix (e.g., "gemini_cli", "antigravity")
        """
        pass

    # =========================================================================
    # CACHE DIRECTORY HELPERS
    # =========================================================================

    def _get_cache_dir(self) -> Path:
        """Get the cache directory for this provider."""
        return get_cache_dir(subdir=self.cache_subdir)

    def _get_learned_costs_file(self) -> Path:
        """Get the file path for storing learned quota costs."""
        return self._get_cache_dir() / "learned_quota_costs.json"

    # =========================================================================
    # LEARNED COSTS MANAGEMENT
    # =========================================================================

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
                            f"learned {self.cache_subdir} quota costs"
                        )
            except Exception as e:
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
            lib_logger.debug(f"Saved learned quota costs to {costs_file}")
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
        tier_costs = self.default_quota_costs.get(
            tier, self.default_quota_costs.get("standard-tier", {})
        )
        return tier_costs.get(clean_model, self.default_quota_cost_unknown)

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

    # =========================================================================
    # MODEL NAME MAPPINGS
    # =========================================================================

    def _user_to_api_model(self, model: str) -> str:
        """
        Convert user-facing model name to API model name for quota lookup.

        Args:
            model: User-facing model name (without provider prefix)

        Returns:
            API model name to look up in quota response
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        return self.user_to_api_model_map.get(clean_model, clean_model)

    def _api_to_user_model(self, model: str) -> str:
        """
        Convert API model name to user-facing model name.

        Args:
            model: API model name from quota response

        Returns:
            User-facing model name
        """
        return self.api_to_user_model_map.get(model, model)

    # =========================================================================
    # CREDENTIAL DISCOVERY
    # =========================================================================

    def discover_all_credentials(
        self,
        oauth_base_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Discover all credentials for this provider (file-based and env-based).

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
        # Check for {PREFIX}_1_ACCESS_TOKEN, {PREFIX}_2_ACCESS_TOKEN, etc.
        env_prefix = self.provider_env_prefix
        provider_name = self.cache_subdir  # e.g., "gemini_cli", "antigravity"

        for i in range(1, ENV_CREDENTIAL_DISCOVERY_LIMIT):  # Upper limit
            if os.getenv(f"{env_prefix}_{i}_ACCESS_TOKEN"):
                credentials.append(f"env://{provider_name}/{i}")
            else:
                break  # Stop at first gap

        # Also check legacy single credential (if no numbered ones found)
        if not credentials and os.getenv(f"{env_prefix}_ACCESS_TOKEN"):
            credentials.append(f"env://{provider_name}/0")

        return credentials

    # =========================================================================
    # QUOTA BASELINE FETCHING
    # =========================================================================

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        Fetches quota data from the provider's API for all provided credentials
        with limited concurrency to avoid rate limiting.

        Args:
            credential_paths: All credential paths to fetch baselines for

        Returns:
            Dict mapping credential_path -> fetched quota data
        """
        if not credential_paths:
            return {}

        lib_logger.debug(
            f"Fetching {self.cache_subdir} quota baselines for "
            f"{len(credential_paths)} credentials..."
        )

        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self._fetch_quota_for_credential(cred_path)

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
            f"{self.cache_subdir} baseline fetch complete: "
            f"{success_count}/{len(credential_paths)} successful"
        )

        return results

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
            interval_seconds: Consider "active" if used within this time
                             (default: _quota_refresh_interval)

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
                f"No recently active {self.cache_subdir} credentials to refresh"
            )
            return {}

        lib_logger.debug(
            f"Refreshing {self.cache_subdir} quota baselines for "
            f"{len(active_credentials)} recently active credentials"
        )

        results = {}
        for cred_path in active_credentials:
            quota_data = await self._fetch_quota_for_credential(cred_path)
            results[cred_path] = quota_data

        return results

    # =========================================================================
    # BASELINE STORAGE TO USAGE MANAGER
    # =========================================================================

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
    ) -> int:
        """
        Store fetched quota baselines into UsageManager.

        Args:
            quota_results: Dict from _fetch_quota_for_credential or fetch_initial_baselines
            usage_manager: UsageManager instance to store baselines in

        Returns:
            Number of baselines successfully stored
        """
        stored_count = 0
        provider_prefix = self._get_provider_prefix()

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            # Get tier for this credential
            tier = self.project_tier_cache.get(cred_path, "standard-tier")

            # Extract model quota data using subclass implementation
            model_quotas = self._extract_model_quota_from_response(quota_data, tier)

            for user_model, remaining, max_requests in model_quotas:
                # Add provider prefix for consistency with usage tracking
                prefixed_model = f"{provider_prefix}/{user_model}"

                # If max_requests not provided, calculate from tier-based cost
                if max_requests is None:
                    max_requests = self.get_max_requests_for_model(user_model, tier)

                # Store baseline
                await usage_manager.update_quota_baseline(
                    cred_path, prefixed_model, remaining, max_requests=max_requests
                )
                stored_count += 1

        return stored_count

    # =========================================================================
    # QUOTA COST DISCOVERY
    # =========================================================================

    async def discover_quota_costs(
        self,
        credential_path: str,
        models_to_test: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Discover quota costs by making test requests and measuring before/after.

        MANUAL USE ONLY - This makes actual API requests that consume quota.
        Use once per new tier to establish baseline costs for unknown tiers.

        The method tests one model per quota group, measures the quota consumption,
        and stores the discovered costs in the learned_costs.json file.

        Args:
            credential_path: Credential to test with (file path or env:// URI)
            models_to_test: Specific models to test (None = one representative per quota group)

        Returns:
            {
                "status": "success" | "partial" | "error",
                "tier": str,
                "credential": str,
                "discovered_costs": {"model": cost_percent, ...},
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
            "discovered_costs": {},
            "updated_groups": [],
            "errors": [],
            "message": "",
        }

        # 1. Get tier for this credential
        tier = self.project_tier_cache.get(credential_path)
        if not tier:
            # Try to load from file metadata (only for file-based credentials)
            if not credential_path.startswith("env://"):
                try:
                    with open(credential_path, "r") as f:
                        cred_data = json.load(f)
                        tier = cred_data.get("_proxy_metadata", {}).get("tier")
                except Exception:
                    pass

        if not tier or tier == "unknown":
            # Try to discover tier by making a fetch first
            try:
                quota_data = await self._fetch_quota_for_credential(credential_path)
                if quota_data.get("status") == "success":
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
        discovered_costs: Dict[str, float] = {}
        updated_groups: List[str] = []

        for model in models_to_test:
            try:
                # Fetch quota before
                before_quota = await self._fetch_quota_for_credential(credential_path)
                if before_quota.get("status") != "success":
                    result["errors"].append(
                        f"{model}: Failed to fetch before quota: {before_quota.get('error')}"
                    )
                    continue

                # Find the remaining fraction for this model
                before_remaining = self._get_model_remaining_from_quota(
                    before_quota, model
                )

                if before_remaining is None:
                    result["errors"].append(
                        f"{model}: Model not found in quota response"
                    )
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
                after_quota = await self._fetch_quota_for_credential(credential_path)
                if after_quota.get("status") != "success":
                    result["errors"].append(
                        f"{model}: Failed to fetch after quota: {after_quota.get('error')}"
                    )
                    continue

                after_remaining = self._get_model_remaining_from_quota(
                    after_quota, model
                )

                if after_remaining is None:
                    # Quota exhausted after our request
                    after_remaining = 0.0

                # Calculate cost
                delta = before_remaining - after_remaining
                if delta < 0:
                    result["errors"].append(
                        f"{model}: Negative delta (quota reset during test?)"
                    )
                    continue

                cost_percent = round(delta * 100.0, 4)

                if cost_percent < 0.001:
                    result["errors"].append(
                        f"{model}: Cost too small ({cost_percent}%) - API may not have updated yet"
                    )
                    continue

                discovered_costs[model] = cost_percent
                lib_logger.info(
                    f"Discovered cost for {model}: {cost_percent}% per request "
                    f"(~{int(100.0 / cost_percent)} requests per 100%)"
                )

                # Update all models in the same group
                quota_group = self._get_quota_group_for_model(model)
                if quota_group:
                    groups = self._get_effective_quota_groups()
                    for group_model in groups.get(quota_group, []):
                        discovered_costs[group_model] = cost_percent
                    updated_groups.append(quota_group)

            except Exception as e:
                result["errors"].append(f"{model}: Exception: {e}")
                lib_logger.warning(f"Error testing {model}: {e}")

        # 4. Save discovered costs to file
        if discovered_costs:
            self._load_learned_costs()
            if tier not in self._learned_costs:
                self._learned_costs[tier] = {}
            self._learned_costs[tier].update(discovered_costs)
            self._save_learned_costs()

            result["status"] = "success" if not result["errors"] else "partial"
            result["discovered_costs"] = discovered_costs
            result["updated_groups"] = updated_groups
            result["message"] = (
                f"Discovered costs for {len(discovered_costs)} models in tier '{tier}'. "
                f"Saved to learned_quota_costs.json"
            )
            lib_logger.info(result["message"])
        else:
            result["message"] = "No costs discovered"

        return result

    def _get_model_remaining_from_quota(
        self,
        quota_data: Dict[str, Any],
        model: str,
    ) -> Optional[float]:
        """
        Get remaining quota fraction for a specific model from quota response.

        Default implementation extracts from _extract_model_quota_from_response.
        Subclasses can override for more efficient lookup.

        Args:
            quota_data: Response from _fetch_quota_for_credential
            model: Model name to look up

        Returns:
            Remaining fraction (0.0 to 1.0) or None if not found
        """
        tier = quota_data.get("tier", "standard-tier")
        model_quotas = self._extract_model_quota_from_response(quota_data, tier)

        clean_model = model.split("/")[-1] if "/" in model else model
        api_model = self._user_to_api_model(clean_model)

        for user_model, remaining, _ in model_quotas:
            if (
                user_model == clean_model
                or self._user_to_api_model(user_model) == api_model
            ):
                return remaining

        return None

    def _get_quota_group_for_model(self, model: str) -> Optional[str]:
        """
        Get the quota group name for a model.

        Uses the inherited _find_model_quota_group from ProviderInterface.
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        return self._find_model_quota_group(clean_model)

    @abstractmethod
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
        pass
