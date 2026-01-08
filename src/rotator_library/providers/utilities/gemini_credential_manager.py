# src/rotator_library/providers/utilities/gemini_credential_manager.py
"""
Shared credential and tier management mixin for Gemini-based providers.

Provides tier loading, caching, and background job methods used by both
GeminiCliProvider and AntigravityProvider.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")


class GeminiCredentialManager:
    """
    Mixin for OAuth credential tier management for Gemini-based providers.

    Provides shared methods for:
    - Loading tier info from credential files
    - Caching tier/project info in memory
    - Initializing credentials at startup
    - Background job management for quota refresh

    Providers must define these attributes:
    - project_tier_cache: Dict[str, str] - Credential path → tier name
    - project_id_cache: Dict[str, str] - Credential path → project ID
    - _quota_refresh_interval: int - Seconds between quota refreshes
    - _initial_quota_fetch_done: bool - Track if initial fetch completed

    Providers must implement:
    - _parse_env_credential_path(path: str) -> Optional[str] - Parse env:// paths
    - get_auth_header(credential_path: str) -> Dict[str, str] - Get auth header
    - _discover_project_id(path, token, params) -> str - Discover project ID
    - fetch_initial_baselines(credentials) -> Dict - Fetch quota for all credentials
    - refresh_active_quota_baselines(credentials, usage_data) -> Dict - Refresh active
    - _store_baselines_to_usage_manager(results, manager) -> int - Store baselines
    """

    # Type hints for attributes that must be defined by providers
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]
    _quota_refresh_interval: int
    _initial_quota_fetch_done: bool

    def _load_tier_from_file(self, credential_path: str) -> Optional[str]:
        """
        Load tier from credential file's _proxy_metadata and cache it.

        This is used as a fallback when the tier isn't in the memory cache,
        typically on first access before initialize_credentials() has run.

        Args:
            credential_path: Path to the credential file

        Returns:
            Tier string if found, None otherwise
        """
        # Skip env:// paths (environment-based credentials)
        if self._parse_env_credential_path(credential_path) is not None:
            return None

        try:
            with open(credential_path, "r") as f:
                creds = json.load(f)

            metadata = creds.get("_proxy_metadata", {})
            tier = metadata.get("tier")
            project_id = metadata.get("project_id")

            if tier:
                self.project_tier_cache[credential_path] = tier
                lib_logger.debug(
                    f"Lazy-loaded tier '{tier}' for credential: {Path(credential_path).name}"
                )

            if project_id and credential_path not in self.project_id_cache:
                self.project_id_cache[credential_path] = project_id

            return tier
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            lib_logger.debug(f"Could not lazy-load tier from {credential_path}: {e}")
            return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the human-readable tier name for a credential.

        Args:
            credential: The credential path

        Returns:
            Tier name string (e.g., "free-tier") or None if unknown
        """
        tier = self.project_tier_cache.get(credential)
        if not tier:
            tier = self._load_tier_from_file(credential)
        return tier

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Load persisted tier information from credential files at startup.

        This ensures all credential priorities are known before any API calls,
        preventing unknown credentials from getting priority 999.

        For credentials without persisted tier info (new or corrupted), performs
        full discovery to ensure proper prioritization in sequential rotation mode.

        Args:
            credential_paths: List of credential file paths to initialize
        """
        # Step 1: Load persisted tiers from files
        await self._load_persisted_tiers(credential_paths)

        # Step 2: Identify credentials still missing tier info
        credentials_needing_discovery = [
            path
            for path in credential_paths
            if path not in self.project_tier_cache
            and self._parse_env_credential_path(path) is None  # Skip env:// paths
        ]

        if not credentials_needing_discovery:
            return  # All credentials have tier info

        # Get provider name for logging
        provider_name = getattr(self, "provider_env_name", "Provider")
        lib_logger.info(
            f"{provider_name}: Discovering tier info for {len(credentials_needing_discovery)} credential(s)..."
        )

        # Step 3: Perform discovery for each missing credential (sequential to avoid rate limits)
        for credential_path in credentials_needing_discovery:
            try:
                auth_header = await self.get_auth_header(credential_path)
                access_token = auth_header["Authorization"].split(" ")[1]
                await self._discover_project_id(
                    credential_path, access_token, litellm_params={}
                )
                discovered_tier = self.project_tier_cache.get(
                    credential_path, "unknown"
                )
                lib_logger.debug(
                    f"Discovered tier '{discovered_tier}' for {Path(credential_path).name}"
                )
            except Exception as e:
                lib_logger.warning(
                    f"Failed to discover tier for {Path(credential_path).name}: {e}. "
                    f"Credential will use default priority."
                )

    async def _load_persisted_tiers(
        self, credential_paths: List[str]
    ) -> Dict[str, str]:
        """
        Load persisted tier information from credential files into memory cache.

        Args:
            credential_paths: List of credential file paths

        Returns:
            Dict mapping credential path to tier name for logging purposes
        """
        loaded = {}
        for path in credential_paths:
            # Skip env:// paths (environment-based credentials)
            if self._parse_env_credential_path(path) is not None:
                continue

            # Skip if already in cache
            if path in self.project_tier_cache:
                continue

            try:
                with open(path, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})
                tier = metadata.get("tier")
                project_id = metadata.get("project_id")

                if tier:
                    self.project_tier_cache[path] = tier
                    loaded[path] = tier
                    lib_logger.debug(
                        f"Loaded persisted tier '{tier}' for credential: {Path(path).name}"
                    )

                if project_id:
                    self.project_id_cache[path] = project_id

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                lib_logger.debug(f"Could not load persisted tier from {path}: {e}")

        if loaded:
            # Log summary at debug level
            provider_name = getattr(self, "provider_env_name", "Provider")
            tier_counts: Dict[str, int] = {}
            for tier in loaded.values():
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            lib_logger.debug(
                f"{provider_name}: Loaded {len(loaded)} credential tiers from disk: "
                + ", ".join(
                    f"{tier}={count}" for tier, count in sorted(tier_counts.items())
                )
            )

        return loaded

    # =========================================================================
    # BACKGROUND JOB INTERFACE
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return background job configuration for quota baseline refresh.

        The quota baseline refresh fetches current quota status from the API
        and stores it in UsageManager for accurate quota estimation.

        Returns:
            Dict with job configuration, or None to disable background jobs.
        """
        job_name = getattr(self, "provider_env_name", "provider") + "_quota_refresh"
        return {
            "interval": self._quota_refresh_interval,  # default 300s (5 min)
            "name": job_name,
            "run_on_start": True,  # fetch baselines immediately at startup
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota baselines for credentials.

        On first run (startup): Fetches quota for ALL credentials to establish baselines.
        On subsequent runs: Only fetches for credentials used since last refresh.

        Handles both file paths and env:// credential formats.

        Args:
            usage_manager: UsageManager instance to store baselines
            credentials: List of credential paths (file paths or env:// URIs)
        """
        if not credentials:
            return

        provider_name = getattr(self, "provider_env_name", "Provider")

        if not self._initial_quota_fetch_done:
            # First run: fetch ALL credentials to establish baselines
            lib_logger.info(
                f"{provider_name}: Fetching initial quota baselines for {len(credentials)} credentials..."
            )
            quota_results = await self.fetch_initial_baselines(credentials)
            self._initial_quota_fetch_done = True
        else:
            # Subsequent runs: only recently used credentials (incremental updates)
            usage_data = await usage_manager._get_usage_data_snapshot()
            quota_results = await self.refresh_active_quota_baselines(
                credentials, usage_data
            )

        if not quota_results:
            return

        # Store new baselines in UsageManager
        stored = await self._store_baselines_to_usage_manager(
            quota_results, usage_manager
        )
        if stored > 0:
            lib_logger.debug(
                f"{provider_name} quota refresh: updated {stored} model baselines"
            )

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by providers
    # =========================================================================

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """Parse env:// credential path. Must be implemented by auth base."""
        raise NotImplementedError("Subclass must implement _parse_env_credential_path")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Get OAuth authorization header. Must be implemented by provider."""
        raise NotImplementedError("Subclass must implement get_auth_header")

    async def _discover_project_id(
        self, credential_path: str, access_token: str, litellm_params: Dict
    ) -> str:
        """Discover project ID for credential. Must be implemented by auth base."""
        raise NotImplementedError("Subclass must implement _discover_project_id")

    async def fetch_initial_baselines(
        self, credential_paths: List[str]
    ) -> Dict[str, Any]:
        """Fetch quota baselines for all credentials. Must be implemented by quota tracker."""
        raise NotImplementedError("Subclass must implement fetch_initial_baselines")

    async def refresh_active_quota_baselines(
        self, credentials: List[str], usage_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refresh quota for active credentials. Must be implemented by quota tracker."""
        raise NotImplementedError(
            "Subclass must implement refresh_active_quota_baselines"
        )

    async def _store_baselines_to_usage_manager(
        self, quota_results: Dict[str, Any], usage_manager: "UsageManager"
    ) -> int:
        """Store quota baselines to usage manager. Must be implemented by quota tracker."""
        raise NotImplementedError(
            "Subclass must implement _store_baselines_to_usage_manager"
        )
