import json
import os
import time
import logging
import asyncio
import random
from datetime import date, datetime, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import aiofiles
import litellm

from .error_handler import ClassifiedError, NoAvailableKeysError, mask_credential
from .providers import PROVIDER_PLUGINS
from .utils.resilient_io import ResilientStateWriter
from .utils.paths import get_data_file

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class UsageManager:
    """
    Manages usage statistics and cooldowns for API keys with asyncio-safe locking,
    asynchronous file I/O, lazy-loading mechanism, and weighted random credential rotation.

    The credential rotation strategy can be configured via the `rotation_tolerance` parameter:

    - **tolerance = 0.0**: Deterministic least-used selection. The credential with
      the lowest usage count is always selected. This provides predictable, perfectly balanced
      load distribution but may be vulnerable to fingerprinting.

    - **tolerance = 2.0 - 4.0 (default, recommended)**: Balanced weighted randomness. Credentials are selected
      randomly with weights biased toward less-used ones. Credentials within 2 uses of the
      maximum can still be selected with reasonable probability. This provides security through
      unpredictability while maintaining good load balance.

    - **tolerance = 5.0+**: High randomness. Even heavily-used credentials have significant
      selection probability. Useful for stress testing or maximum unpredictability, but may
      result in less balanced load distribution.

    The weight formula is: `weight = (max_usage - credential_usage) + tolerance + 1`

    This ensures lower-usage credentials are preferred while tolerance controls how much
    randomness is introduced into the selection process.

    Additionally, providers can specify a rotation mode:
    - "balanced" (default): Rotate credentials to distribute load evenly
    - "sequential": Use one credential until exhausted (preserves caching)
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        daily_reset_time_utc: Optional[str] = "03:00",
        rotation_tolerance: float = 0.0,
        provider_rotation_modes: Optional[Dict[str, str]] = None,
        provider_plugins: Optional[Dict[str, Any]] = None,
        priority_multipliers: Optional[Dict[str, Dict[int, int]]] = None,
        priority_multipliers_by_mode: Optional[
            Dict[str, Dict[str, Dict[int, int]]]
        ] = None,
        sequential_fallback_multipliers: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the UsageManager.

        Args:
            file_path: Path to the usage data JSON file. If None, uses get_data_file("key_usage.json").
                       Can be absolute Path, relative Path, or string.
            daily_reset_time_utc: Time in UTC when daily stats should reset (HH:MM format)
            rotation_tolerance: Tolerance for weighted random credential rotation.
                - 0.0: Deterministic, least-used credential always selected
                - tolerance = 2.0 - 4.0 (default, recommended): Balanced randomness, can pick credentials within 2 uses of max
                - 5.0+: High randomness, more unpredictable selection patterns
            provider_rotation_modes: Dict mapping provider names to rotation modes.
                - "balanced": Rotate credentials to distribute load evenly (default)
                - "sequential": Use one credential until exhausted (preserves caching)
            provider_plugins: Dict mapping provider names to provider plugin instances.
                Used for per-provider usage reset configuration (window durations, field names).
            priority_multipliers: Dict mapping provider -> priority -> multiplier.
                Universal multipliers that apply regardless of rotation mode.
                Example: {"antigravity": {1: 5, 2: 3}}
            priority_multipliers_by_mode: Dict mapping provider -> mode -> priority -> multiplier.
                Mode-specific overrides. Example: {"antigravity": {"balanced": {3: 1}}}
            sequential_fallback_multipliers: Dict mapping provider -> fallback multiplier.
                Used in sequential mode when priority not in priority_multipliers.
                Example: {"antigravity": 2}
        """
        # Resolve file_path - use default if not provided
        if file_path is None:
            self.file_path = str(get_data_file("key_usage.json"))
        elif isinstance(file_path, Path):
            self.file_path = str(file_path)
        else:
            # String path - could be relative or absolute
            self.file_path = file_path
        self.rotation_tolerance = rotation_tolerance
        self.provider_rotation_modes = provider_rotation_modes or {}
        self.provider_plugins = provider_plugins or PROVIDER_PLUGINS
        self.priority_multipliers = priority_multipliers or {}
        self.priority_multipliers_by_mode = priority_multipliers_by_mode or {}
        self.sequential_fallback_multipliers = sequential_fallback_multipliers or {}
        self._provider_instances: Dict[str, Any] = {}  # Cache for provider instances
        self.key_states: Dict[str, Dict[str, Any]] = {}

        self._data_lock = asyncio.Lock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()

        self._timeout_lock = asyncio.Lock()
        self._claimed_on_timeout: Set[str] = set()

        # Resilient writer for usage data persistence
        self._state_writer = ResilientStateWriter(file_path, lib_logger)

        if daily_reset_time_utc:
            hour, minute = map(int, daily_reset_time_utc.split(":"))
            self.daily_reset_time_utc = dt_time(
                hour=hour, minute=minute, tzinfo=timezone.utc
            )
        else:
            self.daily_reset_time_utc = None

    def _get_rotation_mode(self, provider: str) -> str:
        """
        Get the rotation mode for a provider.

        Args:
            provider: Provider name (e.g., "antigravity", "gemini_cli")

        Returns:
            "balanced" or "sequential"
        """
        return self.provider_rotation_modes.get(provider, "balanced")

    def _get_priority_multiplier(
        self, provider: str, priority: int, rotation_mode: str
    ) -> int:
        """
        Get the concurrency multiplier for a provider/priority/mode combination.

        Lookup order:
        1. Mode-specific tier override: priority_multipliers_by_mode[provider][mode][priority]
        2. Universal tier multiplier: priority_multipliers[provider][priority]
        3. Sequential fallback (if mode is sequential): sequential_fallback_multipliers[provider]
        4. Global default: 1 (no multiplier effect)

        Args:
            provider: Provider name (e.g., "antigravity")
            priority: Priority level (1 = highest priority)
            rotation_mode: Current rotation mode ("sequential" or "balanced")

        Returns:
            Multiplier value
        """
        provider_lower = provider.lower()

        # 1. Check mode-specific override
        if provider_lower in self.priority_multipliers_by_mode:
            mode_multipliers = self.priority_multipliers_by_mode[provider_lower]
            if rotation_mode in mode_multipliers:
                if priority in mode_multipliers[rotation_mode]:
                    return mode_multipliers[rotation_mode][priority]

        # 2. Check universal tier multiplier
        if provider_lower in self.priority_multipliers:
            if priority in self.priority_multipliers[provider_lower]:
                return self.priority_multipliers[provider_lower][priority]

        # 3. Sequential fallback (only for sequential mode)
        if rotation_mode == "sequential":
            if provider_lower in self.sequential_fallback_multipliers:
                return self.sequential_fallback_multipliers[provider_lower]

        # 4. Global default
        return 1

    def _get_provider_from_credential(self, credential: str) -> Optional[str]:
        """
        Extract provider name from credential path or identifier.

        Supports multiple credential formats:
        - OAuth: "oauth_creds/antigravity_oauth_15.json" -> "antigravity"
        - OAuth: "C:\\...\\oauth_creds\\gemini_cli_oauth_1.json" -> "gemini_cli"
        - OAuth filename only: "antigravity_oauth_1.json" -> "antigravity"
        - API key style: stored with provider prefix metadata

        Args:
            credential: The credential identifier (path or key)

        Returns:
            Provider name string or None if cannot be determined
        """
        import re

        # Normalize path separators
        normalized = credential.replace("\\", "/")

        # Pattern: path ending with {provider}_oauth_{number}.json
        match = re.search(r"/([a-z_]+)_oauth_\d+\.json$", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: oauth_creds/{provider}_...
        match = re.search(r"oauth_creds/([a-z_]+)_", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: filename only {provider}_oauth_{number}.json (no path)
        match = re.match(r"([a-z_]+)_oauth_\d+\.json$", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        return None

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """
        Get or create a provider plugin instance.

        Args:
            provider: The provider name

        Returns:
            Provider plugin instance or None
        """
        if not provider:
            return None

        plugin_class = self.provider_plugins.get(provider)
        if not plugin_class:
            return None

        # Get or create provider instance from cache
        if provider not in self._provider_instances:
            # Instantiate the plugin if it's a class, or use it directly if already an instance
            if isinstance(plugin_class, type):
                self._provider_instances[provider] = plugin_class()
            else:
                self._provider_instances[provider] = plugin_class

        return self._provider_instances[provider]

    def _get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Get the usage reset configuration for a credential from its provider plugin.

        Args:
            credential: The credential identifier

        Returns:
            Configuration dict with window_seconds, field_name, etc.
            or None to use default daily reset.
        """
        provider = self._get_provider_from_credential(credential)
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_usage_reset_config"):
            return plugin_instance.get_usage_reset_config(credential)

        return None

    def _get_reset_mode(self, credential: str) -> str:
        """
        Get the reset mode for a credential: 'credential' or 'per_model'.

        Args:
            credential: The credential identifier

        Returns:
            "per_model" or "credential" (default)
        """
        config = self._get_usage_reset_config(credential)
        return config.get("mode", "credential") if config else "credential"

    def _get_model_quota_group(self, credential: str, model: str) -> Optional[str]:
        """
        Get the quota group for a model, if the provider defines one.

        Args:
            credential: The credential identifier
            model: Model name (with or without provider prefix)

        Returns:
            Group name (e.g., "claude") or None if not grouped
        """
        provider = self._get_provider_from_credential(credential)
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_model_quota_group"):
            return plugin_instance.get_model_quota_group(model)

        return None

    def _get_grouped_models(self, credential: str, group: str) -> List[str]:
        """
        Get all model names in a quota group (with provider prefix).

        Args:
            credential: The credential identifier
            group: Group name (e.g., "claude")

        Returns:
            List of full model names (e.g., ["antigravity/claude-opus-4-5", ...])
        """
        provider = self._get_provider_from_credential(credential)
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_models_in_quota_group"):
            models = plugin_instance.get_models_in_quota_group(group)
            # Add provider prefix
            return [f"{provider}/{m}" for m in models]

        return []

    def _get_model_usage_weight(self, credential: str, model: str) -> int:
        """
        Get the usage weight for a model when calculating grouped usage.

        Args:
            credential: The credential identifier
            model: Model name (with or without provider prefix)

        Returns:
            Weight multiplier (default 1 if not configured)
        """
        provider = self._get_provider_from_credential(credential)
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_model_usage_weight"):
            return plugin_instance.get_model_usage_weight(model)

        return 1

    # Providers where request_count should be used for credential selection
    # instead of success_count (because failed requests also consume quota)
    _REQUEST_COUNT_PROVIDERS = {"antigravity"}

    def _get_grouped_usage_count(self, key: str, model: str) -> int:
        """
        Get usage count for credential selection, considering quota groups.

        For providers in _REQUEST_COUNT_PROVIDERS (e.g., antigravity), uses
        request_count instead of success_count since failed requests also
        consume quota.

        If the model belongs to a quota group, the request_count is already
        synced across all models in the group (by record_success/record_failure),
        so we just read from the requested model directly.

        Args:
            key: Credential identifier
            model: Model name (with provider prefix, e.g., "antigravity/claude-sonnet-4-5")

        Returns:
            Usage count for the model (synced across group if applicable)
        """
        # Determine usage field based on provider
        # Some providers (antigravity) count failed requests against quota
        provider = self._get_provider_from_credential(key)
        usage_field = (
            "request_count"
            if provider in self._REQUEST_COUNT_PROVIDERS
            else "success_count"
        )

        # For providers with synced quota groups (antigravity), request_count
        # is already synced across all models in the group, so just read directly.
        # For other providers, we still need to sum success_count across group.
        if provider in self._REQUEST_COUNT_PROVIDERS:
            # request_count is synced - just read the model's value
            return self._get_usage_count(key, model, usage_field)

        # For non-synced providers, check if model is in a quota group and sum
        group = self._get_model_quota_group(key, model)

        if group:
            # Get all models in the group
            grouped_models = self._get_grouped_models(key, group)

            # Sum weighted usage across all models in the group
            total_weighted_usage = 0
            for grouped_model in grouped_models:
                usage = self._get_usage_count(key, grouped_model, usage_field)
                weight = self._get_model_usage_weight(key, grouped_model)
                total_weighted_usage += usage * weight
            return total_weighted_usage

        # Not grouped - return individual model usage (no weight applied)
        return self._get_usage_count(key, model, usage_field)

    def _get_quota_display(self, key: str, model: str) -> str:
        """
        Get a formatted quota display string for logging.

        For antigravity (providers in _REQUEST_COUNT_PROVIDERS), returns:
            "quota: 170/250 [32%]" format

        For other providers, returns:
            "usage: 170" format (no max available)

        Args:
            key: Credential identifier
            model: Model name (with provider prefix)

        Returns:
            Formatted string for logging
        """
        provider = self._get_provider_from_credential(key)

        if provider not in self._REQUEST_COUNT_PROVIDERS:
            # Non-antigravity: just show usage count
            usage = self._get_usage_count(key, model, "success_count")
            return f"usage: {usage}"

        # Antigravity: show quota display with remaining percentage
        if self._usage_data is None:
            return "quota: 0/? [100%]"

        key_data = self._usage_data.get(key, {})
        model_data = key_data.get("models", {}).get(model, {})

        request_count = model_data.get("request_count", 0)
        max_requests = model_data.get("quota_max_requests")

        if max_requests:
            remaining = max_requests - request_count
            remaining_pct = (
                int((remaining / max_requests) * 100) if max_requests > 0 else 0
            )
            return f"quota: {request_count}/{max_requests} [{remaining_pct}%]"
        else:
            return f"quota: {request_count}"

    def _get_usage_field_name(self, credential: str) -> str:
        """
        Get the usage tracking field name for a credential.

        Returns the provider-specific field name if configured,
        otherwise falls back to "daily".

        Args:
            credential: The credential identifier

        Returns:
            Field name string (e.g., "5h_window", "weekly", "daily")
        """
        config = self._get_usage_reset_config(credential)
        if config and "field_name" in config:
            return config["field_name"]

        # Check provider default
        provider = self._get_provider_from_credential(credential)
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_default_usage_field_name"):
            return plugin_instance.get_default_usage_field_name()

        return "daily"

    def _get_usage_count(
        self, key: str, model: str, field: str = "success_count"
    ) -> int:
        """
        Get the current usage count for a model from the appropriate usage structure.

        Supports both:
        - New per-model structure: {"models": {"model_name": {"success_count": N, ...}}}
        - Legacy structure: {"daily": {"models": {"model_name": {"success_count": N, ...}}}}

        Args:
            key: Credential identifier
            model: Model name
            field: The field to read for usage count (default: "success_count").
                   Use "request_count" for providers where failed requests also
                   consume quota (e.g., antigravity).

        Returns:
            Usage count for the model in the current window/period
        """
        if self._usage_data is None:
            return 0

        key_data = self._usage_data.get(key, {})
        reset_mode = self._get_reset_mode(key)

        if reset_mode == "per_model":
            # New per-model structure: key_data["models"][model][field]
            return key_data.get("models", {}).get(model, {}).get(field, 0)
        else:
            # Legacy structure: key_data["daily"]["models"][model][field]
            return (
                key_data.get("daily", {}).get("models", {}).get(model, {}).get(field, 0)
            )

    # =========================================================================
    # TIMESTAMP FORMATTING HELPERS
    # =========================================================================

    def _format_timestamp_local(self, ts: Optional[float]) -> Optional[str]:
        """
        Format Unix timestamp as local time string with timezone offset.

        Args:
            ts: Unix timestamp or None

        Returns:
            Formatted string like "2025-12-07 14:30:17 +0100" or None
        """
        if ts is None:
            return None
        try:
            dt = datetime.fromtimestamp(ts).astimezone()  # Local timezone
            # Use UTC offset for conciseness (works on all platforms)
            return dt.strftime("%Y-%m-%d %H:%M:%S %z")
        except (OSError, ValueError, OverflowError):
            return None

    def _add_readable_timestamps(self, data: Dict) -> Dict:
        """
        Add human-readable timestamp fields to usage data before saving.

        Adds 'window_started' and 'quota_resets' fields derived from
        Unix timestamps for easier debugging and monitoring.

        Args:
            data: The usage data dict to enhance

        Returns:
            The same dict with readable timestamp fields added
        """
        for key, key_data in data.items():
            # Handle per-model structure
            models = key_data.get("models", {})
            for model_name, model_stats in models.items():
                if not isinstance(model_stats, dict):
                    continue

                # Add readable window start time
                window_start = model_stats.get("window_start_ts")
                if window_start:
                    model_stats["window_started"] = self._format_timestamp_local(
                        window_start
                    )
                elif "window_started" in model_stats:
                    del model_stats["window_started"]

                # Add readable reset time
                quota_reset = model_stats.get("quota_reset_ts")
                if quota_reset:
                    model_stats["quota_resets"] = self._format_timestamp_local(
                        quota_reset
                    )
                elif "quota_resets" in model_stats:
                    del model_stats["quota_resets"]

        return data

    def _sort_sequential(
        self,
        candidates: List[Tuple[str, int]],
        credential_priorities: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Sort credentials for sequential mode with position retention.

        Credentials maintain their position based on established usage patterns,
        ensuring that actively-used credentials remain primary until exhausted.

        Sorting order (within each sort key, lower value = higher priority):
        1. Priority tier (lower number = higher priority)
        2. Usage count (higher = more established in rotation, maintains position)
        3. Last used timestamp (higher = more recent, tiebreaker for stickiness)
        4. Credential ID (alphabetical, stable ordering)

        Args:
            candidates: List of (credential_id, usage_count) tuples
            credential_priorities: Optional dict mapping credentials to priority levels

        Returns:
            Sorted list of candidates (same format as input)
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        def sort_key(item: Tuple[str, int]) -> Tuple[int, int, float, str]:
            cred, usage_count = item
            priority = (
                credential_priorities.get(cred, 999) if credential_priorities else 999
            )
            last_used = (
                self._usage_data.get(cred, {}).get("last_used_ts", 0)
                if self._usage_data
                else 0
            )
            return (
                priority,  # ASC: lower priority number = higher priority
                -usage_count,  # DESC: higher usage = more established
                -last_used,  # DESC: more recent = preferred for ties
                cred,  # ASC: stable alphabetical ordering
            )

        sorted_candidates = sorted(candidates, key=sort_key)

        # Debug logging - show top 3 credentials in ordering
        if lib_logger.isEnabledFor(logging.DEBUG):
            order_info = [
                f"{mask_credential(c)}(p={credential_priorities.get(c, 999) if credential_priorities else 'N/A'}, u={u})"
                for c, u in sorted_candidates[:3]
            ]
            lib_logger.debug(f"Sequential ordering: {' → '.join(order_info)}")

        return sorted_candidates

    async def _lazy_init(self):
        """Initializes the usage data by loading it from the file asynchronously."""
        async with self._init_lock:
            if not self._initialized.is_set():
                await self._load_usage()
                await self._reset_daily_stats_if_needed()
                self._initialized.set()

    async def _load_usage(self):
        """Loads usage data from the JSON file asynchronously with resilience."""
        async with self._data_lock:
            if not os.path.exists(self.file_path):
                self._usage_data = {}
                return

            try:
                async with aiofiles.open(self.file_path, "r") as f:
                    content = await f.read()
                    self._usage_data = json.loads(content) if content.strip() else {}
            except FileNotFoundError:
                # File deleted between exists check and open
                self._usage_data = {}
            except json.JSONDecodeError as e:
                lib_logger.warning(
                    f"Corrupted usage file {self.file_path}: {e}. Starting fresh."
                )
                self._usage_data = {}
            except (OSError, PermissionError, IOError) as e:
                lib_logger.warning(
                    f"Cannot read usage file {self.file_path}: {e}. Using empty state."
                )
                self._usage_data = {}

    async def _save_usage(self):
        """Saves the current usage data using the resilient state writer."""
        if self._usage_data is None:
            return

        async with self._data_lock:
            # Add human-readable timestamp fields before saving
            self._add_readable_timestamps(self._usage_data)
            # Hand off to resilient writer - handles retries and disk failures
            self._state_writer.write(self._usage_data)

    async def _get_usage_data_snapshot(self) -> Dict[str, Any]:
        """
        Get a shallow copy of the current usage data.

        Returns:
            Copy of usage data dict (safe for reading without lock)
        """
        await self._lazy_init()
        async with self._data_lock:
            return dict(self._usage_data) if self._usage_data else {}

    async def get_available_credentials_for_model(
        self, credentials: List[str], model: str
    ) -> List[str]:
        """
        Get credentials that are not on cooldown for a specific model.

        Filters out credentials where:
        - key_cooldown_until > now (key-level cooldown)
        - model_cooldowns[model] > now (model-specific cooldown, includes quota exhausted)

        Args:
            credentials: List of credential identifiers to check
            model: Model name to check cooldowns for

        Returns:
            List of credentials that are available (not on cooldown) for this model
        """
        await self._lazy_init()
        now = time.time()
        available = []

        async with self._data_lock:
            for key in credentials:
                key_data = self._usage_data.get(key, {})

                # Skip if key-level cooldown is active
                if (key_data.get("key_cooldown_until") or 0) > now:
                    continue

                # Skip if model-specific cooldown is active
                if (key_data.get("model_cooldowns", {}).get(model) or 0) > now:
                    continue

                available.append(key)

        return available

    async def _reset_daily_stats_if_needed(self):
        """
        Checks if usage stats need to be reset for any key.

        Supports three reset modes:
        1. per_model: Each model has its own window, resets based on quota_reset_ts or fallback window
        2. credential: One window per credential (legacy with custom window duration)
        3. daily: Legacy daily reset at daily_reset_time_utc
        """
        if self._usage_data is None:
            return

        now_utc = datetime.now(timezone.utc)
        now_ts = time.time()
        today_str = now_utc.date().isoformat()
        needs_saving = False

        for key, data in self._usage_data.items():
            reset_config = self._get_usage_reset_config(key)

            if reset_config:
                reset_mode = reset_config.get("mode", "credential")

                if reset_mode == "per_model":
                    # Per-model window reset
                    needs_saving |= await self._check_per_model_resets(
                        key, data, reset_config, now_ts
                    )
                else:
                    # Credential-level window reset (legacy)
                    needs_saving |= await self._check_window_reset(
                        key, data, reset_config, now_ts
                    )
            elif self.daily_reset_time_utc:
                # Legacy daily reset
                needs_saving |= await self._check_daily_reset(
                    key, data, now_utc, today_str, now_ts
                )

        if needs_saving:
            await self._save_usage()

    async def _check_per_model_resets(
        self,
        key: str,
        data: Dict[str, Any],
        reset_config: Dict[str, Any],
        now_ts: float,
    ) -> bool:
        """
        Check and perform per-model resets for a credential.

        Each model resets independently based on:
        1. quota_reset_ts (authoritative, from quota exhausted error) if set
        2. window_start_ts + window_seconds (fallback) otherwise

        Grouped models reset together - all models in a group must be ready.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            reset_config: Provider's reset configuration
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        window_seconds = reset_config.get("window_seconds", 86400)
        models_data = data.get("models", {})

        if not models_data:
            return False

        modified = False
        processed_groups = set()

        for model, model_data in list(models_data.items()):
            # Check if this model is in a quota group
            group = self._get_model_quota_group(key, model)

            if group:
                if group in processed_groups:
                    continue  # Already handled this group

                # Check if entire group should reset
                if self._should_group_reset(
                    key, group, models_data, window_seconds, now_ts
                ):
                    # Archive and reset all models in group
                    grouped_models = self._get_grouped_models(key, group)
                    archived_count = 0

                    for grouped_model in grouped_models:
                        if grouped_model in models_data:
                            gm_data = models_data[grouped_model]
                            self._archive_model_to_global(data, grouped_model, gm_data)
                            self._reset_model_data(gm_data)
                            archived_count += 1

                    if archived_count > 0:
                        lib_logger.info(
                            f"Reset model group '{group}' ({archived_count} models) for {mask_credential(key)}"
                        )
                        modified = True

                processed_groups.add(group)

            else:
                # Ungrouped model - check individually
                if self._should_model_reset(model_data, window_seconds, now_ts):
                    self._archive_model_to_global(data, model, model_data)
                    self._reset_model_data(model_data)
                    lib_logger.info(f"Reset model {model} for {mask_credential(key)}")
                    modified = True

        # Preserve unexpired cooldowns
        if modified:
            self._preserve_unexpired_cooldowns(key, data, now_ts)
            if "failures" in data:
                data["failures"] = {}

        return modified

    def _should_model_reset(
        self, model_data: Dict[str, Any], window_seconds: int, now_ts: float
    ) -> bool:
        """
        Check if a single model should reset.

        Returns True if:
        - quota_reset_ts is set AND now >= quota_reset_ts, OR
        - quota_reset_ts is NOT set AND now >= window_start_ts + window_seconds
        """
        quota_reset = model_data.get("quota_reset_ts")
        window_start = model_data.get("window_start_ts")

        if quota_reset:
            return now_ts >= quota_reset
        elif window_start:
            return now_ts >= window_start + window_seconds
        return False

    def _should_group_reset(
        self,
        key: str,
        group: str,
        models_data: Dict[str, Dict],
        window_seconds: int,
        now_ts: float,
    ) -> bool:
        """
        Check if all models in a group should reset.

        All models in the group must be ready to reset.
        If any model has an active cooldown/window, the whole group waits.
        """
        grouped_models = self._get_grouped_models(key, group)

        # Track if any model in group has data
        any_has_data = False

        for grouped_model in grouped_models:
            model_data = models_data.get(grouped_model, {})

            if not model_data or (
                model_data.get("window_start_ts") is None
                and model_data.get("success_count", 0) == 0
            ):
                continue  # No stats for this model yet

            any_has_data = True

            if not self._should_model_reset(model_data, window_seconds, now_ts):
                return False  # At least one model not ready

        return any_has_data

    def _archive_model_to_global(
        self, data: Dict[str, Any], model: str, model_data: Dict[str, Any]
    ) -> None:
        """Archive a single model's stats to global."""
        global_data = data.setdefault("global", {"models": {}})
        global_model = global_data["models"].setdefault(
            model,
            {
                "success_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "approx_cost": 0.0,
            },
        )

        global_model["success_count"] += model_data.get("success_count", 0)
        global_model["prompt_tokens"] += model_data.get("prompt_tokens", 0)
        global_model["completion_tokens"] += model_data.get("completion_tokens", 0)
        global_model["approx_cost"] += model_data.get("approx_cost", 0.0)

    def _reset_model_data(self, model_data: Dict[str, Any]) -> None:
        """Reset a model's window and stats."""
        model_data["window_start_ts"] = None
        model_data["quota_reset_ts"] = None
        model_data["success_count"] = 0
        model_data["failure_count"] = 0
        model_data["request_count"] = 0
        model_data["prompt_tokens"] = 0
        model_data["completion_tokens"] = 0
        model_data["approx_cost"] = 0.0
        # Reset quota baseline fields only if they exist (Antigravity-specific)
        # These are added by update_quota_baseline(), only called for Antigravity
        if "baseline_remaining_fraction" in model_data:
            model_data["baseline_remaining_fraction"] = None
            model_data["baseline_fetched_at"] = None
            model_data["requests_at_baseline"] = None
            # Reset quota display but keep max_requests (it doesn't change between periods)
            max_req = model_data.get("quota_max_requests")
            if max_req:
                model_data["quota_display"] = f"0/{max_req}"

    async def _check_window_reset(
        self,
        key: str,
        data: Dict[str, Any],
        reset_config: Dict[str, Any],
        now_ts: float,
    ) -> bool:
        """
        Check and perform rolling window reset for a credential.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            reset_config: Provider's reset configuration
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        window_seconds = reset_config.get("window_seconds", 86400)  # Default 24h
        field_name = reset_config.get("field_name", "window")
        description = reset_config.get("description", "rolling window")

        # Get current window data
        window_data = data.get(field_name, {})
        window_start = window_data.get("start_ts")

        # No window started yet - nothing to reset
        if window_start is None:
            return False

        # Check if window has expired
        window_end = window_start + window_seconds
        if now_ts < window_end:
            # Window still active
            return False

        # Window expired - perform reset
        hours_elapsed = (now_ts - window_start) / 3600
        lib_logger.info(
            f"Resetting {field_name} for {mask_credential(key)} - "
            f"{description} expired after {hours_elapsed:.1f}h"
        )

        # Archive to global
        self._archive_to_global(data, window_data)

        # Preserve unexpired cooldowns
        self._preserve_unexpired_cooldowns(key, data, now_ts)

        # Reset window stats (but don't start new window until first request)
        data[field_name] = {"start_ts": None, "models": {}}

        # Reset consecutive failures
        if "failures" in data:
            data["failures"] = {}

        return True

    async def _check_daily_reset(
        self,
        key: str,
        data: Dict[str, Any],
        now_utc: datetime,
        today_str: str,
        now_ts: float,
    ) -> bool:
        """
        Check and perform legacy daily reset for a credential.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            now_utc: Current datetime in UTC
            today_str: Today's date as ISO string
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        last_reset_str = data.get("last_daily_reset", "")

        if last_reset_str == today_str:
            return False

        last_reset_dt = None
        if last_reset_str:
            try:
                last_reset_dt = datetime.fromisoformat(last_reset_str).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        # Determine the reset threshold for today
        reset_threshold_today = datetime.combine(
            now_utc.date(), self.daily_reset_time_utc
        )

        if not (
            last_reset_dt is None or last_reset_dt < reset_threshold_today <= now_utc
        ):
            return False

        lib_logger.debug(f"Performing daily reset for key {mask_credential(key)}")

        # Preserve unexpired cooldowns
        self._preserve_unexpired_cooldowns(key, data, now_ts)

        # Reset consecutive failures
        if "failures" in data:
            data["failures"] = {}

        # Archive daily stats to global
        daily_data = data.get("daily", {})
        if daily_data:
            self._archive_to_global(data, daily_data)

        # Reset daily stats
        data["daily"] = {"date": today_str, "models": {}}
        data["last_daily_reset"] = today_str

        return True

    def _archive_to_global(
        self, data: Dict[str, Any], source_data: Dict[str, Any]
    ) -> None:
        """
        Archive usage stats from a source field (daily/window) to global.

        Args:
            data: The credential's usage data
            source_data: The source field data to archive (has "models" key)
        """
        global_data = data.setdefault("global", {"models": {}})
        for model, stats in source_data.get("models", {}).items():
            global_model_stats = global_data["models"].setdefault(
                model,
                {
                    "success_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "approx_cost": 0.0,
                },
            )
            global_model_stats["success_count"] += stats.get("success_count", 0)
            global_model_stats["prompt_tokens"] += stats.get("prompt_tokens", 0)
            global_model_stats["completion_tokens"] += stats.get("completion_tokens", 0)
            global_model_stats["approx_cost"] += stats.get("approx_cost", 0.0)

    def _preserve_unexpired_cooldowns(
        self, key: str, data: Dict[str, Any], now_ts: float
    ) -> None:
        """
        Preserve unexpired cooldowns during reset (important for long quota cooldowns).

        Args:
            key: Credential identifier (for logging)
            data: The credential's usage data
            now_ts: Current timestamp
        """
        # Preserve unexpired model cooldowns
        if "model_cooldowns" in data:
            active_cooldowns = {
                model: end_time
                for model, end_time in data["model_cooldowns"].items()
                if end_time > now_ts
            }
            if active_cooldowns:
                max_remaining = max(
                    end_time - now_ts for end_time in active_cooldowns.values()
                )
                hours_remaining = max_remaining / 3600
                lib_logger.info(
                    f"Preserving {len(active_cooldowns)} active cooldown(s) "
                    f"for key {mask_credential(key)} during reset "
                    f"(longest: {hours_remaining:.1f}h remaining)"
                )
            data["model_cooldowns"] = active_cooldowns
        else:
            data["model_cooldowns"] = {}

        # Preserve unexpired key-level cooldown
        if data.get("key_cooldown_until"):
            if data["key_cooldown_until"] <= now_ts:
                data["key_cooldown_until"] = None
            else:
                hours_remaining = (data["key_cooldown_until"] - now_ts) / 3600
                lib_logger.info(
                    f"Preserving key-level cooldown for {mask_credential(key)} "
                    f"during reset ({hours_remaining:.1f}h remaining)"
                )
        else:
            data["key_cooldown_until"] = None

    def _initialize_key_states(self, keys: List[str]):
        """Initializes state tracking for all provided keys if not already present."""
        for key in keys:
            if key not in self.key_states:
                self.key_states[key] = {
                    "lock": asyncio.Lock(),
                    "condition": asyncio.Condition(),
                    "models_in_use": {},  # Dict[model_name, concurrent_count]
                }

    def _select_weighted_random(self, candidates: List[tuple], tolerance: float) -> str:
        """
        Selects a credential using weighted random selection based on usage counts.

        Args:
            candidates: List of (credential_id, usage_count) tuples
            tolerance: Tolerance value for weight calculation

        Returns:
            Selected credential ID

        Formula:
            weight = (max_usage - credential_usage) + tolerance + 1

        This formula ensures:
            - Lower usage = higher weight = higher selection probability
            - Tolerance adds variability: higher tolerance means more randomness
            - The +1 ensures all credentials have at least some chance of selection
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        if len(candidates) == 1:
            return candidates[0][0]

        # Extract usage counts
        usage_counts = [usage for _, usage in candidates]
        max_usage = max(usage_counts)

        # Calculate weights using the formula: (max - current) + tolerance + 1
        weights = []
        for credential, usage in candidates:
            weight = (max_usage - usage) + tolerance + 1
            weights.append(weight)

        # Log weight distribution for debugging
        if lib_logger.isEnabledFor(logging.DEBUG):
            total_weight = sum(weights)
            weight_info = ", ".join(
                f"{mask_credential(cred)}: w={w:.1f} ({w / total_weight * 100:.1f}%)"
                for (cred, _), w in zip(candidates, weights)
            )
            # lib_logger.debug(f"Weighted selection candidates: {weight_info}")

        # Random selection with weights
        selected_credential = random.choices(
            [cred for cred, _ in candidates], weights=weights, k=1
        )[0]

        return selected_credential

    async def acquire_key(
        self,
        available_keys: List[str],
        model: str,
        deadline: float,
        max_concurrent: int = 1,
        credential_priorities: Optional[Dict[str, int]] = None,
        credential_tier_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Acquires the best available key using a tiered, model-aware locking strategy,
        respecting a global deadline and credential priorities.

        Priority Logic:
        - Groups credentials by priority level (1=highest, 2=lower, etc.)
        - Always tries highest priority (lowest number) first
        - Within same priority, sorts by usage count (load balancing)
        - Only moves to next priority if all higher-priority keys exhausted/busy

        Args:
            available_keys: List of credential identifiers to choose from
            model: Model name being requested
            deadline: Timestamp after which to stop trying
            max_concurrent: Maximum concurrent requests allowed per credential
            credential_priorities: Optional dict mapping credentials to priority levels (1=highest)
            credential_tier_names: Optional dict mapping credentials to tier names (for logging)

        Returns:
            Selected credential identifier

        Raises:
            NoAvailableKeysError: If no key could be acquired within the deadline
        """
        await self._lazy_init()
        await self._reset_daily_stats_if_needed()
        self._initialize_key_states(available_keys)

        # This loop continues as long as the global deadline has not been met.
        while time.time() < deadline:
            now = time.time()

            # Group credentials by priority level (if priorities provided)
            if credential_priorities:
                # Group keys by priority level
                priority_groups = {}
                async with self._data_lock:
                    for key in available_keys:
                        key_data = self._usage_data.get(key, {})

                        # Skip keys on cooldown
                        if (key_data.get("key_cooldown_until") or 0) > now or (
                            key_data.get("model_cooldowns", {}).get(model) or 0
                        ) > now:
                            continue

                        # Get priority for this key (default to 999 if not specified)
                        priority = credential_priorities.get(key, 999)

                        # Get usage count for load balancing within priority groups
                        # Uses grouped usage if model is in a quota group
                        usage_count = self._get_grouped_usage_count(key, model)

                        # Group by priority
                        if priority not in priority_groups:
                            priority_groups[priority] = []
                        priority_groups[priority].append((key, usage_count))

                # Try priority groups in order (1, 2, 3, ...)
                sorted_priorities = sorted(priority_groups.keys())

                for priority_level in sorted_priorities:
                    keys_in_priority = priority_groups[priority_level]

                    # Determine selection method based on provider's rotation mode
                    provider = model.split("/")[0] if "/" in model else ""
                    rotation_mode = self._get_rotation_mode(provider)

                    # Calculate effective concurrency based on priority tier
                    multiplier = self._get_priority_multiplier(
                        provider, priority_level, rotation_mode
                    )
                    effective_max_concurrent = max_concurrent * multiplier

                    # Within each priority group, use existing tier1/tier2 logic
                    tier1_keys, tier2_keys = [], []
                    for key, usage_count in keys_in_priority:
                        key_state = self.key_states[key]

                        # Tier 1: Completely idle keys (preferred)
                        if not key_state["models_in_use"]:
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Keys that can accept more concurrent requests
                        elif (
                            key_state["models_in_use"].get(model, 0)
                            < effective_max_concurrent
                        ):
                            tier2_keys.append((key, usage_count))

                    if rotation_mode == "sequential":
                        # Sequential mode: sort credentials by priority, usage, recency
                        # Keep all candidates in sorted order (no filtering to single key)
                        selection_method = "sequential"
                        if tier1_keys:
                            tier1_keys = self._sort_sequential(
                                tier1_keys, credential_priorities
                            )
                        if tier2_keys:
                            tier2_keys = self._sort_sequential(
                                tier2_keys, credential_priorities
                            )
                    elif self.rotation_tolerance > 0:
                        # Balanced mode with weighted randomness
                        selection_method = "weighted-random"
                        if tier1_keys:
                            selected_key = self._select_weighted_random(
                                tier1_keys, self.rotation_tolerance
                            )
                            tier1_keys = [
                                (k, u) for k, u in tier1_keys if k == selected_key
                            ]
                        if tier2_keys:
                            selected_key = self._select_weighted_random(
                                tier2_keys, self.rotation_tolerance
                            )
                            tier2_keys = [
                                (k, u) for k, u in tier2_keys if k == selected_key
                            ]
                    else:
                        # Deterministic: sort by usage within each tier
                        selection_method = "least-used"
                        tier1_keys.sort(key=lambda x: x[1])
                        tier2_keys.sort(key=lambda x: x[1])

                    # Try to acquire from Tier 1 first
                    for key, usage in tier1_keys:
                        state = self.key_states[key]
                        async with state["lock"]:
                            if not state["models_in_use"]:
                                state["models_in_use"][model] = 1
                                tier_name = (
                                    credential_tier_names.get(key, "unknown")
                                    if credential_tier_names
                                    else "unknown"
                                )
                                quota_display = self._get_quota_display(key, model)
                                lib_logger.info(
                                    f"Acquired key {mask_credential(key)} for model {model} "
                                    f"(tier: {tier_name}, priority: {priority_level}, selection: {selection_method}, {quota_display})"
                                )
                                return key

                    # Then try Tier 2
                    for key, usage in tier2_keys:
                        state = self.key_states[key]
                        async with state["lock"]:
                            current_count = state["models_in_use"].get(model, 0)
                            if current_count < effective_max_concurrent:
                                state["models_in_use"][model] = current_count + 1
                                tier_name = (
                                    credential_tier_names.get(key, "unknown")
                                    if credential_tier_names
                                    else "unknown"
                                )
                                quota_display = self._get_quota_display(key, model)
                                lib_logger.info(
                                    f"Acquired key {mask_credential(key)} for model {model} "
                                    f"(tier: {tier_name}, priority: {priority_level}, selection: {selection_method}, concurrent: {state['models_in_use'][model]}/{effective_max_concurrent}, {quota_display})"
                                )
                                return key

                # If we get here, all priority groups were exhausted but keys might become available
                # Collect all keys across all priorities for waiting
                all_potential_keys = []
                for keys_list in priority_groups.values():
                    all_potential_keys.extend(keys_list)

                if not all_potential_keys:
                    lib_logger.warning(
                        "No keys are eligible (all on cooldown or filtered out). Waiting before re-evaluating."
                    )
                    await asyncio.sleep(1)
                    continue

                # Wait for the highest priority key with lowest usage
                best_priority = min(priority_groups.keys())
                best_priority_keys = priority_groups[best_priority]
                best_wait_key = min(best_priority_keys, key=lambda x: x[1])[0]
                wait_condition = self.key_states[best_wait_key]["condition"]

                lib_logger.info(
                    f"All Priority-{best_priority} keys are busy. Waiting for highest priority credential to become available..."
                )

            else:
                # Original logic when no priorities specified

                # Determine selection method based on provider's rotation mode
                provider = model.split("/")[0] if "/" in model else ""
                rotation_mode = self._get_rotation_mode(provider)

                # Calculate effective concurrency for default priority (999)
                # When no priorities are specified, all credentials get default priority
                default_priority = 999
                multiplier = self._get_priority_multiplier(
                    provider, default_priority, rotation_mode
                )
                effective_max_concurrent = max_concurrent * multiplier

                tier1_keys, tier2_keys = [], []

                # First, filter the list of available keys to exclude any on cooldown.
                async with self._data_lock:
                    for key in available_keys:
                        key_data = self._usage_data.get(key, {})

                        if (key_data.get("key_cooldown_until") or 0) > now or (
                            key_data.get("model_cooldowns", {}).get(model) or 0
                        ) > now:
                            continue

                        # Prioritize keys based on their current usage to ensure load balancing.
                        # Uses grouped usage if model is in a quota group
                        usage_count = self._get_grouped_usage_count(key, model)
                        key_state = self.key_states[key]

                        # Tier 1: Completely idle keys (preferred).
                        if not key_state["models_in_use"]:
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Keys that can accept more concurrent requests for this model.
                        elif (
                            key_state["models_in_use"].get(model, 0)
                            < effective_max_concurrent
                        ):
                            tier2_keys.append((key, usage_count))

                if rotation_mode == "sequential":
                    # Sequential mode: sort credentials by priority, usage, recency
                    # Keep all candidates in sorted order (no filtering to single key)
                    selection_method = "sequential"
                    if tier1_keys:
                        tier1_keys = self._sort_sequential(
                            tier1_keys, credential_priorities
                        )
                    if tier2_keys:
                        tier2_keys = self._sort_sequential(
                            tier2_keys, credential_priorities
                        )
                elif self.rotation_tolerance > 0:
                    # Balanced mode with weighted randomness
                    selection_method = "weighted-random"
                    if tier1_keys:
                        selected_key = self._select_weighted_random(
                            tier1_keys, self.rotation_tolerance
                        )
                        tier1_keys = [
                            (k, u) for k, u in tier1_keys if k == selected_key
                        ]
                    if tier2_keys:
                        selected_key = self._select_weighted_random(
                            tier2_keys, self.rotation_tolerance
                        )
                        tier2_keys = [
                            (k, u) for k, u in tier2_keys if k == selected_key
                        ]
                else:
                    # Deterministic: sort by usage within each tier
                    selection_method = "least-used"
                    tier1_keys.sort(key=lambda x: x[1])
                    tier2_keys.sort(key=lambda x: x[1])

                # Attempt to acquire a key from Tier 1 first.
                for key, usage in tier1_keys:
                    state = self.key_states[key]
                    async with state["lock"]:
                        if not state["models_in_use"]:
                            state["models_in_use"][model] = 1
                            tier_name = (
                                credential_tier_names.get(key)
                                if credential_tier_names
                                else None
                            )
                            tier_info = f"tier: {tier_name}, " if tier_name else ""
                            quota_display = self._get_quota_display(key, model)
                            lib_logger.info(
                                f"Acquired key {mask_credential(key)} for model {model} "
                                f"({tier_info}selection: {selection_method}, {quota_display})"
                            )
                            return key

                # If no Tier 1 keys are available, try Tier 2.
                for key, usage in tier2_keys:
                    state = self.key_states[key]
                    async with state["lock"]:
                        current_count = state["models_in_use"].get(model, 0)
                        if current_count < effective_max_concurrent:
                            state["models_in_use"][model] = current_count + 1
                            tier_name = (
                                credential_tier_names.get(key)
                                if credential_tier_names
                                else None
                            )
                            tier_info = f"tier: {tier_name}, " if tier_name else ""
                            quota_display = self._get_quota_display(key, model)
                            lib_logger.info(
                                f"Acquired key {mask_credential(key)} for model {model} "
                                f"({tier_info}selection: {selection_method}, concurrent: {state['models_in_use'][model]}/{effective_max_concurrent}, {quota_display})"
                            )
                            return key

                # If all eligible keys are locked, wait for a key to be released.
                lib_logger.info(
                    "All eligible keys are currently locked for this model. Waiting..."
                )

                all_potential_keys = tier1_keys + tier2_keys
                if not all_potential_keys:
                    lib_logger.warning(
                        "No keys are eligible (all on cooldown). Waiting before re-evaluating."
                    )
                    await asyncio.sleep(1)
                    continue

                # Wait on the condition of the key with the lowest current usage.
                best_wait_key = min(all_potential_keys, key=lambda x: x[1])[0]
                wait_condition = self.key_states[best_wait_key]["condition"]

            try:
                async with wait_condition:
                    remaining_budget = deadline - time.time()
                    if remaining_budget <= 0:
                        break  # Exit if the budget has already been exceeded.
                    # Wait for a notification, but no longer than the remaining budget or 1 second.
                    await asyncio.wait_for(
                        wait_condition.wait(), timeout=min(1, remaining_budget)
                    )
                lib_logger.info("Notified that a key was released. Re-evaluating...")
            except asyncio.TimeoutError:
                # This is not an error, just a timeout for the wait. The main loop will re-evaluate.
                lib_logger.info("Wait timed out. Re-evaluating for any available key.")

        # If the loop exits, it means the deadline was exceeded.
        raise NoAvailableKeysError(
            f"Could not acquire a key for model {model} within the global time budget."
        )

    async def release_key(self, key: str, model: str):
        """Releases a key's lock for a specific model and notifies waiting tasks."""
        if key not in self.key_states:
            return

        state = self.key_states[key]
        async with state["lock"]:
            if model in state["models_in_use"]:
                state["models_in_use"][model] -= 1
                remaining = state["models_in_use"][model]
                if remaining <= 0:
                    del state["models_in_use"][model]  # Clean up when count reaches 0
                lib_logger.info(
                    f"Released credential {mask_credential(key)} from model {model} "
                    f"(remaining concurrent: {max(0, remaining)})"
                )
            else:
                lib_logger.warning(
                    f"Attempted to release credential {mask_credential(key)} for model {model}, but it was not in use."
                )

        # Notify all tasks waiting on this key's condition
        async with state["condition"]:
            state["condition"].notify_all()

    async def record_success(
        self,
        key: str,
        model: str,
        completion_response: Optional[litellm.ModelResponse] = None,
    ):
        """
        Records a successful API call, resetting failure counters.
        It safely handles cases where token usage data is not available.

        Supports two modes based on provider configuration:
        - per_model: Each model has its own window_start_ts and stats in key_data["models"]
        - credential: Legacy mode with key_data["daily"]["models"]
        """
        await self._lazy_init()
        async with self._data_lock:
            now_ts = time.time()
            today_utc_str = datetime.now(timezone.utc).date().isoformat()

            reset_config = self._get_usage_reset_config(key)
            reset_mode = (
                reset_config.get("mode", "credential") if reset_config else "credential"
            )

            if reset_mode == "per_model":
                # New per-model structure
                key_data = self._usage_data.setdefault(
                    key,
                    {
                        "models": {},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )

                # Ensure models dict exists
                if "models" not in key_data:
                    key_data["models"] = {}

                # Get or create per-model data with window tracking
                model_data = key_data["models"].setdefault(
                    model,
                    {
                        "window_start_ts": None,
                        "quota_reset_ts": None,
                        "success_count": 0,
                        "failure_count": 0,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "approx_cost": 0.0,
                    },
                )

                # Start window on first request for this model
                if model_data.get("window_start_ts") is None:
                    model_data["window_start_ts"] = now_ts

                    # Set expected quota reset time from provider config
                    window_seconds = (
                        reset_config.get("window_seconds", 0) if reset_config else 0
                    )
                    if window_seconds > 0:
                        model_data["quota_reset_ts"] = now_ts + window_seconds

                    window_hours = window_seconds / 3600 if window_seconds else 0
                    lib_logger.info(
                        f"Started {window_hours:.1f}h window for model {model} on {mask_credential(key)}"
                    )

                # Record stats
                model_data["success_count"] += 1
                model_data["request_count"] = model_data.get("request_count", 0) + 1

                # Sync request_count across quota group (for providers with shared quota pools)
                new_request_count = model_data["request_count"]
                group = self._get_model_quota_group(key, model)
                if group:
                    grouped_models = self._get_grouped_models(key, group)
                    for grouped_model in grouped_models:
                        if grouped_model != model:
                            other_model_data = key_data["models"].setdefault(
                                grouped_model,
                                {
                                    "window_start_ts": None,
                                    "quota_reset_ts": None,
                                    "success_count": 0,
                                    "failure_count": 0,
                                    "request_count": 0,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "approx_cost": 0.0,
                                },
                            )
                            other_model_data["request_count"] = new_request_count
                            # Also sync quota_max_requests if set
                            max_req = model_data.get("quota_max_requests")
                            if max_req:
                                other_model_data["quota_max_requests"] = max_req
                                other_model_data["quota_display"] = (
                                    f"{new_request_count}/{max_req}"
                                )

                # Update quota_display if max_requests is set (Antigravity-specific)
                max_req = model_data.get("quota_max_requests")
                if max_req:
                    model_data["quota_display"] = (
                        f"{model_data['request_count']}/{max_req}"
                    )

                usage_data_ref = model_data  # For token/cost recording below

            else:
                # Legacy credential-level structure
                key_data = self._usage_data.setdefault(
                    key,
                    {
                        "daily": {"date": today_utc_str, "models": {}},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )

                if "last_daily_reset" not in key_data:
                    key_data["last_daily_reset"] = today_utc_str

                # Get or create model data in daily structure
                usage_data_ref = key_data["daily"]["models"].setdefault(
                    model,
                    {
                        "success_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "approx_cost": 0.0,
                    },
                )
                usage_data_ref["success_count"] += 1

            # Reset failures for this model
            model_failures = key_data.setdefault("failures", {}).setdefault(model, {})
            model_failures["consecutive_failures"] = 0

            # Clear transient cooldown on success (but NOT quota_reset_ts)
            if model in key_data.get("model_cooldowns", {}):
                del key_data["model_cooldowns"][model]

            # Record token and cost usage
            if (
                completion_response
                and hasattr(completion_response, "usage")
                and completion_response.usage
            ):
                usage = completion_response.usage
                usage_data_ref["prompt_tokens"] += usage.prompt_tokens
                usage_data_ref["completion_tokens"] += getattr(
                    usage, "completion_tokens", 0
                )
                lib_logger.info(
                    f"Recorded usage from response object for key {mask_credential(key)}"
                )
                try:
                    provider_name = model.split("/")[0]
                    provider_instance = self._get_provider_instance(provider_name)

                    if provider_instance and getattr(
                        provider_instance, "skip_cost_calculation", False
                    ):
                        lib_logger.debug(
                            f"Skipping cost calculation for provider '{provider_name}' (custom provider)."
                        )
                    else:
                        if isinstance(completion_response, litellm.EmbeddingResponse):
                            model_info = litellm.get_model_info(model)
                            input_cost = model_info.get("input_cost_per_token")
                            if input_cost:
                                cost = (
                                    completion_response.usage.prompt_tokens * input_cost
                                )
                            else:
                                cost = None
                        else:
                            cost = litellm.completion_cost(
                                completion_response=completion_response, model=model
                            )

                        if cost is not None:
                            usage_data_ref["approx_cost"] += cost
                except Exception as e:
                    lib_logger.warning(
                        f"Could not calculate cost for model {model}: {e}"
                    )
            elif isinstance(completion_response, asyncio.Future) or hasattr(
                completion_response, "__aiter__"
            ):
                pass  # Stream - usage recorded from chunks
            else:
                lib_logger.warning(
                    f"No usage data found in completion response for model {model}. Recording success without token count."
                )

            key_data["last_used_ts"] = now_ts

        await self._save_usage()

    async def record_failure(
        self,
        key: str,
        model: str,
        classified_error: ClassifiedError,
        increment_consecutive_failures: bool = True,
    ):
        """Records a failure and applies cooldowns based on error type.

        Distinguishes between:
        - quota_exceeded: Long cooldown with exact reset time (from quota_reset_timestamp)
          Sets quota_reset_ts on model (and group) - this becomes authoritative stats reset time
        - rate_limit: Short transient cooldown (just wait and retry)
          Only sets model_cooldowns - does NOT affect stats reset timing

        Args:
            key: The API key or credential identifier
            model: The model name
            classified_error: The classified error object
            increment_consecutive_failures: Whether to increment the failure counter.
                Set to False for provider-level errors that shouldn't count against the key.
        """
        await self._lazy_init()
        async with self._data_lock:
            now_ts = time.time()
            today_utc_str = datetime.now(timezone.utc).date().isoformat()

            reset_config = self._get_usage_reset_config(key)
            reset_mode = (
                reset_config.get("mode", "credential") if reset_config else "credential"
            )

            # Initialize key data with appropriate structure
            if reset_mode == "per_model":
                key_data = self._usage_data.setdefault(
                    key,
                    {
                        "models": {},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )
            else:
                key_data = self._usage_data.setdefault(
                    key,
                    {
                        "daily": {"date": today_utc_str, "models": {}},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )

            # Provider-level errors (transient issues) should not count against the key
            provider_level_errors = {"server_error", "api_connection"}

            # Determine if we should increment the failure counter
            should_increment = (
                increment_consecutive_failures
                and classified_error.error_type not in provider_level_errors
            )

            # Calculate cooldown duration based on error type
            cooldown_seconds = None
            model_cooldowns = key_data.setdefault("model_cooldowns", {})

            if classified_error.error_type == "quota_exceeded":
                # Quota exhausted - use authoritative reset timestamp if available
                quota_reset_ts = classified_error.quota_reset_timestamp
                cooldown_seconds = classified_error.retry_after or 60

                if quota_reset_ts and reset_mode == "per_model":
                    # Set quota_reset_ts on model - this becomes authoritative stats reset time
                    models_data = key_data.setdefault("models", {})
                    model_data = models_data.setdefault(
                        model,
                        {
                            "window_start_ts": None,
                            "quota_reset_ts": None,
                            "success_count": 0,
                            "failure_count": 0,
                            "request_count": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "approx_cost": 0.0,
                        },
                    )
                    model_data["quota_reset_ts"] = quota_reset_ts
                    # Track failure for quota estimation (request still consumes quota)
                    model_data["failure_count"] = model_data.get("failure_count", 0) + 1
                    model_data["request_count"] = model_data.get("request_count", 0) + 1
                    new_request_count = model_data["request_count"]

                    # Apply to all models in the same quota group
                    group = self._get_model_quota_group(key, model)
                    if group:
                        grouped_models = self._get_grouped_models(key, group)
                        for grouped_model in grouped_models:
                            group_model_data = models_data.setdefault(
                                grouped_model,
                                {
                                    "window_start_ts": None,
                                    "quota_reset_ts": None,
                                    "success_count": 0,
                                    "failure_count": 0,
                                    "request_count": 0,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "approx_cost": 0.0,
                                },
                            )
                            group_model_data["quota_reset_ts"] = quota_reset_ts
                            # Sync request_count across quota group
                            group_model_data["request_count"] = new_request_count
                            # Also sync quota_max_requests if set
                            max_req = model_data.get("quota_max_requests")
                            if max_req:
                                group_model_data["quota_max_requests"] = max_req
                                group_model_data["quota_display"] = (
                                    f"{new_request_count}/{max_req}"
                                )
                            # Also set transient cooldown for selection logic
                            model_cooldowns[grouped_model] = quota_reset_ts

                        reset_dt = datetime.fromtimestamp(
                            quota_reset_ts, tz=timezone.utc
                        )
                        lib_logger.info(
                            f"Quota exhausted for group '{group}' ({len(grouped_models)} models) "
                            f"on {mask_credential(key)}. Resets at {reset_dt.isoformat()}"
                        )
                    else:
                        reset_dt = datetime.fromtimestamp(
                            quota_reset_ts, tz=timezone.utc
                        )
                        hours = (quota_reset_ts - now_ts) / 3600
                        lib_logger.info(
                            f"Quota exhausted for model {model} on {mask_credential(key)}. "
                            f"Resets at {reset_dt.isoformat()} ({hours:.1f}h)"
                        )

                    # Set transient cooldown for selection logic
                    model_cooldowns[model] = quota_reset_ts
                else:
                    # No authoritative timestamp or legacy mode - just use retry_after
                    model_cooldowns[model] = now_ts + cooldown_seconds
                    hours = cooldown_seconds / 3600
                    lib_logger.info(
                        f"Quota exhausted on {mask_credential(key)} for model {model}. "
                        f"Cooldown: {cooldown_seconds}s ({hours:.1f}h)"
                    )

            elif classified_error.error_type == "rate_limit":
                # Transient rate limit - just set short cooldown (does NOT set quota_reset_ts)
                cooldown_seconds = classified_error.retry_after or 60
                model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.info(
                    f"Rate limit on {mask_credential(key)} for model {model}. "
                    f"Transient cooldown: {cooldown_seconds}s"
                )

            elif classified_error.error_type == "authentication":
                # Apply a 5-minute key-level lockout for auth errors
                key_data["key_cooldown_until"] = now_ts + 300
                cooldown_seconds = 300
                model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.warning(
                    f"Authentication error on key {mask_credential(key)}. Applying 5-minute key-level lockout."
                )

            # If we should increment failures, calculate escalating backoff
            if should_increment:
                failures_data = key_data.setdefault("failures", {})
                model_failures = failures_data.setdefault(
                    model, {"consecutive_failures": 0}
                )
                model_failures["consecutive_failures"] += 1
                count = model_failures["consecutive_failures"]

                # If cooldown wasn't set by specific error type, use escalating backoff
                if cooldown_seconds is None:
                    backoff_tiers = {1: 10, 2: 30, 3: 60, 4: 120}
                    cooldown_seconds = backoff_tiers.get(count, 7200)
                    model_cooldowns[model] = now_ts + cooldown_seconds
                    lib_logger.warning(
                        f"Failure #{count} for key {mask_credential(key)} with model {model}. "
                        f"Error type: {classified_error.error_type}, cooldown: {cooldown_seconds}s"
                    )
            else:
                # Provider-level errors: apply short cooldown but don't count against key
                if cooldown_seconds is None:
                    cooldown_seconds = 30
                    model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.info(
                    f"Provider-level error ({classified_error.error_type}) for key {mask_credential(key)} "
                    f"with model {model}. NOT incrementing failures. Cooldown: {cooldown_seconds}s"
                )

            # Check for key-level lockout condition
            await self._check_key_lockout(key, key_data)

            # Track failure count for quota estimation (all failures consume quota)
            # This is separate from consecutive_failures which is for backoff logic
            if reset_mode == "per_model":
                models_data = key_data.setdefault("models", {})
                model_data = models_data.setdefault(
                    model,
                    {
                        "window_start_ts": None,
                        "quota_reset_ts": None,
                        "success_count": 0,
                        "failure_count": 0,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "approx_cost": 0.0,
                    },
                )
                # Only increment if not already incremented in quota_exceeded branch
                if classified_error.error_type != "quota_exceeded":
                    model_data["failure_count"] = model_data.get("failure_count", 0) + 1
                    model_data["request_count"] = model_data.get("request_count", 0) + 1

                    # Sync request_count across quota group
                    new_request_count = model_data["request_count"]
                    group = self._get_model_quota_group(key, model)
                    if group:
                        grouped_models = self._get_grouped_models(key, group)
                        for grouped_model in grouped_models:
                            if grouped_model != model:
                                other_model_data = models_data.setdefault(
                                    grouped_model,
                                    {
                                        "window_start_ts": None,
                                        "quota_reset_ts": None,
                                        "success_count": 0,
                                        "failure_count": 0,
                                        "request_count": 0,
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "approx_cost": 0.0,
                                    },
                                )
                                other_model_data["request_count"] = new_request_count
                                # Also sync quota_max_requests if set
                                max_req = model_data.get("quota_max_requests")
                                if max_req:
                                    other_model_data["quota_max_requests"] = max_req
                                    other_model_data["quota_display"] = (
                                        f"{new_request_count}/{max_req}"
                                    )

            key_data["last_failure"] = {
                "timestamp": now_ts,
                "model": model,
                "error": str(classified_error.original_exception),
            }

        await self._save_usage()

    async def update_quota_baseline(
        self,
        credential: str,
        model: str,
        remaining_fraction: float,
        max_requests: Optional[int] = None,
        reset_timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update quota baseline data for a credential/model after fetching from API.

        This stores the current quota state as a baseline, which is used to
        estimate remaining quota based on subsequent request counts.

        When quota is exhausted (remaining_fraction <= 0.0) and a valid reset_timestamp
        is provided, this also sets model_cooldowns to prevent wasted requests.

        Args:
            credential: Credential identifier (file path or env:// URI)
            model: Model name (with or without provider prefix)
            remaining_fraction: Current remaining quota as fraction (0.0 to 1.0)
            max_requests: Maximum requests allowed per quota period (e.g., 250 for Claude)
            reset_timestamp: Unix timestamp when quota resets. Only trusted when
                remaining_fraction < 1.0 (quota has been used). API returns garbage
                reset times for unused quota (100%).

        Returns:
            None if no cooldown was set/updated, otherwise:
            {
                "group_or_model": str,  # quota group name or model name if ungrouped
                "hours_until_reset": float,
            }
        """
        await self._lazy_init()
        async with self._data_lock:
            now_ts = time.time()

            # Get or create key data structure
            key_data = self._usage_data.setdefault(
                credential,
                {
                    "models": {},
                    "global": {"models": {}},
                    "model_cooldowns": {},
                    "failures": {},
                },
            )

            # Ensure models dict exists
            if "models" not in key_data:
                key_data["models"] = {}

            # Get or create per-model data
            model_data = key_data["models"].setdefault(
                model,
                {
                    "window_start_ts": None,
                    "quota_reset_ts": None,
                    "success_count": 0,
                    "failure_count": 0,
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "approx_cost": 0.0,
                    "baseline_remaining_fraction": None,
                    "baseline_fetched_at": None,
                    "requests_at_baseline": None,
                },
            )

            # Calculate actual used requests from API's remaining fraction
            # The API is authoritative - sync our local count to match reality
            if max_requests is not None:
                used_requests = int((1.0 - remaining_fraction) * max_requests)
            else:
                # Estimate max_requests from provider's quota cost
                # This matches how get_max_requests_for_model() calculates it
                provider = self._get_provider_from_credential(credential)
                plugin_instance = self._get_provider_instance(provider)
                if plugin_instance and hasattr(
                    plugin_instance, "get_max_requests_for_model"
                ):
                    # Get tier from provider's cache
                    tier = getattr(plugin_instance, "project_tier_cache", {}).get(
                        credential, "standard-tier"
                    )
                    # Strip provider prefix from model if present
                    clean_model = model.split("/")[-1] if "/" in model else model
                    max_requests = plugin_instance.get_max_requests_for_model(
                        clean_model, tier
                    )
                    used_requests = int((1.0 - remaining_fraction) * max_requests)
                else:
                    # Fallback: keep existing count if we can't calculate
                    used_requests = model_data.get("request_count", 0)
                    max_requests = model_data.get("quota_max_requests")

            # Sync local request count to API's authoritative value
            model_data["request_count"] = used_requests
            model_data["requests_at_baseline"] = used_requests

            # Update baseline fields
            model_data["baseline_remaining_fraction"] = remaining_fraction
            model_data["baseline_fetched_at"] = now_ts

            # Update max_requests and quota_display
            if max_requests is not None:
                model_data["quota_max_requests"] = max_requests
                model_data["quota_display"] = f"{used_requests}/{max_requests}"

            # Handle reset_timestamp: only trust it when quota has been used (< 100%)
            # API returns garbage reset times for unused quota
            valid_reset_ts = (
                reset_timestamp is not None
                and remaining_fraction < 1.0
                and reset_timestamp > now_ts
            )

            if valid_reset_ts:
                model_data["quota_reset_ts"] = reset_timestamp

            # Set cooldowns when quota is exhausted
            model_cooldowns = key_data.setdefault("model_cooldowns", {})
            is_exhausted = remaining_fraction <= 0.0
            cooldown_set_info = (
                None  # Will be returned if cooldown was newly set/updated
            )

            if is_exhausted and valid_reset_ts:
                # Only update cooldown if not set or differs by more than 5 minutes
                existing_cooldown = model_cooldowns.get(model)
                should_update = (
                    existing_cooldown is None
                    or abs(existing_cooldown - reset_timestamp) > 300
                )
                if should_update:
                    model_cooldowns[model] = reset_timestamp
                    hours_until_reset = (reset_timestamp - now_ts) / 3600
                    # Determine group or model name for logging
                    group = self._get_model_quota_group(credential, model)
                    cooldown_set_info = {
                        "group_or_model": group if group else model.split("/")[-1],
                        "hours_until_reset": hours_until_reset,
                    }

            # Sync baseline fields and quota info across quota group
            group = self._get_model_quota_group(credential, model)
            if group:
                grouped_models = self._get_grouped_models(credential, group)
                for grouped_model in grouped_models:
                    if grouped_model != model:
                        other_model_data = key_data["models"].setdefault(
                            grouped_model,
                            {
                                "window_start_ts": None,
                                "quota_reset_ts": None,
                                "success_count": 0,
                                "failure_count": 0,
                                "request_count": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "approx_cost": 0.0,
                            },
                        )
                        # Sync request tracking
                        other_model_data["request_count"] = used_requests
                        if max_requests is not None:
                            other_model_data["quota_max_requests"] = max_requests
                            other_model_data["quota_display"] = (
                                f"{used_requests}/{max_requests}"
                            )
                        # Sync baseline fields
                        other_model_data["baseline_remaining_fraction"] = (
                            remaining_fraction
                        )
                        other_model_data["baseline_fetched_at"] = now_ts
                        other_model_data["requests_at_baseline"] = used_requests
                        # Sync reset timestamp if valid
                        if valid_reset_ts:
                            other_model_data["quota_reset_ts"] = reset_timestamp
                        # Sync cooldown if exhausted (with ±5 min check)
                        if is_exhausted and valid_reset_ts:
                            existing_grouped = model_cooldowns.get(grouped_model)
                            should_update_grouped = (
                                existing_grouped is None
                                or abs(existing_grouped - reset_timestamp) > 300
                            )
                            if should_update_grouped:
                                model_cooldowns[grouped_model] = reset_timestamp

            lib_logger.debug(
                f"Updated quota baseline for {mask_credential(credential)} model={model}: "
                f"remaining={remaining_fraction:.2%}, synced_request_count={used_requests}"
            )

        await self._save_usage()
        return cooldown_set_info

    async def _check_key_lockout(self, key: str, key_data: Dict):
        """
        Checks if a key should be locked out due to multiple model failures.

        NOTE: This check is currently disabled. The original logic counted individual
        models in long-term lockout, but this caused issues with quota groups - when
        a single quota group (e.g., "claude" with 5 models) was exhausted, it would
        count as 5 lockouts and trigger key-level lockout, blocking other quota groups
        (like gemini) that were still available.

        The per-model and per-group cooldowns already handle quota exhaustion properly.
        """
        # Disabled - see docstring above
        pass

    async def get_stats_for_endpoint(
        self,
        provider_filter: Optional[str] = None,
        include_global: bool = True,
    ) -> Dict[str, Any]:
        """
        Get usage stats formatted for the /v1/quota-stats endpoint.

        Aggregates data from key_usage.json grouped by provider.
        Includes both current period stats and global (lifetime) stats.

        Args:
            provider_filter: If provided, only return stats for this provider
            include_global: If True, include global/lifetime stats alongside current

        Returns:
            {
                "providers": {
                    "provider_name": {
                        "credential_count": int,
                        "active_count": int,
                        "on_cooldown_count": int,
                        "total_requests": int,
                        "tokens": {
                            "input_cached": int,
                            "input_uncached": int,
                            "input_cache_pct": float,
                            "output": int
                        },
                        "approx_cost": float | None,
                        "credentials": [...],
                        "global": {...}  # If include_global is True
                    }
                },
                "summary": {...},
                "global_summary": {...},  # If include_global is True
                "timestamp": float
            }
        """
        await self._lazy_init()

        now_ts = time.time()
        providers: Dict[str, Dict[str, Any]] = {}
        # Track global stats separately
        global_providers: Dict[str, Dict[str, Any]] = {}

        async with self._data_lock:
            if not self._usage_data:
                return {
                    "providers": {},
                    "summary": {
                        "total_providers": 0,
                        "total_credentials": 0,
                        "active_credentials": 0,
                        "exhausted_credentials": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_total_cost": 0.0,
                    },
                    "global_summary": {
                        "total_providers": 0,
                        "total_credentials": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_total_cost": 0.0,
                    },
                    "data_source": "cache",
                    "timestamp": now_ts,
                }

            for credential, cred_data in self._usage_data.items():
                # Extract provider from credential path
                provider = self._get_provider_from_credential(credential)
                if not provider:
                    continue

                # Apply filter if specified
                if provider_filter and provider != provider_filter:
                    continue

                # Initialize provider entry
                if provider not in providers:
                    providers[provider] = {
                        "credential_count": 0,
                        "active_count": 0,
                        "on_cooldown_count": 0,
                        "exhausted_count": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_cost": 0.0,
                        "credentials": [],
                    }
                    global_providers[provider] = {
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_cost": 0.0,
                    }

                prov_stats = providers[provider]
                prov_stats["credential_count"] += 1

                # Determine credential status and cooldowns
                key_cooldown = cred_data.get("key_cooldown_until", 0) or 0
                model_cooldowns = cred_data.get("model_cooldowns", {})

                # Build active cooldowns with remaining time
                active_cooldowns = {}
                for model, cooldown_ts in model_cooldowns.items():
                    if cooldown_ts > now_ts:
                        remaining_seconds = int(cooldown_ts - now_ts)
                        active_cooldowns[model] = {
                            "until_ts": cooldown_ts,
                            "remaining_seconds": remaining_seconds,
                        }

                key_cooldown_remaining = None
                if key_cooldown > now_ts:
                    key_cooldown_remaining = int(key_cooldown - now_ts)

                has_active_cooldown = key_cooldown > now_ts or len(active_cooldowns) > 0

                # Check if exhausted (all quota groups exhausted for Antigravity)
                is_exhausted = False
                models_data = cred_data.get("models", {})
                if models_data:
                    # Check if any model has remaining quota
                    all_exhausted = True
                    for model_stats in models_data.values():
                        if isinstance(model_stats, dict):
                            baseline = model_stats.get("baseline_remaining_fraction")
                            if baseline is None or baseline > 0:
                                all_exhausted = False
                                break
                    if all_exhausted and len(models_data) > 0:
                        is_exhausted = True

                if is_exhausted:
                    prov_stats["exhausted_count"] += 1
                    status = "exhausted"
                elif has_active_cooldown:
                    prov_stats["on_cooldown_count"] += 1
                    status = "cooldown"
                else:
                    prov_stats["active_count"] += 1
                    status = "active"

                # Aggregate token stats (current period)
                cred_tokens = {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "output": 0,
                }
                cred_requests = 0
                cred_cost = 0.0

                # Aggregate global token stats
                cred_global_tokens = {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "output": 0,
                }
                cred_global_requests = 0
                cred_global_cost = 0.0

                # Handle per-model structure (current period)
                if models_data:
                    for model_name, model_stats in models_data.items():
                        if not isinstance(model_stats, dict):
                            continue
                        # Prefer request_count if available and non-zero, else fall back to success+failure
                        req_count = model_stats.get("request_count", 0)
                        if req_count > 0:
                            cred_requests += req_count
                        else:
                            cred_requests += model_stats.get("success_count", 0)
                            cred_requests += model_stats.get("failure_count", 0)
                        # Token stats - track cached separately
                        cred_tokens["input_cached"] += model_stats.get(
                            "prompt_tokens_cached", 0
                        )
                        cred_tokens["input_uncached"] += model_stats.get(
                            "prompt_tokens", 0
                        )
                        cred_tokens["output"] += model_stats.get("completion_tokens", 0)
                        cred_cost += model_stats.get("approx_cost", 0.0)

                # Handle legacy daily structure
                daily_data = cred_data.get("daily", {})
                daily_models = daily_data.get("models", {})
                for model_name, model_stats in daily_models.items():
                    if not isinstance(model_stats, dict):
                        continue
                    cred_requests += model_stats.get("success_count", 0)
                    cred_tokens["input_cached"] += model_stats.get(
                        "prompt_tokens_cached", 0
                    )
                    cred_tokens["input_uncached"] += model_stats.get("prompt_tokens", 0)
                    cred_tokens["output"] += model_stats.get("completion_tokens", 0)
                    cred_cost += model_stats.get("approx_cost", 0.0)

                # Handle global stats
                global_data = cred_data.get("global", {})
                global_models = global_data.get("models", {})
                for model_name, model_stats in global_models.items():
                    if not isinstance(model_stats, dict):
                        continue
                    cred_global_requests += model_stats.get("success_count", 0)
                    cred_global_tokens["input_cached"] += model_stats.get(
                        "prompt_tokens_cached", 0
                    )
                    cred_global_tokens["input_uncached"] += model_stats.get(
                        "prompt_tokens", 0
                    )
                    cred_global_tokens["output"] += model_stats.get(
                        "completion_tokens", 0
                    )
                    cred_global_cost += model_stats.get("approx_cost", 0.0)

                # Add current period stats to global totals
                cred_global_requests += cred_requests
                cred_global_tokens["input_cached"] += cred_tokens["input_cached"]
                cred_global_tokens["input_uncached"] += cred_tokens["input_uncached"]
                cred_global_tokens["output"] += cred_tokens["output"]
                cred_global_cost += cred_cost

                # Build credential entry
                # Mask credential identifier for display
                if credential.startswith("env://"):
                    identifier = credential
                else:
                    identifier = Path(credential).name

                cred_entry = {
                    "identifier": identifier,
                    "full_path": credential,
                    "status": status,
                    "last_used_ts": cred_data.get("last_used_ts"),
                    "requests": cred_requests,
                    "tokens": cred_tokens,
                    "approx_cost": cred_cost if cred_cost > 0 else None,
                }

                # Add cooldown info
                if key_cooldown_remaining is not None:
                    cred_entry["key_cooldown_remaining"] = key_cooldown_remaining
                if active_cooldowns:
                    cred_entry["model_cooldowns"] = active_cooldowns

                # Add global stats for this credential
                if include_global:
                    # Calculate global cache percentage
                    global_total_input = (
                        cred_global_tokens["input_cached"]
                        + cred_global_tokens["input_uncached"]
                    )
                    global_cache_pct = (
                        round(
                            cred_global_tokens["input_cached"]
                            / global_total_input
                            * 100,
                            1,
                        )
                        if global_total_input > 0
                        else 0
                    )

                    cred_entry["global"] = {
                        "requests": cred_global_requests,
                        "tokens": {
                            "input_cached": cred_global_tokens["input_cached"],
                            "input_uncached": cred_global_tokens["input_uncached"],
                            "input_cache_pct": global_cache_pct,
                            "output": cred_global_tokens["output"],
                        },
                        "approx_cost": cred_global_cost
                        if cred_global_cost > 0
                        else None,
                    }

                # Add model-specific data for providers with per-model tracking
                if models_data:
                    cred_entry["models"] = {}
                    for model_name, model_stats in models_data.items():
                        if not isinstance(model_stats, dict):
                            continue
                        cred_entry["models"][model_name] = {
                            "requests": model_stats.get("success_count", 0)
                            + model_stats.get("failure_count", 0),
                            "request_count": model_stats.get("request_count", 0),
                            "success_count": model_stats.get("success_count", 0),
                            "failure_count": model_stats.get("failure_count", 0),
                            "prompt_tokens": model_stats.get("prompt_tokens", 0),
                            "prompt_tokens_cached": model_stats.get(
                                "prompt_tokens_cached", 0
                            ),
                            "completion_tokens": model_stats.get(
                                "completion_tokens", 0
                            ),
                            "approx_cost": model_stats.get("approx_cost", 0.0),
                            "window_start_ts": model_stats.get("window_start_ts"),
                            "quota_reset_ts": model_stats.get("quota_reset_ts"),
                            # Quota baseline fields (Antigravity-specific)
                            "baseline_remaining_fraction": model_stats.get(
                                "baseline_remaining_fraction"
                            ),
                            "baseline_fetched_at": model_stats.get(
                                "baseline_fetched_at"
                            ),
                            "quota_max_requests": model_stats.get("quota_max_requests"),
                            "quota_display": model_stats.get("quota_display"),
                        }

                prov_stats["credentials"].append(cred_entry)

                # Aggregate to provider totals (current period)
                prov_stats["total_requests"] += cred_requests
                prov_stats["tokens"]["input_cached"] += cred_tokens["input_cached"]
                prov_stats["tokens"]["input_uncached"] += cred_tokens["input_uncached"]
                prov_stats["tokens"]["output"] += cred_tokens["output"]
                if cred_cost > 0:
                    prov_stats["approx_cost"] += cred_cost

                # Aggregate to global provider totals
                global_providers[provider]["total_requests"] += cred_global_requests
                global_providers[provider]["tokens"]["input_cached"] += (
                    cred_global_tokens["input_cached"]
                )
                global_providers[provider]["tokens"]["input_uncached"] += (
                    cred_global_tokens["input_uncached"]
                )
                global_providers[provider]["tokens"]["output"] += cred_global_tokens[
                    "output"
                ]
                global_providers[provider]["approx_cost"] += cred_global_cost

        # Calculate cache percentages for each provider
        for provider, prov_stats in providers.items():
            total_input = (
                prov_stats["tokens"]["input_cached"]
                + prov_stats["tokens"]["input_uncached"]
            )
            if total_input > 0:
                prov_stats["tokens"]["input_cache_pct"] = round(
                    prov_stats["tokens"]["input_cached"] / total_input * 100, 1
                )
            # Set cost to None if 0
            if prov_stats["approx_cost"] == 0:
                prov_stats["approx_cost"] = None

            # Calculate global cache percentages
            if include_global and provider in global_providers:
                gp = global_providers[provider]
                global_total = (
                    gp["tokens"]["input_cached"] + gp["tokens"]["input_uncached"]
                )
                if global_total > 0:
                    gp["tokens"]["input_cache_pct"] = round(
                        gp["tokens"]["input_cached"] / global_total * 100, 1
                    )
                if gp["approx_cost"] == 0:
                    gp["approx_cost"] = None
                prov_stats["global"] = gp

        # Build summary (current period)
        total_creds = sum(p["credential_count"] for p in providers.values())
        active_creds = sum(p["active_count"] for p in providers.values())
        exhausted_creds = sum(p["exhausted_count"] for p in providers.values())
        total_requests = sum(p["total_requests"] for p in providers.values())
        total_input_cached = sum(
            p["tokens"]["input_cached"] for p in providers.values()
        )
        total_input_uncached = sum(
            p["tokens"]["input_uncached"] for p in providers.values()
        )
        total_output = sum(p["tokens"]["output"] for p in providers.values())
        total_cost = sum(p["approx_cost"] or 0 for p in providers.values())

        total_input = total_input_cached + total_input_uncached
        input_cache_pct = (
            round(total_input_cached / total_input * 100, 1) if total_input > 0 else 0
        )

        result = {
            "providers": providers,
            "summary": {
                "total_providers": len(providers),
                "total_credentials": total_creds,
                "active_credentials": active_creds,
                "exhausted_credentials": exhausted_creds,
                "total_requests": total_requests,
                "tokens": {
                    "input_cached": total_input_cached,
                    "input_uncached": total_input_uncached,
                    "input_cache_pct": input_cache_pct,
                    "output": total_output,
                },
                "approx_total_cost": total_cost if total_cost > 0 else None,
            },
            "data_source": "cache",
            "timestamp": now_ts,
        }

        # Build global summary
        if include_global:
            global_total_requests = sum(
                gp["total_requests"] for gp in global_providers.values()
            )
            global_total_input_cached = sum(
                gp["tokens"]["input_cached"] for gp in global_providers.values()
            )
            global_total_input_uncached = sum(
                gp["tokens"]["input_uncached"] for gp in global_providers.values()
            )
            global_total_output = sum(
                gp["tokens"]["output"] for gp in global_providers.values()
            )
            global_total_cost = sum(
                gp["approx_cost"] or 0 for gp in global_providers.values()
            )

            global_total_input = global_total_input_cached + global_total_input_uncached
            global_input_cache_pct = (
                round(global_total_input_cached / global_total_input * 100, 1)
                if global_total_input > 0
                else 0
            )

            result["global_summary"] = {
                "total_providers": len(global_providers),
                "total_credentials": total_creds,
                "total_requests": global_total_requests,
                "tokens": {
                    "input_cached": global_total_input_cached,
                    "input_uncached": global_total_input_uncached,
                    "input_cache_pct": global_input_cache_pct,
                    "output": global_total_output,
                },
                "approx_total_cost": global_total_cost
                if global_total_cost > 0
                else None,
            }

        return result

    async def reload_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when another process may have updated the file.
        """
        async with self._init_lock:
            self._initialized.clear()
            await self._load_usage()
            await self._reset_daily_stats_if_needed()
            self._initialized.set()
