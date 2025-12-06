import json
import os
import time
import logging
import asyncio
import random
from datetime import date, datetime, timezone, time as dt_time
from typing import Any, Dict, List, Optional, Set, Tuple
import aiofiles
import litellm

from .error_handler import ClassifiedError, NoAvailableKeysError, mask_credential
from .providers import PROVIDER_PLUGINS

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
        file_path: str = "key_usage.json",
        daily_reset_time_utc: Optional[str] = "03:00",
        rotation_tolerance: float = 0.0,
        provider_rotation_modes: Optional[Dict[str, str]] = None,
        provider_plugins: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the UsageManager.

        Args:
            file_path: Path to the usage data JSON file
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
        """
        self.file_path = file_path
        self.rotation_tolerance = rotation_tolerance
        self.provider_rotation_modes = provider_rotation_modes or {}
        self.provider_plugins = provider_plugins or PROVIDER_PLUGINS
        self.key_states: Dict[str, Dict[str, Any]] = {}

        self._data_lock = asyncio.Lock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()

        self._timeout_lock = asyncio.Lock()
        self._claimed_on_timeout: Set[str] = set()

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

    def _get_provider_from_credential(self, credential: str) -> Optional[str]:
        """
        Extract provider name from credential path or identifier.

        Supports multiple credential formats:
        - OAuth: "oauth_creds/antigravity_oauth_15.json" -> "antigravity"
        - OAuth: "C:\\...\\oauth_creds\\gemini_cli_oauth_1.json" -> "gemini_cli"
        - API key style: stored with provider prefix metadata

        Args:
            credential: The credential identifier (path or key)

        Returns:
            Provider name string or None if cannot be determined
        """
        import re

        # Normalize path separators
        normalized = credential.replace("\\", "/")

        # Pattern: {provider}_oauth_{number}.json
        match = re.search(r"/([a-z_]+)_oauth_\d+\.json$", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: oauth_creds/{provider}_...
        match = re.search(r"oauth_creds/([a-z_]+)_", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        return None

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
        if not provider:
            return None

        plugin = self.provider_plugins.get(provider)
        if not plugin:
            return None

        if hasattr(plugin, "get_usage_reset_config"):
            return plugin.get_usage_reset_config(credential)

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
        if not provider:
            return None

        plugin = self.provider_plugins.get(provider)
        if not plugin:
            return None

        if hasattr(plugin, "get_model_quota_group"):
            return plugin.get_model_quota_group(model)

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
        if not provider:
            return []

        plugin = self.provider_plugins.get(provider)
        if not plugin:
            return []

        if hasattr(plugin, "get_models_in_quota_group"):
            models = plugin.get_models_in_quota_group(group)
            # Add provider prefix
            return [f"{provider}/{m}" for m in models]

        return []

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
        if provider:
            plugin = self.provider_plugins.get(provider)
            if plugin and hasattr(plugin, "get_default_usage_field_name"):
                return plugin.get_default_usage_field_name()

        return "daily"

    def _get_usage_count(self, key: str, model: str) -> int:
        """
        Get the current usage count for a model from the appropriate usage structure.

        Supports both:
        - New per-model structure: {"models": {"model_name": {"success_count": N, ...}}}
        - Legacy structure: {"daily": {"models": {"model_name": {"success_count": N, ...}}}}

        Args:
            key: Credential identifier
            model: Model name

        Returns:
            Usage count (success_count) for the model in the current window/period
        """
        if self._usage_data is None:
            return 0

        key_data = self._usage_data.get(key, {})
        reset_mode = self._get_reset_mode(key)

        if reset_mode == "per_model":
            # New per-model structure: key_data["models"][model]["success_count"]
            return key_data.get("models", {}).get(model, {}).get("success_count", 0)
        else:
            # Legacy structure: key_data["daily"]["models"][model]["success_count"]
            return (
                key_data.get("daily", {})
                .get("models", {})
                .get(model, {})
                .get("success_count", 0)
            )

    def _select_sequential(
        self,
        candidates: List[Tuple[str, int]],
        credential_priorities: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Select credential in strict sequential order for cache-preserving rotation.

        This method ensures the same credential is reused until it hits a cooldown,
        which preserves provider-side caching (e.g., thinking signature caches).

        Selection logic:
        1. Sort by priority (lowest number = highest priority)
        2. Within same priority, sort by last_used_ts (most recent first = sticky)
        3. Return the first candidate

        Args:
            candidates: List of (credential_id, usage_count) tuples
            credential_priorities: Optional dict mapping credentials to priority levels

        Returns:
            Selected credential ID
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        if len(candidates) == 1:
            return candidates[0][0]

        def sort_key(item: Tuple[str, int]) -> Tuple[int, float]:
            cred, _ = item
            # Priority: lower is better (1 = highest priority)
            priority = (
                credential_priorities.get(cred, 999) if credential_priorities else 999
            )
            # Last used: higher (more recent) is better for stickiness
            last_used = (
                self._usage_data.get(cred, {}).get("last_used_ts", 0)
                if self._usage_data
                else 0
            )
            # Negative last_used so most recent sorts first
            return (priority, -last_used)

        sorted_candidates = sorted(candidates, key=sort_key)
        selected = sorted_candidates[0][0]

        lib_logger.debug(
            f"Sequential selection: chose {mask_credential(selected)} "
            f"(priority={credential_priorities.get(selected, 999) if credential_priorities else 'N/A'})"
        )

        return selected

    async def _lazy_init(self):
        """Initializes the usage data by loading it from the file asynchronously."""
        async with self._init_lock:
            if not self._initialized.is_set():
                await self._load_usage()
                await self._reset_daily_stats_if_needed()
                self._initialized.set()

    async def _load_usage(self):
        """Loads usage data from the JSON file asynchronously."""
        async with self._data_lock:
            if not os.path.exists(self.file_path):
                self._usage_data = {}
                return
            try:
                async with aiofiles.open(self.file_path, "r") as f:
                    content = await f.read()
                    self._usage_data = json.loads(content)
            except (json.JSONDecodeError, IOError, FileNotFoundError):
                self._usage_data = {}

    async def _save_usage(self):
        """Saves the current usage data to the JSON file asynchronously."""
        if self._usage_data is None:
            return
        async with self._data_lock:
            async with aiofiles.open(self.file_path, "w") as f:
                await f.write(json.dumps(self._usage_data, indent=2))

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
        model_data["prompt_tokens"] = 0
        model_data["completion_tokens"] = 0
        model_data["approx_cost"] = 0.0

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
                        usage_count = self._get_usage_count(key, model)

                        # Group by priority
                        if priority not in priority_groups:
                            priority_groups[priority] = []
                        priority_groups[priority].append((key, usage_count))

                # Try priority groups in order (1, 2, 3, ...)
                sorted_priorities = sorted(priority_groups.keys())

                for priority_level in sorted_priorities:
                    keys_in_priority = priority_groups[priority_level]

                    # Within each priority group, use existing tier1/tier2 logic
                    tier1_keys, tier2_keys = [], []
                    for key, usage_count in keys_in_priority:
                        key_state = self.key_states[key]

                        # Tier 1: Completely idle keys (preferred)
                        if not key_state["models_in_use"]:
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Keys that can accept more concurrent requests
                        elif key_state["models_in_use"].get(model, 0) < max_concurrent:
                            tier2_keys.append((key, usage_count))

                    # Determine selection method based on provider's rotation mode
                    provider = model.split("/")[0] if "/" in model else ""
                    rotation_mode = self._get_rotation_mode(provider)

                    if rotation_mode == "sequential":
                        # Sequential mode: stick with same credential until exhausted
                        selection_method = "sequential"
                        if tier1_keys:
                            selected_key = self._select_sequential(
                                tier1_keys, credential_priorities
                            )
                            tier1_keys = [
                                (k, u) for k, u in tier1_keys if k == selected_key
                            ]
                        if tier2_keys:
                            selected_key = self._select_sequential(
                                tier2_keys, credential_priorities
                            )
                            tier2_keys = [
                                (k, u) for k, u in tier2_keys if k == selected_key
                            ]
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
                                lib_logger.info(
                                    f"Acquired key {mask_credential(key)} for model {model} "
                                    f"(tier: {tier_name}, priority: {priority_level}, selection: {selection_method}, usage: {usage})"
                                )
                                return key

                    # Then try Tier 2
                    for key, usage in tier2_keys:
                        state = self.key_states[key]
                        async with state["lock"]:
                            current_count = state["models_in_use"].get(model, 0)
                            if current_count < max_concurrent:
                                state["models_in_use"][model] = current_count + 1
                                tier_name = (
                                    credential_tier_names.get(key, "unknown")
                                    if credential_tier_names
                                    else "unknown"
                                )
                                lib_logger.info(
                                    f"Acquired key {mask_credential(key)} for model {model} "
                                    f"(tier: {tier_name}, priority: {priority_level}, selection: {selection_method}, concurrent: {state['models_in_use'][model]}/{max_concurrent}, usage: {usage})"
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
                        usage_count = self._get_usage_count(key, model)
                        key_state = self.key_states[key]

                        # Tier 1: Completely idle keys (preferred).
                        if not key_state["models_in_use"]:
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Keys that can accept more concurrent requests for this model.
                        elif key_state["models_in_use"].get(model, 0) < max_concurrent:
                            tier2_keys.append((key, usage_count))

                # Determine selection method based on provider's rotation mode
                provider = model.split("/")[0] if "/" in model else ""
                rotation_mode = self._get_rotation_mode(provider)

                if rotation_mode == "sequential":
                    # Sequential mode: stick with same credential until exhausted
                    selection_method = "sequential"
                    if tier1_keys:
                        selected_key = self._select_sequential(
                            tier1_keys, credential_priorities
                        )
                        tier1_keys = [
                            (k, u) for k, u in tier1_keys if k == selected_key
                        ]
                    if tier2_keys:
                        selected_key = self._select_sequential(
                            tier2_keys, credential_priorities
                        )
                        tier2_keys = [
                            (k, u) for k, u in tier2_keys if k == selected_key
                        ]
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
                            lib_logger.info(
                                f"Acquired key {mask_credential(key)} for model {model} "
                                f"({tier_info}selection: {selection_method}, usage: {usage})"
                            )
                            return key

                # If no Tier 1 keys are available, try Tier 2.
                for key, usage in tier2_keys:
                    state = self.key_states[key]
                    async with state["lock"]:
                        current_count = state["models_in_use"].get(model, 0)
                        if current_count < max_concurrent:
                            state["models_in_use"][model] = current_count + 1
                            tier_name = (
                                credential_tier_names.get(key)
                                if credential_tier_names
                                else None
                            )
                            tier_info = f"tier: {tier_name}, " if tier_name else ""
                            lib_logger.info(
                                f"Acquired key {mask_credential(key)} for model {model} "
                                f"({tier_info}selection: {selection_method}, concurrent: {state['models_in_use'][model]}/{max_concurrent}, usage: {usage})"
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
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "approx_cost": 0.0,
                    },
                )

                # Start window on first request for this model
                if model_data.get("window_start_ts") is None:
                    model_data["window_start_ts"] = now_ts
                    window_hours = (
                        reset_config.get("window_seconds", 0) / 3600
                        if reset_config
                        else 0
                    )
                    lib_logger.info(
                        f"Started {window_hours:.1f}h window for model {model} on {mask_credential(key)}"
                    )

                # Record stats
                model_data["success_count"] += 1
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
                    provider_plugin = self.provider_plugins.get(provider_name)

                    if provider_plugin and getattr(
                        provider_plugin, "skip_cost_calculation", False
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
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "approx_cost": 0.0,
                        },
                    )
                    model_data["quota_reset_ts"] = quota_reset_ts

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
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "approx_cost": 0.0,
                                },
                            )
                            group_model_data["quota_reset_ts"] = quota_reset_ts
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

            key_data["last_failure"] = {
                "timestamp": now_ts,
                "model": model,
                "error": str(classified_error.original_exception),
            }

        await self._save_usage()

    async def _check_key_lockout(self, key: str, key_data: Dict):
        """Checks if a key should be locked out due to multiple model failures."""
        long_term_lockout_models = 0
        now = time.time()

        for model, cooldown_end in key_data.get("model_cooldowns", {}).items():
            if cooldown_end - now >= 7200:  # Check for 2-hour lockouts
                long_term_lockout_models += 1

        if long_term_lockout_models >= 3:
            key_data["key_cooldown_until"] = now + 300  # 5-minute key lockout
            lib_logger.error(
                f"Key {mask_credential(key)} has {long_term_lockout_models} models in long-term lockout. Applying 5-minute key-level lockout."
            )
