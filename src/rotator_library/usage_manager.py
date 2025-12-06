import json
import os
import time
import logging
import asyncio
import random
from datetime import date, datetime, timezone, time as dt_time
from typing import Any, Dict, List, Optional, Set
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
    """

    def __init__(
        self,
        file_path: str = "key_usage.json",
        daily_reset_time_utc: Optional[str] = "03:00",
        rotation_tolerance: float = 0.0,
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
        """
        self.file_path = file_path
        self.rotation_tolerance = rotation_tolerance
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
        """Checks if daily stats need to be reset for any key."""
        if self._usage_data is None or not self.daily_reset_time_utc:
            return

        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.date().isoformat()
        needs_saving = False

        for key, data in self._usage_data.items():
            last_reset_str = data.get("last_daily_reset", "")

            if last_reset_str != today_str:
                last_reset_dt = None
                if last_reset_str:
                    # Ensure the parsed datetime is timezone-aware (UTC)
                    last_reset_dt = datetime.fromisoformat(last_reset_str).replace(
                        tzinfo=timezone.utc
                    )

                # Determine the reset threshold for today
                reset_threshold_today = datetime.combine(
                    now_utc.date(), self.daily_reset_time_utc
                )

                if (
                    last_reset_dt is None
                    or last_reset_dt < reset_threshold_today <= now_utc
                ):
                    lib_logger.debug(
                        f"Performing daily reset for key {mask_credential(key)}"
                    )
                    needs_saving = True

                    # Reset cooldowns
                    data["model_cooldowns"] = {}
                    data["key_cooldown_until"] = None

                    # Reset consecutive failures
                    if "failures" in data:
                        data["failures"] = {}

                    # Archive global stats from the previous day's 'daily'
                    daily_data = data.get("daily", {})
                    if daily_data:
                        global_data = data.setdefault("global", {"models": {}})
                        for model, stats in daily_data.get("models", {}).items():
                            global_model_stats = global_data["models"].setdefault(
                                model,
                                {
                                    "success_count": 0,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "approx_cost": 0.0,
                                },
                            )
                            global_model_stats["success_count"] += stats.get(
                                "success_count", 0
                            )
                            global_model_stats["prompt_tokens"] += stats.get(
                                "prompt_tokens", 0
                            )
                            global_model_stats["completion_tokens"] += stats.get(
                                "completion_tokens", 0
                            )
                            global_model_stats["approx_cost"] += stats.get(
                                "approx_cost", 0.0
                            )

                    # Reset daily stats
                    data["daily"] = {"date": today_str, "models": {}}
                    data["last_daily_reset"] = today_str

        if needs_saving:
            await self._save_usage()

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
                        usage_count = (
                            key_data.get("daily", {})
                            .get("models", {})
                            .get(model, {})
                            .get("success_count", 0)
                        )

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

                    # Apply weighted random selection or deterministic sorting
                    selection_method = (
                        "weighted-random"
                        if self.rotation_tolerance > 0
                        else "least-used"
                    )

                    if self.rotation_tolerance > 0:
                        # Weighted random selection within each tier
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
                        usage_count = (
                            key_data.get("daily", {})
                            .get("models", {})
                            .get(model, {})
                            .get("success_count", 0)
                        )
                        key_state = self.key_states[key]

                        # Tier 1: Completely idle keys (preferred).
                        if not key_state["models_in_use"]:
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Keys that can accept more concurrent requests for this model.
                        elif key_state["models_in_use"].get(model, 0) < max_concurrent:
                            tier2_keys.append((key, usage_count))

                # Apply weighted random selection or deterministic sorting
                selection_method = (
                    "weighted-random" if self.rotation_tolerance > 0 else "least-used"
                )

                if self.rotation_tolerance > 0:
                    # Weighted random selection within each tier
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
        """
        await self._lazy_init()
        async with self._data_lock:
            today_utc_str = datetime.now(timezone.utc).date().isoformat()
            key_data = self._usage_data.setdefault(
                key,
                {
                    "daily": {"date": today_utc_str, "models": {}},
                    "global": {"models": {}},
                    "model_cooldowns": {},
                    "failures": {},
                },
            )

            # If the key is new, ensure its reset date is initialized to prevent an immediate reset.
            if "last_daily_reset" not in key_data:
                key_data["last_daily_reset"] = today_utc_str

            # Always record a success and reset failures
            model_failures = key_data.setdefault("failures", {}).setdefault(model, {})
            model_failures["consecutive_failures"] = 0
            if model in key_data.get("model_cooldowns", {}):
                del key_data["model_cooldowns"][model]

            daily_model_data = key_data["daily"]["models"].setdefault(
                model,
                {
                    "success_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "approx_cost": 0.0,
                },
            )
            daily_model_data["success_count"] += 1

            # Safely attempt to record token and cost usage
            if (
                completion_response
                and hasattr(completion_response, "usage")
                and completion_response.usage
            ):
                usage = completion_response.usage
                daily_model_data["prompt_tokens"] += usage.prompt_tokens
                daily_model_data["completion_tokens"] += getattr(
                    usage, "completion_tokens", 0
                )  # Not present in embedding responses
                lib_logger.info(
                    f"Recorded usage from response object for key {mask_credential(key)}"
                )
                try:
                    provider_name = model.split("/")[0]
                    provider_plugin = PROVIDER_PLUGINS.get(provider_name)

                    # Check class attribute directly - no need to instantiate
                    if provider_plugin and getattr(
                        provider_plugin, "skip_cost_calculation", False
                    ):
                        lib_logger.debug(
                            f"Skipping cost calculation for provider '{provider_name}' (custom provider)."
                        )
                    else:
                        # Differentiate cost calculation based on response type
                        if isinstance(completion_response, litellm.EmbeddingResponse):
                            # Manually calculate cost for embeddings
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
                            daily_model_data["approx_cost"] += cost
                except Exception as e:
                    lib_logger.warning(
                        f"Could not calculate cost for model {model}: {e}"
                    )
            elif isinstance(completion_response, asyncio.Future) or hasattr(
                completion_response, "__aiter__"
            ):
                # This is an unconsumed stream object. Do not log a warning, as usage will be recorded from the chunks.
                pass
            else:
                lib_logger.warning(
                    f"No usage data found in completion response for model {model}. Recording success without token count."
                )

            key_data["last_used_ts"] = time.time()

        await self._save_usage()

    async def record_failure(
        self,
        key: str,
        model: str,
        classified_error: ClassifiedError,
        increment_consecutive_failures: bool = True,
    ):
        """Records a failure and applies cooldowns based on an escalating backoff strategy.

        Args:
            key: The API key or credential identifier
            model: The model name
            classified_error: The classified error object
            increment_consecutive_failures: Whether to increment the failure counter.
                Set to False for provider-level errors that shouldn't count against the key.
        """
        await self._lazy_init()
        async with self._data_lock:
            today_utc_str = datetime.now(timezone.utc).date().isoformat()
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

            if classified_error.error_type in ["rate_limit", "quota_exceeded"]:
                # Rate limit / Quota errors: use retry_after if available, otherwise default to 60s
                cooldown_seconds = classified_error.retry_after or 60
                lib_logger.info(
                    f"Rate limit error on key {mask_credential(key)} for model {model}. "
                    f"Using {'provided' if classified_error.retry_after else 'default'} retry_after: {cooldown_seconds}s"
                )
            elif classified_error.error_type == "authentication":
                # Apply a 5-minute key-level lockout for auth errors
                key_data["key_cooldown_until"] = time.time() + 300
                lib_logger.warning(
                    f"Authentication error on key {mask_credential(key)}. Applying 5-minute key-level lockout."
                )
                # Auth errors still use escalating backoff for the specific model
                cooldown_seconds = 300  # 5 minutes for model cooldown

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
                    cooldown_seconds = backoff_tiers.get(
                        count, 7200
                    )  # Default to 2 hours for "spent" keys
                    lib_logger.warning(
                        f"Failure #{count} for key {mask_credential(key)} with model {model}. "
                        f"Error type: {classified_error.error_type}"
                    )
            else:
                # Provider-level errors: apply short cooldown but don't count against key
                if cooldown_seconds is None:
                    cooldown_seconds = 30  # 30s cooldown for provider issues
                lib_logger.info(
                    f"Provider-level error ({classified_error.error_type}) for key {mask_credential(key)} with model {model}. "
                    f"NOT incrementing consecutive failures. Applying {cooldown_seconds}s cooldown."
                )

            # Apply the cooldown
            model_cooldowns = key_data.setdefault("model_cooldowns", {})
            model_cooldowns[model] = time.time() + cooldown_seconds
            lib_logger.warning(
                f"Cooldown applied for key {mask_credential(key)} with model {model}: {cooldown_seconds}s. "
                f"Error type: {classified_error.error_type}"
            )

            # Check for key-level lockout condition
            await self._check_key_lockout(key, key_data)

            key_data["last_failure"] = {
                "timestamp": time.time(),
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
