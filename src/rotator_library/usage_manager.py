import json
import os
import time
import logging
import asyncio
from datetime import date, datetime, timezone, time as dt_time
from typing import Dict, List, Optional, Set
from filelock import FileLock
import aiofiles
import litellm

from .error_handler import ClassifiedError

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class UsageManager:
    """
    Manages usage statistics and cooldowns for API keys with asyncio-safe locking,
    asynchronous file I/O, and a lazy-loading mechanism for usage data.
    """
    def __init__(self, file_path: str = "key_usage.json", wait_timeout: int = 5, daily_reset_time_utc: Optional[str] = "00:00"):
        self.file_path = file_path
        self.file_lock = FileLock(f"{self.file_path}.lock")
        self.key_locks: Dict[str, asyncio.Lock] = {}
        self.condition = asyncio.Condition()
        self.wait_timeout = wait_timeout
        
        self._data_lock = asyncio.Lock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()

        self._timeout_lock = asyncio.Lock()
        self._claimed_on_timeout: Set[str] = set()

        if daily_reset_time_utc:
            hour, minute = map(int, daily_reset_time_utc.split(':'))
            self.daily_reset_time_utc = dt_time(hour=hour, minute=minute, tzinfo=timezone.utc)
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
                async with aiofiles.open(self.file_path, 'r') as f:
                    content = await f.read()
                    self._usage_data = json.loads(content)
            except (json.JSONDecodeError, IOError, FileNotFoundError):
                self._usage_data = {}

    async def _save_usage(self):
        """Saves the current usage data to the JSON file asynchronously."""
        if self._usage_data is None:
            return
        async with self._data_lock:
            with self.file_lock:
                async with aiofiles.open(self.file_path, 'w') as f:
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
                last_reset_dt = datetime.fromisoformat(last_reset_str) if last_reset_str else None
                
                # Determine the reset threshold for today
                reset_threshold_today = datetime.combine(now_utc.date(), self.daily_reset_time_utc)

                if last_reset_dt is None or last_reset_dt < reset_threshold_today <= now_utc:
                    lib_logger.info(f"Performing daily reset for key ...{key[-4:]}")
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
                            global_model_stats = global_data["models"].setdefault(model, {"success_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "approx_cost": 0.0})
                            global_model_stats["success_count"] += stats.get("success_count", 0)
                            global_model_stats["prompt_tokens"] += stats.get("prompt_tokens", 0)
                            global_model_stats["completion_tokens"] += stats.get("completion_tokens", 0)
                            global_model_stats["approx_cost"] += stats.get("approx_cost", 0.0)
                    
                    # Reset daily stats
                    data["daily"] = {"date": today_str, "models": {}}
                    data["last_daily_reset"] = today_str

        if needs_saving:
            await self._save_usage()

    def _initialize_locks(self, keys: List[str]):
        """Initializes asyncio locks for all provided keys if not already present."""
        for key in keys:
            if key not in self.key_locks:
                self.key_locks[key] = asyncio.Lock()

    async def acquire_key(self, available_keys: List[str], model: str) -> str:
        """Acquires the best available key with robust locking and a fair timeout mechanism."""
        await self._lazy_init()
        self._initialize_locks(available_keys)

        async with self.condition:
            while True:
                eligible_keys = []
                async with self._data_lock:
                    now = time.time()
                    for key in available_keys:
                        key_data = self._usage_data.get(key, {})
                        
                        # Check for key-level lockout
                        key_cooldown = key_data.get("key_cooldown_until")
                        if key_cooldown and now < key_cooldown:
                            continue

                        # Check for model-specific cooldown
                        model_cooldown = key_data.get("model_cooldowns", {}).get(model)
                        if model_cooldown and now < model_cooldown:
                            continue
                        
                        usage_count = key_data.get("daily", {}).get("models", {}).get(model, {}).get("success_count", 0)
                        eligible_keys.append((key, usage_count))
                
                if not eligible_keys:
                    lib_logger.warning("All keys are on cooldown. Waiting...")
                    await asyncio.sleep(5)
                    continue

                eligible_keys.sort(key=lambda x: x[1])
                
                for key, _ in eligible_keys:
                    lock = self.key_locks[key]
                    if not lock.locked():
                        await lock.acquire()
                        lib_logger.info(f"Acquired lock for available key: ...{key[-4:]}")
                        return key

                lib_logger.info("All eligible keys are locked. Waiting for a key to be released.")
                
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=self.wait_timeout)
                    lib_logger.info("Notified that a key was released. Re-evaluating...")
                    continue
                except asyncio.TimeoutError:
                    lib_logger.warning("Wait timed out. Attempting to acquire a key via fair timeout logic.")
                    async with self._timeout_lock:
                        for key, _ in eligible_keys:
                            if key not in self._claimed_on_timeout:
                                self._claimed_on_timeout.add(key)
                                lib_logger.info(f"Acquired key ...{key[-4:]} via timeout claim.")
                                return key
                    lib_logger.error("Timeout occurred, but all eligible keys were already claimed by other timed-out tasks.")
                    await asyncio.sleep(1)

    async def release_key(self, key: str):
        """Releases the lock for a given key and notifies waiting tasks."""
        async with self.condition:
            async with self._timeout_lock:
                if key in self._claimed_on_timeout:
                    self._claimed_on_timeout.remove(key)

            if key in self.key_locks and self.key_locks[key].locked():
                self.key_locks[key].release()
                lib_logger.info(f"Released lock for key ...{key[-4:]}")
                self.condition.notify()

    async def record_success(self, key: str, model: str, completion_response: Optional[litellm.ModelResponse] = None):
        """
        Records a successful API call, resetting failure counters.
        It safely handles cases where token usage data is not available.
        """
        await self._lazy_init()
        async with self._data_lock:
            key_data = self._usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}, "failures": {}})
            
            # Perform a just-in-time daily reset if the date has changed.
            if key_data["daily"].get("date") != date.today().isoformat():
                key_data["daily"] = {"date": date.today().isoformat(), "models": {}}

            # Always record a success and reset failures
            model_failures = key_data.setdefault("failures", {}).setdefault(model, {})
            model_failures["consecutive_failures"] = 0
            if model in key_data.get("model_cooldowns", {}):
                del key_data["model_cooldowns"][model]

            daily_model_data = key_data["daily"]["models"].setdefault(model, {"success_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "approx_cost": 0.0})
            daily_model_data["success_count"] += 1

            # Safely attempt to record token and cost usage
            if completion_response and hasattr(completion_response, 'usage') and completion_response.usage:
                usage = completion_response.usage
                daily_model_data["prompt_tokens"] += usage.prompt_tokens
                daily_model_data["completion_tokens"] += usage.completion_tokens
                
                try:
                    cost = litellm.completion_cost(completion_response=completion_response)
                    daily_model_data["approx_cost"] += cost
                except Exception as e:
                    lib_logger.warning(f"Could not calculate cost for model {model}: {e}")
            else:
                lib_logger.warning(f"No usage data found in completion response for model {model}. Recording success without token count.")

            key_data["last_used_ts"] = time.time()
        
        await self._save_usage()

    async def record_failure(self, key: str, model: str, classified_error: ClassifiedError):
        """Records a failure and applies cooldowns based on an escalating backoff strategy."""
        await self._lazy_init()
        async with self._data_lock:
            key_data = self._usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}, "failures": {}})
            
            # Handle specific error types first
            if classified_error.error_type == 'rate_limit' and classified_error.retry_after:
                cooldown_seconds = classified_error.retry_after
            elif classified_error.error_type == 'authentication':
                # Apply a 5-minute key-level lockout for auth errors
                key_data["key_cooldown_until"] = time.time() + 300
                lib_logger.warning(f"Authentication error on key ...{key[-4:]}. Applying 5-minute key-level lockout.")
                await self._save_usage()
                return # No further backoff logic needed
            else:
                # General backoff logic for other errors
                failures_data = key_data.setdefault("failures", {})
                model_failures = failures_data.setdefault(model, {"consecutive_failures": 0})
                model_failures["consecutive_failures"] += 1
                count = model_failures["consecutive_failures"]

                backoff_tiers = {1: 10, 2: 30, 3: 60, 4: 120}
                cooldown_seconds = backoff_tiers.get(count, 7200) # Default to 2 hours

            # Apply the cooldown
            model_cooldowns = key_data.setdefault("model_cooldowns", {})
            model_cooldowns[model] = time.time() + cooldown_seconds
            lib_logger.warning(f"Failure recorded for key ...{key[-4:]} with model {model}. Applying {cooldown_seconds}s cooldown.")

            # Check for key-level lockout condition
            await self._check_key_lockout(key, key_data)

            key_data["last_failure"] = {
                "timestamp": time.time(),
                "model": model,
                "error": str(classified_error.original_exception)
            }
        
        await self._save_usage()

    async def _check_key_lockout(self, key: str, key_data: Dict):
        """Checks if a key should be locked out due to multiple model failures."""
        long_term_lockout_models = 0
        now = time.time()
        
        for model, cooldown_end in key_data.get("model_cooldowns", {}).items():
            if cooldown_end - now >= 7200: # Check for 2-hour lockouts
                long_term_lockout_models += 1
        
        if long_term_lockout_models >= 3:
            key_data["key_cooldown_until"] = now + 300 # 5-minute key lockout
            lib_logger.error(f"Key ...{key[-4:]} has {long_term_lockout_models} models in long-term lockout. Applying 5-minute key-level lockout.")
