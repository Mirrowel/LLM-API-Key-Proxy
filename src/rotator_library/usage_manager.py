import json
import os
import time
import logging
import asyncio
from datetime import date
from typing import Dict, List, Optional, Set
from filelock import FileLock
import aiofiles
import litellm
import re

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class UsageManager:
    """
    Manages usage statistics and cooldowns for API keys with asyncio-safe locking,
    asynchronous file I/O, and a lazy-loading mechanism for usage data.
    """
    def __init__(self, file_path: str = "key_usage.json", wait_timeout: int = 5):
        self.file_path = file_path
        self.file_lock = FileLock(f"{self.file_path}.lock")
        self.key_locks: Dict[str, asyncio.Lock] = {}
        self.condition = asyncio.Condition()
        self.wait_timeout = wait_timeout
        
        # Data-related locks and state
        self._data_lock = asyncio.Lock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()

        # For "fair timeout" logic
        self._timeout_lock = asyncio.Lock()
        self._claimed_on_timeout: Set[str] = set()

    async def _lazy_init(self):
        """
        Initializes the usage data by loading it from the file asynchronously.
        This method is called on the first access to ensure data is loaded
        before any operations are performed.
        """
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
            with self.file_lock: # Use filelock to prevent multi-process race conditions
                async with aiofiles.open(self.file_path, 'w') as f:
                    await f.write(json.dumps(self._usage_data, indent=2))

    async def _reset_daily_stats_if_needed(self):
        """Checks if daily stats need to be reset for any key (async version)."""
        if self._usage_data is None:
            return

        today_str = date.today().isoformat()
        needs_saving = False
        for key, data in self._usage_data.items():
            daily_data = data.get("daily", {})
            if daily_data.get("date") != today_str:
                needs_saving = True
                global_data = data.setdefault("global", {"models": {}})
                for model, stats in daily_data.get("models", {}).items():
                    global_model_stats = global_data["models"].setdefault(model, {"success_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "approx_cost": 0.0})
                    global_model_stats["success_count"] += stats.get("success_count", 0)
                    global_model_stats["prompt_tokens"] += stats.get("prompt_tokens", 0)
                    global_model_stats["completion_tokens"] += stats.get("completion_tokens", 0)
                    global_model_stats["approx_cost"] += stats.get("approx_cost", 0.0)
                data["daily"] = {"date": today_str, "models": {}}
        
        if needs_saving:
            await self._save_usage()

    def _initialize_locks(self, keys: List[str]):
        """Initializes asyncio locks for all provided keys if not already present."""
        for key in keys:
            if key not in self.key_locks:
                self.key_locks[key] = asyncio.Lock()

    async def acquire_key(self, available_keys: List[str], model: str) -> str:
        """
        Acquires the best available key with robust locking and a fair timeout mechanism.
        """
        await self._lazy_init()
        self._initialize_locks(available_keys)

        async with self.condition:
            while True:
                eligible_keys = []
                async with self._data_lock:
                    for key in available_keys:
                        key_data = self._usage_data.get(key, {})
                        cooldown_until = key_data.get("model_cooldowns", {}).get(model)
                        if not cooldown_until or time.time() > cooldown_until:
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
                    # Fallback to waiting again if all keys were claimed
                    await asyncio.sleep(1)


    async def release_key(self, key: str):
        """Releases the lock for a given key and notifies waiting tasks."""
        async with self.condition:
            # Also release from timeout claim set if it's there
            async with self._timeout_lock:
                if key in self._claimed_on_timeout:
                    self._claimed_on_timeout.remove(key)

            if key in self.key_locks and self.key_locks[key].locked():
                self.key_locks[key].release()
                lib_logger.info(f"Released lock for key ...{key[-4:]}")
                self.condition.notify()

    async def record_success(self, key: str, model: str, completion_response: litellm.ModelResponse):
        """Records a successful API call asynchronously."""
        await self._lazy_init()
        async with self._data_lock:
            key_data = self._usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}})
            
            if model in key_data.get("model_cooldowns", {}):
                del key_data["model_cooldowns"][model]

            if key_data["daily"].get("date") != date.today().isoformat():
                # This is a simplified reset for the current key. A full reset is done in _lazy_init.
                key_data["daily"] = {"date": date.today().isoformat(), "models": {}}

            daily_model_data = key_data["daily"]["models"].setdefault(model, {"success_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "approx_cost": 0.0})
            
            usage = completion_response.usage
            daily_model_data["success_count"] += 1
            daily_model_data["prompt_tokens"] += usage.prompt_tokens
            daily_model_data["completion_tokens"] += usage.completion_tokens
            
            try:
                cost = litellm.completion_cost(completion_response=completion_response)
                daily_model_data["approx_cost"] += cost
            except Exception as e:
                lib_logger.warning(f"Could not calculate cost for model {model}: {e}")

            key_data["last_used_ts"] = time.time()
        
        await self._save_usage()

    async def record_rotation_error(self, key: str, model: str, error: Exception):
        """Records a rotation error and sets a cooldown asynchronously."""
        await self._lazy_init()
        async with self._data_lock:
            key_data = self._usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}})
            
            cooldown_seconds = 86400
            error_str = str(error).lower()
            
            patterns = [
                r'retry_delay.*?(\d+)',
                r'retrydelay.*?(\d+)s',
                r'wait.*?(\d+)\s*seconds?'
            ]
            for pattern in patterns:
                match = re.search(pattern, error_str, re.IGNORECASE)
                if match:
                    try:
                        cooldown_seconds = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            model_cooldowns = key_data.setdefault("model_cooldowns", {})
            model_cooldowns[model] = time.time() + cooldown_seconds

            key_data["last_rotation_error"] = {
                "timestamp": time.time(),
                "model": model,
                "error": str(error)
            }
        
        await self._save_usage()
