import json
import os
import time
import logging
import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from filelock import FileLock
import litellm
import re

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class UsageManager:
    """
    Manages usage statistics and cooldowns for API keys with asyncio-safe locking.
    """
    def __init__(self, file_path: str = "key_usage.json", wait_timeout: int = 5):
        self.file_path = file_path
        self.file_lock = FileLock(f"{self.file_path}.lock")
        self.key_locks: Dict[str, asyncio.Lock] = {}
        self.condition = asyncio.Condition()
        self.wait_timeout = wait_timeout
        self.usage_data = self._load_usage()
        self._reset_daily_stats_if_needed()

    def _load_usage(self) -> Dict:
        with self.file_lock:
            if not os.path.exists(self.file_path):
                return {}
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

    def _save_usage(self):
        with self.file_lock:
            with open(self.file_path, 'w') as f:
                json.dump(self.usage_data, f, indent=2)

    def _reset_daily_stats_if_needed(self):
        """Checks if daily stats need to be reset for any key."""
        today_str = date.today().isoformat()
        needs_saving = False
        for key, data in self.usage_data.items():
            daily_data = data.get("daily", {})
            last_date_str = daily_data.get("date")
            if last_date_str != today_str:
                needs_saving = True
                # Add yesterday's daily stats to global stats
                global_data = data.setdefault("global", {"models": {}})
                for model, stats in daily_data.get("models", {}).items():
                    global_model_stats = global_data["models"].setdefault(model, {"success_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "approx_cost": 0.0})
                    global_model_stats["success_count"] += stats.get("success_count", 0)
                    global_model_stats["prompt_tokens"] += stats.get("prompt_tokens", 0)
                    global_model_stats["completion_tokens"] += stats.get("completion_tokens", 0)
                    global_model_stats["approx_cost"] += stats.get("approx_cost", 0.0)
                
                # Reset daily stats
                data["daily"] = {"date": today_str, "models": {}}
        
        if needs_saving:
            self._save_usage()

    def _initialize_locks(self, keys: List[str]):
        """Initializes asyncio locks for all provided keys if not already present."""
        for key in keys:
            if key not in self.key_locks:
                self.key_locks[key] = asyncio.Lock()

    async def acquire_key(self, available_keys: List[str], model: str) -> str:
        """
        Acquires the best available key. If all are locked, waits for one to be
        released or times out and returns the best-ranked key anyway.
        """
        self._initialize_locks(available_keys)

        async with self.condition:
            while True:
                # Rank all keys that are not on cooldown
                eligible_keys = []
                for key in available_keys:
                    key_data = self.usage_data.get(key, {})
                    cooldown_until = key_data.get("model_cooldowns", {}).get(model)
                    if not cooldown_until or time.time() > cooldown_until:
                        usage_count = key_data.get("daily", {}).get("models", {}).get(model, {}).get("success_count", 0)
                        eligible_keys.append((key, usage_count))
                
                if not eligible_keys:
                    lib_logger.warning("All keys are on cooldown. Waiting...")
                    await asyncio.sleep(5)
                    continue

                # Sort by usage count (ascending)
                eligible_keys.sort(key=lambda x: x[1])
                
                # Try to acquire the lock for the first unlocked key in the ranked list
                for key, _ in eligible_keys:
                    lock = self.key_locks[key]
                    if not lock.locked():
                        await lock.acquire()
                        lib_logger.info(f"Acquired lock for available key: ...{key[-4:]}")
                        return key

                # If all eligible keys are locked, wait for a notification or timeout
                best_locked_key = eligible_keys[0][0]
                lib_logger.info(f"All eligible keys are locked. Waiting for a key to be released. Best candidate: ...{best_locked_key[-4:]}")
                
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=self.wait_timeout)
                    # If wait() returns, it means we were notified, so we re-run the loop
                    lib_logger.info("Notified that a key was released. Re-evaluating...")
                    continue
                except asyncio.TimeoutError:
                    # If we time out, we take the best-ranked key, even if it's locked
                    lib_logger.warning(f"Wait timed out. Proceeding with best-ranked locked key: ...{best_locked_key[-4:]}")
                    return best_locked_key

    async def release_key(self, key: str):
        """Releases the lock for a given key and notifies waiting tasks."""
        async with self.condition:
            if key in self.key_locks and self.key_locks[key].locked():
                self.key_locks[key].release()
                lib_logger.info(f"Released lock for key ...{key[-4:]}")
                self.condition.notify() # Notify one waiting task

    def record_success(self, key: str, model: str, completion_response: litellm.ModelResponse):
        key_data = self.usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}})
        
        # Clear any cooldown for this specific model on success
        if model in key_data.get("model_cooldowns", {}):
            del key_data["model_cooldowns"][model]

        # Ensure daily stats are for today
        if key_data["daily"].get("date") != date.today().isoformat():
            self._reset_daily_stats_if_needed()
            key_data = self.usage_data[key]

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
        self._save_usage()

    def record_rotation_error(self, key: str, model: str, error: Exception):
        key_data = self.usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}})
        
        cooldown_seconds = 86400  # Default cooldown of 24 hours
        
        error_str = str(error).lower()
        if "retry_delay" in error_str:
            try:
                # Try multiple patterns to extract delay from error message
                delay_str = None
                
                # Pattern 1: retry_delay...seconds format
                if "retry_delay" in error_str and "seconds:" in error_str:
                    try:
                        delay_str = error_str.split("retry_delay")[1].split("seconds:")[1].strip().split("}")[0]
                    except (IndexError, AttributeError):
                        pass
                
                # Pattern 2: retryDelay with 's' suffix (Gemini format)
                if not delay_str and "retrydelay" in error_str:
                    try:
                        match = re.search(r'"retrydelay":\s*"(\d+)s"', error_str)
                        if match:
                            delay_str = match.group(1)
                    except Exception:
                        pass
                
                # Pattern 3: Generic numeric extraction for retry/delay contexts
                if not delay_str:
                    try:
                        # Look for numbers followed by 's' or 'seconds' in retry/delay context
                        patterns = [
                            r'retry.*?(\d+)s',
                            r'delay.*?(\d+)s', 
                            r'wait.*?(\d+)\s*seconds?'
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, error_str, re.IGNORECASE)
                            if match:
                                delay_str = match.group(1)
                                break
                    except Exception:
                        pass
                
                if delay_str:
                    cooldown_seconds = int(delay_str)
                cooldown_seconds = int(delay_str)
            except (IndexError, ValueError):
                pass

        model_cooldowns = key_data.setdefault("model_cooldowns", {})
        model_cooldowns[model] = time.time() + cooldown_seconds

        key_data["last_rotation_error"] = {
            "timestamp": time.time(),
            "model": model,
            "error": str(error)
        }
        self._save_usage()
