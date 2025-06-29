import json
import os
import time
import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from filelock import FileLock
import litellm

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class UsageManager:
    """
    Manages daily and global usage statistics and cooldowns for API keys.
    """
    def __init__(self, file_path: str = "key_usage.json"):
        self.file_path = file_path
        self.lock = FileLock(f"{self.file_path}.lock")
        self.usage_data = self._load_usage()
        self._reset_daily_stats_if_needed()

    def _load_usage(self) -> Dict:
        with self.lock:
            if not os.path.exists(self.file_path):
                return {}
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

    def _save_usage(self):
        with self.lock:
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

    def get_next_smart_key(self, available_keys: List[str], model: str) -> Optional[str]:
        """
        Gets the least-used, available key for a specific model, considering model-specific cooldowns.
        """
        best_key = None
        min_usage = float('inf')

        active_keys = []
        for key in available_keys:
            key_data = self.usage_data.get(key, {})
            model_cooldowns = key_data.get("model_cooldowns", {})
            cooldown_until = model_cooldowns.get(model)

            if not cooldown_until or time.time() > cooldown_until:
                active_keys.append(key)

        if not active_keys:
            return None

        # Find the key with the minimum daily success_count for the given model
        for key in active_keys:
            key_data = self.usage_data.setdefault(key, {"daily": {"date": date.today().isoformat(), "models": {}}, "global": {"models": {}}, "model_cooldowns": {}})
            daily_model_usage = key_data.get("daily", {}).get("models", {}).get(model, {})
            usage_count = daily_model_usage.get("success_count", 0)

            if usage_count < min_usage:
                min_usage = usage_count
                best_key = key
        
        return best_key if best_key else active_keys[0]

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
                delay_str = error_str.split("retry_delay")[1].split("seconds:")[1].strip().split("}")[0]
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
