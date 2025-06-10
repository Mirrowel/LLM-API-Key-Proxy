import json
import os
import time
from typing import Dict, List, Optional
from filelock import FileLock

class UsageManager:
    """
    Manages detailed usage and failure data for API keys, stored in a JSON file.
    """
    def __init__(self, file_path: str = "key_usage.json"):
        self.file_path = file_path
        self.lock = FileLock(f"{self.file_path}.lock")
        self.usage_data = self._load_usage()

    def _load_usage(self) -> Dict:
        """Loads usage data from the JSON file."""
        with self.lock:
            if not os.path.exists(self.file_path):
                return {}
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

    def _save_usage(self):
        """Saves the current usage data to the JSON file."""
        with self.lock:
            with open(self.file_path, 'w') as f:
                json.dump(self.usage_data, f, indent=2)

    def get_next_smart_key(self, available_keys: List[str], model: str, excluded_keys: List[str]) -> Optional[str]:
        """
        Finds the best key to use based on the lowest usage count for the given model.
        """
        best_key = None
        min_usage = float('inf')

        eligible_keys = [k for k in available_keys if k not in excluded_keys]

        if not eligible_keys:
            return None

        # Initialize all available keys in usage data if they aren't present
        for key in eligible_keys:
            self.usage_data.setdefault(key, {"models": {}, "last_rotation_error": None})

        # Find the key with the minimum success_count for the given model
        for key in eligible_keys:
            model_usage = self.usage_data[key].get("models", {}).get(model, {})
            usage_count = model_usage.get("success_count", 0)

            if usage_count < min_usage:
                min_usage = usage_count
                best_key = key
        
        # If all have the same usage count, it will pick the first one in the list
        return best_key if best_key else eligible_keys[0]


    def record_success(self, key: str, model: str, usage: Dict):
        """Records a successful API call and its token usage."""
        key_data = self.usage_data.setdefault(key, {"models": {}, "last_rotation_error": None})
        model_data = key_data["models"].setdefault(model, {
            "success_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        })
        
        model_data["success_count"] += 1
        model_data["prompt_tokens"] += usage.get("prompt_tokens", 0)
        model_data["completion_tokens"] += usage.get("completion_tokens", 0)
        
        key_data["last_used_ts"] = time.time()
        self._save_usage()

    def record_rotation_error(self, key: str, model: str, error: str):
        """Records the error that caused a key to be rotated."""
        key_data = self.usage_data.setdefault(key, {"models": {}, "last_rotation_error": None})
            
        key_data["last_rotation_error"] = {
            "timestamp": time.time(),
            "model": model,
            "error": error
        }
        self._save_usage()
