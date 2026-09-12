import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def run_async(coro):
    return asyncio.run(coro)


class FakeCredentialContext:
    def __init__(self, credential: str):
        self.credential = credential
        self.success_headers: Optional[Dict[str, Any]] = None
        self.success_tokens: Optional[Dict[str, int]] = None
        self.success_cost: Optional[float] = None
        self.failure_error: Optional[Any] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(
        self,
        response: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
        response_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.success_headers = response_headers
        self.success_tokens = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "thinking": thinking_tokens,
            "prompt_cached": prompt_tokens_cache_read,
            "prompt_cache_write": prompt_tokens_cache_write,
        }
        self.success_cost = approx_cost

    def mark_failure(self, error: Any) -> None:
        self.failure_error = error


class FakeUsageManager:
    def __init__(self, credential: str = "mock-key"):
        self.credential = credential
        self.initialized = False
        self.last_initialize_args: Optional[Dict[str, Any]] = None
        self.last_acquire_args: Optional[Dict[str, Any]] = None
        self.last_context: Optional[FakeCredentialContext] = None
        self.states = {}
        self.window_manager = SimpleNamespace(get_primary_definition=lambda: None)

    async def initialize(
        self,
        credentials: List[str],
        priorities: Optional[Dict[str, int]] = None,
        tiers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.initialized = True
        self.last_initialize_args = {
            "credentials": credentials,
            "priorities": priorities,
            "tiers": tiers,
        }

    async def acquire_credential(
        self,
        model: str,
        quota_group: Optional[str] = None,
        exclude: Optional[Any] = None,
        candidates: Optional[List[str]] = None,
        priorities: Optional[Dict[str, int]] = None,
        deadline: float = 0.0,
        session_id: Optional[str] = None,
        session_affinity_key: Optional[str] = None,
    ) -> FakeCredentialContext:
        self.last_acquire_args = {
            "model": model,
            "quota_group": quota_group,
            "exclude": exclude,
            "candidates": candidates,
            "priorities": priorities,
            "deadline": deadline,
            "session_id": session_id,
            "session_affinity_key": session_affinity_key,
        }
        credential = candidates[0] if candidates else self.credential
        context = FakeCredentialContext(credential)
        context.stable_id = "stable-id"
        self.last_context = context
        return context

    def get_model_quota_group(self, model: str) -> Optional[str]:
        return None

    async def get_availability_stats(
        self, model: str, quota_group: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "total": 1,
            "available": 1,
            "blocked": 0,
            "blocked_by": {
                "cooldowns": 0,
                "window_limits": 0,
                "custom_caps": 0,
                "fair_cycle": 0,
                "concurrent": 0,
            },
            "rotation_mode": "sequential",
        }


class FakeResponse:
    def __init__(self, usage: Any, headers: Dict[str, Any]):
        self.usage = usage
        self.response = SimpleNamespace(headers=headers)
