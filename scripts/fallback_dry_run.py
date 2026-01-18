import asyncio
import asyncio
import json
import logging
from typing import Any, Dict

try:
    import litellm
except ModuleNotFoundError as exc:
    print("Missing dependency: litellm. Activate venv and install requirements.")
    raise SystemExit(1) from exc

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_root = os.path.join(repo_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

try:
    from rotator_library.client import RotatingClient
    from rotator_library.error_handler import NoAvailableKeysError
except ModuleNotFoundError as exc:
    print("Missing rotator_library module. Run from repository root with venv active.")
    raise SystemExit(1) from exc


class DummyProvider:
    def __init__(self, model_name: str):
        self._model_name = model_name

    async def get_models(self, credential: str, http_client: Any):
        return [self._model_name]

    def has_custom_logic(self) -> bool:
        return False



async def _fake_acompletion(**kwargs: Any):
    model = kwargs.get("model")
    key = kwargs.get("_dry_run_key")
    print(f"Dry run: acompletion attempt for {model} with {key}")
    failure_budget = kwargs.get("_dry_run_failure_budget") or {}
    remaining = failure_budget.get(key, 0)
    if remaining > 0:
        failure_budget[key] = remaining - 1
        print(f"Dry run: simulating rate limit for {model} with {key}")
        raise litellm.RateLimitError("Dry run: simulated rate limit")
    print(f"Dry run: returning success for {model} with {key}")
    return {
        "id": "dry-run",
        "object": "chat.completion",
        "created": 0,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "dry-run"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _fake_streaming_acompletion(**kwargs: Any):
    model = kwargs.get("model")
    if model and model.startswith("gemini_cli/"):
        raise NoAvailableKeysError(f"Dry run: simulated exhaustion for {model}")

    async def _generator():
        yield {
            "id": "dry-run",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model or "unknown",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "dry"}}
            ],
        }
        yield {
            "id": "dry-run",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model or "unknown",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    return _generator()


def _configure_demo_provider(client: RotatingClient, provider: str, model: str):
    client._provider_instances[provider] = DummyProvider(model)
    client._provider_plugins[provider] = DummyProvider


async def run_demo() -> int:
    logging.basicConfig(level=logging.INFO)
    client = RotatingClient(
        api_keys={
            "gemini_cli": [
                "dummy-gemini-key-1",
                "dummy-gemini-key-2",
            ],
            "antigravity": ["dummy-antigravity-key-1"],
        },
        configure_logging=False,
    )

    model_name = "gemini-pro"
    primary_model = f"gemini_cli/{model_name}"
    fallback_model = f"antigravity/{model_name}"

    _configure_demo_provider(client, "gemini_cli", primary_model)
    _configure_demo_provider(client, "antigravity", fallback_model)

    client._model_list_cache["gemini_cli"] = [primary_model]
    client._model_list_cache["antigravity"] = [fallback_model]

    client._convert_model_params_for_litellm = lambda **kwargs: kwargs

    last_key: Dict[str, str] = {}

    def _fake_get_provider_kwargs(**kwargs: Any) -> Dict[str, Any]:
        payload = dict(kwargs)
        payload["_dry_run_key"] = last_key.get("value")
        payload["_dry_run_failure_budget"] = failure_budget
        return payload

    client.all_providers.get_provider_kwargs = _fake_get_provider_kwargs

    failure_budget = {
        "dummy-gemini-key-1": 1,
        "dummy-gemini-key-2": 1,
        "dummy-antigravity-key-1": 0,
    }

    async def _fake_release_key(key: str, model: str):
        return None

    key_cursor: Dict[str, int] = {}

    async def _fake_acquire_key(**kwargs: Any):
        available_keys = kwargs.get("available_keys") or []
        if not available_keys:
            return "dummy-key"
        key = available_keys[key_cursor.get("idx", 0) % len(available_keys)]
        key_cursor["idx"] = key_cursor.get("idx", 0) + 1
        last_key["value"] = key
        return key

    async def _fake_availability_stats(
        creds: Any, model: str, credential_priorities: Any
    ) -> Dict[str, int]:
        return {"available": len(creds), "on_cooldown": 0, "fair_cycle_excluded": 0}

    async def _fake_record_success(*_args: Any, **_kwargs: Any):
        return None

    async def _fake_record_failure(*_args: Any, **_kwargs: Any):
        return None

    async def _fake_acquire_key_proxy(*args: Any, **kwargs: Any):
        return await _fake_acquire_key(**kwargs)

    original_acquire_key = client.usage_manager.acquire_key
    original_release_key = client.usage_manager.release_key
    original_availability = client.usage_manager.get_credential_availability_stats
    original_record_success = client.usage_manager.record_success
    original_record_failure = client.usage_manager.record_failure
    client.usage_manager.acquire_key = _fake_acquire_key_proxy
    client.usage_manager.release_key = _fake_release_key
    client.usage_manager.get_credential_availability_stats = _fake_availability_stats
    client.usage_manager.record_success = _fake_record_success
    client.usage_manager.record_failure = _fake_record_failure

    async def _fake_streaming_impl(
        request: Any, pre_request_callback: Any = None, **kwargs: Any
    ):
        model = kwargs.get("model")
        print(f"Dry run: streaming attempt for {model}")
        try:
            stream_source = await _fake_streaming_acompletion(**kwargs)
        except NoAvailableKeysError:
            if model and model.startswith("gemini_cli/"):
                fallback_model = f"antigravity/{model.split('/', 1)[1]}"
            else:
                fallback_model = "antigravity/gemini-pro"
            print(
                f"Dry run: streaming fallback from {model} to {fallback_model}"
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = fallback_model
            stream_source = await _fake_streaming_acompletion(**fallback_kwargs)

        async for chunk in stream_source:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    original_streaming_impl = client._streaming_acompletion_with_retry
    client._streaming_acompletion_with_retry = _fake_streaming_impl

    original_acompletion = litellm.acompletion
    litellm.acompletion = _fake_acompletion
    try:
        print("Dry run: triggering non-streaming fallback...")
        try:
            response = await client._execute_with_retry(
                litellm.acompletion, request=None, model=primary_model
            )
            if isinstance(response, dict):
                print(
                    "Dry run: non-streaming result model = "
                    f"{response.get('model')}"
                )
        except NoAvailableKeysError:
            print("Non-streaming: fallback exhausted (unexpected in dry run)")

        remaining = await client.cooldown_manager.get_cooldown_remaining("gemini_cli")
        print(f"Primary cooldown remaining: {remaining:.1f}s")
    finally:
        litellm.acompletion = original_acompletion

    original_streaming = litellm.acompletion
    try:
        print("Dry run: triggering streaming fallback...")
        stream = client._streaming_acompletion_with_retry(
            request=None,
            model=primary_model,
            stream=True,
            messages=[{"role": "user", "content": "ping"}],
        )
        saw_done = False
        async for chunk in stream:
            payload = chunk.strip()
            if payload:
                print(payload)
            if payload.endswith("[DONE]"):
                saw_done = True
                break
        if not saw_done:
            print("Streaming: no completion emitted (unexpected in dry run)")
    finally:
        litellm.acompletion = original_streaming
        client._streaming_acompletion_with_retry = original_streaming_impl

        client.usage_manager.acquire_key = original_acquire_key
        client.usage_manager.release_key = original_release_key
        client.usage_manager.get_credential_availability_stats = original_availability
        client.usage_manager.record_success = original_record_success
        client.usage_manager.record_failure = original_record_failure

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_demo()))
