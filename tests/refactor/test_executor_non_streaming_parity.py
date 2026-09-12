import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from rotator_library.providers.provider_interface import ProviderInterface

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext

from tests.refactor.helpers import FakeResponse, FakeUsageManager, run_async


class DummyProvider(ProviderInterface):
    provider_env_name = "dummy"

    async def get_models(self, api_key, client):
        return []


class FakeUsage:
    prompt_tokens = 5
    completion_tokens = 7
    prompt_tokens_details = SimpleNamespace(cached_tokens=2)


def test_non_streaming_merges_provider_params_and_logger():
    usage_manager = FakeUsageManager("cred-1")
    usage_manager.states["stable-id"] = SimpleNamespace(
        tier="standard-tier",
        priority=2,
        totals=SimpleNamespace(request_count=0, success_count=0, failure_count=0),
        model_usage={},
        group_usage={},
        get_usage_for_scope=lambda *args, **kwargs: None,
    )
    provider_params = {"dummy": {"custom": "ok"}}
    logger_calls = []

    def logger_fn(payload):
        logger_calls.append(payload)

    executor = RequestExecutor(
        usage_managers={"dummy": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_provider_params=provider_params,
        litellm_logger_fn=logger_fn,
    )

    response = FakeResponse(FakeUsage(), {"x-test": "1"})
    captured_kwargs = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return response

    async def run_test():
        context = RequestContext(
            model="dummy/test",
            provider="dummy",
            kwargs={"model": "dummy/test"},
            streaming=False,
            credentials=["cred-1"],
            deadline=9999999999.0,
        )

        with patch(
            "rotator_library.client.executor.litellm.acompletion", fake_acompletion
        ):
            await executor._execute_non_streaming(context)

    run_async(run_test())

    assert captured_kwargs["api_key"] == "cred-1"
    assert captured_kwargs["litellm_params"]["custom"] == "ok"
    assert captured_kwargs["logger_fn"] is logger_fn
    assert usage_manager.last_context is not None
    assert usage_manager.last_context.success_headers == {"x-test": "1"}
