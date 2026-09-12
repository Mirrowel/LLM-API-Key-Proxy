import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import httpx

from rotator_library.providers.provider_interface import ProviderInterface

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext

from tests.refactor.helpers import FakeUsageManager, run_async


class DummyProvider(ProviderInterface):
    provider_env_name = "dummy"

    async def get_models(self, api_key, client):
        return []


class TrackingExecutor(RequestExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensure_called = False
        self.validate_called = False

    async def _ensure_initialized(self, usage_manager, context, filter_result):
        self.ensure_called = True
        await super()._ensure_initialized(usage_manager, context, filter_result)

    async def _validate_request(self, provider, model, kwargs):
        self.validate_called = True
        await super()._validate_request(provider, model, kwargs)


async def stream_gen():
    yield {"choices": [{"delta": {}}]}
    yield {
        "choices": [{"delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_streaming_initialization_and_params():
    usage_manager = FakeUsageManager("cred-2")
    usage_manager.states["stable-id"] = SimpleNamespace(
        tier="standard-tier",
        priority=2,
        totals=SimpleNamespace(request_count=0, success_count=0, failure_count=0),
        model_usage={},
        group_usage={},
        get_usage_for_scope=lambda *args, **kwargs: None,
    )
    provider_params = {"dummy": {"stream_flag": True}}

    executor = TrackingExecutor(
        usage_managers={"dummy": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_provider_params=provider_params,
        litellm_logger_fn=lambda payload: None,
    )

    captured_kwargs = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return stream_gen()

    async def run_test():
        context = RequestContext(
            model="dummy/test",
            provider="dummy",
            kwargs={"model": "dummy/test", "stream": True},
            streaming=True,
            credentials=["cred-2"],
            deadline=9999999999.0,
        )

        with patch(
            "rotator_library.client.executor.litellm.acompletion", fake_acompletion
        ):
            stream = executor._execute_streaming(context)
            chunks = [chunk async for chunk in stream]
            assert chunks[-1] == "data: [DONE]\n\n"

    run_async(run_test())

    assert executor.ensure_called is True
    assert executor.validate_called is True
    assert captured_kwargs["litellm_params"]["stream_flag"] is True
