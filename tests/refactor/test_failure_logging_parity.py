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
from rotator_library.core.errors import ClassifiedError

from tests.refactor.helpers import FakeUsageManager, run_async


class DummyProvider(ProviderInterface):
    provider_env_name = "dummy"

    async def get_models(self, api_key, client):
        return []


def test_log_failure_called_on_non_streaming_error():
    usage_manager = FakeUsageManager("cred-4")
    usage_manager.states["stable-id"] = SimpleNamespace(
        tier="standard-tier",
        priority=2,
        totals=SimpleNamespace(request_count=0, success_count=0, failure_count=0),
        model_usage={},
        group_usage={},
        get_usage_for_scope=lambda *args, **kwargs: None,
    )
    executor = RequestExecutor(
        usage_managers={"dummy": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
    )

    async def fake_acompletion(**kwargs):
        raise httpx.HTTPStatusError("boom", request=SimpleNamespace(), response=None)

    async def run_test():
        context = RequestContext(
            model="dummy/test",
            provider="dummy",
            kwargs={"model": "dummy/test"},
            streaming=False,
            credentials=["cred-4"],
            deadline=9999999999.0,
        )

        with (
            patch(
                "rotator_library.client.executor.litellm.acompletion", fake_acompletion
            ),
            patch("rotator_library.client.executor.log_failure") as log_failure_mock,
            patch(
                "rotator_library.client.executor.classify_error",
                return_value=ClassifiedError(
                    error_type="rate_limit", original_exception=Exception("boom")
                ),
            ),
        ):
            try:
                await executor._execute_non_streaming(context)
            except Exception:
                pass
            assert log_failure_mock.called

    run_async(run_test())


def test_log_failure_called_on_streaming_error():
    usage_manager = FakeUsageManager("cred-5")
    usage_manager.states["stable-id"] = SimpleNamespace(
        tier="standard-tier",
        priority=2,
        totals=SimpleNamespace(request_count=0, success_count=0, failure_count=0),
        model_usage={},
        group_usage={},
        get_usage_for_scope=lambda *args, **kwargs: None,
    )
    executor = RequestExecutor(
        usage_managers={"dummy": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
    )

    async def fake_acompletion(**kwargs):
        raise httpx.HTTPStatusError("boom", request=SimpleNamespace(), response=None)

    async def run_test():
        context = RequestContext(
            model="dummy/test",
            provider="dummy",
            kwargs={"model": "dummy/test", "stream": True},
            streaming=True,
            credentials=["cred-5"],
            deadline=9999999999.0,
        )

        with (
            patch(
                "rotator_library.client.executor.litellm.acompletion", fake_acompletion
            ),
            patch("rotator_library.client.executor.log_failure") as log_failure_mock,
            patch(
                "rotator_library.client.executor.classify_error",
                return_value=ClassifiedError(
                    error_type="rate_limit", original_exception=Exception("boom")
                ),
            ),
        ):
            stream = executor._execute_streaming(context)
            chunks = [chunk async for chunk in stream]
            assert chunks[-1] == "data: [DONE]\n\n"
            assert log_failure_mock.called

    run_async(run_test())
