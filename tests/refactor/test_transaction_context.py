import sys
from pathlib import Path

import httpx
from unittest.mock import patch
from types import SimpleNamespace

from rotator_library.providers.provider_interface import ProviderInterface

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext

from tests.refactor.helpers import FakeUsageManager, run_async


class CustomProvider(ProviderInterface):
    provider_env_name = "custom"

    def __init__(self):
        self.kwargs_seen = None

    def has_custom_logic(self):
        return True

    async def get_models(self, api_key, client):
        return []

    async def acompletion(self, client, **kwargs):
        self.kwargs_seen = kwargs

        class Usage:
            prompt_tokens = 1
            completion_tokens = 1
            prompt_tokens_details = None

        class Response:
            usage = Usage()

        return Response()


class DummyLogger:
    def get_context(self):
        return {"trace_id": "abc"}

    # No-op stubs for the transaction-logging calls the executor makes.
    def log_request(self, *args, **kwargs):
        pass

    def log_transformed_request(self, *args, **kwargs):
        pass

    def log_transform_pass(self, *args, **kwargs):
        pass

    def log_response(self, *args, **kwargs):
        pass

    def update_metadata(self, *args, **kwargs):
        pass

    def set_trace_context(self, *args, **kwargs):
        pass


def test_transaction_context_passed_to_custom_provider():
    provider = CustomProvider()
    usage_manager = FakeUsageManager("cred-3")
    usage_manager.states["stable-id"] = SimpleNamespace(
        tier="standard-tier",
        priority=2,
        totals=SimpleNamespace(request_count=0, success_count=0, failure_count=0),
        model_usage={},
        group_usage={},
        get_usage_for_scope=lambda *args, **kwargs: None,
    )
    executor = RequestExecutor(
        usage_managers={"custom": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"custom": provider}),
        provider_transforms=ProviderTransforms({"custom": provider}, None),
        provider_plugins={"custom": provider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
    )

    async def run_test():
        context = RequestContext(
            model="custom/model",
            provider="custom",
            kwargs={"model": "custom/model"},
            streaming=False,
            credentials=["cred-3"],
            deadline=9999999999.0,
            transaction_logger=DummyLogger(),
        )
        await executor._execute_non_streaming(context)

    run_async(run_test())

    assert provider.kwargs_seen is not None
    assert provider.kwargs_seen.get("transaction_context") == {"trace_id": "abc"}
