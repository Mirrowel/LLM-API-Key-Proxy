"""G1 pin: a deterministic client error fails fast and surfaces itself.

Old behavior: the credential-loop cleanup catch-all swallowed the FAIL
decision, retried the request against every key, and finally returned a
misleading all-credentials-exhausted aggregate. The client never saw their
own schema error.
"""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from rotator_library.providers.provider_interface import ProviderInterface

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext


class DummyProvider(ProviderInterface):
    provider_env_name = "dummy"

    async def get_models(self, api_key, client):
        return []


class FakeCredentialContext:
    def __init__(self, credential: str):
        self.credential = credential
        self.stable_id = "stable-id"
        self.failure_error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(self, **kwargs) -> None:
        pass

    def mark_failure(self, classified) -> None:
        self.failure_error = classified


class FakeUsageManager:
    def __init__(self, credential: str = "cred-1"):
        self.credential = credential
        self.initialized = True
        self.window_manager = SimpleNamespace(get_primary_definition=lambda: None)

    async def initialize(self, credentials=None, priorities=None, tiers=None):
        self.initialized = True

    async def acquire_credential(self, *args, **kwargs):
        return FakeCredentialContext(self.credential)

    def get_model_quota_group(self, model):
        return None

    async def get_availability_stats(self, model, quota_group=None):
        return {"available": 1, "total": 1}


def _provider_error(status: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.example/v1/chat")
    response = httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error"}}, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


def _executor(max_retries: int = 3) -> RequestExecutor:
    return RequestExecutor(
        usage_managers={"dummy": FakeUsageManager("cred-1")},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=max_retries,
        global_timeout=5,
    )


@pytest.mark.asyncio
async def test_deterministic_client_error_escapes_instead_of_exhausting() -> None:
    executor = _executor()
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(1)
        raise _provider_error(400, "tool schema invalid")

    context = RequestContext(
        model="dummy/test",
        provider="dummy",
        kwargs={"model": "dummy/test"},
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
    )
    with patch("rotator_library.client.executor.litellm.acompletion", fake_acompletion):
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await executor._execute_non_streaming(context)

    # The client's own error surfaces verbatim — one attempt, no rotation
    # storm, no all-credentials-exhausted disguise.
    assert "tool schema invalid" in str(raised.value)
    assert len(calls) == 1


def test_fail_decision_unwraps_for_route_classification() -> None:
    from rotator_library.client.executor import _FailDecision, _route_error_type

    original = _provider_error(400, "tool schema invalid")
    assert _route_error_type(_FailDecision(original)) == "invalid_request"
