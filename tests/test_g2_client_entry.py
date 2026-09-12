"""G2 client entry: R1-R4 stages, callback preservation, startup validation.

Pins the client-layer slice of the hookable pipeline:
- the public ``pre_request_callback`` keeps its per-attempt in-place kwargs
  mutation contract on BOTH the non-native and native paths;
- a separately-declared ``request_received`` hook fires exactly once per
  request on the same run the native executor uses;
- routing_resolved / credential_selected / session_resolved fire with the
  resolved facts on that run;
- unknown hook names/stages fail at startup, never per request.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.request_builder import RequestContextBuilder
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.config.experimental import load_config_from_mapping
from rotator_library.core.types import RequestContext
from rotator_library.hooks.registry import register_hook
from rotator_library.hooks.runner import PipelineRun
from rotator_library.hooks.types import PipelineHook
from rotator_library.providers import validate_provider_hooks
from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.protocols import get_protocol
from rotator_library.routing import parse_route_target


class EntryRecorder(PipelineHook):
    """Records every client-entry stage on the shared per-request run."""

    name = "entry_recorder"
    stages = (
        "request_received",
        "routing_resolved",
        "credential_selected",
        "session_resolved",
    )

    def __init__(self):
        self.seen: list[tuple[str, object]] = []

    async def __call__(self, invocation, context):
        self.seen.append((invocation.stage, invocation.payload))
        return None


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


class FakeNativeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self):
        self.calls = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return FakeNativeResponse(
            {
                "id": "chat_native",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        )


class NativePlugin:
    def has_custom_logic(self):
        return False

    def get_protocol_name(self, model=""):
        return "openai_chat"

    def get_native_endpoint(self, model="", operation="chat"):
        return "https://native.test/chat"

    def get_native_headers(self, credential_identifier, model="", operation="chat"):
        return {"Authorization": f"Bearer {credential_identifier}"}

    def get_native_operation(self, model="", request=None, stream=False):
        return "chat"

    def normalize_native_model(self, model=""):
        return model.split("/", 1)[1] if "/" in model else model

    def get_adapter_names(self, model=""):
        return ()

    def get_adapter_config(self, model=""):
        return {}

    def get_field_cache_rules(self, model=""):
        return ()


def _non_native_executor(provider_plugin=None) -> RequestExecutor:
    plugin = provider_plugin or DummyProvider
    return RequestExecutor(
        usage_managers={"dummy": FakeUsageManager("cred-1")},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": plugin}),
        provider_transforms=ProviderTransforms({"dummy": plugin}, None),
        provider_plugins={"dummy": plugin},
        http_client=httpx.AsyncClient(),
        max_retries=2,
        global_timeout=5,
    )


def _native_executor(plugin, http_client: FakeHTTPClient) -> RequestExecutor:
    return RequestExecutor(
        usage_managers={},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"provider": plugin}),
        provider_transforms=ProviderTransforms({"provider": plugin}, None),
        provider_plugins={"provider": plugin},
        http_client=http_client,
        max_retries=1,
        global_timeout=5,
    )


def _native_context() -> RequestContext:
    kwargs = {"model": "provider/gpt-test", "messages": [{"role": "user", "content": "hi"}]}
    return RequestContext(
        model="provider/gpt-test",
        provider="provider",
        kwargs=kwargs,
        streaming=False,
        credentials=["cred"],
        deadline=9999999999.0,
        routing_targets=(parse_route_target("provider/gpt-test"),),
        protocol_request=dict(kwargs),
        unified_request=get_protocol("openai_chat").parse_request(kwargs),
        input_provider="provider",
        session_id="session-a",
    )


# --- pre_request_callback contract (kept, not rewritten) -------------------


@pytest.mark.asyncio
async def test_callback_mutation_reaches_non_native_provider() -> None:
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {"id": "ok"}

    async def callback(request, kwargs):
        kwargs["marker"] = "from-callback"

    context = RequestContext(
        model="dummy/test",
        provider="dummy",
        kwargs={"model": "dummy/test", "messages": []},
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
        pre_request_callback=callback,
    )
    with patch("rotator_library.client.executor.litellm.acompletion", fake_acompletion):
        await _non_native_executor()._execute_non_streaming(context)

    assert calls and calls[0]["marker"] == "from-callback"


@pytest.mark.asyncio
async def test_callback_mutation_reaches_native_provider() -> None:
    async def callback(request, kwargs):
        kwargs["messages"] = [{"role": "user", "content": "MUTATED"}]

    context = _native_context()
    context.routing_target_index = 0
    context.pre_request_callback = callback
    http = FakeHTTPClient()
    executor = _native_executor(NativePlugin(), http)
    plugin = executor._get_plugin_instance("provider")
    kwargs = dict(context.kwargs)

    await executor._run_pre_request_callback(context, kwargs)
    await executor._execute_provider_request(
        "provider", "provider/gpt-test", plugin, "secret", "stable", kwargs, context
    )

    assert http.calls[0]["json"]["messages"][0]["content"] == "MUTATED"


# --- R1-R4 stage firing -----------------------------------------------------


class FakeModelResolver:
    def resolve_model_id(self, model, provider):
        return model


class FakeSession:
    session_id = "session"
    affinity_key = "affinity"
    tracking_namespace = "namespace"
    confidence = 0.9


class FakeSessionTracker:
    def infer_session(self, *args, **kwargs):
        return FakeSession()


async def _scope(provider, classifier, request_api_keys, request_providers, private):
    return {
        "credentials": [f"{provider}-cred"],
        "usage_manager_key": provider,
        "provider_config": {"provider": provider},
        "credential_secrets": {f"{provider}-cred": f"{provider}-secret"},
        "classifier": classifier or "global",
    }


@pytest.mark.asyncio
async def test_entry_stages_fire_once_on_one_run(monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_GROUPS", "entry_chain")
    monkeypatch.setenv("FALLBACK_GROUP_ENTRY_CHAIN", "hooked/hooked-model")
    monkeypatch.setenv("MODEL_ROUTE_HOOKED", "group:entry_chain")

    recorder = EntryRecorder()

    class HookedPlugin:
        hooks = (recorder,)

    builder = RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=FakeModelResolver(),
        session_tracker=FakeSessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
        get_provider_instance=lambda provider: HookedPlugin(),
    )

    context = await builder.build_completion_context(
        None, None, {"model": "hooked", "messages": []}
    )

    # Provider-bound hooks bind at enrichment — AFTER request_received
    # fired (provider unknown at entry). They see the post-routing stages.
    stages = [stage for stage, _ in recorder.seen]
    assert stages == ["routing_resolved", "session_resolved"]
    assert context.pipeline_run is not None
    payloads = dict(recorder.seen)
    assert payloads["routing_resolved"]["targets"][0]["provider"] == "hooked"
    assert payloads["routing_resolved"]["targets"][0]["model"] == "hooked/hooked-model"
    assert payloads["session_resolved"]["session_id"] == "session"


@pytest.mark.asyncio
async def test_request_received_hook_edits_are_authoritative(monkeypatch) -> None:
    """Global hooks intercept request_received and their edits reach the wire."""
    from rotator_library.hooks.registry import register_hook
    from rotator_library.hooks.types import PipelineHook

    class EntryMutator(PipelineHook):
        name = "entry_mutator_g2"
        stages = ("request_received",)
        stateful = False

        async def __call__(self, invocation, context):
            payload = dict(invocation.payload)
            payload["messages"] = [{"role": "user", "content": "rewritten by entry hook"}]
            payload["temperature"] = 0.123
            return payload

    register_hook(EntryMutator, replace=True)

    class _StubConfig:
        hooks = {"global": ["entry_mutator_g2"]}

    monkeypatch.setenv("FALLBACK_GROUPS", "entry_chain")
    monkeypatch.setenv("FALLBACK_GROUP_ENTRY_CHAIN", "hooked/hooked-model")
    monkeypatch.setenv("MODEL_ROUTE_HOOKED", "group:entry_chain")

    builder = RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=FakeModelResolver(),
        session_tracker=FakeSessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
        get_provider_instance=lambda provider: None,
        experimental_config=_StubConfig(),
    )

    context = await builder.build_completion_context(
        None, None, {"model": "hooked", "messages": []}
    )
    # The edit is authoritative: the request kwargs carry the rewritten
    # payload downstream (model resolution, routing, and the native raw
    # basis all see it).
    assert context.kwargs["messages"] == [{"role": "user", "content": "rewritten by entry hook"}]
    assert context.kwargs["temperature"] == 0.123
    # And the protocol snapshot (taken AFTER the entry slot) sees it too.
    assert context.protocol_request["messages"] == [{"role": "user", "content": "rewritten by entry hook"}]
    # The rewrite is recorded — nothing invisible.
    assert any(o.get("kind") == "request_received_edit" for o in context.pipeline_run.overlays)


@pytest.mark.asyncio
async def test_request_pipeline_run_is_shared_with_native_executor() -> None:
    run = PipelineRun(request_id="req-1")
    context = _native_context()
    context.pipeline_run = run
    context.routing_target_index = 0
    http = FakeHTTPClient()
    executor = _native_executor(NativePlugin(), http)
    plugin = executor._get_plugin_instance("provider")

    native_context = executor._build_native_provider_context(
        "provider",
        "provider/gpt-test",
        plugin,
        "secret",
        "stable",
        context,
        context.routing_targets[0],
    )

    assert native_context.pipeline_run is run
    assert context.pipeline_run is run
    assert native_context.hook_global_names == ()


@pytest.mark.asyncio
async def test_credential_selected_fires_with_resolved_credential() -> None:
    recorder = EntryRecorder()

    class HookedDummy(DummyProvider):
        provider_env_name = "dummy"
        hooks = (recorder,)

    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {"id": "ok"}

    context = RequestContext(
        model="dummy/test",
        provider="dummy",
        kwargs={"model": "dummy/test", "messages": []},
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
    )
    with patch("rotator_library.client.executor.litellm.acompletion", fake_acompletion):
        await _non_native_executor(HookedDummy)._execute_non_streaming(context)

    selected = [payload for stage, payload in recorder.seen if stage == "credential_selected"]
    assert selected == [
        {"credential_id": "stable-id", "provider": "dummy", "model": "dummy/test"}
    ]


# --- startup validation -----------------------------------------------------


def test_startup_validation_rejects_unknown_hook_name() -> None:
    config = load_config_from_mapping(
        {"providers": {"dummy": {"hooks": ["definitely_missing_hook"]}}}
    )
    with pytest.raises(KeyError):
        validate_provider_hooks(config)


def test_startup_validation_rejects_unknown_stage() -> None:
    config = load_config_from_mapping(
        {
            "providers": {
                "dummy": {
                    "hooks": [{"name": "definitely_missing_hook", "stages": ["not_a_stage"]}]
                }
            }
        }
    )
    with pytest.raises(ValueError):
        validate_provider_hooks(config)


def test_startup_validation_accepts_registered_global_and_provider_hooks() -> None:
    register_hook(EntryRecorder, replace=True, name="g2_entry_recorder")
    config = load_config_from_mapping(
        {
            "hooks": {"global": ["g2_entry_recorder"]},
            "providers": {"dummy": {"hooks": ["g2_entry_recorder"]}},
        }
    )
    validate_provider_hooks(config)
