import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library import RotatingClient
from rotator_library.usage import UsageManager


class FakeUsage:
    prompt_tokens = 3
    completion_tokens = 5
    prompt_tokens_details = SimpleNamespace(cached_tokens=1)
    completion_tokens_details = SimpleNamespace(reasoning_tokens=0)


class FakeResponse:
    def __init__(self):
        self.usage = FakeUsage()
        self.response = SimpleNamespace(headers={"x-test": "ok"})


def run_async(coro):
    return asyncio.run(coro)


async def _close(client):
    await client.close()


def _make_client(tmp_path, **kwargs):
    return RotatingClient(
        data_dir=tmp_path,
        configure_logging=False,
        **kwargs,
    )


def test_default_completion_keeps_backward_compatible_global_pool(tmp_path):
    captured = {}
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    async def run_test():
        try:
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="openai/test-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            await _close(client)

    run_async(run_test())

    assert captured["api_key"] == "global-openai-key"
    assert captured["model"] == "openai/test-model"
    assert "openai" in client.usage_managers
    assert all(not key.startswith("classifier:") for key in client.usage_managers)


def test_stateless_private_completion_uses_only_scoped_secret_and_provider_overlay(tmp_path):
    captured = {}
    client = _make_client(
        tmp_path,
        api_keys={
            "openai": ["global-openai-key"],
            "logfare": ["global-logfare-key"],
        },
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    async def run_test():
        try:
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="logfare/scoped-model",
                    messages=[{"role": "user", "content": "hi"}],
                    classifier="user/one",
                    api_keys={
                        "logfare": ["scoped-logfare-secret"],
                        "openai": ["irrelevant-openai-secret"],
                    },
                    providers={
                        "logfare": {"base_url": "https://tenant.example/v1"}
                    },
                    private=True,
                )
        finally:
            await _close(client)

    run_async(run_test())

    assert captured["api_key"] == "scoped-logfare-secret"
    assert captured["model"] == "openai/scoped-model"
    assert captured["api_base"] == "https://tenant.example/v1"
    assert captured["custom_llm_provider"] == "openai"
    assert captured["api_key"] != "global-logfare-key"
    assert captured["api_key"] != "irrelevant-openai-secret"

    safe_classifier = client._safe_scope_name("user/one")
    usage_file = (
        Path(tmp_path)
        / "usage"
        / "classifiers"
        / safe_classifier
        / "usage_logfare.json"
    )
    usage_data = json.loads(usage_file.read_text())
    serialized = json.dumps(usage_data)
    assert "scoped-logfare-secret" not in serialized
    assert "global-logfare-key" not in serialized
    assert usage_data["accessor_index"] == {}

    credential_state = next(iter(usage_data["credentials"].values()))
    assert credential_state["accessor"].startswith("private:")
    assert credential_state["private"] is True


def test_streaming_scoped_completion_resolves_secret_at_call_boundary(tmp_path):
    captured = {}
    wrapped = {}
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_stream():
        yield {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return fake_stream()

    async def fake_wrap_stream(
        self, stream, credential, model, request, cred_context, **kwargs
    ):
        wrapped["credential"] = credential
        wrapped["model"] = model
        wrapped["stream"] = stream
        yield "data: [DONE]\n\n"

    async def run_test():
        try:
            with (
                patch(
                    "rotator_library.client.executor.litellm.acompletion",
                    fake_acompletion,
                ),
                patch(
                    "rotator_library.client.executor.StreamingHandler.wrap_stream",
                    fake_wrap_stream,
                ),
            ):
                stream = await client.acompletion(
                    model="logfare/stream-model",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                    classifier="stream-user",
                    api_keys={"logfare": ["stream-secret"]},
                    providers={"logfare": {"base_url": "https://stream.example/v1"}},
                    private=True,
                )
                chunks = [chunk async for chunk in stream]
                assert chunks == ["data: [DONE]\n\n"]
        finally:
            await _close(client)

    run_async(run_test())

    assert captured["api_key"] == "stream-secret"
    assert captured["stream"] is True
    assert captured["api_base"] == "https://stream.example/v1"
    assert wrapped["credential"].startswith("private:")
    assert wrapped["credential"] != "stream-secret"


def test_classifier_never_falls_back_to_global_keys(tmp_path):
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def run_test():
        try:
            try:
                await client.acompletion(
                    model="openai/test-model",
                    messages=[{"role": "user", "content": "hi"}],
                    classifier="user-without-openai",
                )
            except ValueError as exc:
                assert "no credentials" in str(exc)
            else:
                raise AssertionError("classified request unexpectedly used global key")

            models = await client.get_available_models(
                "openai", classifier="user-without-openai"
            )
            assert models == []
        finally:
            await _close(client)

    run_async(run_test())


def test_request_overlay_overrides_registered_scope_without_mutating_it(tmp_path):
    captured = {}
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    async def run_test():
        try:
            await client.register_scope(
                "tenant",
                providers={"logfare": {"base_url": "https://registered.example/v1"}},
                api_keys={"logfare": ["registered-secret"]},
                private=True,
            )
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="logfare/model",
                    messages=[{"role": "user", "content": "hi"}],
                    classifier="tenant",
                    api_keys={"logfare": ["request-secret"]},
                    providers={"logfare": {"base_url": "https://request.example/v1"}},
                    private=True,
                )
            registered = await client.get_scope("tenant", include_secrets=True)
            assert registered["providers"]["logfare"]["base_url"] == "https://registered.example/v1"
            assert registered["credentials"]["logfare"]["keys"] == ["registered-secret"]
        finally:
            await _close(client)

    run_async(run_test())

    assert captured["api_key"] == "request-secret"
    assert captured["api_base"] == "https://request.example/v1"


def test_registered_scope_management_add_set_remove_fetches_without_secret_leak(tmp_path):
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def run_test():
        try:
            await client.register_scope(
                "tenant",
                providers={"logfare": {"base_url": "https://one.example/v1"}},
                api_keys={"logfare": ["secret-one"]},
                private=True,
            )
            public_scope = await client.get_scope("tenant")
            assert public_scope["credentials"]["logfare"] == {
                "count": 1,
                "private": True,
            }
            assert "keys" not in public_scope["credentials"]["logfare"]

            await client.add_scope_provider(
                "tenant", "other", {"base_url": "https://other.example/v1"}
            )
            providers = await client.list_scope_providers("tenant")
            assert providers["other"]["base_url"] == "https://other.example/v1"

            await client.update_scope_provider(
                "tenant", "other", {"base_url": "https://updated.example/v1"}
            )
            providers = await client.list_scope_providers("tenant")
            assert providers["other"]["base_url"] == "https://updated.example/v1"

            await client.add_scope_credentials(
                "tenant", "logfare", ["secret-two"], private=True
            )
            credentials = await client.list_scope_credentials("tenant", "logfare")
            assert len(credentials["logfare"]["credential_ids"]) == 2
            assert all(
                credential_id.startswith("private:")
                for credential_id in credentials["logfare"]["credential_ids"]
            )

            await client.remove_scope_credentials(
                "tenant",
                "logfare",
                [credentials["logfare"]["credential_ids"][0]],
            )
            credentials = await client.list_scope_credentials("tenant", "logfare")
            assert len(credentials["logfare"]["credential_ids"]) == 1

            await client.set_scope_credentials(
                "tenant", "logfare", ["replacement"], private=False
            )
            credentials = await client.list_scope_credentials("tenant", "logfare")
            assert credentials["logfare"] == {
                "credential_ids": ["replacement"],
                "private": False,
            }

            await client.remove_scope_credentials("tenant", "logfare")
            assert (await client.list_scope_credentials("tenant", "logfare"))["logfare"] == {
                "credential_ids": [],
                "private": True,
            }

            await client.remove_scope_provider("tenant", "other")
            assert "other" not in await client.list_scope_providers("tenant")

            await client.remove_scope("tenant")
            assert await client.get_scope("tenant") == {
                "classifier": "tenant",
                "providers": {},
                "credentials": {},
            }
        finally:
            await _close(client)

    run_async(run_test())


def test_model_discovery_is_scoped_cached_and_filterable_per_classifier(tmp_path):
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["authorization"]))
        if "tenant-a" in str(request.url):
            return httpx.Response(
                200,
                json={"data": [{"id": "visible-a"}, {"id": "hidden-a"}]},
            )
        return httpx.Response(200, json={"data": [{"id": "visible-b"}]})

    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def run_test():
        try:
            await client.http_client.aclose()
            client.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            await client.register_scope(
                "tenant-a",
                providers={"logfare": {"base_url": "https://tenant-a.example/v1"}},
                api_keys={"logfare": ["secret-a"]},
                private=True,
            )
            await client.register_scope(
                "tenant-b",
                providers={"logfare": {"base_url": "https://tenant-b.example/v1"}},
                api_keys={"logfare": ["secret-b"]},
                private=True,
            )

            models_a = await client.get_available_models(
                "logfare",
                classifier="tenant-a",
                model_filters={"logfare": {"blacklist": ["*/hidden-a"]}},
            )
            models_a_cached = await client.get_available_models(
                "logfare",
                classifier="tenant-a",
                model_filters={"logfare": {"blacklist": ["*/hidden-a"]}},
            )
            models_b = await client.get_available_models("logfare", classifier="tenant-b")
            grouped = await client.get_all_available_models(grouped=True, classifier="tenant-a")
            flat = await client.get_all_available_models(grouped=False, classifier="tenant-a")

            assert models_a == ["logfare/visible-a"]
            assert models_a_cached == ["logfare/visible-a"]
            assert models_b == ["logfare/visible-b"]
            assert grouped == {"logfare": ["logfare/visible-a", "logfare/hidden-a"]}
            assert flat == ["logfare/visible-a", "logfare/hidden-a"]
        finally:
            await _close(client)

    run_async(run_test())

    assert seen.count(("https://tenant-a.example/v1/models", "Bearer secret-a")) == 2
    assert seen.count(("https://tenant-b.example/v1/models", "Bearer secret-b")) == 1


def test_model_filters_do_not_propagate_to_classifiers_by_default(tmp_path):
    client = _make_client(
        tmp_path,
        api_keys={"logfare": ["global-logfare-key"]},
        ignore_models={"logfare": ["*/hidden"]},
        whitelist_models={"logfare": ["*/always-allowed"]},
    )

    try:
        assert client._is_scoped_model_allowed(
            "logfare/hidden", "logfare", None, None
        ) is False
        assert client._is_scoped_model_allowed(
            "logfare/always-allowed", "logfare", None, None
        ) is True

        assert client._is_scoped_model_allowed(
            "logfare/hidden", "logfare", "tenant", None
        ) is True
        assert client._is_scoped_model_allowed(
            "logfare/hidden",
            "logfare",
            "tenant",
            {"logfare": {"blacklist": ["*/hidden"]}},
        ) is False
        assert client._is_scoped_model_allowed(
            "logfare/visible",
            "logfare",
            "tenant",
            {"logfare": {"whitelist": ["*/visible"]}},
        ) is True
        assert client._is_scoped_model_allowed(
            "logfare/other",
            "logfare",
            "tenant",
            {"logfare": {"whitelist": ["*/visible"]}},
        ) is False
    finally:
        run_async(_close(client))


def test_private_stats_hide_full_path_and_quota_stats_are_classifier_scoped(tmp_path):
    captured = {}
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    async def run_test():
        try:
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="logfare/scoped-model",
                    messages=[{"role": "user", "content": "hi"}],
                    classifier="stats-user",
                    api_keys={"logfare": ["stats-secret"]},
                    providers={"logfare": {"base_url": "https://stats.example/v1"}},
                    private=True,
                )

            default_stats = await client.get_quota_stats()
            scoped_stats = await client.get_quota_stats(classifier="stats-user")
            assert default_stats["providers"] == {}
            assert len(scoped_stats["providers"]) == 1
            provider_stats = next(iter(scoped_stats["providers"].values()))
            credential_stats = next(iter(provider_stats["credentials"].values()))
            assert credential_stats["private"] is True
            assert credential_stats["full_path"] is None
            assert "stats-secret" not in json.dumps(scoped_stats)
        finally:
            await _close(client)

    run_async(run_test())

    assert captured["api_key"] == "stats-secret"


def test_usage_manager_resyncs_active_credentials_while_preserving_history(tmp_path):
    usage_file = Path(tmp_path) / "usage.json"
    manager = UsageManager(provider="openai", file_path=usage_file)

    async def run_test():
        await manager.initialize(["key-one"])
        await manager.record_usage("key-one", "openai/model", success=True, prompt_tokens=1)
        assert len(manager.states) == 1

        await manager.initialize(["key-two"])
        assert len(manager.states) == 2
        stats = await manager.get_stats_for_endpoint()
        assert stats["credential_count"] == 1
        active = next(iter(stats["credentials"].values()))
        assert active["full_path"] == "key-two"
        await manager.shutdown()

    run_async(run_test())


def test_executor_resyncs_stateless_scoped_credentials_after_first_request(tmp_path):
    captured_keys = []
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_acompletion(**kwargs):
        captured_keys.append(kwargs["api_key"])
        return FakeResponse()

    async def run_test():
        try:
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="logfare/model",
                    messages=[{"role": "user", "content": "first"}],
                    classifier="stateless-user",
                    api_keys={"logfare": ["first-secret"]},
                    providers={"logfare": {"base_url": "https://tenant.example/v1"}},
                    private=True,
                )
                await client.acompletion(
                    model="logfare/model",
                    messages=[{"role": "user", "content": "second"}],
                    classifier="stateless-user",
                    api_keys={"logfare": ["second-secret"]},
                    providers={"logfare": {"base_url": "https://tenant.example/v1"}},
                    private=True,
                )
        finally:
            await _close(client)

    run_async(run_test())

    assert captured_keys == ["first-secret", "second-secret"]


def test_executor_resyncs_registered_scope_after_set_credentials(tmp_path):
    captured_keys = []
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def fake_acompletion(**kwargs):
        captured_keys.append(kwargs["api_key"])
        return FakeResponse()

    async def run_test():
        try:
            await client.register_scope(
                "tenant-resync",
                providers={"logfare": {"base_url": "https://tenant.example/v1"}},
                api_keys={"logfare": ["registered-one"]},
                private=True,
            )
            with patch(
                "rotator_library.client.executor.litellm.acompletion",
                fake_acompletion,
            ):
                await client.acompletion(
                    model="logfare/model",
                    messages=[{"role": "user", "content": "first"}],
                    classifier="tenant-resync",
                )
                await client.set_scope_credentials(
                    "tenant-resync", "logfare", ["registered-two"], private=True
                )
                await client.acompletion(
                    model="logfare/model",
                    messages=[{"role": "user", "content": "second"}],
                    classifier="tenant-resync",
                )
        finally:
            await _close(client)

    run_async(run_test())

    assert captured_keys == ["registered-one", "registered-two"]


def test_scoped_usage_manager_inherits_rotation_tolerance_and_reset_config(tmp_path):
    class ResettableProvider:
        def get_usage_reset_config(self, credential):
            assert credential == "scoped-reset-secret"
            return {"window_seconds": 7200, "mode": "credential"}

    client = _make_client(
        tmp_path,
        api_keys={"openai": ["global-openai-key"]},
        rotation_tolerance=0.42,
    )

    async def run_test():
        try:
            with patch.dict(
                "rotator_library.client.rotating_client.PROVIDER_PLUGINS",
                {"resettable": ResettableProvider},
            ):
                scope = await client._resolve_scope_for_provider(
                    "resettable",
                    "reset-user",
                    {"resettable": ["scoped-reset-secret"]},
                    None,
                    True,
                )
                manager = client.usage_managers[scope["usage_manager_key"]]
                assert manager._config.rotation_tolerance == 0.42
                assert manager._config.windows[0].duration_seconds == 7200
                assert manager._config.windows[0].name == "2h"
        finally:
            await _close(client)

    run_async(run_test())


def test_model_discovery_cache_key_changes_with_scoped_credentials(tmp_path):
    seen = []

    def handler(request):
        auth = request.headers["authorization"]
        seen.append(auth)
        model_id = "model-a" if auth == "Bearer secret-a" else "model-b"
        return httpx.Response(200, json={"data": [{"id": model_id}]})

    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    async def run_test():
        try:
            await client.http_client.aclose()
            client.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            first = await client.get_available_models(
                "logfare",
                classifier="cache-user",
                api_keys={"logfare": ["secret-a"]},
                providers={"logfare": {"base_url": "https://tenant.example/v1"}},
                private=True,
            )
            second = await client.get_available_models(
                "logfare",
                classifier="cache-user",
                api_keys={"logfare": ["secret-b"]},
                providers={"logfare": {"base_url": "https://tenant.example/v1"}},
                private=True,
            )
            assert first == ["logfare/model-a"]
            assert second == ["logfare/model-b"]
        finally:
            await _close(client)

    run_async(run_test())

    assert seen == ["Bearer secret-a", "Bearer secret-b"]


def test_provider_config_override_routes_without_global_mutation(tmp_path):
    client = _make_client(tmp_path, api_keys={"openai": ["global-openai-key"]})

    try:
        kwargs = client.provider_config.convert_for_litellm(model="logfare/model")
        assert kwargs == {"model": "logfare/model"}

        converted = client.provider_config.convert_for_litellm(
            provider_override={"base_url": "https://override.example/v1"},
            model="logfare/model",
        )
        assert converted["model"] == "openai/model"
        assert converted["api_base"] == "https://override.example/v1"
        assert converted["custom_llm_provider"] == "openai"

        kwargs_after = client.provider_config.convert_for_litellm(model="logfare/model")
        assert kwargs_after == {"model": "logfare/model"}
    finally:
        run_async(_close(client))
