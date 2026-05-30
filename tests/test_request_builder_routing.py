from __future__ import annotations

import pytest

from rotator_library.client.request_builder import RequestContextBuilder


class FakeModelResolver:
    def resolve_model_id(self, model, provider):
        return model


class FakeSession:
    session_id = "session"
    affinity_key = "affinity"
    possible_compaction = False
    lineage_parent_session_id = None
    tracking_namespace = "namespace"


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


def _builder() -> RequestContextBuilder:
    return RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=FakeModelResolver(),
        session_tracker=FakeSessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
    )


@pytest.mark.asyncio
async def test_request_builder_leaves_no_config_provider_model_unrouted(monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_GROUPS", raising=False)
    context = await _builder().build_completion_context(None, None, {"model": "openai/gpt-5.1", "messages": []})

    assert context.routing_targets is None
    assert context.provider == "openai"
    assert context.credentials == ["openai-cred"]


@pytest.mark.asyncio
async def test_request_builder_populates_fallback_group_targets_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_GROUPS", "code_chain")
    monkeypatch.setenv("FALLBACK_GROUP_CODE_CHAIN", "codex/gpt-5.1-codex,openai/gpt-5.1")
    monkeypatch.setenv("MODEL_ROUTE_CODEX", "group:code_chain")

    context = await _builder().build_completion_context(None, None, {"model": "codex", "messages": []})

    assert context.provider == "codex"
    assert context.model == "codex/gpt-5.1-codex"
    assert context.routing_group_name == "code_chain"
    assert [target.prefixed_model for target in context.routing_targets] == ["codex/gpt-5.1-codex", "openai/gpt-5.1"]
    assert context.routing_targets[1].metadata["request_scope"]["credentials"] == ["openai-cred"]


@pytest.mark.asyncio
async def test_request_builder_rejects_unprefixed_model_without_route(monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_GROUPS", raising=False)

    with pytest.raises(ValueError):
        await _builder().build_completion_context(None, None, {"model": "gpt-5.1", "messages": []})
