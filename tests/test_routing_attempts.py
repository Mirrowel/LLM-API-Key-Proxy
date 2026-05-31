from __future__ import annotations

from rotator_library.core.types import RequestContext
from rotator_library.routing import clone_context_for_target, parse_route_target


def test_clone_context_for_target_updates_model_provider_without_mutating_original() -> None:
    original = RequestContext(
        model="requested",
        provider="original",
        kwargs={"model": "requested", "messages": []},
        streaming=False,
        credentials=["cred-a"],
        deadline=123.0,
        usage_manager_key="original",
    )
    target = parse_route_target("codex/gpt-5.1-codex@native")

    cloned = clone_context_for_target(original, target, credentials=["cred-b"], target_index=1)

    assert cloned.model == "codex/gpt-5.1-codex"
    assert cloned.provider == "codex"
    assert cloned.kwargs["model"] == "codex/gpt-5.1-codex"
    assert cloned.credentials == ["cred-b"]
    assert cloned.usage_manager_key == "codex"
    assert cloned.routing_target_index == 1
    assert original.model == "requested"
    assert original.kwargs["model"] == "requested"
    assert original.credentials == ["cred-a"]


def test_clone_context_for_target_preserves_request_metadata() -> None:
    original = RequestContext(
        model="requested",
        provider="original",
        kwargs={"model": "requested"},
        streaming=True,
        credentials=["cred-a"],
        deadline=123.0,
        session_id="session-1",
        classifier="global",
        routing_group_name="chain",
    )

    cloned = clone_context_for_target(original, parse_route_target("openai/gpt-5.1"))

    assert cloned.streaming is True
    assert cloned.session_id == "session-1"
    assert cloned.classifier == "global"
    assert cloned.routing_group_name == "chain"


def test_clone_context_for_target_rewrites_standard_session_namespace() -> None:
    original = RequestContext(
        model="openai/gpt-5",
        provider="openai",
        kwargs={"model": "openai/gpt-5"},
        streaming=False,
        credentials=["cred-a"],
        deadline=123.0,
        session_tracking_namespace="scope:openai:provider:openai:model:openai/gpt-5",
    )

    cloned = clone_context_for_target(original, parse_route_target("anthropic/claude"))

    assert cloned.session_tracking_namespace == "scope:openai:provider:anthropic:model:anthropic/claude"


def test_clone_context_for_target_preserves_custom_session_namespace() -> None:
    original = RequestContext(
        model="openai/gpt-5",
        provider="openai",
        kwargs={"model": "openai/gpt-5"},
        streaming=False,
        credentials=["cred-a"],
        deadline=123.0,
        session_tracking_namespace="custom-namespace",
    )

    cloned = clone_context_for_target(original, parse_route_target("anthropic/claude"))

    assert cloned.session_tracking_namespace == "custom-namespace"
