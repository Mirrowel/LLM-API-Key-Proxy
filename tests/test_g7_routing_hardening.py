"""G7 routing-hardening pins: identity normalization across sinks + config load.

Covers the fix-pass G7 deliverables 3-4: per-segment whitespace stripping,
provider casing normalize-or-reject, profile survival through scope
attachment, Ollama digest survival, empty-policy defaults, group/env-key
hardening, explicit-config-path strictness, and skip-unservicable targets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.client.request_builder import (
    RequestContextBuilder,
    _normalize_profile_reference,
)
from rotator_library.config.experimental import (
    ExperimentalConfigError,
    load_experimental_config,
)
from rotator_library.field_cache import FieldCacheContext, FieldCacheRule, build_cache_key
from rotator_library.field_cache.compat import parse_model_ref
from rotator_library.routing.config import (
    RoutingConfigError,
    load_routing_config_from_env,
    parse_route_target,
)
from rotator_library.routing.profiles import (
    ModelReferenceError,
    parse_model_reference,
)
from rotator_library.routing.types import (
    DEFAULT_FAILOVER_ON,
    FallbackGroup,
    RouteTarget,
    RoutingDecision,
)


# --- (1) Per-segment whitespace -------------------------------------------


def test_provider_and_profile_segments_are_stripped_model_is_not() -> None:
    ref = parse_model_reference(" openai : chat / gpt-4 ")
    assert (ref.provider, ref.profile, ref.model) == ("openai", "chat", " gpt-4 ")
    assert ref.bare == "openai/ gpt-4 "


def test_normalize_profile_reference_writes_back_the_stripped_form() -> None:
    kwargs = {"model": " openai : chat /gpt-4"}
    profile = _normalize_profile_reference(kwargs)
    assert profile == "chat"
    assert kwargs["model"] == "openai/gpt-4"


# --- (2) Field-cache model canonicalization -------------------------------


def test_field_cache_model_dimension_canonicalizes_provider_prefix() -> None:
    rule = FieldCacheRule(
        name="canonical",
        source="response",
        path="x",
        scope=("provider", "model"),
    )
    prefixed = build_cache_key(
        rule, FieldCacheContext(provider="openai", model="openai/gpt-4")
    )
    stripped = build_cache_key(
        rule, FieldCacheContext(provider="openai", model="gpt-4")
    )
    assert prefixed is not None
    assert prefixed == stripped


def test_field_cache_nested_model_ids_keep_their_slashes() -> None:
    rule = FieldCacheRule(
        name="canonical-nested",
        source="response",
        path="x",
        scope=("provider", "model"),
    )
    prefixed = build_cache_key(
        rule, FieldCacheContext(provider="openrouter", model="openrouter/meta/llama")
    )
    stripped = build_cache_key(
        rule, FieldCacheContext(provider="openrouter", model="meta/llama")
    )
    assert prefixed is not None
    assert prefixed == stripped


def test_compat_member_refs_strip_the_profile() -> None:
    ref = parse_model_ref("openai:chat/gpt-4")
    assert ref is not None
    assert (ref.provider, ref.model) == ("openai", "gpt-4")
    assert ref.key == "openai/gpt-4"
    # Malformed profile segments are rejected, never folded into the provider.
    assert parse_model_ref("openai:/gpt-4") is None
    assert parse_model_ref(":chat/gpt-4") is None


# --- (3) Casing / provider grammar ----------------------------------------


def test_provider_casing_is_preserved_and_illegal_names_rejected() -> None:
    # Normalize-OR-reject: reject, never silently lowercase. Case survives.
    assert parse_model_reference("OpenAI/gpt-4").provider == "OpenAI"
    with pytest.raises(ModelReferenceError) as excinfo:
        parse_model_reference("OpenAI!/gpt-4")
    assert "!" in str(excinfo.value)
    # A leading separator is not a valid first character either.
    with pytest.raises(ModelReferenceError):
        parse_model_reference("-openai/gpt-4")


def test_model_segment_case_is_untouched() -> None:
    assert parse_model_reference("openai/GPT-4").model == "GPT-4"


# --- (4) New routing pins --------------------------------------------------


def test_ollama_digest_survives_execution_split() -> None:
    target = parse_route_target("ollama/llama3:8b@sha256:abc")
    assert target.provider == "ollama"
    assert target.model == "llama3:8b@sha256:abc"
    assert target.execution == "auto"


def test_explicit_execution_mode_still_parses() -> None:
    target = parse_route_target("openai/gpt-5@litellm_fallback")
    assert target.execution == "litellm_fallback"
    assert target.model == "gpt-5"


def test_profile_survives_request_scope_attachment() -> None:
    target = RouteTarget(provider="myserver", model="model", profile="responses")
    scope = {
        "credentials": ["cred-1", "cred-2"],
        "usage_manager_key": "myserver",
        "provider_config": {"api_base": "https://example.test"},
        "credential_secrets": {"cred-1": "secret-1"},
    }
    scoped = RequestContextBuilder._with_request_scope(target, scope)
    assert scoped.profile == "responses"
    assert (scoped.provider, scoped.model) == ("myserver", "model")
    assert scoped.metadata["request_scope"]["credentials"] == ["cred-1", "cred-2"]
    # Frozen target untouched; the helper returns a new instance.
    assert target.metadata == {}
    assert scoped is not target


def test_empty_failover_on_env_falls_back_to_default() -> None:
    config = load_routing_config_from_env(
        env={
            "FALLBACK_GROUPS": "g",
            "FALLBACK_GROUP_G": "openai/gpt-4",
            "FALLBACK_GROUP_G_FAILOVER_ON": "",
        }
    )
    assert config.fallback_groups["g"].failover_on == DEFAULT_FAILOVER_ON


def test_group_names_with_colon_rejected() -> None:
    with pytest.raises(RoutingConfigError, match="must not contain"):
        load_routing_config_from_env(env={"FALLBACK_GROUPS": "a:b"})


def test_group_names_colliding_on_env_key_rejected() -> None:
    with pytest.raises(RoutingConfigError, match="collide"):
        load_routing_config_from_env(
            env={
                "FALLBACK_GROUPS": "a-b,a_b",
                "FALLBACK_GROUP_A_B": "openai/gpt-4",
            }
        )


def test_missing_explicit_config_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ExperimentalConfigError, match="not found"):
        load_experimental_config(path=tmp_path / "definitely-missing.json")


@pytest.mark.asyncio
async def test_unserviceable_middle_target_is_skipped() -> None:
    """D17 breaker: a later target without credentials is skipped and the
    chain proceeds; only a fully unserviceable decision raises."""
    targets = (
        RouteTarget(provider="alpha", model="m"),
        RouteTarget(provider="beta", model="m"),
        RouteTarget(provider="gamma", model="m"),
    )
    group = FallbackGroup(name="g", targets=targets)
    decision = RoutingDecision(
        requested_model="alpha/m",
        targets=targets,
        group_name="g",
        group=group,
        reason="model_route_group",
    )

    scopes = {
        "alpha": _scope(["cred-alpha"]),
        "beta": _scope([]),
        "gamma": _scope(["cred-gamma"]),
    }

    async def resolve_scope(provider, classifier, api_keys, providers, private):
        return scopes[provider]

    builder = RequestContextBuilder(
        resolve_scope_for_provider=resolve_scope,
        model_resolver=SimpleNamespace(resolve_model_id=lambda model, provider: model),
        session_tracker=SimpleNamespace(
            infer_session=lambda *args, **kwargs: SimpleNamespace(
                session_id="session-1",
                affinity_key=None,
                tracking_namespace=None,
                confidence=None,
            )
        ),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
        get_provider_instance=None,
        experimental_config=None,
    )
    builder._resolve_routing_decision = lambda model: decision

    context = await builder.build_completion_context(
        request=None,
        pre_request_callback=None,
        kwargs={
            "model": "alpha/m",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert context.provider == "alpha"
    assert [target.provider for target in context.routing_targets] == ["alpha", "gamma"]


def _scope(credentials: list[str]) -> dict:
    return {
        "credentials": credentials,
        "usage_manager_key": "usage",
        "provider_config": {},
        "credential_secrets": {},
        "classifier": None,
    }
