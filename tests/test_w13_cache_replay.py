"""W13 acceptance: cache-and-replay + compatibility classes (D11/D12/D14).

Scope relaxation (provider+model required, credential/session optional
refinements), bound-vs-portable inheritance with default-deny groups,
declarative cache_replay compilation, and transform-on-inject.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.field_cache.compat import (
    CompatibilityRegistry,
    ModelRef,
    get_compatibility_registry,
    reset_compatibility_registry,
)
from rotator_library.field_cache.engine import FieldCacheEngine, build_cache_key
from rotator_library.field_cache.replay import compile_cache_replay, parse_cache_replay_config
from rotator_library.field_cache.types import FieldCacheContext, FieldCacheRule
from rotator_library.field_cache.store import InMemoryFieldCacheStore


def _ctx(provider="prov", model="m1", credential=None, session=None):
    return FieldCacheContext(provider=provider, model=model, credential_id=credential, session_id=session)


# --- D11: scope relaxation ---


def test_missing_credential_and_session_no_longer_disable_caching() -> None:
    """provider+model are the required identity; credential/session are
    optional refinements — a rule with the default scope caches without
    them (single-operator semantics; no fail-closed on unknown creds)."""

    rule = FieldCacheRule(name="reasoning", source="response", path="reasoning_content")
    key = build_cache_key(rule, _ctx())
    assert key is not None
    # Refinements participate when present.
    refined = build_cache_key(rule, _ctx(credential="c1", session="s1"))
    assert refined is not None and refined != key


def test_provider_or_model_absence_still_disables() -> None:
    rule = FieldCacheRule(name="reasoning", source="response", path="reasoning_content")
    assert build_cache_key(rule, FieldCacheContext(provider="p")) is None
    assert build_cache_key(rule, FieldCacheContext(model="m")) is None


def test_provider_continuation_rules_keep_strict_session_binding() -> None:
    rule = FieldCacheRule(
        name="continuation",
        source="response",
        path="id",
        inject=None,
        allow_missing_session=False,
        metadata={"provider_continuation": True},
    )
    assert build_cache_key(rule, _ctx()) is None
    assert build_cache_key(rule, _ctx(session="s1")) is not None


# --- D12: compatibility classes ---


def test_bound_fields_identity_match_only() -> None:
    registry = CompatibilityRegistry({"open-models": ["prov/m1", "prov/m2"]})
    source = ModelRef(provider="prov", model="m1")
    same = registry.can_inherit(source, ModelRef("prov", "m1"), field_class="bound")
    sibling = registry.can_inherit(source, ModelRef("prov", "m2"), field_class="bound")
    assert same is True
    assert sibling is False


def test_portable_fields_inherit_within_groups() -> None:
    registry = CompatibilityRegistry({"open-models": ["prov/m1", "prov/m2"]})
    source = ModelRef(provider="prov", model="m1")
    assert registry.can_inherit(source, ModelRef("prov", "m2"), field_class="portable") is True
    # Default-deny outside groups; the model:<name> identity group still
    # carries the SAME model name across providers.
    assert registry.can_inherit(source, ModelRef("prov", "other"), field_class="portable") is False
    assert registry.can_inherit(source, ModelRef("other", "m1"), field_class="portable") is True
    assert registry.can_inherit(source, ModelRef("other", "other"), field_class="portable") is False


def test_env_configured_groups(monkeypatch) -> None:
    reset_compatibility_registry()
    monkeypatch.setenv(
        "FIELD_CACHE_COMPAT_GROUPS",
        '{"reasoning-family": ["openai/gpt-test", "openai/gpt-test-mini"]}',
    )
    registry = get_compatibility_registry()
    assert (
        registry.can_inherit(
            ModelRef("openai", "gpt-test"),
            ModelRef("openai", "gpt-test-mini"),
            field_class="portable",
        )
        is True
    )
    reset_compatibility_registry()


def test_engine_portable_inheritance_on_miss() -> None:
    """A miss on the target model's key inherits from a declared group
    sibling's cached value, with provenance recorded on the operation."""

    reset_compatibility_registry()
    import os

    os.environ["FIELD_CACHE_COMPAT_GROUPS"] = '{"family": ["prov/m1", "prov/m2"]}'
    try:
        store = InMemoryFieldCacheStore()
        rule = FieldCacheRule(
            name="reasoning",
            source="response",
            path="reasoning_content",
            mode="last",
            inject=None,
            metadata={"compatibility": "portable"},
        )
        # Cache from m1's key.
        m1_key = build_cache_key(rule, _ctx(model="m1", session="s1"))
        engine = FieldCacheEngine([rule], store)
        import asyncio

        asyncio.run(store.set(m1_key, {"v": "plaintext reasoning"}))
        # Inject into m2 (same session dimension) — sibling walk hits m1.
        target_payload = {"messages": [{"role": "user", "content": "hi"}]}
        rule_inject = FieldCacheRule(
            name="reasoning",
            source="response",
            path="reasoning_content",
            mode="last",
            inject=type(
                "Inj",
                (),
                {"target": "request", "path": "reasoning_hint", "when_missing_only": True, "insert": False, "as_list": False},
            )(),
            metadata={"compatibility": "portable"},
        )
        engine2 = FieldCacheEngine([rule_inject], store)
        updated, ops = asyncio.run(engine2.inject("request", target_payload, _ctx(model="m2", session="s1")))
        assert ops[0].hit is True
        # Inheritance through a _none credential dimension is still
        # transparent (merged provenance per D11).
        assert ops[0].reason == "inherited_from_compatible_model;optional_scope_none:credential"
        assert updated["reasoning_hint"] == {"v": "plaintext reasoning"}
    finally:
        os.environ.pop("FIELD_CACHE_COMPAT_GROUPS", None)
        reset_compatibility_registry()


def test_engine_bound_rules_never_inherit() -> None:
    reset_compatibility_registry()
    import os

    os.environ["FIELD_CACHE_COMPAT_GROUPS"] = '{"family": ["prov/m1", "prov/m2"]}'
    try:
        store = InMemoryFieldCacheStore()
        rule = FieldCacheRule(
            name="signature",
            source="response",
            path="signature",
            inject=type(
                "Inj",
                (),
                {"target": "request", "path": "sig", "when_missing_only": True, "insert": False, "as_list": False},
            )(),
            # No compatibility metadata: bound by default.
        )
        m1_key = build_cache_key(rule, _ctx(model="m1", session="s1"))
        import asyncio

        asyncio.run(store.set(m1_key, "sig-value"))
        engine = FieldCacheEngine([rule], store)
        updated, ops = asyncio.run(engine.inject("request", {}, _ctx(model="m2", session="s1")))
        assert ops[0].hit is False
        assert "sig" not in updated
    finally:
        os.environ.pop("FIELD_CACHE_COMPAT_GROUPS", None)
        reset_compatibility_registry()


# --- D14: declarative cache_replay ---


def test_cache_replay_compiles_modes_and_inject_options() -> None:
    rules = parse_cache_replay_config(
        '[{"name": "reasoning", "source": "response", "path": "reasoning_content", '
        '"keep": "turns:2", "compatibility": "portable", "inject": {"path": "messages[-1].reasoning_content", "if": "auto"}}]',
        provider="prov",
    )
    (rule,) = rules
    assert rule.mode == "all"
    assert rule.max_values == 2
    assert rule.metadata["compatibility"] == "portable"
    assert rule.inject.when_missing_only is True


def test_cache_replay_always_means_overwrite_operator_choice() -> None:
    rules = compile_cache_replay(
        [{"name": "sig", "source": "response", "path": "signature", "keep": "last", "inject": {"path": "sig", "if": "always"}}],
        provider="prov",
    )
    assert rules[0].inject.when_missing_only is False


def test_cache_replay_rejects_bad_declarations() -> None:
    with pytest.raises(ValueError, match="keep"):
        parse_cache_replay_config('[{"name": "x", "source": "response", "path": "p", "keep": "forever"}]', provider="prov")
    with pytest.raises(ValueError, match="transform requires"):
        parse_cache_replay_config(
            '[{"name": "x", "source": "response", "path": "p", "transform": "identity", "compatibility": "bound"}]',
            provider="prov",
        )
    with pytest.raises(ValueError, match="inject.if"):
        parse_cache_replay_config(
            '[{"name": "x", "source": "response", "path": "p", "inject": {"path": "q", "if": "maybe"}}]',
            provider="prov",
        )


def test_env_cache_replay_wires_into_merged_rules(monkeypatch) -> None:
    from rotator_library.client.executor import _merged_field_cache_rules

    class _Plugin:
        protocol_name = "openai_chat"

        def get_field_cache_rules(self, model):
            return ()

    monkeypatch.setenv(
        "PROV_CACHE_REPLAY",
        '[{"name": "reasoning", "source": "response", "path": "reasoning_content", "keep": "turn"}]',
    )
    rules = _merged_field_cache_rules("prov", "prov/m1", _Plugin(), config=None)
    assert any(getattr(rule, "name", "") == "reasoning" for rule in rules)


# --- Transform-on-inject ---


def test_transform_converts_chat_reasoning_to_anthropic_thinking() -> None:
    from rotator_library.protocols.transforms import apply_transform

    block = apply_transform("chat_reasoning_to_anthropic_thinking", "plain reasoning text")
    assert block == {"type": "thinking", "thinking": "plain reasoning text"}
    assert apply_transform("anthropic_thinking_to_chat_reasoning", block) == "plain reasoning text"


def test_engine_applies_transform_on_inject() -> None:
    store = InMemoryFieldCacheStore()
    rule = FieldCacheRule(
        name="reasoning",
        source="response",
        path="reasoning_content",
        inject=type(
            "Inj",
            (),
            {"target": "request", "path": "thinking_block", "when_missing_only": True, "insert": False, "as_list": False},
        )(),
        metadata={"compatibility": "portable", "transform": "chat_reasoning_to_anthropic_thinking"},
    )
    import asyncio

    engine = FieldCacheEngine([rule], store)
    key = build_cache_key(rule, _ctx())
    asyncio.run(store.set(key, "plain reasoning"))
    updated, ops = asyncio.run(engine.inject("request", {}, _ctx()))
    assert ops[0].changed is True
    assert updated["thinking_block"] == {"type": "thinking", "thinking": "plain reasoning"}


def test_bound_rule_rejects_transform_declaration() -> None:
    with pytest.raises(ValueError, match="never changes shape"):
        FieldCacheRule(
            name="sig",
            source="response",
            path="signature",
            metadata={"compatibility": "bound", "transform": "identity"},
        )
    # Implicit class defaults to bound: transform still rejected.
    with pytest.raises(ValueError, match="never changes shape"):
        FieldCacheRule(
            name="sig2",
            source="response",
            path="signature",
            metadata={"transform": "identity"},
        )


def test_inject_auto_preserves_client_value_and_always_overwrites() -> None:
    """auto (default) adds only when absent; always overwrites — an
    operator choice honored regardless of field class."""

    store = InMemoryFieldCacheStore()
    base_rule = FieldCacheRule(
        name="hint",
        source="response",
        path="hint",
        inject=type(
            "Inj",
            (),
            {"target": "request", "path": "hint", "when_missing_only": True, "insert": False, "as_list": False},
        )(),
    )
    always_rule = FieldCacheRule(
        name="hint",
        source="response",
        path="hint",
        inject=type(
            "Inj",
            (),
            {"target": "request", "path": "hint", "when_missing_only": False, "insert": False, "as_list": False},
        )(),
    )
    import asyncio

    key = build_cache_key(base_rule, _ctx())
    asyncio.run(store.set(key, "cached-value"))

    auto_engine = FieldCacheEngine([base_rule], store)
    updated, ops = asyncio.run(auto_engine.inject("request", {"hint": "client-value"}, _ctx()))
    assert updated["hint"] == "client-value"
    assert ops[0].changed is False

    always_engine = FieldCacheEngine([always_rule], store)
    updated, ops = asyncio.run(always_engine.inject("request", {"hint": "client-value"}, _ctx()))
    assert updated["hint"] == "cached-value"
    assert ops[0].changed is True


def test_unknown_credential_hit_records_none_provenance() -> None:
    """D11 provenance: a hit through a missing optional dimension is
    visible on the operation (no fail-closed, but never silent)."""

    store = InMemoryFieldCacheStore()
    rule = FieldCacheRule(
        name="reasoning",
        source="response",
        path="reasoning_content",
        inject=type(
            "Inj",
            (),
            {"target": "request", "path": "reasoning_hint", "when_missing_only": True, "insert": False, "as_list": False},
        )(),
    )
    import asyncio

    engine = FieldCacheEngine([rule], store)
    asyncio.run(engine.extract("response", {"reasoning_content": "text"}, _ctx(credential=None)))
    updated, ops = asyncio.run(engine.inject("request", {}, _ctx(credential=None)))
    assert ops[0].hit is True
    assert ops[0].reason == "optional_scope_none:credential+session"


def test_env_shadowing_a_plugin_rule_passes_the_weakening_guard_or_rejects() -> None:
    """Same-name env rules replace plugin rules only without weakening
    isolation; name collisions never crash the engine."""

    from rotator_library.client.executor import _merged_field_cache_rules, RoutingExecutionError
    from rotator_library.field_cache.types import FieldCacheInjection

    class _Plugin:
        protocol_name = "openai_chat"

        def get_field_cache_rules(self, model):
            # Same behavioral shape as the env replacement (the guard
            # allows identical-behavior swaps; only weakening rejects).
            return parse_cache_replay_config(
                '[{"name": "sig", "source": "response", "path": "signature", "keep": "last",'
                ' "inject": {"path": "sig", "if": "auto"}}]',
                provider="shadowprov",
            )

    import os

    # Identical-scope env rule replaces cleanly (no crash, no weakening).
    os.environ["SHADOWPROV_CACHE_REPLAY"] = (
        '[{"name": "sig", "source": "response", "path": "signature", "keep": "last",'
        ' "inject": {"path": "sig", "if": "auto"}}]'
    )
    try:
        rules = _merged_field_cache_rules("shadowprov", "shadowprov/m", _Plugin(), config=None)
        assert len([r for r in rules if r.name == "sig"]) == 1
        # Weakened scope (dropping model isolation) is rejected loudly.
        os.environ["SHADOWPROV_CACHE_REPLAY"] = (
            '[{"name": "sig", "source": "response", "path": "signature", "keep": "last",'
            ' "scope": ["provider"], "inject": {"path": "sig", "if": "auto"}}]'
        )
        from rotator_library.client.executor import _env_cache_replay_cached

        _env_cache_replay_cached.cache_clear()
        with pytest.raises(RoutingExecutionError, match="cannot weaken"):            _merged_field_cache_rules("shadowprov", "shadowprov/m", _Plugin(), config=None)
    finally:
        os.environ.pop("SHADOWPROV_CACHE_REPLAY", None)
        from rotator_library.client.executor import _env_cache_replay_cached

        _env_cache_replay_cached.cache_clear()
