# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 example envelope pins: the teaching template's declarations resolve.

The template is intentionally unregistered (leading underscore); these pins
import it directly and exercise the same machinery a registered provider
rides: the three-form ``speaks`` grammar, the ``model_rules`` CSS cascade and
effort chain, field-addressed cache-rule derivation (plus the loud refusal on
families without registry locations), the always-on param engine, inherited
shared listing, hook validation, and the session/quota seams.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _module():
    return importlib.import_module("rotator_library.providers._example_provider")


def _plugin():
    return _module().ExampleProvider()


# --- registration honesty -------------------------------------------------------


def test_example_stays_out_of_registration() -> None:
    from rotator_library.providers import PROVIDER_PLUGINS

    provider_class = _module().ExampleProvider
    assert "example" not in PROVIDER_PLUGINS
    assert provider_class not in set(PROVIDER_PLUGINS.values())
    assert provider_class.__module__ == "rotator_library.providers._example_provider"


def test_identity_and_transport_base_declared() -> None:
    plugin = _plugin()
    assert plugin.provider_env_name == "example"
    assert plugin.skip_cost_calculation is True
    assert plugin._provider_config_key() == "example"
    assert plugin.default_api_base == "https://api.example-vendor.example/v1"
    assert plugin.get_provider_api_base() == "https://api.example-vendor.example/v1"
    assert plugin.native_streaming_supported is True


# --- speaks: all three entry forms ---------------------------------------------


def test_all_three_speaks_forms_resolve() -> None:
    from rotator_library.protocols.defaults import validate_speaks

    plugin = _plugin()
    # Unknown protocols raise here; the three declared forms are legal.
    legal = validate_speaks(plugin.speaks)
    assert {"openai_chat", "responses", "anthropic_messages"} <= set(legal)

    profiles = plugin._speaks_profiles()
    assert set(profiles) - {"__default__"} == {
        "openai_chat",
        "responses",
        "anthropic",
    }
    assert profiles["__default__"] is profiles["openai_chat"]
    assert profiles["openai_chat"]["protocol"] == "openai_chat"
    assert profiles["responses"]["protocol"] == "responses"
    assert profiles["anthropic"]["protocol"] == "anthropic_messages"

    view, default = plugin.get_declared_profiles()
    assert default == "openai_chat"
    assert {name: entry["protocol"] for name, entry in view.items()} == {
        "openai_chat": "openai_chat",
        "responses": "responses",
        "anthropic": "anthropic_messages",
    }


def test_face_endpoints_inherit_then_override() -> None:
    plugin = _plugin()
    base = plugin.get_provider_api_base()

    # Form 1 (string): every path inherits from the protocol defaults.
    assert plugin.get_protocol_name("m") == "openai_chat"
    assert (
        plugin.get_native_endpoint(model="m", operation="chat")
        == f"{base}/chat/completions"
    )

    # Form 2 (protocol, overrides): the responses route itself inherits;
    # only the diverging token-count route is written down.
    assert plugin.get_protocol_name("m", profile="responses") == "responses"
    assert (
        plugin.get_native_endpoint(model="m", operation="responses", profile="responses")
        == f"{base}/responses"
    )
    assert (
        plugin.get_native_endpoint(
            model="m", operation="count_tokens", profile="responses"
        )
        == f"{base}/responses/input_tokens"
    )

    # Form 3 (name, protocol, overrides): explicit profile identity with
    # overridden paths; the operation vocabulary follows the protocol.
    assert plugin.get_protocol_name("m", profile="anthropic") == "anthropic_messages"
    assert (
        plugin.get_native_endpoint(model="m", operation="messages", profile="anthropic")
        == f"{base}/anthropic/v1/messages"
    )
    assert (
        plugin.get_native_endpoint(
            model="m", operation="count_tokens", profile="anthropic"
        )
        == f"{base}/anthropic/v1/count_tokens"
    )
    assert (
        plugin.get_native_operation("m", None, stream=False, profile="anthropic")
        == "messages"
    )
    assert plugin.supports_native_operation("m", "messages", profile="anthropic") is True


def test_face_auth_inherits_from_the_protocol_registry() -> None:
    plugin = _plugin()
    # openai_chat and responses inherit Bearer; the anthropic face inherits
    # x-api-key without ever restating auth.
    assert plugin.get_native_headers("sk-test") == {"Authorization": "Bearer sk-test"}
    assert plugin.get_native_headers("sk-test", profile="responses") == {
        "Authorization": "Bearer sk-test"
    }
    assert plugin.get_native_headers("sk-test", profile="anthropic") == {
        "x-api-key": "sk-test"
    }


def test_unknown_profile_fails_loudly() -> None:
    from rotator_library.routing.profiles import ModelReferenceError

    plugin = _plugin()
    with pytest.raises(ModelReferenceError):
        plugin.get_protocol_name("m", profile="nope")


# --- model_rules: the CSS cascade and the compiled tables -----------------------


def test_model_rules_cascade_overrides_and_inherits() -> None:
    from rotator_library.adapters.param_rules import resolve_model_rules

    plugin = _plugin()
    rows = plugin.model_rules

    base = resolve_model_rules(rows, "example-chat-v1", "example")
    assert base["strip"] == [
        "reasoning_effort",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    ]
    assert base["clamp"] == {"temperature": [0.0, 2.0]}
    assert base["map"] == {"tool_choice": {"required": "any"}}
    assert base["rename"] == {"max_completion_tokens": "max_tokens"}

    # Row 2's strip_override REPLACES row 1's strip (terminal); the
    # non-conflicting clamp/map/rename inherit.
    family = resolve_model_rules(rows, "example-reasoner-1", "example")
    assert family["strip_override"] == ["logit_bias", "logprobs", "top_logprobs"]
    assert family["effort_accept"] == ["off", "low", "medium", "high"]
    assert family["toggle"] is True
    assert family["clamp"] == {"temperature": [0.0, 2.0]}
    assert family["rename"] == {"max_completion_tokens": "max_tokens"}

    # Row 3 overrides its family's effort_accept and inherits the rest.
    v2 = resolve_model_rules(rows, "example-reasoner-v2", "example")
    assert v2["effort_accept"] == ["off", "low", "medium", "high", "xhigh"]
    assert v2["toggle"] is True
    assert v2["strip_override"] == ["logit_bias", "logprobs", "top_logprobs"]


def test_strip_override_compiles_terminal_and_capability_keys_stay_out() -> None:
    from rotator_library.adapters.param_rules import declared_param_rules

    plugin = _plugin()

    reasoner = declared_param_rules(plugin, "example-reasoner-v2")
    assert reasoner["strip"] == ["logit_bias", "logprobs", "top_logprobs"]
    assert reasoner["rename"] == {"max_completion_tokens": "max_tokens"}
    assert reasoner["clamp"] == {"temperature": [0.0, 2.0]}
    assert reasoner["map"] == {"tool_choice": {"required": "any"}}
    # Capability keys are declarations the effort system consumes; the
    # param engine never sees them.
    assert "effort_accept" not in reasoner and "toggle" not in reasoner

    chat = declared_param_rules(plugin, "example-chat-v1")
    assert "reasoning_effort" in chat["strip"]


def test_effort_chain_resolves_through_rows() -> None:
    from rotator_library.protocols.effort import (
        normalize_effort,
        resolve_accepted_effort,
        resolve_effort_toggle,
    )

    plugin = _plugin()

    # No row declares a vocabulary for plain chat models, so the protocol
    # base decides (and the * strip removes the control anyway).
    accepted, source = resolve_accepted_effort(
        plugin, "example-chat-v1", protocol_family="openai_chat"
    )
    assert source == "protocol_base"
    assert accepted == ("off", "low", "medium", "high")
    assert resolve_effort_toggle(plugin, "example-chat-v1") is False

    accepted, source = resolve_accepted_effort(
        plugin, "example-reasoner-1", protocol_family="openai_chat"
    )
    assert source == "model_rules:example-reasoner-*"
    assert accepted == ("off", "low", "medium", "high")
    assert resolve_effort_toggle(plugin, "example-reasoner-1") is True

    accepted, source = resolve_accepted_effort(
        plugin, "example-reasoner-v2", protocol_family="openai_chat"
    )
    assert source == "model_rules:example-reasoner-v2"
    assert accepted == ("off", "low", "medium", "high", "xhigh")
    # The ladder folds an unaccepted rung to its nearest accepted rung:
    # xhigh rides natively on v2, folds to high on its family predecessor.
    assert normalize_effort("xhigh", accepted)[0] == "xhigh"
    family_accepted, _ = resolve_accepted_effort(
        plugin, "example-reasoner-1", protocol_family="openai_chat"
    )
    assert normalize_effort("xhigh", family_accepted)[0] == "high"


# --- field_cache_rules: field addressing and derivation -------------------------


def test_field_rules_declared_by_field() -> None:
    plugin = _plugin()
    rules = plugin.field_cache_rules
    assert [rule.name for rule in rules] == ["reasoning", "signature"]

    reasoning, signature = rules
    assert reasoning.field == "reasoning"
    assert reasoning.sources == ("response", "stream_event")
    assert reasoning.source is None
    assert reasoning.mode == "all"
    assert reasoning.placeholder == "Reasoning content unavailable."
    assert reasoning.ttl_seconds is None and reasoning.cache_key is None
    assert reasoning.inject.when_missing_only is True
    assert reasoning.inject.target == "request"
    # The registry derivation owns the path (the empty path is the marker).
    assert reasoning.inject.path == ""

    # The turn_count variant: last-two-regions restoration.
    assert signature.field == "signature"
    assert signature.source == "response"
    assert signature.mode == "turns"
    assert signature.turn_count == 2
    assert signature.inject.when_missing_only is True


def test_field_rule_derives_paths_and_round_trips() -> None:
    from rotator_library.field_cache import (
        FieldCacheContext,
        FieldCacheEngine,
        build_cache_key,
    )

    plugin = _plugin()
    reasoning = plugin.field_cache_rules[0]
    engine = FieldCacheEngine([reasoning])
    context = FieldCacheContext(
        provider="example",
        model="example-reasoner-v2",
        credential_id="cred",
        session_id="sess",
        protocol_family="openai_chat",
    )

    # Paths, injection, and correlation all resolve from FIELD_LOCATIONS;
    # derivation runs on the expanded sources twin (source="response").
    response_twin = engine._expanded_rules[0]
    derived = engine._derived_rule(response_twin, "openai_chat")
    assert derived.path == "choices.*.message.reasoning_content"
    assert derived.inject.path == "messages.*.reasoning_content"
    assert derived.metadata["tool_call_id_path"] == "tool_calls.*.id"

    # Both source twins share ONE auto-derived cache key.
    assert len(engine._expanded_rules) == 2
    assert build_cache_key(engine._expanded_rules[0], context) == build_cache_key(
        engine._expanded_rules[1], context
    )

    # Round trip: response -> store -> request.
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "think-a",
                    "tool_calls": [{"id": "call_a", "type": "function"}],
                }
            }
        ]
    }
    asyncio.run(engine.extract("response", response, context))

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "answer",
                "tool_calls": [{"id": "call_a", "type": "function"}],
            },
        ]
    }
    updated, operations = asyncio.run(engine.inject("request", request, context))
    assert updated["messages"][1]["reasoning_content"] == "think-a"
    assert any(operation.hit for operation in operations)

    signature_engine = FieldCacheEngine([plugin.field_cache_rules[1]])
    signature = signature_engine._derived_rule(plugin.field_cache_rules[1], "openai_chat")
    assert (
        signature.path
        == "choices.*.message.tool_calls.*.extra_content.google.thought_signature"
    )
    assert (
        signature.inject.path
        == "messages.*.tool_calls.*.extra_content.google.thought_signature"
    )


@pytest.mark.parametrize("family", ["responses", "anthropic_messages"])
def test_field_rule_refuses_families_without_locations(family: str) -> None:
    """The loud boundary the template's COVERAGE note documents: a
    field-addressed rule must never silently pretend on a structural face."""

    from rotator_library.field_cache import FieldCacheContext, FieldCacheEngine

    plugin = _plugin()
    engine = FieldCacheEngine([plugin.field_cache_rules[0]])
    context = FieldCacheContext(
        provider="example",
        model="m",
        session_id="s",
        protocol_family=family,
    )
    with pytest.raises(ValueError, match="declares no field locations"):
        asyncio.run(engine.extract("response", {"choices": []}, context))


# --- adapters, hooks, listing, sessions, quota ----------------------------------


def test_param_engine_heads_the_chain_without_declaration() -> None:
    plugin = _plugin()
    assert plugin.adapter_names == ()
    assert plugin.get_adapter_names("example/model") == ("param_rules",)


def test_hook_declaration_validates() -> None:
    from rotator_library.hooks.registry import validate_declared_names

    plugin = _plugin()
    hooks = plugin.get_hooks("m")
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.name == "example_request_observer"
    assert hook.stages == ("request_received",)
    # Raises on an unknown name or stage — the same startup gate a
    # registered provider passes.
    validate_declared_names(class_hooks=hooks)


def test_listing_is_inherited_and_honest() -> None:
    provider_class = _module().ExampleProvider
    # The shared listing implementation is inherited, not reimplemented.
    assert "get_models" not in vars(provider_class)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "example-frontier-v2"}, {"object": "model"}]}

    class _Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, headers=None, **kwargs):
            self.calls.append((url, dict(headers or {})))
            return _Response()

    client = _Client()
    models = asyncio.run(provider_class().get_models("k", client))
    assert models == ["example/example-frontier-v2"]
    assert client.calls[0][0] == "https://api.example-vendor.example/v1/models"
    assert client.calls[0][1] == {"Authorization": "Bearer k"}

    class _BrokenClient:
        async def get(self, url, headers=None, **kwargs):
            raise RuntimeError("network down")

    assert asyncio.run(provider_class().get_models("k", _BrokenClient())) == []


def test_session_hint_seam_defaults_to_the_generic_tracker() -> None:
    plugin = _plugin()
    assert plugin.get_session_tracking_hints({"messages": []}, model="m") is None


def test_quota_declarations_resolve() -> None:
    plugin = _plugin()
    assert plugin.get_model_quota_group("example/example-frontier-v2") == "frontier"
    assert plugin.get_model_usage_weight("example-reasoner-v2") == 2
    assert plugin.default_rotation_mode == "sequential"
    assert plugin.get_background_job_config()["name"] == "quota_refresh"


def test_execution_is_declaration_not_custom() -> None:
    plugin = _plugin()
    assert plugin.has_custom_logic() is False
