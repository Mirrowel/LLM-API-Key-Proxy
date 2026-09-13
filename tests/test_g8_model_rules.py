# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""The model_rules capability table (G8): an ordered cascade replacing
model_param_rules — CSS-style inheritance/override per row key, effort_map
sugar, per-model face limiting, and JSON-config merging.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakePlugin:
    provider_env_name = "fake"
    param_rules = {"strip": ["logprobs"]}


class CascadePlugin(FakePlugin):
    model_rules = (
        {"match": "*", "strip": ["logit_bias"], "clamp": {"temperature": [0.0, 1.0]}},
        {"match": "reasoner-*", "strip": ["logprobs", "tool_choice"]},
    )


def _declared(plugin, model, runtime=None):
    from rotator_library.adapters.param_rules import declared_param_rules

    return declared_param_rules(plugin, model, runtime)


def test_cascade_overrides_and_inherits():
    """A specific row overrides the * row's conflicting key (strip) and
    inherits its non-conflicting key (clamp); unmatched models keep the
    provider-default row. Row content deep-merges onto the provider base
    (strip lists union — the existing model-level contract)."""

    reasoner = _declared(CascadePlugin(), "reasoner-pro")
    assert reasoner["strip"] == ["logprobs", "tool_choice"]  # later row wins over the * row
    assert reasoner["clamp"] == {"temperature": [0.0, 1.0]}  # inherited

    plain = _declared(CascadePlugin(), "plain-model")
    assert plain["strip"] == ["logprobs", "logit_bias"]  # provider base ∪ * row
    assert plain["clamp"] == {"temperature": [0.0, 1.0]}


def test_cascade_matches_are_case_insensitive_and_prefix_aware():
    rules = _declared(CascadePlugin(), "FAKE/REASONER-PRO")
    assert rules["strip"] == ["logprobs", "tool_choice"]


def test_effort_map_sugar_compiles_to_reasoning_effort_map():
    class EffortPlugin(FakePlugin):
        model_rules = ({"match": "*", "effort_map": {"low": "none", "medium": "high"}},)

    rules = _declared(EffortPlugin(), "any-model")
    assert rules["map"]["reasoning_effort"] == {"low": "none", "medium": "high"}
    assert "effort_map" not in rules

    # Explicit map on the same knob beats the sugar.
    class ExplicitPlugin(FakePlugin):
        model_rules = (
            {"match": "*", "effort_map": {"low": "none"}, "map": {"reasoning_effort": {"low": "keep"}}},
        )

    merged = _declared(ExplicitPlugin(), "any-model")
    assert merged["map"]["reasoning_effort"] == {"low": "keep"}


def test_effort_map_applies_on_the_wire():
    from rotator_library.adapters.base import AdapterContext
    from rotator_library.adapters.param_rules import ParamRulesAdapter

    class WirePlugin(FakePlugin):
        model_rules = ({"match": "*", "effort_map": {"medium": "high"}},)

    adapter = ParamRulesAdapter()
    context = AdapterContext(
        provider="fake",
        model="m",
        adapter_config={"param_rules": _declared(WirePlugin(), "m")},
        metadata={},
    )
    result = asyncio.run(adapter.transform_request({"reasoning_effort": "medium"}, context))
    assert result["reasoning_effort"] == "high"


def test_no_rows_is_identity():
    assert _declared(FakePlugin(), "any") == {"strip": ["logprobs"]}

    class EmptyPlugin:
        provider_env_name = "empty"

    assert _declared(EmptyPlugin(), "any") == {}


def test_strip_override_row_stays_terminal():
    class OverridePlugin(FakePlugin):
        model_rules = (
            {"match": "*", "strip": ["logit_bias", "logprobs"]},
            {"match": "wide", "strip_override": ["logit_bias"]},
        )

    assert _declared(OverridePlugin(), "wide")["strip"] == ["logit_bias"]
    assert _declared(OverridePlugin(), "other")["strip"] == ["logprobs", "logit_bias"]


def test_json_config_model_rules_merge_and_override_code_rows():
    runtime = {"model_rules": [{"match": "*", "strip": ["from-config"]}]}
    merged = _declared(CascadePlugin(), "plain-model", runtime)
    # JSON rows append after class rows: config's strip replaces the class
    # row's strip in the cascade; clamp inherits; provider base unions.
    assert merged["strip"] == ["logprobs", "from-config"]
    assert merged["clamp"] == {"temperature": [0.0, 1.0]}


def test_json_schema_accepts_and_validates_model_rules():
    from rotator_library.config.experimental import (
        ExperimentalConfigError,
        get_provider_runtime_config,
        load_config_from_mapping,
    )

    config = load_config_from_mapping(
        {"providers": {"fake": {"model_rules": [{"match": "*", "effort_map": {"low": "none"}}]}}}
    )
    runtime_config = get_provider_runtime_config("fake", config=config)
    assert runtime_config.model_rules == ({"match": "*", "effort_map": {"low": "none"}},)

    try:
        load_config_from_mapping(
            {"providers": {"fake": {"model_rules": [{"strip": ["x"]}]}}}
        )
    except ExperimentalConfigError as exc:
        assert "match" in str(exc)
    else:
        raise AssertionError("model_rules rows without a match wildcard must fail")

    try:
        load_config_from_mapping(
            {"providers": {"fake": {"model_rules": [{"match": "*", "banana": 1}]}}}
        )
    except ExperimentalConfigError as exc:
        assert "unsupported keys" in str(exc)
    else:
        raise AssertionError("unknown model_rules row keys must fail")


def test_allow_deny_refuse_and_admit_faces():
    from rotator_library.adapters.param_rules import ModelRulesFaceError
    from rotator_library.providers.provider_interface import ProviderInterface

    class FaceLimitedProvider(ProviderInterface):
        provider_env_name = "facelimited"
        speaks = ("openai_chat", "responses")
        model_rules = (
            {"match": "mini*", "deny": ["responses"]},
            {"match": "pro-*", "allow": ["openai_chat"]},
        )

    plugin = FaceLimitedProvider()

    # Denied face: refused, error names the deciding row.
    try:
        plugin.get_protocol_name("mini", profile="responses")
    except ModelRulesFaceError as exc:
        assert "mini*" in str(exc) and "responses" in str(exc)
    else:
        raise AssertionError("denied face must be refused")

    # Allowed faces keep resolving (the deny row never matched them).
    assert plugin.get_protocol_name("mini", profile="openai_chat") == "openai_chat"
    assert plugin.get_protocol_name("mini") == "openai_chat"  # default face

    # allow-list refusal names the specific row.
    try:
        plugin.get_protocol_name("pro-max", profile="responses")
    except ModelRulesFaceError as exc:
        assert "pro-*" in str(exc)
    else:
        raise AssertionError("face outside allow must be refused")

    # The refusals surface through the operation gate the client executor
    # consults (_supports_profile_operation -> supports_native_operation).
    try:
        plugin.supports_native_operation("pro-max", "chat", profile="responses")
    except ModelRulesFaceError:
        pass
    else:
        raise AssertionError("operation gate must propagate the face refusal")


def test_allow_matches_protocol_family():
    from rotator_library.providers.provider_interface import ProviderInterface

    class FamilyLimitedProvider(ProviderInterface):
        provider_env_name = "familylimited"
        speaks = ("openai_chat", "responses", ("stateful", "responses_stateful", {}))
        model_rules = ({"match": "*", "allow": ["openai_chat", "responses"]},)

    plugin = FamilyLimitedProvider()
    # A sibling variant resolves through its family: allowed.
    assert plugin.get_protocol_name("m", profile="stateful") == "responses_stateful"


def test_runtime_json_rows_feed_face_limiting():
    from rotator_library.adapters.param_rules import ModelRulesFaceError
    from rotator_library.config.experimental import load_config_from_mapping
    from rotator_library.providers.provider_interface import ProviderInterface

    class ConfiguredFaceProvider(ProviderInterface):
        provider_env_name = "configuredface"
        speaks = ("openai_chat", "responses")

    plugin = ConfiguredFaceProvider()
    config = load_config_from_mapping(
        {"providers": {"configuredface": {"model_rules": [{"match": "*", "deny": ["responses"]}]}}}
    )
    plugin.bind_runtime_config(config)
    assert plugin.get_protocol_name("m", profile="openai_chat") == "openai_chat"
    try:
        plugin.get_protocol_name("m", profile="responses")
    except ModelRulesFaceError:
        pass
    else:
        raise AssertionError("JSON-configured deny must refuse the face")


def test_model_param_rules_bridge_keeps_working():
    class BridgePlugin(FakePlugin):
        model_param_rules = {"legacy": {"map": {"reasoning_effort": {"low": "high"}}}}
        model_rules = ({"match": "modern", "effort_map": {"low": "none"}},)

    # Legacy mapping still resolves for its models...
    legacy = _declared(BridgePlugin(), "legacy")
    assert legacy["map"]["reasoning_effort"] == {"low": "high"}
    assert legacy["strip"] == ["logprobs"]
    # ...and the capability table governs its own models.
    modern = _declared(BridgePlugin(), "modern")
    assert modern["map"]["reasoning_effort"] == {"low": "none"}
