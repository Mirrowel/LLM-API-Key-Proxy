# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""param_rules adapter pins (G8): declared request-parameter hygiene."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ctx(provider, model, rules_config):
    from rotator_library.adapters.base import AdapterContext

    return AdapterContext(
        provider=provider,
        model=model,
        adapter_config={"param_rules": rules_config},
        metadata={},
    )


FULL_RULES = {
    "param_rules": {
        "strip": ["logit_bias", "logprobs"],
        "clamp": {"temperature": [0.0, 1.0]},
        "map": {"reasoning_effort": {"medium": "high", "xhigh": "high"}},
        "rename": {"max_completion_tokens": "max_tokens"},
    },
    "model_param_rules": {
        "mistral-medium-3-5": {"map": {"reasoning_effort": {"medium": "none"}}},
    },
}


def _adapter():
    from rotator_library.adapters.param_rules import ParamRulesAdapter

    return ParamRulesAdapter()


def test_strip_clamp_map_rename():
    adapter = _adapter()
    context = _ctx("mistral", "mistral-medium-3-5", FULL_RULES)
    payload = {
        "model": "mistral-medium-3-5",
        "messages": [],
        "temperature": 1.7,
        "logit_bias": {"1": 2},
        "reasoning_effort": "xhigh",
        "max_completion_tokens": 512,
    }
    result = asyncio.run(adapter.transform_request(payload, context))
    assert result["temperature"] == 1.0
    assert "logit_bias" not in result
    assert result["reasoning_effort"] == "high"
    assert result["max_tokens"] == 512 and "max_completion_tokens" not in result


def test_unmapped_values_pass_through():
    adapter = _adapter()
    context = _ctx("p", "m", {"param_rules": {"map": {"reasoning_effort": {"medium": "high"}}}})
    payload = {"reasoning_effort": "low"}
    result = asyncio.run(adapter.transform_request(payload, context))
    assert result["reasoning_effort"] == "low"


def test_model_overrides_beat_provider_defaults():
    adapter = _adapter()
    overridden = _ctx("mistral", "mistral-medium-3-5", FULL_RULES)
    result = asyncio.run(adapter.transform_request({"reasoning_effort": "medium"}, overridden))
    assert result["reasoning_effort"] == "none"  # model override
    base = _ctx("mistral", "base-model", FULL_RULES)
    result = asyncio.run(adapter.transform_request({"reasoning_effort": "medium"}, base))
    assert result["reasoning_effort"] == "high"  # provider default


def test_no_rules_is_identity():
    adapter = _adapter()
    context = _ctx("p", "m", {})
    payload = {"model": "m", "temperature": 2.0}
    result = asyncio.run(adapter.transform_request(payload, context))
    assert result is payload


def test_declared_resolution_from_plugin_class():
    from rotator_library.adapters.param_rules import declared_param_rules

    class FakePlugin:
        provider_env_name = "fake"
        param_rules = {"strip": ["logprobs"]}
        model_param_rules = {"fake-reasoner": {"map": {"reasoning_effort": {"low": "high"}}}}

    base = declared_param_rules(FakePlugin(), "fake-base")
    assert base == {"strip": ["logprobs"]}
    reasoner = declared_param_rules(FakePlugin(), "fake-reasoner")
    assert reasoner["map"] == {"reasoning_effort": {"low": "high"}}
    assert reasoner["strip"] == ["logprobs"]
    extended = declared_param_rules(FakePlugin(), "fake-base", {"param_rules": {"strip": ["n"]}})
    assert extended["strip"] == ["n"]


def test_resolution_is_idempotent_over_resolved_flat_tables():
    """get_adapter_config stores the RESOLVED provider+model tables under
    the adapter's config key; the adapter's own resolution pass must apply
    them unchanged (the documented idempotence contract)."""

    adapter = _adapter()
    resolved = {
        "rename": {"max_completion_tokens": "max_tokens"},
        "map": {"reasoning_effort": {"medium": "high"}},
    }
    context = _ctx("fake", "fake-reasoner", resolved)
    result = asyncio.run(
        adapter.transform_request(
            {"reasoning_effort": "medium", "max_completion_tokens": 8}, context
        )
    )
    assert result == {"reasoning_effort": "high", "max_tokens": 8}


def test_protocol_and_profile_scoped_tables():
    """by_protocol / by_profile sections overlay the flat base only on
    their face (the extensibility ruling: same param, different rules
    per protocol/profile)."""

    from rotator_library.adapters.base import AdapterContext

    adapter = _adapter()
    rules_config = {
        "param_rules": {"strip": ["logprobs"], "clamp": {"temperature": [0.0, 2.0]}},
        "by_protocol": {
            "anthropic_messages": {"strip": ["logit_bias"], "clamp": {"temperature": [0.0, 1.0]}},
        },
        "by_profile": {
            "openai": {"map": {"reasoning_effort": {"medium": "high"}}},
        },
    }
    # anthropic face: strip extends, clamp narrows
    ctx = AdapterContext(provider="gemini", model="m", protocol="anthropic_messages", adapter_config={"param_rules": rules_config}, metadata={})
    result = asyncio.run(adapter.transform_request({"logprobs": True, "logit_bias": {}, "temperature": 1.5}, ctx))
    assert "logprobs" not in result and "logit_bias" not in result
    assert result["temperature"] == 1.0
    # openai face (profile): the map applies, the base clamp stays 0..2
    ctx = AdapterContext(provider="gemini", model="m", protocol="openai_chat", profile="openai", adapter_config={"param_rules": rules_config}, metadata={})
    result = asyncio.run(adapter.transform_request({"temperature": 1.5, "reasoning_effort": "medium"}, ctx))
    assert result["temperature"] == 1.5
    assert result["reasoning_effort"] == "high"
    # unrelated face: neither scoped section applies
    ctx = AdapterContext(provider="other", model="m", protocol="openai_chat", adapter_config={"param_rules": rules_config}, metadata={})
    result = asyncio.run(adapter.transform_request({"reasoning_effort": "medium"}, ctx))
    assert result["reasoning_effort"] == "medium"


def test_strip_override_replaces_provider_strip_list():
    """A model-level strip_override REPLACES the provider strip list for
    that model — the escape hatch for "provider strips X globally, this
    model allows it". Terminal: no union with the inherited list."""

    adapter = _adapter()
    rules_config = {
        "param_rules": {"strip": ["reasoning_effort", "logit_bias", "logprobs"]},
        "model_param_rules": {
            "reasoner": {"strip_override": ["logit_bias"]},
        },
    }
    # plain model: the provider strip list applies in full
    base = _ctx("p", "plain", rules_config)
    result = asyncio.run(
        adapter.transform_request({"reasoning_effort": "low", "logit_bias": {}, "logprobs": True}, base)
    )
    assert result == {}
    # reasoner: only the override list strips — effort and logprobs survive
    overridden = _ctx("p", "reasoner", rules_config)
    result = asyncio.run(
        adapter.transform_request({"reasoning_effort": "low", "logit_bias": {}, "logprobs": True}, overridden)
    )
    assert result["reasoning_effort"] == "low"
    assert result["logprobs"] is True
    assert "logit_bias" not in result


def test_strip_override_empty_list_allows_everything():
    adapter = _adapter()
    rules_config = {
        "param_rules": {"strip": ["logit_bias"]},
        "model_param_rules": {"wide": {"strip_override": []}},
    }
    context = _ctx("p", "wide", rules_config)
    result = asyncio.run(adapter.transform_request({"logit_bias": {}, "logprobs": True}, context))
    assert result == {"logit_bias": {}, "logprobs": True}


def test_strip_override_coexists_with_other_model_rules():
    adapter = _adapter()
    rules_config = {
        "param_rules": {"strip": ["reasoning_effort"], "clamp": {"temperature": [0.0, 1.0]}},
        "model_param_rules": {
            "reasoner": {"strip_override": [], "map": {"reasoning_effort": {"medium": "high"}}},
        },
    }
    context = _ctx("p", "reasoner", rules_config)
    result = asyncio.run(
        adapter.transform_request({"reasoning_effort": "medium", "temperature": 2.5}, context)
    )
    # strip_override re-admitted the effort knob; the map and the clamp
    # (separate rule kinds) still apply.
    assert result["reasoning_effort"] == "high"
    assert result["temperature"] == 1.0


def test_declared_resolution_consumes_strip_override():
    """declared_param_rules resolves strip_override into the ordinary
    ``strip`` table (what get_adapter_config stores for the adapter) —
    the resolved form never carries the escape-hatch key, keeping the
    resolution pass idempotent over stored flat tables."""

    from rotator_library.adapters.param_rules import declared_param_rules

    class FakePlugin:
        provider_env_name = "fake"
        param_rules = {"strip": ["a"]}
        model_param_rules = {"m": {"strip_override": ["b"]}}

    overridden = declared_param_rules(FakePlugin(), "m")
    assert overridden["strip"] == ["b"]
    assert "strip_override" not in overridden
    plain = declared_param_rules(FakePlugin(), "other")
    assert plain["strip"] == ["a"]
