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
