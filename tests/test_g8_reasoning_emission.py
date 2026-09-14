# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Declared dotted-target reasoning emission (G8 nvidia prep).

The emission extension: model_rules rows name WHERE the folded effort
word and the thinking toggle land — top-level by default, nested
(chat_template_kwargs.*) for the vLLM families — and what the toggle's
on/off values are (booleans for enable_thinking-style, the DeepSeek
object pair via the legacy preset).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.native_provider.effort_emission import (
    apply_reasoning_emission,
    normalize_wire_effort,
    resolve_reasoning_targets,
)


class _CTKFamily:
    """kimi-k2-style: boolean toggle + nested effort inside ctk."""

    model_rules = (
        {
            "match": "*kimi-k2*",
            "effort_accept": ["off", "low", "high"],
            "toggle_field": "chat_template_kwargs.thinking",
            "toggle_on": True,
            "toggle_off": False,
        },
    )


class _EnableFamily:
    """enable_thinking-style boolean, no effort word."""

    model_rules = (
        {
            "match": "*diffusion*",
            "toggle_field": "chat_template_kwargs.enable_thinking",
        },
    )


class _NestedEffort:
    """effort rides inside ctk, no toggle."""

    model_rules = (
        {
            "match": "*weird*",
            "effort_field": "chat_template_kwargs.reasoning_effort",
        },
    )


def test_boolean_toggle_family_emits_nested_on():
    payload = {"model": "moonshotai/kimi-k2.6", "reasoning_effort": "high"}
    normalize_wire_effort(
        payload,
        provider_plugin=_CTKFamily(),
        model="moonshotai/kimi-k2.6",
        protocol_name="openai_chat",
    )
    assert apply_reasoning_emission(
        payload,
        provider_plugin=_CTKFamily(),
        model="moonshotai/kimi-k2.6",
        protocol_name="openai_chat",
    )
    assert payload["chat_template_kwargs"] == {"thinking": True}
    assert "reasoning_effort" not in payload


def test_boolean_toggle_family_off_disables():
    payload = {"model": "moonshotai/kimi-k2.6", "reasoning_effort": "off"}
    normalize_wire_effort(
        payload,
        provider_plugin=_CTKFamily(),
        model="moonshotai/kimi-k2.6",
        protocol_name="openai_chat",
    )
    apply_reasoning_emission(
        payload,
        provider_plugin=_CTKFamily(),
        model="moonshotai/kimi-k2.6",
        protocol_name="openai_chat",
    )
    assert payload["chat_template_kwargs"] == {"thinking": False}


def test_enable_thinking_family_default_values():
    payload = {"model": "google/diffusiongemma-26b", "reasoning_effort": "high"}
    normalize_wire_effort(
        payload,
        provider_plugin=_EnableFamily(),
        model="google/diffusiongemma-26b",
        protocol_name="openai_chat",
    )
    apply_reasoning_emission(
        payload,
        provider_plugin=_EnableFamily(),
        model="google/diffusiongemma-26b",
        protocol_name="openai_chat",
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_nested_effort_without_toggle():
    payload = {"model": "x/weird-1", "reasoning_effort": "high"}
    normalize_wire_effort(
        payload, provider_plugin=_NestedEffort(), model="x/weird-1", protocol_name="openai_chat"
    )
    apply_reasoning_emission(
        payload, provider_plugin=_NestedEffort(), model="x/weird-1", protocol_name="openai_chat"
    )
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_targets_resolution_and_no_legacy_preset():
    targets = resolve_reasoning_targets(_CTKFamily(), "moonshotai/kimi-k2.6")
    assert targets["toggle_field"] == "chat_template_kwargs.thinking"
    assert targets["toggle_on"] is True and targets["toggle_off"] is False
    # No legacy toggle declared -> no object preset
    assert resolve_reasoning_targets(_NestedEffort(), "x/weird-1").get("toggle_field") is None


def test_undeclared_provider_is_untouched():
    payload = {"model": "m", "reasoning_effort": "high"}
    assert not apply_reasoning_emission(
        payload, provider_plugin=None, model="m", protocol_name="openai_chat"
    )
    assert payload == {"model": "m", "reasoning_effort": "high"}
