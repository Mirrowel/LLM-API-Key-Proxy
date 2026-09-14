# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""NVIDIA NIM remake pins (G8): the capability matrix as behavior.

Every family row is exercised through the real machinery — wire
normalization (the ladder) plus reasoning emission (declared targets)
— replacing the retired hand-coded handler's extra_body shapes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.native_provider.effort_emission import (
    apply_reasoning_emission,
    normalize_wire_effort,
)
from rotator_library.providers.nvidia_provider import NvidiaProvider

PLUGIN = NvidiaProvider()


def _emit(model: str, word: str) -> dict:
    """One request through the pipeline: param engine (strip/clamp),
    wire normalization (the ladder), then reasoning emission."""

    payload = {"model": model, "reasoning_effort": word}
    ctx_model = f"nvidia_nim/{model}"
    # The always-on param engine consumes the matrix's strip/clamp rows.
    from rotator_library.adapters.base import AdapterContext
    from rotator_library.adapters.param_rules import ParamRulesAdapter

    import asyncio

    payload = asyncio.run(
        ParamRulesAdapter().transform_request(
            payload,
            AdapterContext(
                provider="nvidia_nim",
                model=model,
                protocol="openai_chat",
                adapter_config=PLUGIN.get_adapter_config(model),
                metadata={},
            ),
        )
    )
    normalize_wire_effort(payload, provider_plugin=PLUGIN, model=ctx_model, protocol_name="openai_chat")
    apply_reasoning_emission(
        payload, provider_plugin=PLUGIN, model=ctx_model, protocol_name="openai_chat"
    )
    return payload


def test_kimi_k2_boolean_toggle_family():
    assert _emit("moonshotai/kimi-k2.6", "high")["chat_template_kwargs"] == {"thinking": True}
    assert "reasoning_effort" not in _emit("moonshotai/kimi-k2.6", "high")
    assert _emit("moonshotai/kimi-k2.6", "off")["chat_template_kwargs"] == {"thinking": False}


def test_kimi_k3_always_on_no_off():
    payload = _emit("moonshotai/kimi-k3", "low")
    assert payload["reasoning_effort"] == "low"
    # No off value exists: OFF drops the control (thinking stays on).
    assert "reasoning_effort" not in _emit("moonshotai/kimi-k3", "off")


def test_deepseek_v4_off_word_is_none():
    payload = _emit("deepseek-ai/deepseek-v4-flash-0731", "off")
    assert payload["reasoning_effort"] == "none"
    assert _emit("deepseek-ai/deepseek-v4-flash-0731", "max")["reasoning_effort"] == "max"
    # Ladder fold: medium -> high (tie rounds up on {off,high,max}).
    assert _emit("deepseek-ai/deepseek-v4-flash-0731", "medium")["reasoning_effort"] == "high"


def test_glm_vocabulary_split():
    assert _emit("zai-org/glm-5.3-flash", "low")["reasoning_effort"] == "low"
    # 5.2's narrower set folds low up to high.
    assert _emit("zai-org/glm-5.2", "low")["reasoning_effort"] == "high"


def test_gpt_oss_live_vocabulary():
    assert _emit("openai/gpt-oss-20b", "medium")["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in _emit("openai/gpt-oss-20b", "off")


def test_muse_full_seven_rung_vocabulary():
    for word in ("minimal", "xhigh", "max"):
        assert _emit("meta/muse-glimmer-30b", word)["reasoning_effort"] == word
    assert _emit("meta/muse-glimmer-30b", "off")["reasoning_effort"] == "none"


def test_nemotron_budget_clamp_and_effort():
    payload = _emit("nvidia/nemotron-3-ultra-550b", "medium")
    assert payload["reasoning_effort"] == "medium"
    assert _emit("nvidia/nemotron-3-ultra-550b", "off")["reasoning_effort"] == "none"


def test_diffusiongemma_and_qwen_enable_thinking():
    assert _emit("google/diffusiongemma-26b-a4b-it", "high")["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert _emit("google/diffusiongemma-26b-a4b-it", "off")["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert _emit("qwen/qwen3.5-480b", "high")["chat_template_kwargs"] == {"enable_thinking": True}


def test_plain_families_strip_the_effort_word():
    assert "reasoning_effort" not in _emit("meta/llama-3.3-70b-instruct", "high")
    assert "reasoning_effort" not in _emit("stepfun/step-3.7-flash", "medium")


def test_unknown_model_falls_to_base_with_folds():
    # The "only some" ruling: unknown ids keep the protocol base
    # vocabulary (off,low,medium,high) — nothing fabricated.
    assert _emit("someorg/new-model-x", "medium")["reasoning_effort"] == "medium"


def test_declaration_surface():
    assert PLUGIN.get_protocol_name() == "openai_chat"
    assert PLUGIN.config_key_alias == "nvidia_nim"
    assert not hasattr(PLUGIN, "handle_thinking_parameter")
    assert not hasattr(PLUGIN, "V4_EFFORT_MAP")
    # The param engine heads the chain; no custom adapter exists.
    assert PLUGIN.get_adapter_names("nvidia_nim/x") == ("param_rules",)
