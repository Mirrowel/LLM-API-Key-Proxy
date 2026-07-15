from __future__ import annotations

import pytest

from rotator_library.providers.nvidia_provider import NvidiaProvider


MODEL = "nvidia_nim/google/diffusiongemma-26b-a4b-it"


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        (None, True),
        ("low", True),
        ("medium", True),
        ("high", True),
        ("off", False),
        ("disable", False),
        ("disabled", False),
        ("none", False),
    ],
)
def test_diffusion_gemma_maps_reasoning_effort_to_enable_thinking(
    reasoning_effort: str | None,
    expected: bool,
) -> None:
    """DiffusionGemma receives NVIDIA's required top-level template toggle."""
    payload = {"chat_template_kwargs": {"preserved": True}}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    NvidiaProvider().handle_thinking_parameter(payload, MODEL)

    assert payload["chat_template_kwargs"] == {
        "preserved": True,
        "enable_thinking": expected,
    }
    assert "reasoning_effort" not in payload
    assert "extra_body" not in payload


def test_diffusion_gemma_mapping_does_not_affect_other_nvidia_models() -> None:
    """The exact model match prevents template flags leaking to other models."""
    payload = {"reasoning_effort": "high"}

    NvidiaProvider().handle_thinking_parameter(
        payload,
        "nvidia_nim/google/gemma-3-27b-it",
    )

    assert payload == {"reasoning_effort": "high"}


@pytest.mark.parametrize(
    "initial_template_kwargs",
    [
        None,
        [],
        {"enable_thinking": False, "preserved": True},
    ],
)
def test_diffusion_gemma_builds_or_overwrites_template_toggle(
    initial_template_kwargs: object,
) -> None:
    """Malformed/missing template containers cannot suppress the model toggle."""

    payload = {}
    if initial_template_kwargs is not None:
        payload["chat_template_kwargs"] = initial_template_kwargs

    NvidiaProvider().handle_thinking_parameter(payload, MODEL)

    expected = {"enable_thinking": True}
    if isinstance(initial_template_kwargs, dict):
        expected["preserved"] = True
    assert payload["chat_template_kwargs"] == expected


@pytest.mark.parametrize(
    ("model", "assert_disabled"),
    [
        (
            "nvidia_nim/moonshotai/kimi-k2.5",
            lambda payload: payload["chat_template_kwargs"]["thinking"] is False,
        ),
        (
            "nvidia_nim/mistralai/mistral-medium-3.5-instruct",
            lambda payload: "extra_body" not in payload,
        ),
        (
            "nvidia_nim/deepseek-ai/deepseek-v3.2",
            lambda payload: payload["extra_body"]["chat_template_kwargs"]["thinking"]
            is False,
        ),
        (
            "nvidia_nim/deepseek-ai/deepseek-v4-flash",
            lambda payload: payload["extra_body"]["chat_template_kwargs"]["thinking"]
            is False,
        ),
    ],
)
def test_disabled_alias_applies_to_existing_reasoning_branches(
    model: str,
    assert_disabled,
) -> None:
    """The shared disabled alias intentionally applies to every supported family."""

    payload = {"reasoning_effort": "disabled"}

    NvidiaProvider().handle_thinking_parameter(payload, model)

    assert "reasoning_effort" not in payload
    assert assert_disabled(payload)
