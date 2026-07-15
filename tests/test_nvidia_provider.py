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
