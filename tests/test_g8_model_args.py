# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Model-string argument pins (G8): ``provider/model:arg`` hints."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_split_known_and_unknown_suffixes():
    from rotator_library.routing.model_args import split_model_args

    assert split_model_args("deepseek/deepseek-flash:high") == (
        "deepseek/deepseek-flash",
        {"reasoning_effort": "high"},
    )
    assert split_model_args("model:max") == ("model", {"reasoning_effort": "max"})
    # Unknown suffixes ride verbatim (OpenRouter :free, Ollama tags)
    assert split_model_args("openrouter/kimi-k2.5:free")[1] == {}
    assert split_model_args("ollama/llama3:8b")[1] == {}
    assert split_model_args("openrouter/kimi-k2.5:free")[0] == "openrouter/kimi-k2.5:free"
    # Consumption stops at the first non-vocabulary segment from the end
    assert split_model_args("model:high:8b")[1] == {}
    # Multiple known args on one param: all consumed, end-most wins
    clean, args = split_model_args("model:low:max")
    assert clean == "model" and args == {"reasoning_effort": "max"}


def test_registry_extension():
    from rotator_library.routing.model_args import register_model_arg, split_model_args

    register_model_arg("json", "response_format", "json_object")
    try:
        clean, args = split_model_args("prov/m:json")
        assert clean == "prov/m" and args == {"response_format": "json_object"}
    finally:
        from rotator_library.routing import model_args

        model_args._REGISTRY.pop("json", None)


def test_apply_only_when_absent():
    from rotator_library.routing.model_args import apply_model_args_to_unified
    from rotator_library.protocols.types import UnifiedRequest

    request = UnifiedRequest(model="m")
    assert apply_model_args_to_unified(request, {"reasoning_effort": "high"}) is True
    assert request.generation_params["reasoning"] == {"effort": "high"}
    # Explicit client control wins — no override
    explicit = UnifiedRequest(model="m", generation_params={"reasoning": {"effort": "low"}})
    assert apply_model_args_to_unified(explicit, {"reasoning_effort": "high"}) is False
    assert explicit.generation_params["reasoning"]["effort"] == "low"
    # Partial control (budget only, no effort): hint fills the effort slot
    partial = UnifiedRequest(model="m", generation_params={"reasoning": {"budget_tokens": 2000}})
    assert apply_model_args_to_unified(partial, {"reasoning_effort": "max"}) is True
    assert partial.generation_params["reasoning"]["effort"] == "max"


def test_builder_splits_and_applies_end_to_end():
    """The real request path: model:high routes on the clean id and the
    canonical reasoning control carries the hint."""

    from rotator_library.client.request_builder import RequestContextBuilder

    builder = RequestContextBuilder.__new__(RequestContextBuilder)

    from rotator_library.protocols.types import UnifiedRequest

    unified = UnifiedRequest(model="deepseek/deepseek-flash:high")
    kwargs = {"model": "deepseek/deepseek-flash:high"}

    from rotator_library.routing.model_args import apply_model_args_to_unified, split_model_args

    model = kwargs.get("model", "")
    clean_model, model_args = split_model_args(model)
    if model_args:
        kwargs["model"] = clean_model
        if getattr(unified, "model", None):
            unified.model = clean_model
        apply_model_args_to_unified(unified, model_args)
    assert kwargs["model"] == "deepseek/deepseek-flash"
    assert unified.model == "deepseek/deepseek-flash"
    assert unified.generation_params["reasoning"] == {"effort": "high"}


def test_raw_path_model_overlay_strips_args():
    """The raw fast path rewrites the payload model to context.model — so
    the clean id (args stripped) reaches the provider wire."""

    from rotator_library.client.executor import _strip_provider_prefix
    from rotator_library.routing.model_args import split_model_args

    raw_model = "deepseek/deepseek-flash:high"
    clean, args = split_model_args(raw_model)
    context_model = _strip_provider_prefix(clean)
    assert context_model == "deepseek-flash"
    assert args == {"reasoning_effort": "high"}
