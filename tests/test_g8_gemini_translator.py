# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Translator-not-oracle pins (G8): the gemini protocol serves any model.

An undeclared model is translated literally with a disclosure; a
declared model is fully declaration-driven; a custom provider on the
gemini protocol gets no Google-family assumptions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.protocols.canonical import format_reasoning_controls
from rotator_library.protocols.types import UnifiedRequest


def _request(effort):
    req = UnifiedRequest(model="m")
    req.parameters = {"reasoning": {"effort": effort}}
    req.warnings = []
    return req


def _codes(warnings):
    out = []
    for w in warnings or []:
        code = getattr(w, "code", None)
        if code is None and isinstance(w, dict):
            code = w.get("code")
        out.append(code)
    return out


def test_undeclared_model_translates_literally_with_disclosure():
    req = _request("high")
    config = format_reasoning_controls(req.parameters.get("reasoning"), "gemini", req)
    assert config["generation_config"]["thinkingConfig"]["thinkingLevel"] == "high"
    assert "reasoning_undeclared_model" in _codes(req.warnings)


def test_declared_model_has_no_disclosure():
    req = _request("high")
    config = format_reasoning_controls(
        req.parameters.get("reasoning"),
        "gemini",
        req,
        capabilities={"thinking_dialect": "level"},
    )
    assert config["generation_config"]["thinkingConfig"]["thinkingLevel"] == "high"
    assert "reasoning_undeclared_model" not in _codes(req.warnings)


def test_budget_dialect_model_never_emits_level():
    req = _request("high")
    config = format_reasoning_controls(
        req.parameters.get("reasoning"),
        "gemini",
        req,
        capabilities={"thinking_dialect": "budget", "thinking_budget_range": [128, 32768]},
    )
    assert "thinkingLevel" not in config["generation_config"]["thinkingConfig"]
    assert config["generation_config"]["thinkingConfig"]["thinkingBudget"] >= 128


def test_off_accepted_suppresses_the_model_dependent_caveat():
    req = _request("off")
    config = format_reasoning_controls(req.parameters.get("reasoning"), "gemini", req)
    assert config["generation_config"]["thinkingConfig"]["thinkingBudget"] == 0
    assert "reasoning_disabled_model_dependent" in _codes(req.warnings)

    req2 = _request("off")
    config2 = format_reasoning_controls(
        req2.parameters.get("reasoning"),
        "gemini",
        req2,
        capabilities={"effort_accept": ["off", "low", "high"]},
    )
    assert config2["generation_config"]["thinkingConfig"]["thinkingBudget"] == 0
    assert "reasoning_disabled_model_dependent" not in _codes(req2.warnings)
