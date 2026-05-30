from __future__ import annotations

import pytest

from rotator_library.providers.gemini_cli_provider import GeminiCliProvider


@pytest.mark.asyncio
async def test_gemini_cli_declares_gemini_protocol_without_changing_custom_logic() -> None:
    provider = GeminiCliProvider()

    assert provider.has_custom_logic() is True
    assert provider.get_protocol_name("gemini-3-flash-preview") == "gemini"
    assert provider.get_adapter_names("gemini-3-flash-preview") == ()


@pytest.mark.asyncio
async def test_gemini_cli_declares_thought_signature_cache_rule() -> None:
    provider = GeminiCliProvider()

    rules = provider.get_field_cache_rules("gemini-3-flash-preview")

    assert len(rules) == 1
    assert rules[0].name == "gemini_cli_thought_signature"
    assert rules[0].path == "candidates.*.content.parts.*.thoughtSignature"
    assert rules[0].inject.path == "metadata.thoughtSignatures"
    assert rules[0].scope == ("provider", "model", "credential", "session")
