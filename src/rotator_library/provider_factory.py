# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/provider_factory.py

from .providers.gemini_auth_base import GeminiAuthBase
from .providers.openai_oauth_base import OpenAIOAuthBase
from .providers.anthropic_oauth_base import AnthropicOAuthBase
from .providers.copilot_auth_base import CopilotAuthBase
from .providers.x_ai_auth_base import XAiAuthBase

PROVIDER_MAP = {
    "gemini_cli": GeminiAuthBase,
    "codex": OpenAIOAuthBase,
    "anthropic": AnthropicOAuthBase,
    "copilot": CopilotAuthBase,
    "x-ai": XAiAuthBase,
}

def get_provider_auth_class(provider_name: str):
    """
    Returns the authentication class for a given provider.
    """
    provider_class = PROVIDER_MAP.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class

def get_available_providers():
    """
    Returns a list of available provider names.
    """
    return list(PROVIDER_MAP.keys())
