# src/rotator_library/provider_factory.py

from .providers.gemini_auth_base import GeminiAuthBase
from .providers.qwen_auth_base import QwenAuthBase
from .providers.iflow_auth_base import IFlowAuthBase
from .providers.antigravity_auth_base import AntigravityAuthBase
from .providers.codex_auth_base import CodexAuthBase

PROVIDER_MAP = {
    "gemini_cli": GeminiAuthBase,
    "qwen_code": QwenAuthBase,
    "iflow": IFlowAuthBase,
    "antigravity": AntigravityAuthBase,
    "codex": CodexAuthBase,
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
