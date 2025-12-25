from typing import TYPE_CHECKING, Dict, Type

from .client import RotatingClient

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface
    from .model_info_service import ModelInfoService, ModelInfo, ModelMetadata
    from .anthropic_compat import (
        AnthropicMessagesRequest,
        AnthropicMessagesResponse,
        AnthropicCountTokensRequest,
        AnthropicCountTokensResponse,
    )

__all__ = [
    "RotatingClient",
    "PROVIDER_PLUGINS",
    "ModelInfoService",
    "ModelInfo",
    "ModelMetadata",
    # Anthropic compatibility
    "AnthropicMessagesRequest",
    "AnthropicMessagesResponse",
    "AnthropicCountTokensRequest",
    "AnthropicCountTokensResponse",
]


def __getattr__(name):
    """Lazy-load PROVIDER_PLUGINS, ModelInfoService, and Anthropic compat to speed up module import."""
    if name == "PROVIDER_PLUGINS":
        from .providers import PROVIDER_PLUGINS

        return PROVIDER_PLUGINS
    if name == "ModelInfoService":
        from .model_info_service import ModelInfoService

        return ModelInfoService
    if name == "ModelInfo":
        from .model_info_service import ModelInfo

        return ModelInfo
    if name == "ModelMetadata":
        from .model_info_service import ModelMetadata

        return ModelMetadata
    # Anthropic compatibility models
    if name == "AnthropicMessagesRequest":
        from .anthropic_compat import AnthropicMessagesRequest

        return AnthropicMessagesRequest
    if name == "AnthropicMessagesResponse":
        from .anthropic_compat import AnthropicMessagesResponse

        return AnthropicMessagesResponse
    if name == "AnthropicCountTokensRequest":
        from .anthropic_compat import AnthropicCountTokensRequest

        return AnthropicCountTokensRequest
    if name == "AnthropicCountTokensResponse":
        from .anthropic_compat import AnthropicCountTokensResponse

        return AnthropicCountTokensResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
