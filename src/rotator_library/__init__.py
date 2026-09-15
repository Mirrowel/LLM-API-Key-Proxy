# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import TYPE_CHECKING, Type

# For type checkers (Pylance, mypy), import statically. At runtime every
# public name resolves lazily via __getattr__: `import rotator_library`
# stays milliseconds-fast (the client package pulls litellm, ~8s) so the
# proxy's launcher/load-screen phasing keeps heavy imports behind the
# loading screens (see utils.paths / startup invariants).
if TYPE_CHECKING:
    from .client import RotatingClient
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface
    from .model_info_service import ModelInfoService, ModelInfo, ModelMetadata

__all__ = [
    "RotatingClient",
    "PROVIDER_PLUGINS",
    "ModelInfoService",
    "ModelInfo",
    "ModelMetadata",
]


def __getattr__(name):
    """Lazy-load public names to keep `import rotator_library` fast."""
    if name == "RotatingClient":
        from .client import RotatingClient

        return RotatingClient
    if name == "ProviderInterface":
        from .providers.provider_interface import ProviderInterface

        return ProviderInterface
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
