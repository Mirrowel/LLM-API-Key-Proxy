from typing import TYPE_CHECKING, Dict, Type

from .client import RotatingClient

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface
    from .model_info_service import ModelInfoService, ModelInfo

__all__ = ["RotatingClient", "PROVIDER_PLUGINS", "ModelInfoService", "ModelInfo"]

def __getattr__(name):
    """Lazy-load PROVIDER_PLUGINS and ModelInfoService to speed up module import."""
    if name == "PROVIDER_PLUGINS":
        from .providers import PROVIDER_PLUGINS
        return PROVIDER_PLUGINS
    if name == "ModelInfoService":
        from .model_info_service import ModelInfoService
        return ModelInfoService
    if name == "ModelInfo":
        from .model_info_service import ModelInfo
        return ModelInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
