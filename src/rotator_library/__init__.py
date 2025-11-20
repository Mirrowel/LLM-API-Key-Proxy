from typing import TYPE_CHECKING, Dict, Type

from .client import RotatingClient

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface

__all__ = ["RotatingClient", "PROVIDER_PLUGINS"]

def __getattr__(name):
    """Lazy-load PROVIDER_PLUGINS to speed up module import."""
    if name == "PROVIDER_PLUGINS":
        from .providers import PROVIDER_PLUGINS
        return PROVIDER_PLUGINS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
