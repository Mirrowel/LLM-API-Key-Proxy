import importlib
import pkgutil
from typing import Dict, Type
from .provider_interface import ProviderInterface

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}

def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    """
    package_path = __path__
    package_name = __name__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Construct the full module path
        full_module_path = f"{package_name}.{module_name}"
        
        # Import the module
        module = importlib.import_module(full_module_path)

        # Look for a class that inherits from ProviderInterface
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, ProviderInterface) and attribute is not ProviderInterface:
                # Derives 'gemini_cli' from 'gemini_cli_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                #print(f"Registered provider: {provider_name}")

# Discover and register providers when the package is imported
_register_providers()
