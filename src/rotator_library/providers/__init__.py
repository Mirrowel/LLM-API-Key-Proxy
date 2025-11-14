import importlib
import pkgutil
import os
from typing import Dict, Type
from .provider_interface import ProviderInterface

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}


class DynamicOpenAICompatibleProvider:
    """
    Dynamic provider class for custom OpenAI-compatible providers.
    Created at runtime for providers with API_BASE environment variables.
    """

    # Class attribute - no need to instantiate
    skip_cost_calculation: bool = True

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment
        self.api_base = os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for OpenAI-compatible provider"
            )

        # Import model definitions
        from ..model_definitions import ModelDefinitions

        self.model_definitions = ModelDefinitions()

    def get_models(self, api_key: str, client):
        """Delegate to OpenAI-compatible provider implementation."""
        from .openai_compatible_provider import OpenAICompatibleProvider

        # Create temporary instance to reuse logic
        temp_provider = OpenAICompatibleProvider(self.provider_name)
        return temp_provider.get_models(api_key, client)

    def get_model_options(self, model_name: str) -> Dict[str, any]:
        """Get model options from static definitions."""
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """Returns False since we want to use the standard litellm flow."""
        return False

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Returns the standard Bearer token header."""
        return {"Authorization": f"Bearer {credential_identifier}"}


def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    Also creates dynamic plugins for custom OpenAI-compatible providers.
    """
    package_path = __path__
    package_name = __name__

    # First, register file-based providers
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Construct the full module path
        full_module_path = f"{package_name}.{module_name}"

        # Import the module
        module = importlib.import_module(full_module_path)

        # Look for a class that inherits from ProviderInterface
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, ProviderInterface)
                and attribute is not ProviderInterface
            ):
                # Derives 'gemini_cli' from 'gemini_cli_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                # print(f"Registered provider: {provider_name}")

    # Then, create dynamic plugins for custom OpenAI-compatible providers
    # Load environment variables to find custom providers
    from dotenv import load_dotenv

    load_dotenv()

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip known providers that already have file-based plugins
            if provider_name in [
                "openai",
                "anthropic",
                "google",
                "gemini",
                "nvidia",
                "mistral",
                "cohere",
                "groq",
                "openrouter",
                "chutes",
            ]:
                continue

            # Create a dynamic plugin class
            def create_plugin_class(name):
                class DynamicPlugin(DynamicOpenAICompatibleProvider):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            PROVIDER_PLUGINS[provider_name] = plugin_class
            # print(f"Registered dynamic provider: {provider_name}")


# Discover and register providers when the package is imported
_register_providers()
