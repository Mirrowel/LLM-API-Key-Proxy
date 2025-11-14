import json
import os
import logging
from typing import Dict, Any, Optional

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class ModelDefinitions:
    """
    Simple model definitions loader from environment variables.
    Format: PROVIDER_MODELS={"model1": {"id": "id1"}, "model2": {"id": "id2", "options": {"reasoning_effort": "high"}}}
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize model definitions loader."""
        self.config_path = config_path
        self.definitions = {}
        self._load_definitions()

    def _load_definitions(self):
        """Load model definitions from environment variables."""
        for env_var, env_value in os.environ.items():
            if env_var.endswith("_MODELS"):
                provider_name = env_var[:-7].lower()  # Remove "_MODELS" (7 characters)
                try:
                    models_json = json.loads(env_value)
                    if isinstance(models_json, dict):
                        self.definitions[provider_name] = models_json
                        lib_logger.info(
                            f"Loaded {len(models_json)} models for provider: {provider_name}"
                        )
                except (json.JSONDecodeError, TypeError) as e:
                    lib_logger.warning(f"Invalid JSON in {env_var}: {e}")

    def get_provider_models(self, provider_name: str) -> Dict[str, Any]:
        """Get all models for a provider."""
        return self.definitions.get(provider_name, {})

    def get_model_definition(
        self, provider_name: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific model definition."""
        provider_models = self.get_provider_models(provider_name)
        return provider_models.get(model_name)

    def get_model_options(self, provider_name: str, model_name: str) -> Dict[str, Any]:
        """Get options for a specific model."""
        model_def = self.get_model_definition(provider_name, model_name)
        return model_def.get("options", {}) if model_def else {}

    def get_model_id(self, provider_name: str, model_name: str) -> Optional[str]:
        """Get model ID for a specific model."""
        model_def = self.get_model_definition(provider_name, model_name)
        return model_def.get("id") if model_def else None

    def get_all_provider_models(self, provider_name: str) -> list:
        """Get all model names with provider prefix."""
        provider_models = self.get_provider_models(provider_name)
        return [f"{provider_name}/{model}" for model in provider_models.keys()]

    def reload_definitions(self):
        """Reload model definitions from environment variables."""
        self.definitions.clear()
        self._load_definitions()
