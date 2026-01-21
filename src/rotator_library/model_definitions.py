# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

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

    Supports two formats:
    1. Array format (simple): PROVIDER_MODELS=["model-1", "model-2", "model-3"]
       - Each model name is used as both name and ID
    2. Dict format (advanced): PROVIDER_MODELS={"model-name": {"id": "model-id", "options": {...}}}
       - The 'id' field is optional - if not provided, the model name (key) is used as the ID

    Examples:
    - IFLOW_MODELS='["glm-4.6", "qwen3-max"]' - simple array format
    - IFLOW_MODELS='{"glm-4.6": {}}' - dict format, uses "glm-4.6" as both name and ID
    - IFLOW_MODELS='{"custom-name": {"id": "actual-id"}}' - dict format with custom ID
    - IFLOW_MODELS='{"model": {"id": "id", "options": {"temperature": 0.7}}}' - with options

    This class is a singleton - instantiated once and shared across all providers.
    """

    _instance: Optional["ModelDefinitions"] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize model definitions loader (only runs once due to singleton)."""
        if ModelDefinitions._initialized:
            return
        ModelDefinitions._initialized = True
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

                    # Handle dict format: {"model-name": {"id": "...", "options": {...}}}
                    if isinstance(models_json, dict):
                        self.definitions[provider_name] = models_json
                        lib_logger.info(
                            f"Loaded {len(models_json)} models for provider: {provider_name}"
                        )
                    # Handle array format: ["model-1", "model-2", "model-3"]
                    elif isinstance(models_json, list):
                        # Convert array to dict format with empty definitions
                        models_dict = {
                            model_name: {}
                            for model_name in models_json
                            if isinstance(model_name, str)
                        }
                        self.definitions[provider_name] = models_dict
                        lib_logger.info(
                            f"Loaded {len(models_dict)} models for provider: {provider_name} (array format)"
                        )
                    else:
                        lib_logger.warning(
                            f"{env_var} must be a JSON object or array, got {type(models_json).__name__}"
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
        """Get model ID for a specific model. Falls back to model_name if 'id' is not specified."""
        model_def = self.get_model_definition(provider_name, model_name)
        if not model_def:
            return None
        # Use 'id' if provided, otherwise use the model_name as the ID
        return model_def.get("id", model_name)

    def get_all_provider_models(self, provider_name: str) -> list:
        """Get all model names with provider prefix."""
        provider_models = self.get_provider_models(provider_name)
        return [f"{provider_name}/{model}" for model in provider_models.keys()]

    def reload_definitions(self):
        """Reload model definitions from environment variables."""
        self.definitions.clear()
        self._load_definitions()
