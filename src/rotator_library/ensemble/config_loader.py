"""
Configuration loader for HiveMind ensemble configs.

Loads and validates configurations from the ensemble_configs directory structure.
"""

import os
import json
import logging
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional

lib_logger = logging.getLogger("rotator_library.ensemble")


class ConfigLoader:
    """Loads and manages ensemble configurations from folder structure."""
    
    def __init__(self, config_dir: str):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Path to ensemble_configs directory (relative to rotator_library)
        """
        self.config_dir = Path(config_dir)
        self.swarms_dir = self.config_dir / "swarms"
        self.fusions_dir = self.config_dir / "fusions"
        self.strategies_dir = self.config_dir / "strategies"
        
        # Loaded configurations
        self.swarm_default: Optional[Dict[str, Any]] = None
        self.swarm_configs: Dict[str, Dict[str, Any]] = {}
        self.fusion_configs: Dict[str, Dict[str, Any]] = {}
        self.strategies: Dict[str, str] = {}
        
    def load_all(self) -> None:
        """Load all configurations from the directory structure."""
        lib_logger.info("[HiveMind] Loading ensemble configurations...")
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Load swarm configurations
        self._load_swarm_configs()
        
        # Load fusion configurations
        self._load_fusion_configs()
        
        # Load strategy templates
        self._load_strategies()
        
        lib_logger.info(
            f"[HiveMind] Loaded {len(self.swarm_configs)} swarm configs, "
            f"{len(self.fusion_configs)} fusion configs, "
            f"{len(self.strategies)} strategies"
        )
    
    def _ensure_directories(self) -> None:
        """Create config directories if they don't exist."""
        for directory in [self.swarms_dir, self.fusions_dir, self.strategies_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_swarm_configs(self) -> None:
        """Load swarm configurations from swarms/ directory."""
        if not self.swarms_dir.exists():
            lib_logger.warning(f"[HiveMind] Swarms directory not found: {self.swarms_dir}")
            return
        
        # Load default.json first
        default_path = self.swarms_dir / "default.json"
        if default_path.exists():
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    self.swarm_default = json.load(f)
                lib_logger.debug("[HiveMind] Loaded default swarm config")
            except Exception as e:
                lib_logger.error(f"[HiveMind] Failed to load default swarm config: {e}")
        else:
            lib_logger.warning("[HiveMind] No default swarm config found")
        
        # Load model-specific configs
        for config_file in self.swarms_dir.glob("*.json"):
            if config_file.name == "default.json":
                continue
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Extract model name from config
                model_name = config.get("model")
                if model_name:
                    self.swarm_configs[model_name] = config
                    lib_logger.debug(f"[HiveMind] Loaded swarm config for '{model_name}'")
                else:
                    lib_logger.warning(
                        f"[HiveMind] Swarm config '{config_file.name}' missing 'model' field"
                    )
            except Exception as e:
                lib_logger.error(f"[HiveMind] Failed to load swarm config '{config_file.name}': {e}")
    
    def _load_fusion_configs(self) -> None:
        """Load fusion configurations from fusions/ directory."""
        if not self.fusions_dir.exists():
            lib_logger.warning(f"[HiveMind] Fusions directory not found: {self.fusions_dir}")
            return
        
        for config_file in self.fusions_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                fusion_id = config.get("id")
                if not fusion_id:
                    lib_logger.warning(
                        f"[HiveMind] Fusion config '{config_file.name}' missing 'id' field"
                    )
                    continue
                
                # Check for duplicate IDs
                if fusion_id in self.fusion_configs:
                    lib_logger.warning(
                        f"[HiveMind] Duplicate fusion ID '{fusion_id}'. "
                        f"Config from '{config_file.name}' will override previous."
                    )
                
                self.fusion_configs[fusion_id] = config
                lib_logger.debug(f"[HiveMind] Loaded fusion config '{fusion_id}'")
                
            except Exception as e:
                lib_logger.error(f"[HiveMind] Failed to load fusion config '{config_file.name}': {e}")
    
    def _load_strategies(self) -> None:
        """Load strategy templates from strategies/ directory."""
        if not self.strategies_dir.exists():
            lib_logger.warning(f"[HiveMind] Strategies directory not found: {self.strategies_dir}")
            return
        
        for strategy_file in self.strategies_dir.glob("*.txt"):
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                strategy_name = strategy_file.stem
                self.strategies[strategy_name] = content
                lib_logger.debug(f"[HiveMind] Loaded strategy '{strategy_name}'")
                
            except Exception as e:
                lib_logger.error(
                    f"[HiveMind] Failed to load strategy '{strategy_file.name}': {e}"
                )
    
    def get_swarm_config(self, model: str) -> Dict[str, Any]:
        """
        Get swarm configuration for a specific model.
        
        Merges default config with model-specific overrides.
        
        Args:
            model: Base model name (without [swarm] suffix)
        
        Returns:
            Merged configuration dictionary
        """
        # BUGFIX: Use deepcopy to prevent mutations to global default config
        config = copy.deepcopy(self.swarm_default) if self.swarm_default else {}
        
        # Apply model-specific overrides
        if model in self.swarm_configs:
            model_config = self.swarm_configs[model]
            # Deep merge
            for key, value in model_config.items():
                if key == "model":
                    continue  # Don't copy the model name
                if isinstance(value, dict) and key in config:
                    config[key] = {**config[key], **value}
                else:
                    config[key] = value
        
        return config
    
    def get_fusion_config(self, fusion_id: str) -> Optional[Dict[str, Any]]:
        """
        Get fusion configuration by ID.
        
        Args:
            fusion_id: Fusion identifier
        
        Returns:
            Fusion configuration or None if not found
        """
        return self.fusion_configs.get(fusion_id)
    
    def get_strategy(self, strategy_name: str) -> Optional[str]:
        """
        Get strategy template by name.
        
        Args:
            strategy_name: Strategy identifier
        
        Returns:
            Strategy template string or None if not found
        """
        return self.strategies.get(strategy_name)
    
    def get_all_fusion_ids(self) -> List[str]:
        """Get list of all fusion IDs."""
        return list(self.fusion_configs.keys())
    
    def get_all_swarm_model_ids(self) -> List[str]:
        """
        Get all discoverable swarm model variants.
        
        Generates model IDs from all swarm configs that define base_models.
        Format: {base_model}-{preset_id}[swarm]
        
        Returns:
            List of swarm model IDs
        """
        swarm_models = []
        
        for config_file in self.swarms_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                    preset_id = config.get("id")
                    base_models = config.get("base_models", [])
                    
                    if not preset_id:
                        lib_logger.debug(f"Swarm config {config_file.name} missing 'id', skipping")
                        continue
                    
                    if not base_models:
                        lib_logger.debug(f"Swarm config {preset_id} has no base_models, not discoverable")
                        continue
                    
                    # Generate model IDs: {base_model}-{preset_id}[swarm]
                    for base_model in base_models:
                        model_id = f"{base_model}-{preset_id}[swarm]"
                        swarm_models.append(model_id)
                        
            except Exception as e:
                lib_logger.warning(f"Failed to process swarm config {config_file.name}: {e}")
        
        lib_logger.info(f"Discovered {len(swarm_models)} swarm model variants")
        return swarm_models
