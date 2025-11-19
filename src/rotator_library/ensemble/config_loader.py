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
        self.roles_dir = self.config_dir / "roles"
        
        # Loaded configurations
        self.swarm_default: Optional[Dict[str, Any]] = None
        self.swarm_configs: Dict[str, Dict[str, Any]] = {}
        self.fusion_configs: Dict[str, Dict[str, Any]] = {}
        self.strategies: Dict[str, str] = {}
        self.role_templates: Dict[str, Dict[str, Any]] = {}
        
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
        
        # Load role templates
        self._load_roles()
        
        lib_logger.info(
            f"[HiveMind] Loaded {len(self.swarm_configs)} swarm configs, "
            f"{len(self.fusion_configs)} fusion configs, "
            f"{len(self.strategies)} strategies, "
            f"{len(self.role_templates)} roles"
        )
    
    def _ensure_directories(self) -> None:
        """Create config directories if they don't exist."""
        for directory in [self.swarms_dir, self.fusions_dir, self.strategies_dir, self.roles_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_swarm_configs(self) -> None:
        """Load swarm configurations from swarms/ directory.
        
        Only supports preset-based format with 'id' and 'base_models'.
        """
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
        
        # All swarm configs now use preset-based format (id + base_models)
        # Discovery is handled by get_all_swarm_model_ids()
        # Individual preset configs loaded on-demand via get_swarm_config()
    
    def _load_fusion_configs(self) -> None:
        """Load fusion configurations from fusions/ directory.
        
        Supports two formats:
        1. Single fusion: {"id": "...", "specialists": [...], ...}
        2. Multiple fusions: {"fusions": [{"id": "...", ...}, ...]}
        """
        if not self.fusions_dir.exists():
            lib_logger.warning(f"[HiveMind] Fusions directory not found: {self.fusions_dir}")
            return
        
        for config_file in self.fusions_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check if this is the new array format
                if "fusions" in config:
                    # New format: {"fusions": [...]}
                    fusions_list = config.get("fusions", [])
                    if not isinstance(fusions_list, list):
                        lib_logger.warning(
                            f"[HiveMind] Config '{config_file.name}' has 'fusions' but it's not a list"
                        )
                        continue
                    
                    for fusion in fusions_list:
                        self._register_fusion(fusion, config_file.name)
                else:
                    # Old format: {"id": "...", "specialists": [...], ...}
                    self._register_fusion(config, config_file.name)
                
            except Exception as e:
                lib_logger.error(f"[HiveMind] Failed to load fusion config '{config_file.name}': {e}")
    
    def _register_fusion(self, fusion: Dict[str, Any], source_file: str) -> None:
        """Register a single fusion configuration."""
        fusion_id = fusion.get("id")
        if not fusion_id:
            lib_logger.warning(
                f"[HiveMind] Fusion in '{source_file}' missing 'id' field"
            )
            return
        
        # Check for duplicate IDs
        if fusion_id in self.fusion_configs:
            lib_logger.warning(
                f"[HiveMind] Duplicate fusion ID '{fusion_id}'. "
                f"Config from '{source_file}' will override previous."
            )
        
        self.fusion_configs[fusion_id] = fusion
        lib_logger.debug(f"[HiveMind] Loaded fusion config '{fusion_id}'")
    
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
    
    def _load_roles(self) -> None:
        """Load role templates from roles/ directory.
        
        Supports two formats:
        1. Single role: {"name": "...", "system_prompt": "...", ...}
        2. Multiple roles: {"roles": [{"name": "...", ...}, ...]}
        """
        if not self.roles_dir.exists():
            lib_logger.warning(f"[HiveMind] Roles directory not found: {self.roles_dir}")
            return
        
        for role_file in self.roles_dir.glob("*.json"):
            try:
                with open(role_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if this is the new array format
                if "roles" in data:
                    # New format: {"roles": [...]}
                    roles_list = data.get("roles", [])
                    if not isinstance(roles_list, list):
                        lib_logger.warning(
                            f"[HiveMind] Role file '{role_file.name}' has 'roles' but it's not a list"
                        )
                        continue
                    
                    for role in roles_list:
                        self._register_role(role, role_file.name)
                else:
                    # Old format: {"name": "...", "system_prompt": "...", ...}
                    # Use filename as role_id
                    role_id = role_file.stem
                    self.role_templates[role_id] = data
                    lib_logger.debug(f"[HiveMind] Loaded role template '{role_id}'")
                
            except Exception as e:
                lib_logger.error(
                    f"[HiveMind] Failed to load role template '{role_file.name}': {e}"
                )
    
    def _register_role(self, role: Dict[str, Any], source_file: str) -> None:
        """Register a single role template."""
        # Use 'name' field as role_id, convert to lowercase with hyphens
        role_name = role.get("name")
        if not role_name:
            lib_logger.warning(
                f"[HiveMind] Role in '{source_file}' missing 'name' field"
            )
            return
        
        # Convert name to role_id (e.g., "Security Expert" -> "security-expert")
        role_id = role_name.lower().replace(" ", "-")
        
        # Check for duplicate IDs
        if role_id in self.role_templates:
            lib_logger.warning(
                f"[HiveMind] Duplicate role ID '{role_id}'. "
                f"Role from '{source_file}' will override previous."
            )
        
        self.role_templates[role_id] = role
        lib_logger.debug(f"[HiveMind] Loaded role template '{role_id}' from array")
    
    def get_swarm_config(self, preset_id: str) -> Dict[str, Any]:
        """
        Get swarm configuration for a specific preset.
        
        Args:
            preset_id: Preset ID (e.g., "default", "aggressive")
        
        Returns:
            Configuration dictionary with defaults applied
        """
        # Try to load preset config file
        config_file = self.swarms_dir / f"{preset_id}.json"
        
        if not config_file.exists():
            lib_logger.warning(f"[HiveMind] Swarm preset '{preset_id}' not found")
            # Return default config if available
            return copy.deepcopy(self.swarm_default) if self.swarm_default else {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate it's a preset-based config
            if "id" not in config or "base_models" not in config:
                lib_logger.warning(
                    f"[HiveMind] Swarm config '{preset_id}' missing 'id' or 'base_models'"
                )
                return copy.deepcopy(self.swarm_default) if self.swarm_default else {}
            
            return config
            
        except Exception as e:
            lib_logger.error(f"[HiveMind] Failed to load swarm preset '{preset_id}': {e}")
            return copy.deepcopy(self.swarm_default) if self.swarm_default else {}
    
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
    
    def get_role_template(self, role_id: str) -> Optional[Dict[str, Any]]:
        """
        Get role template by ID.
        
        Args:
            role_id: Role template identifier (e.g., "architect", "security-expert")
        
        Returns:
            Role template dictionary or None if not found
        """
        return self.role_templates.get(role_id)
    
    def get_all_fusion_ids(self) -> List[str]:
        """Get list of all fusion IDs with [fusion] suffix."""
        return [f"{fusion_id}[fusion]" for fusion_id in self.fusion_configs.keys()]
    
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
