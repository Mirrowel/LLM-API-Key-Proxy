"""
EnsembleManager - Core orchestration for HiveMind (Swarm/Fusion) feature.

This module manages parallel model execution with intelligent arbitration.
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from .config_loader import ConfigLoader

lib_logger = logging.getLogger("rotator_library.ensemble")


class EnsembleManager:
    """
    Manages ensemble execution (Swarm and Fusion modes).
    
    Responsibilities:
    - Detect ensemble requests (swarm suffix or fusion ID)
    - Load and manage configurations
    - Handle naming conflicts
    - Orchestrate parallel execution (implemented in later phases)
    """
    
    def __init__(self, rotating_client, config_dir: Optional[str] = None):
        """
        Initialize the ensemble manager.
        
        Args:
            rotating_client: Reference to RotatingClient for making API calls
            config_dir: Path to ensemble_configs directory (relative to this file)
        """
        self.rotating_client = rotating_client
        
        # Default config directory (relative to this file)
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "ensemble_configs"
            )
        
        # Initialize config loader
        self.config_loader = ConfigLoader(config_dir)
        self.config_loader.load_all()
        
        # Cache for resolved ensemble names (for conflict resolution)
        self._resolved_names: Dict[str, str] = {}
        
        # Cache for provider models (loaded from RotatingClient)
        self._provider_models: Optional[Set[str]] = None
        
        lib_logger.info("[HiveMind] EnsembleManager initialized")
    
    def is_ensemble(self, model_id: str) -> bool:
        """
        Check if a model ID represents an ensemble request.
        
        Args:
            model_id: Full model ID from user request
        
        Returns:
            True if this is an ensemble (swarm or fusion), False otherwise
        """
        # Check for fusion ID (exact match)
        if model_id in self.config_loader.fusion_configs:
            return True
        
        # Check for swarm suffix
        if self._is_swarm_request(model_id):
            return True
        
        return False
    
    def _is_swarm_request(self, model_id: str) -> bool:
        """
        Check if model ID contains swarm suffix.
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if this is a swarm request
        """
        # Get default suffix from config
        default_suffix = "[swarm]"
        if self.config_loader.swarm_default:
            default_suffix = self.config_loader.swarm_default.get("suffix", "[swarm]")
        
        return default_suffix in model_id
    
    def get_base_model(self, swarm_id: str) -> str:
        """
        Extract base model name from swarm ID.
        
        Args:
            swarm_id: Swarm model ID (e.g., "gemini-1.5-flash[swarm]")
        
        Returns:
            Base model name (e.g., "gemini-1.5-flash")
        """
        # Get suffix from config
        default_suffix = "[swarm]"
        if self.config_loader.swarm_default:
            default_suffix = self.config_loader.swarm_default.get("suffix", "[swarm]")
        
        # Remove suffix
        if default_suffix in swarm_id:
            return swarm_id.replace(default_suffix, "")
        
        return swarm_id
    
    def resolve_conflicts(self, ensemble_id: str) -> str:
        """
        Resolve naming conflicts by appending numeric suffixes.
        
        If an ensemble ID conflicts with a real provider model,
        append -1, -2, -3, etc. until unique.
        
        Args:
            ensemble_id: Original ensemble ID (swarm or fusion)
        
        Returns:
            Resolved unique ensemble ID
        """
        # Check cache first
        if ensemble_id in self._resolved_names:
            return self._resolved_names[ensemble_id]
        
        # Load provider models if not cached
        if self._provider_models is None:
            self._load_provider_models()
        
        # Check for conflict
        if ensemble_id not in self._provider_models:
            # No conflict, use original
            self._resolved_names[ensemble_id] = ensemble_id
            return ensemble_id
        
        # Conflict detected, find available suffix
        counter = 1
        while True:
            candidate = f"{ensemble_id}-{counter}"
            if candidate not in self._provider_models:
                lib_logger.warning(
                    f"[HiveMind] Naming conflict detected. "
                    f"Renamed '{ensemble_id}' to '{candidate}'"
                )
                self._resolved_names[ensemble_id] = candidate
                return candidate
            counter += 1
            
            # Safety check (shouldn't happen in practice)
            if counter > 100:
                lib_logger.error(
                    f"[HiveMind] Could not resolve naming conflict for '{ensemble_id}' "
                    f"after 100 attempts"
                )
                return f"{ensemble_id}-{counter}"
    
    def _load_provider_models(self) -> None:
        """
        Load all provider models from RotatingClient.
        
        This is used for conflict detection.
        """
        try:
            # Get all available models (this might be async in the actual implementation)
            # For now, we'll use a synchronous approach
            # TODO: Handle async model loading properly
            self._provider_models = set()
            
            # Note: This will be implemented properly when we integrate with RotatingClient
            # For now, just initialize an empty set
            lib_logger.debug("[HiveMind] Provider models cache initialized (empty)")
            
        except Exception as e:
            lib_logger.error(f"[HiveMind] Failed to load provider models: {e}")
            self._provider_models = set()
    
    def get_fusion_ids(self) -> List[str]:
        """
        Get list of all configured fusion IDs.
        
        Returns:
            List of fusion identifiers
        """
        return self.config_loader.get_all_fusion_ids()
    
    async def handle_request(self, request, **kwargs):
        """
        Handle an ensemble request (swarm or fusion).
        
        This is the main entry point for ensemble execution.
        Will be implemented in Phase 2.
        
        Args:
            request: Original request object
            **kwargs: Request parameters
        
        Returns:
            Response from arbiter (streaming or complete)
        """
        model_id = kwargs.get("model")
        
        if not model_id:
            raise ValueError("Model ID is required")
        
        # Resolve conflicts
        resolved_id = self.resolve_conflicts(model_id)
        
        # Determine type
        if resolved_id in self.config_loader.fusion_configs:
            lib_logger.info(f"[HiveMind] Processing Fusion request: {resolved_id}")
            # TODO: Implement fusion handling in Phase 5
            raise NotImplementedError("Fusion mode not yet implemented")
        
        elif self._is_swarm_request(resolved_id):
            base_model = self.get_base_model(resolved_id)
            config = self.config_loader.get_swarm_config(base_model)
            count = config.get("count", 3)
            
            lib_logger.info(
                f"[HiveMind] Processing Swarm request: {resolved_id} "
                f"(base: {base_model}, {count} drones)"
            )
            # TODO: Implement swarm handling in Phase 2
            raise NotImplementedError("Swarm mode not yet implemented")
        
        else:
            raise ValueError(f"Unknown ensemble type for model: {model_id}")
