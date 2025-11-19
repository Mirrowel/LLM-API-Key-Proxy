"""
EnsembleManager - Core orchestration for HiveMind (Swarm/Fusion) feature.

This module manages parallel model execution with intelligent arbitration.
"""

import os
import logging
import asyncio
import random
import copy
import time
import re
from typing import Dict, List, Any, Optional, Set

import litellm

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
        
        # Initialize provider models
        self._load_provider_models()
        
        lib_logger.info("[HiveMind] EnsembleManager initialized")
    
    def is_ensemble(self, model_id: str) -> bool:
        """
        Check if a model ID represents an ensemble request.
        
        Args:
            model_id: Full model ID from user request
        
        Returns:
            True if this is an ensemble (swarm or fusion), False otherwise
        """
        # BUGFIX: Check for conflict first (Provider Model Shadowing)
        # If the model ID exists in provider models, it's NOT an ensemble request
        # (unless we've already resolved it, but this check is for the raw request)
        if self._provider_models is None:
            self._load_provider_models()
            
        if model_id in self._provider_models:
            return False

        # Check for fusion suffix
        if model_id.endswith("[fusion]"):
            return True
        
        # Check for swarm suffix
        if self._is_swarm_request(model_id):
            return True
        
        return False
    
    def _is_swarm_request(self, model_id: str) -> bool:
        """
        Check if model ID contains swarm suffix.
        
        Supports new preset-based format: {base_model}-{preset_id}[swarm]
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if this is a swarm request
        """
        return model_id.endswith("[swarm]")
    
    def get_base_model(self, swarm_id: str) -> str:
        """
        Extract base model name from swarm ID.
        
        Supports new format: {base_model}-{preset_id}[swarm]
        
        Args:
            swarm_id: Swarm model ID (e.g., "gpt-4o-default[swarm]")
        
        Returns:
            Base model name (e.g., "gpt-4o")
        """
        # Remove [swarm] suffix first
        if swarm_id.endswith("[swarm]"):
            swarm_id = swarm_id[:-7]  # Remove "[swarm]"
        
        # Parse: {base_model}-{preset_id}
        # preset_id is the last segment after the last hyphen
        if "-" in swarm_id:
            # Split and check if last segment is a preset ID
            parts = swarm_id.rsplit("-", 1)
            potential_preset = parts[1]
            
            # Check if it's a valid preset ID in our configs
            # For now, just check if the directory has that config file
            config_file = self.config_loader.swarms_dir / f"{potential_preset}.json"
            if config_file.exists():
                # This is a preset ID, so base_model is everything before it
                return parts[0]
        
        # If no preset found or no hyphen, treat entire thing as base_model
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
            self._provider_models = set()
            
            # BUGFIX: Populate provider models from RotatingClient.model_definitions
            if hasattr(self.rotating_client, 'model_definitions'):
                defs = self.rotating_client.model_definitions.definitions
                for provider, models in defs.items():
                    for model_name in models.keys():
                        self._provider_models.add(model_name)
                        self._provider_models.add(f"{provider}/{model_name}")
            
            lib_logger.debug(f"[HiveMind] Loaded {len(self._provider_models)} provider models for conflict detection")
            
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
    
    def _prepare_drones(
        self,
        config: Dict[str, Any],
        base_model: str,
        request_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare drone configurations for parallel execution.
        
        Creates N identical copies of the request parameters with the base model.
        Advanced features (jitter, adversarial) will be added in Phase 4.
        
        Args:
            config: Swarm configuration
            base_model: Base model to use for all drones
            request_params: Original request parameters
        
        Returns:
            List of drone configurations ready for parallel execution
        """
        count = config.get("count", 3)
        drones = []
        
        # Get temperature jitter config
        temp_jitter_config = config.get("temperature_jitter", {})
        jitter_enabled = temp_jitter_config.get("enabled", False)
        jitter_delta = temp_jitter_config.get("delta", 0.2)
        
        # Get adversarial config
        adversarial_config = config.get("adversarial_config", {})
        adversarial_enabled = adversarial_config.get("enabled", False)
        adversarial_count = adversarial_config.get("count", 1)
        adversarial_prompt = adversarial_config.get("prompt", "")
        
        lib_logger.debug(f"[HiveMind] Preparing {count} drones for base model '{base_model}'")
        if adversarial_enabled:
            lib_logger.debug(f"[HiveMind] Adversarial mode enabled: {adversarial_count} critical drones")
        
        for i in range(count):
            # Clone the request params
            # BUGFIX: Use deepcopy to avoid shared mutable state
            drone_params = copy.deepcopy(request_params)
            
            # Override model with base model (strip [swarm] suffix)
            drone_params["model"] = base_model
            
            # Phase 4: Determine if this drone should be adversarial
            # Last N drones become adversarial
            is_adversarial = False
            if adversarial_enabled and adversarial_prompt:
                adversarial_start_index = count - adversarial_count
                if i >= adversarial_start_index:
                    is_adversarial = True
                    
                    # Inject adversarial system prompt
                    if "messages" in drone_params:
                        # Insert adversarial system message at the beginning
                        adversarial_message = {
                            "role": "system",
                            "content": adversarial_prompt
                        }
                        drone_params["messages"].insert(0, adversarial_message)
                    
                    lib_logger.debug(
                        f"[HiveMind] Drone {i+1}/{count}: ADVERSARIAL - injected critical analysis prompt"
                    )
            
            # Phase 4: Apply temperature jitter if enabled
            if jitter_enabled:
                base_temp = drone_params.get("temperature", 1.0)

                # Apply random jitter
                jitter = random.uniform(-jitter_delta, jitter_delta)
                new_temp = base_temp + jitter
                
                # Clamp to valid range [0.0, 2.0]
                new_temp = max(0.0, min(2.0, new_temp))
                
                drone_params["temperature"] = new_temp
                
                lib_logger.debug(
                    f"[HiveMind] Drone {i+1}/{count}: Applied temperature jitter "
                    f"({base_temp:.2f} → {new_temp:.2f}, delta: {jitter:+.2f})"
                )
            
            # Store drone metadata for logging
            drone_params["_drone_index"] = i + 1
            drone_params["_total_drones"] = count
            drone_params["_is_adversarial"] = is_adversarial
            
            drones.append(drone_params)
            
            temp_display = drone_params.get("temperature", "default")
            if isinstance(temp_display, float):
                temp_display = f"{temp_display:.2f}"
            
            lib_logger.debug(
                f"[HiveMind] Drone {i+1}/{count}: model={base_model}, temp={temp_display}"
            )
        
        return drones
    
    def _prepare_fusion_models(
        self,
        config: Dict[str, Any],
        request_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare specialist model configurations for fusion execution.
        
        Each specialist model gets a role-specific system prompt and 
        processes the same user query.
        
        Args:
            config: Fusion configuration
            request_params: Original request parameters
        
        Returns:
            List of specialist model configurations with metadata
        """
        specialists = config.get("specialists", [])
        models = []
        
        lib_logger.debug(f"[HiveMind] Preparing {len(specialists)} specialist models for fusion")
        
        for i, specialist in enumerate(specialists):
            specialist_num = i + 1
            specialist_model = specialist.get("model")
            specialist_role = specialist.get("role", f"Specialist {specialist_num}")
            specialist_prompt = specialist.get("system_prompt", "")
            specialist_weight = specialist.get("weight", 1.0)
            # MISSING FEATURE FIX: Extract weight description for arbiter context
            specialist_weight_desc = specialist.get("weight_description", "")
            
            if not specialist_model:
                lib_logger.warning(
                    f"[HiveMind] Specialist {specialist_num} missing model, skipping"
                )
                continue
            
            # Clone request params
            # BUGFIX: Use deepcopy
            model_params = copy.deepcopy(request_params)
            
            # Set specialist model
            model_params["model"] = specialist_model
            
            # Inject role-specific system prompt if provided
            if specialist_prompt and "messages" in model_params:
                role_message = {
                    "role": "system",
                    "content": specialist_prompt
                }
                model_params["messages"].insert(0, role_message)
            
            # Store specialist metadata
            model_params["_specialist_index"] = specialist_num
            model_params["_specialist_role"] = specialist_role
            model_params["_specialist_weight"] = specialist_weight
            model_params["_specialist_weight_description"] = specialist_weight_desc
            model_params["_total_specialists"] = len(specialists)
            
            models.append(model_params)
            
            lib_logger.debug(
                f"[HiveMind] Specialist {specialist_num}/{len(specialists)}: "
                f"role={specialist_role}, model={specialist_model}, weight={specialist_weight}"
            )
        
        return models
    
    async def _execute_parallel(
        self,
        drones: List[Dict[str, Any]],
        request: Any
    ) -> tuple:
        """
        Execute all drone requests in parallel.
        
        Uses asyncio.gather to execute all drones concurrently.
        Aggregates usage statistics from all successful responses.
        
        Args:
            drones: List of drone configurations
            request: Original request object
        
        Returns:
            Tuple of (successful_responses, aggregated_usage)
        """
        lib_logger.info(f"[HiveMind] Executing {len(drones)} drones in parallel...")
        
        # Create tasks for all drones
        tasks = []
        for i, drone_params in enumerate(drones):
            # Call acompletion directly (will use RotatingClient's retry logic)
            # Remove metadata fields before calling
            clean_params = {k: v for k, v in drone_params.items() if not k.startswith('_')}
            
            task = self.rotating_client._execute_with_retry(
                litellm.acompletion,  # Use litellm.acompletion directly
                request=request,
                **clean_params
            )
            tasks.append(task)
        
        # Execute all drones in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_responses = []
        failed_count = 0
        aggregated_usage = {}
        
        for i, result in enumerate(results):
            drone_index = i + 1
            
            if isinstance(result, Exception):
                # Drone failed
                failed_count += 1
                lib_logger.error(
                    f"[HiveMind] Drone {drone_index}/{len(drones)} failed: {result}"
                )
                continue
            
            # Drone succeeded
            successful_responses.append(result)
            
            # Aggregate usage - dynamically sum ALL numeric usage fields
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                
                # Iterate through all attributes of the usage object
                for attr_name in dir(usage):
                    # Skip private/magic attributes
                    if attr_name.startswith('_'):
                        continue
                    
                    try:
                        attr_value = getattr(usage, attr_name)
                        
                        # Only aggregate numeric fields (int or float)
                        if isinstance(attr_value, (int, float)) and not isinstance(attr_value, bool):
                            if attr_name not in aggregated_usage:
                                aggregated_usage[attr_name] = 0
                            aggregated_usage[attr_name] += attr_value
                    except (AttributeError, TypeError):
                        # Skip non-accessible or non-numeric attributes
                        continue
            
            lib_logger.debug(
                f"[HiveMind] Drone {drone_index}/{len(drones)} completed successfully"
            )
        
        # Check if we have at least one successful response
        if not successful_responses:
            raise RuntimeError(
                f"[HiveMind] All {len(drones)} drones failed. Cannot proceed with arbitration."
            )
        
        if failed_count > 0:
            lib_logger.warning(
                f"[HiveMind] {failed_count}/{len(drones)} drones failed. "
                f"Proceeding with {len(successful_responses)} successful responses."
            )
        
        lib_logger.info(
            f"[HiveMind] Parallel execution complete: {len(successful_responses)}/{len(drones)} succeeded. "
            f"Total tokens: {aggregated_usage['total_tokens']}"
        )
        
        return successful_responses, aggregated_usage
    
    def _format_for_arbiter(
        self,
        responses: List[Any],
        config: Dict[str, Any],
        specialist_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """
        Format drone/specialist responses for arbiter consumption.
        
        Creates a structured text format with numbered responses.
        Phase 4: Implements Blind Switch to strip model names.
        Phase 5: Adds role labels for fusion specialists.
        MISSING FEATURE FIX: Extracts specialist metadata for arbiter context.
        
        Args:
            responses: List of successful drone/specialist responses
            config: Swarm or fusion configuration
            specialist_metadata: Optional list of specialist metadata (for fusion mode)
        
        Returns:
            Tuple of (formatted_text, metadata_for_arbiter)
            metadata_for_arbiter is None for swarm mode, list of dicts for fusion mode
        """
        lib_logger.debug(f"[HiveMind] Formatting {len(responses)} responses for arbiter")
        
        # Check if blind mode is enabled
        arbiter_config = config.get("arbiter", {})
        blind_mode = arbiter_config.get("blind", True)  # Default ON
        
        formatted_parts = []
        arbiter_metadata = []  # MISSING FEATURE FIX: Collect metadata for arbiter
        
        for i, response in enumerate(responses):
            response_num = i + 1
            
            # Extract content from response
            content = ""
            if hasattr(response, 'choices') and response.choices:
                # Standard OpenAI-style response
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                elif hasattr(choice, 'text'):
                    content = choice.text
            
            if not content:
                lib_logger.warning(
                    f"[HiveMind] Response {response_num} has no content, skipping"
                )
                continue
            
            # Phase 5: Determine label (with fusion role support)
            label = f"Response {response_num}"
            
            # Check if this is fusion mode with specialist metadata
            if specialist_metadata and i < len(specialist_metadata):
                specialist = specialist_metadata[i]
                role = specialist.get("_specialist_role", "Unknown")
                model_name = specialist.get("model", "unknown")
                weight_desc = specialist.get("_specialist_weight_description", "")
                
                # MISSING FEATURE FIX: Build metadata for arbiter context
                arbiter_metadata.append({
                    "role": role,
                    "model": model_name,
                    "weight_description": weight_desc
                })
                
                if blind_mode:
                    # Blind mode: show role but not model
                    label = f"{role}"
                else:
                    # Non-blind: show role and model
                    label = f"{role} ({model_name})"
                    
                lib_logger.debug(
                    f"[HiveMind] Fusion specialist {response_num}: role={role}, blind={blind_mode}"
                )
            else:
                # Swarm mode fallback
                if blind_mode:
                    label = f"Response {response_num}"
                else:
                    model_name = "unknown"
                    if hasattr(response, 'model'):
                        model_name = response.model
                    label = f"Response {response_num} (Model: {model_name})"
            
            # Format: "Label:\n<content>\n"
            formatted_parts.append(f"{label}:\n{content}\n")
        
        # Join all responses
        formatted_text = "\n".join(formatted_parts)
        
        lib_logger.debug(
            f"[HiveMind] Formatted {len(formatted_parts)} responses "
            f"({len(formatted_text)} characters total, blind_mode={blind_mode})"
        )
        
        # Return metadata only if fusion mode
        metadata_for_arbiter = arbiter_metadata if arbiter_metadata else None
        
        return formatted_text, metadata_for_arbiter
    
    def _build_arbiter_prompt(
        self,
        formatted_responses: str,
        config: Dict[str, Any],
        original_messages: List[Dict[str, str]],
        specialist_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build the complete prompt for the arbiter model.
        
        Loads the strategy template and constructs the message array.
        Phase 6: Adds recursive mode instructions for autonomous decision-making.
        MISSING FEATURE FIX: Adds specialist expertise context with weights for fusion mode.
        
        Args:
            formatted_responses: Formatted drone/specialist responses
            config: Swarm or fusion configuration
            original_messages: Original user messages
            specialist_metadata: Optional metadata about specialists (for fusion mode)
        
        Returns:
            Complete messages array for arbiter
        """
        lib_logger.debug("[HiveMind] Building arbiter prompt")
        
        # Get strategy template
        arbiter_config = config.get("arbiter", {})
        strategy_name = arbiter_config.get("strategy", "synthesis")
        
        strategy_template = self.config_loader.get_strategy(strategy_name)
        
        if not strategy_template:
            lib_logger.warning(
                f"[HiveMind] Strategy '{strategy_name}' not found, using default"
            )
            strategy_template = "Synthesize the following responses into a single, high-quality answer:\n{responses}"
        
        # Replace {responses} placeholder
        strategy_prompt = strategy_template.replace("{responses}", formatted_responses)
        
        # MISSING FEATURE FIX: Add specialist expertise context for fusion mode
        if specialist_metadata:
            expertise_lines = ["\n\nSPECIALIST EXPERTISE:"]
            expertise_lines.append("You are synthesizing responses from specialists with the following expertise:\n")
            
            for spec in specialist_metadata:
                role = spec.get('role', 'Unknown')
                model = spec.get('model', 'Unknown')
                weight_desc = spec.get('weight_description', '')
                
                if weight_desc:
                    expertise_lines.append(f"- {role} ({model}): {weight_desc}")
                else:
                    expertise_lines.append(f"- {role} ({model}): Subject matter expert")
            
            expertise_lines.append("\nConsider each specialist's domain expertise when synthesizing your response.")
            strategy_prompt += "\n".join(expertise_lines)
            
            lib_logger.debug(f"[HiveMind] Added specialist expertise context for {len(specialist_metadata)} specialists")
        
        # Phase 6: Add recursive mode instructions if enabled
        recursive_config = config.get("recursive_mode", {})
        if recursive_config.get("enabled", False):
            consensus_threshold = recursive_config.get("consensus_threshold", 7)
            
            recursive_instructions = f"""

AUTONOMOUS DECISION PROTOCOL:
You have autonomous decision-making authority. Follow this protocol:

1. ASSESSMENT PHASE:
   - Analyze the provided responses
   - Rate consensus level (1-10 scale)
   - Output: [CONSENSUS: X/10]

2. DECISION PHASE:
   If consensus >= {consensus_threshold}/10:
     - Proceed directly to synthesis
   
   If consensus < {consensus_threshold}/10:
     - Identify specific conflict points
     - Output: [CONFLICTS: <brief list>]
     - For each response, reason internally about how it addresses the conflicts
     - Output: [CRITIQUE: <your internal reasoning>]

3. SYNTHESIS PHASE:
   - Create final answer incorporating all insights
   - Output: [FINAL SYNTHESIS:]
   - Provide your complete response after this marker

IMPORTANT: Wrap all internal reasoning (CONSENSUS, CONFLICTS, CRITIQUE) in [INTERNAL] tags.
Only the content after [FINAL SYNTHESIS:] will be shown to the user.

Example format:
[INTERNAL]
[CONSENSUS: 5/10]
[CONFLICTS: Response 1 suggests X, Response 2 suggests Y]
[CRITIQUE: Analyzing the conflict...]
[/INTERNAL]
[FINAL SYNTHESIS:]
<your complete answer to the user>
"""
            strategy_prompt += recursive_instructions
            lib_logger.info(
                f"[HiveMind] Recursive mode enabled (consensus threshold: {consensus_threshold}/10)"
            )
        
        # Build messages array
        messages = [
            {
                "role": "system",
                "content": strategy_prompt
            }
        ]
        
        # Add original user query
        if original_messages:
            # Find the last user message
            for msg in reversed(original_messages):
                if msg.get("role") == "user":
                    messages.append({
                        "role": "user",
                        "content": msg.get("content", "")
                    })
                    break
        
        lib_logger.debug(f"[HiveMind] Arbiter prompt built: {len(messages)} messages")
        
        return messages
    
    async def _call_arbiter(
        self,
        messages: List[Dict[str, str]],
        config: Dict[str, Any],
        request: Any
    ) -> tuple:
        """
        Call the arbiter model to synthesize responses.
        
        Non-streaming version for Phase 2.
        Streaming support will be added in Phase 3.
        
        Args:
            messages: Constructed arbiter messages
            config: Swarm or fusion configuration
            request: Original request object
        
        Returns:
            Tuple of (arbiter_response, arbiter_usage)
        """
        # Get arbiter model
        arbiter_config = config.get("arbiter", {})
        arbiter_model = arbiter_config.get("model", "self")
        
        # If "self", we need to determine which model to use
        # For swarm, this will be handled by caller
        # For now, just use as-is
        
        lib_logger.info(f"[HiveMind] Calling arbiter model: {arbiter_model}")
        
        # Build params for arbiter call
        arbiter_params = {
            "model": arbiter_model,
            "messages": messages,
            "stream": False  # Non-streaming for Phase 2
        }
        
        # Call arbiter through RotatingClient
        # Use _execute_with_retry for consistency
        arbiter_response = await self.rotating_client._execute_with_retry(
            litellm.acompletion,
            request=request,
            **arbiter_params
        )
        
        # Extract usage - dynamically capture ALL numeric usage fields
        arbiter_usage = {}
        
        if hasattr(arbiter_response, 'usage') and arbiter_response.usage:
            usage = arbiter_response.usage
            
            # Iterate through all attributes of the usage object
            for attr_name in dir(usage):
                # Skip private/magic attributes
                if attr_name.startswith('_'):
                    continue
                
                try:
                    attr_value = getattr(usage, attr_name)
                    
                    # Only capture numeric fields (int or float)
                    if isinstance(attr_value, (int, float)) and not isinstance(attr_value, bool):
                        arbiter_usage[attr_name] = attr_value
                except (AttributeError, TypeError):
                    # Skip non-accessible or non-numeric attributes
                    continue
        
        lib_logger.info(
            f"[HiveMind] Arbiter completed. Tokens: {arbiter_usage['total_tokens']}"
        )
        
        return arbiter_response, arbiter_usage
    
    async def _call_arbiter_streaming(
        self,
        messages: List[Dict[str, str]],
        config: Dict[str, Any],
        request: Any
    ):
        """
        Call the arbiter model with streaming enabled.
        
        Yields arbiter response chunks while tracking usage.
        Phase 6: Filters [INTERNAL] markers for recursive mode.
        
        Args:
            messages: Constructed arbiter messages
            config: Swarm or fusion configuration
            request: Original request object
        
        Yields:
            Response chunks from arbiter
            Final yield includes usage metadata
        """
        # Get arbiter model
        arbiter_config = config.get("arbiter", {})
        arbiter_model = arbiter_config.get("model", "self")
        
        lib_logger.info(f"[HiveMind] Calling arbiter model (streaming): {arbiter_model}")
        
        # Build params for arbiter call
        arbiter_params = {
            "model": arbiter_model,
            "messages": messages,
            "stream": True  # Enable streaming
        }
        # Call arbiter through RotatingClient's streaming method
        stream_generator = self.rotating_client._streaming_acompletion_with_retry(
            request=request,
            **arbiter_params
        )
        
        # Track usage from stream
        arbiter_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Phase 6: Track recursive mode state
        recursive_enabled = config.get("recursive_mode", {}).get("enabled", False)
        in_internal_block = False
        internal_buffer = []
        
        # Stream chunks and collect usage
        async for chunk in stream_generator:
            # Check if this chunk has usage info (typically the last chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage
                arbiter_usage['prompt_tokens'] = getattr(usage, 'prompt_tokens', 0)
                arbiter_usage['completion_tokens'] = getattr(usage, 'completion_tokens', 0)
                arbiter_usage['total_tokens'] = getattr(usage, 'total_tokens', 0)
                
                # Include other fields
                for field in ['cached_tokens', 'reasoning_tokens']:
                    if hasattr(usage, field):
                        arbiter_usage[field] = getattr(usage, field, 0)
            
            # BUGFIX: Robust handling of [INTERNAL] markers to prevent data loss
            if recursive_enabled and hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta if hasattr(chunk.choices[0], 'delta') else None
                if delta and hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    
                    # Handle [INTERNAL] start
                    if '[INTERNAL]' in content:
                        parts = content.split('[INTERNAL]')
                        before_internal = parts[0]
                        
                        # Yield content before marker
                        if before_internal:
                            chunk.choices[0].delta.content = before_internal
                            yield chunk
                        
                        in_internal_block = True
                        
                        # Handle content after marker (start of internal)
                        if len(parts) > 1:
                            remaining = parts[1]
                            # Check if it also ends in this chunk
                            if '[/INTERNAL]' in remaining:
                                internal_parts = remaining.split('[/INTERNAL]')
                                internal_buffer.append(internal_parts[0])
                                
                                # Process buffer
                                full_internal = ''.join(internal_buffer)
                                self._log_recursive_markers(full_internal, config)
                                internal_buffer = []
                                in_internal_block = False
                                
                                # Yield content after [/INTERNAL]
                                after_internal = internal_parts[1]
                                if after_internal:
                                    chunk.choices[0].delta.content = after_internal
                                    yield chunk
                            else:
                                internal_buffer.append(remaining)
                        
                        continue # Done with this chunk
                    
                    # Handle [/INTERNAL] end (if we are in block)
                    if in_internal_block and '[/INTERNAL]' in content:
                        parts = content.split('[/INTERNAL]')
                        internal_buffer.append(parts[0])
                        
                        # Process buffer
                        full_internal = ''.join(internal_buffer)
                        self._log_recursive_markers(full_internal, config)
                        internal_buffer = []
                        in_internal_block = False
                        
                        # Yield content after marker
                        after_internal = parts[1]
                        if after_internal:
                            chunk.choices[0].delta.content = after_internal
                            yield chunk
                        continue
                    
                    # If inside internal block, buffer it
                    if in_internal_block:
                        internal_buffer.append(content)
                        continue
            
            # Yield the chunk to caller (normal flow or filtered)
            yield chunk
        
        lib_logger.info(
            f"[HiveMind] Arbiter streaming completed. Tokens: {arbiter_usage['total_tokens']}"
        )
        
        # Return usage as final metadata
        # Caller will handle usage aggregation
        yield {"_hivemind_usage": arbiter_usage}
    
    def _log_recursive_markers(self, internal_content: str, config: Dict[str, Any]):
        """
        Parse and log recursive mode markers from internal reasoning.
        
        Phase 6: Extracts consensus scores, conflicts, and critique reasoning.
        
        Args:
            internal_content: Content between [INTERNAL] tags
            config: Configuration with recursive threshold
        """
        
        # Extract consensus score
        consensus_match = re.search(r'\[CONSENSUS:\s*(\d+)/10\]', internal_content)
        if consensus_match:
            consensus_score = int(consensus_match.group(1))
            threshold = config.get("recursive_mode", {}).get("consensus_threshold", 7)
            
            if consensus_score < threshold:
                lib_logger.warning(
                    f"[HiveMind] Recursive mode: Consensus {consensus_score}/10 "
                    f"(below threshold {threshold}/10) - arbiter performing critique"
                )
            else:
                lib_logger.info(
                    f"[HiveMind] Recursive mode: Consensus {consensus_score}/10 "
                    f"(>= threshold {threshold}/10) - proceeding to synthesis"
                )
        
        # Extract conflicts if present
        conflicts_match = re.search(r'\[CONFLICTS:\s*([^\]]+)\]', internal_content)
        if conflicts_match:
            conflicts = conflicts_match.group(1).strip()
            lib_logger.info(f"[HiveMind] Conflicts identified: {conflicts}")
        
        # Log that critique is happening
        if '[CRITIQUE:' in internal_content:
            lib_logger.debug("[HiveMind] Arbiter performing internal critique reasoning")

    
    async def _handle_swarm_streaming(
        self,
        config: Dict[str, Any],
        base_model: str,
        request: Any,
        **kwargs
    ):
        """
        Handle streaming swarm request.
        
        Executes drones in parallel, then streams arbiter response.
        Aggregates usage and injects into stream.
        
        Args:
            config: Swarm configuration
            base_model: Base model name
            request: Original request object
            **kwargs: Request parameters
        
        Yields:
            Arbiter response chunks with aggregated usage
        """
        # Steps 1-4: Same as non-streaming (collect drone responses)
        drones = self._prepare_drones(config, base_model, kwargs)
        drone_responses, drone_usage = await self._execute_parallel(drones, request)
        formatted_responses = self._format_for_arbiter(drone_responses, config)
        
        original_messages = kwargs.get("messages", [])
        arbiter_messages = self._build_arbiter_prompt(
            formatted_responses,
            config,
            original_messages
        )
        
        # Handle "self" arbiter model
        arbiter_config = config.get("arbiter", {})
        arbiter_model = arbiter_config.get("model", "self")
        if arbiter_model == "self":
            arbiter_model = base_model
            lib_logger.debug(f"[HiveMind] Using self-arbiter: {arbiter_model}")
        
        # BUGFIX: Use deepcopy for config
        config_copy = copy.deepcopy(config)
        config_copy["arbiter"] = arbiter_config.copy()
        config_copy["arbiter"]["model"] = arbiter_model
        
        # Call arbiter in streaming mode
        arbiter_usage = {}
        async for chunk in self._call_arbiter_streaming(arbiter_messages, config_copy, request):
            # Check for usage metadata
            if isinstance(chunk, dict) and "_hivemind_usage" in chunk:
                arbiter_usage = chunk["_hivemind_usage"]
                continue  # Don't yield metadata chunk
            
            # For SSE chunks, check if this is the final chunk with usage
            # and update with aggregated usage
            if hasattr(chunk, 'usage') and chunk.usage:
                # This is the final chunk - aggregate total usage
                total_usage = {
                    'prompt_tokens': drone_usage['prompt_tokens'] + arbiter_usage.get('prompt_tokens', 0),
                    'completion_tokens': drone_usage['completion_tokens'] + arbiter_usage.get('completion_tokens', 0),
                    'total_tokens': drone_usage['total_tokens'] + arbiter_usage.get('total_tokens', 0)
                }
                
                for field in ['cached_tokens', 'reasoning_tokens']:
                    if field in drone_usage or field in arbiter_usage:
                        total_usage[field] = drone_usage.get(field, 0) + arbiter_usage.get(field, 0)
                
                # Update chunk usage with aggregated values
                chunk.usage.prompt_tokens = total_usage['prompt_tokens']
                chunk.usage.completion_tokens = total_usage['completion_tokens']
                chunk.usage.total_tokens = total_usage['total_tokens']
                
                for field in ['cached_tokens', 'reasoning_tokens']:
                    if field in total_usage:
                        setattr(chunk.usage, field, total_usage[field])
                
                lib_logger.info(
                    f"[HiveMind] Streaming swarm completed. "
                    f"Total usage: {total_usage['total_tokens']} tokens "
                    f"(Drones: {drone_usage['total_tokens']}, Arbiter: {arbiter_usage.get('total_tokens', 0)})"
                )
            
            yield chunk
    
    async def _handle_fusion_streaming(
        self,
        config: Dict[str, Any],
        request: Any,
        **kwargs
    ):
        """
        Handle streaming fusion request.
        
        Executes specialists in parallel, then streams arbiter response.
        Aggregates usage and injects into stream.
        
        Args:
            config: Fusion configuration
            request: Original request object
            **kwargs: Request parameters
        
        Yields:
            Arbiter response chunks with aggregated usage
        """
        # Prepare specialist models
        specialist_models = self._prepare_fusion_models(config, kwargs)
        
        if not specialist_models:
            raise ValueError("[HiveMind] No valid specialists found for fusion")
        
        # Execute specialists in parallel
        specialist_responses, specialist_usage = await self._execute_parallel(
            specialist_models, request
        )
        
        # Format responses with role labels and extract metadata
        formatted_responses, specialist_metadata_for_arbiter = self._format_for_arbiter(
            specialist_responses,
            config,
            specialist_metadata=specialist_models
        )
        
        # Build arbiter prompt with specialist expertise context
        original_messages = kwargs.get("messages", [])
        arbiter_messages = self._build_arbiter_prompt(
            formatted_responses,
            config,
            original_messages,
            specialist_metadata=specialist_metadata_for_arbiter  # MISSING FEATURE FIX: Pass metadata
        )
        
        # Get arbiter model
        arbiter_config = config.get("arbiter", {})
        arbiter_model = arbiter_config.get("model", "gpt-4o")
        
        lib_logger.debug(f"[HiveMind] Using arbiter model: {arbiter_model}")
        
        # Update config
        # BUGFIX: Use deepcopy
        config_copy = copy.deepcopy(config)
        config_copy["arbiter"] = arbiter_config.copy()
        config_copy["arbiter"]["model"] = arbiter_model
        
        # Stream arbiter
        arbiter_usage = {}
        async for chunk in self._call_arbiter_streaming(arbiter_messages, config_copy, request):
            if isinstance(chunk, dict) and "_hivemind_usage" in chunk:
                arbiter_usage = chunk["_hivemind_usage"]
                continue
            
            if hasattr(chunk, 'usage') and chunk.usage:
                # Final chunk - aggregate usage
                total_usage = {
                    'prompt_tokens': specialist_usage['prompt_tokens'] + arbiter_usage.get('prompt_tokens', 0),
                    'completion_tokens': specialist_usage['completion_tokens'] + arbiter_usage.get('completion_tokens', 0),
                    'total_tokens': specialist_usage['total_tokens'] + arbiter_usage.get('total_tokens', 0)
                }
                
                for field in ['cached_tokens', 'reasoning_tokens']:
                    if field in specialist_usage or field in arbiter_usage:
                        total_usage[field] = specialist_usage.get(field, 0) + arbiter_usage.get(field, 0)
                
                chunk.usage.prompt_tokens = total_usage['prompt_tokens']
                chunk.usage.completion_tokens = total_usage['completion_tokens']
                chunk.usage.total_tokens = total_usage['total_tokens']
                
                for field in ['cached_tokens', 'reasoning_tokens']:
                    if field in total_usage:
                        setattr(chunk.usage, field, total_usage[field])
                
                lib_logger.info(
                    f"[HiveMind] Fusion streaming completed. "
                    f"Total usage: {total_usage['total_tokens']} tokens "
                    f"(Specialists: {specialist_usage['total_tokens']}, Arbiter: {arbiter_usage.get('total_tokens', 0)})"
                )
            
            yield chunk
    
    async def handle_request(self, request, **kwargs):
        """
        Handle an ensemble request (swarm or fusion).
        
        This is the main entry point for ensemble execution.
        
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
            config = self.config_loader.get_fusion_config(resolved_id)
            specialists = config.get("specialists", [])
            is_streaming = kwargs.get("stream", False)
            
            # Phase 6: Track execution start time
            start_time = time.time()
            
            lib_logger.info(
                f"[HiveMind] Processing Fusion request: {resolved_id} "
                f"({len(specialists)} specialists, streaming: {is_streaming})"
            )
            
            # Route based on streaming mode
            if is_streaming:
                # Streaming fusion
                return self._handle_fusion_streaming(
                    config=config,
                    request=request,
                    **kwargs
                )
            
            # Non-streaming fusion
            specialist_models = self._prepare_fusion_models(config, kwargs)
            
            if not specialist_models:
                raise ValueError(f"[HiveMind] No valid specialists found for fusion '{resolved_id}'")
            
            specialist_responses, specialist_usage = await self._execute_parallel(
                specialist_models, request
            )
            
            # Format responses and extract metadata for arbiter
            formatted_responses, specialist_metadata_for_arbiter = self._format_for_arbiter(
                specialist_responses,
                config,
                specialist_metadata=specialist_models
            )
            
            original_messages = kwargs.get("messages", [])
            arbiter_messages = self._build_arbiter_prompt(
                formatted_responses,
                config,
                original_messages,
                specialist_metadata=specialist_metadata_for_arbiter  # MISSING FEATURE FIX: Pass metadata
            )
            
            arbiter_config = config.get("arbiter", {})
            arbiter_model = arbiter_config.get("model", "gpt-4o")
            
            # BUGFIX: Use deepcopy
            config_copy = copy.deepcopy(config)
            config_copy["arbiter"] = arbiter_config.copy()
            config_copy["arbiter"]["model"] = arbiter_model
            
            arbiter_response, arbiter_usage = await self._call_arbiter(
                arbiter_messages,
                config_copy,
                request
            )
            
            # Aggregate usage - dynamically sum ALL numeric fields from both sources
            total_usage = {}
            
            # Helper function to merge usage dictionaries
            for usage_dict in [specialist_usage, arbiter_usage]:
                for field, value in usage_dict.items():
                    if field not in total_usage:
                        total_usage[field] = 0
                    total_usage[field] += value
            
            # Phase 6: Calculate latency and cost
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Try to calculate cost using litellm
            total_cost = 0.0
            try:
                total_cost = litellm.completion_cost(completion_response=arbiter_response)
            except Exception as e:
                lib_logger.debug(f"[HiveMind] Could not calculate cost: {e}")
            
            # Add hivemind_details to usage
            hivemind_details = {
                "mode": "fusion",
                "specialist_count": len(specialists),
                "specialist_tokens": specialist_usage['total_tokens'],
                "arbiter_tokens": arbiter_usage['total_tokens'],
                "total_cost_usd": round(total_cost, 6),
                "latency_ms": round(latency_ms, 2)
            }
            
            
            if hasattr(arbiter_response, 'usage'):
                # IMPORTANT: Standard usage fields contain the TOTAL aggregated usage
                # (specialists + arbiter). This ensures consumers can parse usage normally.
                
                # Dynamically set ALL usage fields from total_usage
                for field, value in total_usage.items():
                    try:
                        setattr(arbiter_response.usage, field, value)
                    except (AttributeError, TypeError):
                        # Skip if field cannot be set
                        lib_logger.debug(f"[HiveMind] Could not set usage field '{field}'")
                
                # Add hivemind_details as SUPPLEMENTARY breakdown information
                # This does NOT replace standard fields, but provides additional context
                arbiter_response.usage.hivemind_details = hivemind_details
            
            lib_logger.info(
                f"[HiveMind] Fusion completed successfully. "
                f"Total usage: {total_usage['total_tokens']} tokens "
                f"(Specialists: {specialist_usage['total_tokens']}, Arbiter: {arbiter_usage['total_tokens']}). "
                f"Latency: {latency_ms:.2f}ms, Cost: ${total_cost:.6f}"
            )
            
            return arbiter_response
        
        elif self._is_swarm_request(resolved_id):
            base_model = self.get_base_model(resolved_id)
            config = self.config_loader.get_swarm_config(base_model)
            count = config.get("count", 3)
            is_streaming = kwargs.get("stream", False)
            
            # Phase 6: Track execution start time
            start_time = time.time()
            
            lib_logger.info(
                f"[HiveMind] Processing Swarm request: {resolved_id} "
                f"(base: {base_model}, {count} drones, streaming: {is_streaming})"
            )
            
            # Phase 3B: Route based on streaming mode
            if is_streaming:
                # Streaming mode - return async generator
                return self._handle_swarm_streaming(
                    config=config,
                    base_model=base_model,
                    request=request,
                    **kwargs
                )
            else:
                # Non-streaming mode - return complete response
                # Step 1: Prepare drones
                drones = self._prepare_drones(config, base_model, kwargs)
                
                # Step 2: Execute drones in parallel
                drone_responses, drone_usage = await self._execute_parallel(drones, request)
                
                # Step 3: Format responses for arbiter
                formatted_responses = self._format_for_arbiter(drone_responses, config)
                
                # Step 4: Build arbiter prompt
                original_messages = kwargs.get("messages", [])
                arbiter_messages = self._build_arbiter_prompt(
                    formatted_responses,
                    config,
                    original_messages
                )
                
                # Step 5: Handle "self" arbiter model
                arbiter_config = config.get("arbiter", {})
                arbiter_model = arbiter_config.get("model", "self")
                if arbiter_model == "self":
                    arbiter_model = base_model
                    lib_logger.debug(f"[HiveMind] Using self-arbiter: {arbiter_model}")
                
                # Update config with resolved arbiter model
                # BUGFIX: Use deepcopy
                config_copy = copy.deepcopy(config)
                config_copy["arbiter"] = arbiter_config.copy()
                config_copy["arbiter"]["model"] = arbiter_model
                
                # Step 6: Call arbiter
                arbiter_response, arbiter_usage = await self._call_arbiter(
                    arbiter_messages,
                    config_copy,
                    request
                )
                
                # Step 7: Aggregate total usage - dynamically sum ALL numeric fields from both sources
                total_usage = {}
                
                # Helper function to merge usage dictionaries
                for usage_dict in [drone_usage, arbiter_usage]:
                    for field, value in usage_dict.items():
                        if field not in total_usage:
                            total_usage[field] = 0
                        total_usage[field] += value
                
                # Phase 6: Calculate latency and cost
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Try to calculate cost using litellm
                total_cost = 0.0
                try:
                    total_cost = litellm.completion_cost(completion_response=arbiter_response)
                except Exception as e:
                    lib_logger.debug(f"[HiveMind] Could not calculate cost: {e}")
                
                # Add hivemind_details to usage
                hivemind_details = {
                    "mode": "swarm",
                    "drone_count": count,
                    "drone_tokens": drone_usage['total_tokens'],
                    "arbiter_tokens": arbiter_usage['total_tokens'],
                    "total_cost_usd": round(total_cost, 6),
                    "latency_ms": round(latency_ms, 2)
                }
                
                # Step 8: Update arbiter response with aggregated usage
                if hasattr(arbiter_response, 'usage'):
                    # IMPORTANT: Standard usage fields contain the TOTAL aggregated usage
                    # (drones + arbiter). This ensures consumers can parse usage normally.
                    
                    # Dynamically set ALL usage fields from total_usage
                    for field, value in total_usage.items():
                        try:
                            setattr(arbiter_response.usage, field, value)
                        except (AttributeError, TypeError):
                            # Skip if field cannot be set
                            lib_logger.debug(f"[HiveMind] Could not set usage field '{field}'")
                    
                    # Add hivemind_details as SUPPLEMENTARY breakdown information
                    # This does NOT replace standard fields, but provides additional context
                    arbiter_response.usage.hivemind_details = hivemind_details
                
                lib_logger.info(
                    f"[HiveMind] Swarm completed successfully. "
                    f"Total usage: {total_usage['total_tokens']} tokens "
                    f"(Drones: {drone_usage['total_tokens']}, Arbiter: {arbiter_usage['total_tokens']}). "
                    f"Latency: {latency_ms:.2f}ms, Cost: ${total_cost:.6f}"
                )
                
                return arbiter_response
        
        else:
            raise ValueError(f"Unknown ensemble type for model: {model_id}")
