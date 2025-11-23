# src/rotator_library/providers/antigravity_provider.py

import json
import httpx
import logging
import time
import asyncio
import random
import uuid
import copy
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple
from .provider_interface import ProviderInterface
from .antigravity_auth_base import AntigravityAuthBase
from ..model_definitions import ModelDefinitions
import litellm
from litellm.exceptions import RateLimitError

lib_logger = logging.getLogger('rotator_library')

# Antigravity base URLs with fallback order
# Priority: daily (sandbox) → autopush (sandbox) → production
BASE_URLS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://cloudcode-pa.googleapis.com/v1internal"  # Production fallback
]

# Hardcoded models available via Antigravity
HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-thinking"
]

# Logging configuration
LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
ANTIGRAVITY_LOGS_DIR = LOGS_DIR / "antigravity_logs"


class _AntigravityFileLogger:
    """A simple file logger for a single Antigravity transaction."""
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = str(uuid.uuid4())
        # Sanitize model name for directory
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        self.log_dir = ANTIGRAVITY_LOGS_DIR / f"{timestamp}_{safe_model_name}_{request_id}"
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create Antigravity log directory: {e}")
            self.enabled = False

    def log_request(self, payload: Dict[str, Any]):
        """Logs the request payload sent to Antigravity."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "request_payload.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"_AntigravityFileLogger: Failed to write request: {e}")

    def log_response_chunk(self, chunk: str):
        """Logs a raw chunk from the Antigravity response stream."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "response_stream.log", "a", encoding="utf-8") as f:
                f.write(chunk + "\n")
        except Exception as e:
            lib_logger.error(f"_AntigravityFileLogger: Failed to write response chunk: {e}")

    def log_error(self, error_message: str):
        """Logs an error message."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "error.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] {error_message}\n")
        except Exception as e:
            lib_logger.error(f"_AntigravityFileLogger: Failed to write error: {e}")

    def log_final_response(self, response_data: Dict[str, Any]):
        """Logs the final, reassembled response."""
        if not self.enabled: return
        try:
            with open(self.log_dir / "final_response.json", "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"_AntigravityFileLogger: Failed to write final response: {e}")

class ThoughtSignatureCache:
    """
    Server-side cache for thoughtSignatures to maintain Gemini 3 conversation context.
    
    Maps tool_call_id → thoughtSignature to preserve encrypted reasoning signatures
    across turns, even if clients don't support the thought_signature field.
    
    Features:
    - TTL-based expiration to prevent memory growth
    - Thread-safe for concurrent access
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the signature cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self._cache: Dict[str, Tuple[str, float]] = {}  # {call_id: (signature, timestamp)}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def store(self, tool_call_id: str, signature: str):
        """
        Store a signature for a tool call ID.
        
        Args:
            tool_call_id: Unique identifier for the tool call
            signature: Encrypted thoughtSignature from Antigravity API
        """
        with self._lock:
            self._cache[tool_call_id] = (signature, time.time())
            self._cleanup_expired()
    
    def retrieve(self, tool_call_id: str) -> Optional[str]:
        """
        Retrieve signature for a tool call ID.
        
        Args:
            tool_call_id: Unique identifier for the tool call
            
        Returns:
            The signature if found and not expired, None otherwise
        """
        with self._lock:
            if tool_call_id not in self._cache:
                return None
            
            signature, timestamp = self._cache[tool_call_id]
            if time.time() - timestamp > self._ttl:
                del self._cache[tool_call_id]
                return None
            
            return signature
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]
    
    def clear(self):
        """Clear all cached signatures."""
        with self._lock:
            self._cache.clear()

class AntigravityProvider(AntigravityAuthBase, ProviderInterface):
    """
    Antigravity provider implementation for Gemini models.
    
    Antigravity is an experimental internal Google API that provides access to Gemini models
    including Gemini 3 with thinking/reasoning capabilities. It wraps standard Gemini API
    requests with additional metadata and uses sandbox endpoints.
    
    Key features:
    - Model aliasing (gemini-3-pro-high ↔ gemini-3-pro-preview)
    - Gemini 3 thinkingLevel support
    - ThoughtSignature preservation for multi-turn conversations
    - Reasoning content separation (thought=true parts)
    - Sophisticated tool response grouping
    - Base URL fallback (sandbox → production)
    
    Gemini 3 Special Mechanics:
    1. ThinkingLevel: Uses thinkingLevel (low/high) instead of thinkingBudget for Gemini 3 models
    2. ThoughtSignature: Function calls include thoughtSignature="skip_thought_signature_validator"
       - This is a CONSTANT validation bypass flag, not a session key
       - Preserved across conversation turns to maintain reasoning continuity
       - Filtered from responses to prevent exposing encrypted internal data
    3. Reasoning Content: Text parts with thought=true flag are separated into reasoning_content
    4. Token Counting: thoughtsTokenCount is included in prompt_tokens and reported as reasoning_tokens
    """
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        self._current_base_url = BASE_URLS[0]  # Start with daily sandbox
        self._base_url_index = 0
        
        # Initialize thoughtSignature cache for Gemini 3 multi-turn conversations
        cache_ttl = int(os.getenv("ANTIGRAVITY_SIGNATURE_CACHE_TTL", "3600"))
        self._signature_cache = ThoughtSignatureCache(ttl_seconds=cache_ttl)
        
        # Check if client passthrough is enabled (default: TRUE for testing)
        self._preserve_signatures_in_client = os.getenv(
            "ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES", 
            "true"  # Default ON for testing
        ).lower() in ("true", "1", "yes")
        
        # Check if server-side cache is enabled (default: TRUE for testing)
        self._enable_signature_cache = os.getenv(
            "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE",
            "true"  # Default ON for testing
        ).lower() in ("true", "1", "yes")
        
        # Check if dynamic model discovery is enabled (default: OFF due to endpoint instability)
        self._enable_dynamic_model_discovery = os.getenv(
            "ANTIGRAVITY_ENABLE_DYNAMIC_MODELS",
            "false"  # Default OFF - use hardcoded list
        ).lower() in ("true", "1", "yes")
        
        if self._preserve_signatures_in_client:
            lib_logger.info("Antigravity: thoughtSignature client passthrough ENABLED")
        else:
            lib_logger.info("Antigravity: thoughtSignature client passthrough DISABLED")
        
        if self._enable_signature_cache:
            lib_logger.info(f"Antigravity: thoughtSignature server-side cache ENABLED (TTL: {cache_ttl}s)")
        else:
            lib_logger.info("Antigravity: thoughtSignature server-side cache DISABLED")
        
        if self._enable_dynamic_model_discovery:
            lib_logger.info("Antigravity: Dynamic model discovery ENABLED (may fail if endpoint unavailable)")
        else:
            lib_logger.info("Antigravity: Dynamic model discovery DISABLED (using hardcoded model list)")

    # ============================================================================
    # MODEL ALIAS SYSTEM
    # ============================================================================

    def _model_name_to_alias(self, model_name: str) -> str:
        """
        Convert internal Antigravity model names to public aliases.
        
        Args:
            model_name: Internal model name
            
        Returns:
            Public alias name, or empty string if model should be excluded
        """
        alias_map = {
            "rev19-uic3-1p": "gemini-2.5-computer-use-preview-10-2025",
            "gemini-3-pro-image": "gemini-3-pro-image-preview",
            "gemini-3-pro-high": "gemini-3-pro-preview",
            # Claude models: no aliasing needed (public name = internal name)
        }
        
        # Filter out excluded models (return empty string to skip)
        excluded = [
            "chat_20706", "chat_23310", "gemini-2.5-flash-thinking",
            "gemini-3-pro-low", "gemini-2.5-pro"
        ]
        if model_name in excluded:
            return ""
        
        return alias_map.get(model_name, model_name)

    def _alias_to_model_name(self, alias: str) -> str:
        """
        Convert public aliases to internal Antigravity model names.
        
        Args:
            alias: Public alias name
            
        Returns:
            Internal model name
        """
        reverse_map = {
            "gemini-2.5-computer-use-preview-10-2025": "rev19-uic3-1p",
            "gemini-3-pro-image-preview": "gemini-3-pro-image",
            "gemini-3-pro-preview": "gemini-3-pro-high",
            # Claude models: no aliasing needed (public name = internal name)
        }
        return reverse_map.get(alias, alias)

    def _is_gemini_3_model(self, model: str) -> bool:
        """
        Check if model is Gemini 3 (requires thoughtSignature preservation).
        
        Args:
            model: Model name (public alias)
            
        Returns:
            True if this is a Gemini 3 model
        """
        internal_model = self._alias_to_model_name(model)
        return internal_model.startswith("gemini-3-") or model.startswith("gemini-3-")

    @staticmethod
    def _normalize_type_arrays(schema: Any) -> Any:
        """
        Normalize type arrays in JSON Schema for Proto-based Antigravity API.
        Converts `"type": ["string", "null"]` → `"type": "string"`.
        """
        if isinstance(schema, dict):
            normalized = {}
            for key, value in schema.items():
                if key == "type" and isinstance(value, list):
                    # Take first non-null type
                    non_null_types = [t for t in value if t != "null"]
                    normalized[key] = non_null_types[0] if non_null_types else value[0]
                else:
                    normalized[key] = AntigravityProvider._normalize_type_arrays(value)
            return normalized
        elif isinstance(schema, list):
            return [AntigravityProvider._normalize_type_arrays(item) for item in schema]
        else:
            return schema

    # ============================================================================
    # RANDOM ID GENERATION
    # ============================================================================

    @staticmethod
    def generate_request_id() -> str:
        """Generate Antigravity request ID: agent-{uuid}"""
        return f"agent-{uuid.uuid4()}"

    @staticmethod
    def generate_session_id() -> str:
        """Generate Antigravity session ID: -{random_number}"""
        # Generate random 19-digit number
        n = random.randint(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)
        return f"-{n}"

    @staticmethod
    def generate_project_id() -> str:
        """Generate fake project ID: {adj}-{noun}-{random}"""
        adjectives = ["useful", "bright", "swift", "calm", "bold"]
        nouns = ["fuze", "wave", "spark", "flow", "core"]
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        random_part = str(uuid.uuid4())[:5].lower()
        return f"{adj}-{noun}-{random_part}"

    # ============================================================================
    # MESSAGE TRANSFORMATION (OpenAI → Gemini CLI format)
    # ============================================================================

    def _transform_messages(self, messages: List[Dict[str, Any]], model: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.
        Handles thoughtSignature preservation with 3-tier fallback:
        1. Use client-provided signature (if present)
        2. Fall back to server-side cache
        3. Use bypass constant as last resort
        
        Args:
            messages: List of OpenAI-formatted messages
            model: Model name for Gemini 3 detection
            
        Returns:
            Tuple of (system_instruction, gemini_contents)
        """
        system_instruction = None
        gemini_contents = []
        
        # Make a copy to avoid modifying original
        messages = copy.deepcopy(messages)
        
        # Separate system prompt from other messages
        if messages and messages[0].get('role') == 'system':
            system_prompt_content = messages.pop(0).get('content', '')
            if system_prompt_content:
                # Handle both string and list-based system content
                system_parts = []
                if isinstance(system_prompt_content, str):
                    system_parts.append({"text": system_prompt_content})
                elif isinstance(system_prompt_content, list):
                    # Multi-part system content (strip cache_control)
                    for item in system_prompt_content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                # Skip cache_control - Claude-specific field
                                system_parts.append({"text": text})
                
                if system_parts:
                    system_instruction = {
                        "role": "user",
                        "parts": system_parts
                    }


        # Build tool call ID to name mapping
        tool_call_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("type") == "function":
                        tool_call_id_to_name[tool_call["id"]] = tool_call["function"]["name"]

        #Convert each message
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            gemini_role = "model" if role == "assistant" else "tool" if role == "tool" else "user"

            if role == "user":
                if isinstance(content, str):
                    # Simple text content
                    if content:
                        parts.append({"text": content})
                elif isinstance(content, list):
                    # Multi-part content (text, images, etc.)
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                # Strip Claude-specific cache_control field
                                # This field causes 400 errors with Antigravity
                                parts.append({"text": text})
                        elif item.get("type") == "image_url":
                            # Handle image data URLs
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                try:
                                    # Parse: data:image/png;base64,iVBORw0KG...
                                    header, data = image_url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": data
                                        }
                                    })
                                except Exception as e:
                                    lib_logger.warning(f"Failed to parse image data URL: {e}")

            elif role == "assistant":
                if isinstance(content, str) and content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            try:
                                args_dict = json.loads(tool_call["function"]["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                args_dict = {}
                            
                            tool_call_id = tool_call.get("id", "")
                            
                            func_call_part = {
                                "functionCall": {
                                    "name": tool_call["function"]["name"],
                                    "args": args_dict
                                }
                            }
                            
                            # PRIORITY 1: Use client-provided signature if available
                            client_signature = tool_call.get("thought_signature")
                            
                            # PRIORITY 2: Fall back to server-side cache
                            if not client_signature and tool_call_id and self._enable_signature_cache:
                                client_signature = self._signature_cache.retrieve(tool_call_id)
                                if client_signature:
                                    lib_logger.debug(f"Retrieved thoughtSignature from cache for {tool_call_id}")
                            
                            # PRIORITY 3: Use bypass constant as last resort
                            if client_signature:
                                func_call_part["thoughtSignature"] = client_signature
                            else:
                                func_call_part["thoughtSignature"] = "skip_thought_signature_validator"
                                
                                # WARNING: Missing signature for Gemini 3
                                if self._is_gemini_3_model(model):
                                    lib_logger.warning(
                                        f"Gemini 3 tool call '{tool_call_id}' missing thoughtSignature. "
                                        f"Client didn't provide it and cache lookup failed. "
                                        f"Using bypass - reasoning quality may degrade."
                                    )
                            
                            parts.append(func_call_part)

            elif role == "tool":
                # Tool responses grouped by function name
                tool_call_id = msg.get("tool_call_id", "")
                function_name = tool_call_id_to_name.get(tool_call_id, "unknown_function")
                tool_content = msg.get("content", "{}")
                
                # Parse tool content - if it's JSON, use parsed value; otherwise use as-is
                try:
                    parsed_content = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    parsed_content = tool_content

                parts.append({
                    "functionResponse": {
                        "name": function_name,
                        "response": {
                            "result": parsed_content
                        }
                    }
                })

            if parts:
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": parts
                })
        
        return system_instruction, gemini_contents

    # ============================================================================
    # REASONING CONFIGURATION (GEMINI 2.5 & 3 ONLY)
    # ============================================================================

    def _map_reasoning_effort_to_thinking_config(
        self,
        reasoning_effort: Optional[str],
        model: str,
        custom_reasoning_budget: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration for Gemini 2.5 and 3 models.
        
        IMPORTANT: This function ONLY applies to Gemini 2.5 and 3 models.
        For other models (e.g., Claude via Antigravity), it returns None.
        
        Gemini 2.5 and 3 use separate budgeting systems:
        - Gemini 2.5: thinkingBudget (integer tokens, based on Gemini CLI logic)
        - Gemini 3: thinkingLevel (string: "low" or "high")
        
        Default behavior (no reasoning_effort):
        - Gemini 2.5: thinkingBudget=-1 (auto mode)
        - Gemini 3: thinkingLevel="high" (always enabled at high level)
        
        Args:
            reasoning_effort: Effort level ('low', 'medium', 'high', 'disable', or None)
            model: Model name (public alias)
            custom_reasoning_budget: If True, use full budgets; if False, divide by 4
            
        Returns:
            Dict with thinkingConfig or None if not a Gemini 2.5/3 model
        """
        internal_model = self._alias_to_model_name(model)
        
        # Detect model family - ONLY support gemini-2.5 and gemini-3
        # For other models (Claude, etc.), return None without filtering
        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = internal_model.startswith("gemini-3-")
        
        # Return None for unsupported models - no reasoning config changes
        if not is_gemini_25 and not is_gemini_3:
            return None
        
        # ========================================================================
        # GEMINI 2.5: Use Gemini CLI logic with thinkingBudget
        # ========================================================================
        if is_gemini_25:
            # Default: auto mode
            if not reasoning_effort:
                return {"thinkingBudget": -1, "include_thoughts": True}
            
            # Disable thinking
            if reasoning_effort == "disable":
                return {"thinkingBudget": 0, "include_thoughts": False}
            
            # Model-specific budgets (same as Gemini CLI)
            if "gemini-2.5-pro" in model:
                budgets = {"low": 8192, "medium": 16384, "high": 32768}
            elif "gemini-2.5-flash" in model:
                budgets = {"low": 6144, "medium": 12288, "high": 24576}
            else:
                # Fallback for other gemini-2.5 models
                budgets = {"low": 1024, "medium": 2048, "high": 4096}
            
            budget = budgets.get(reasoning_effort, -1)  # -1 for invalid/auto
            
            # Apply custom_reasoning_budget toggle
            # If False (default), divide by 4 like Gemini CLI
            if not custom_reasoning_budget:
                budget = budget // 4
            
            return {"thinkingBudget": budget, "include_thoughts": True}
        
        # ========================================================================
        # GEMINI 3: Use STRING thinkingLevel ("low" or "high")
        # ========================================================================
        if is_gemini_3:
            # Default: Always use "high" if not specified
            # Gemini 3 cannot be disabled - always has thinking enabled
            if not reasoning_effort:
                return {"thinkingLevel": "high", "include_thoughts": True}
            
            # Map reasoning effort to string level
            # Note: "disable" is ignored - Gemini 3 cannot disable thinking
            if reasoning_effort == "low":
                level = "low"
            # Medium level not yet available - map to high
            # When medium is released, uncomment the following line:
            # elif reasoning_effort == "medium":
            #     level = "medium"
            else:
                # "medium", "high", "disable", or any invalid value → "high"
                level = "high"
            
            return {"thinkingLevel": level, "include_thoughts": True}
        
        return None

    # ============================================================================
    # TOOL RESPONSE GROUPING
    # ============================================================================

    def _fix_tool_response_grouping(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Antigravity compatibility.
        
        Converts linear format (function call, response, function call, response)
        to grouped format (model with calls, function role with all responses).
        
        Args:
            contents: List of Gemini content objects
            
        Returns:
            List of grouped content objects
        """
        new_contents = []
        pending_groups = []  # Groups awaiting responses
        collected_responses = []  # Standalone responses to match
        
        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])
            
            # Check if this content has function responses
            response_parts = [p for p in parts if "functionResponse" in p]
            
            if response_parts:
                # Collect responses
                collected_responses.extend(response_parts)
                
                # Try to satisfy pending groups
                for i in range(len(pending_groups) - 1, -1, -1):
                    group = pending_groups[i]
                    if len(collected_responses) >= group["responses_needed"]:
                        # Take needed responses
                        group_responses = collected_responses[:group["responses_needed"]]
                        collected_responses = collected_responses[group["responses_needed"]:]
                        
                        # Create merged function response content
                        function_response_content = {
                            "parts": group_responses,
                            "role": "user"
                        }
                        new_contents.append(function_response_content)
                        
                        # Remove satisfied group
                        pending_groups.pop(i)
                        break
                
                continue  # Skip adding this content
            
            # If this is model content with function calls, create a group
            if role == "model":
                function_calls = [p for p in parts if "functionCall" in p]
                
                if function_calls:
                    # Add model content first
                    new_contents.append(content)
                    
                    # Create pending group
                    pending_groups.append({
                        "model_content": content,
                        "function_calls": function_calls,
                        "responses_needed": len(function_calls)
                    })
                else:
                    # Regular model content without function calls
                    new_contents.append(content)
            else:
                # Non-model content (user, etc.)
                new_contents.append(content)
        
        # Handle remaining pending groups
        for group in pending_groups:
            if len(collected_responses) >= group["responses_needed"]:
                group_responses = collected_responses[:group["responses_needed"]]
                collected_responses = collected_responses[group["responses_needed"]:]
                
                function_response_content = {
                    "parts": group_responses,
                    "role": "user"
                }
                new_contents.append(function_response_content)
        
        return new_contents

    # ============================================================================
    # ANTIGRAVITY REQUEST TRANSFORMATION
    # ============================================================================

    def _transform_to_antigravity_format(
        self,
        gemini_cli_payload: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Transform Gemini CLI format to complete Antigravity format.
        
        Args:
            gemini_cli_payload: Request in Gemini CLI format
            model: Model name (public alias)
            
        Returns:
            Complete Antigravity request payload
        """
        internal_model = self._alias_to_model_name(model)
        
        # 1. Wrap in Antigravity envelope
        antigravity_payload = {
            "project": self.generate_project_id(),
            "userAgent": "antigravity",
            "requestId": self.generate_request_id(),
            "model": internal_model,  # Use internal name
            "request": copy.deepcopy(gemini_cli_payload)
        }
        
        # 2. Add session ID
        antigravity_payload["request"]["sessionId"] = self.generate_session_id()
        
        # 3. Remove fields that Antigravity doesn't support
        antigravity_payload["request"].pop("safetySettings", None)
        if "generationConfig" in antigravity_payload["request"]:
            antigravity_payload["request"]["generationConfig"].pop("maxOutputTokens", None)
        
        # 4. Set toolConfig mode
        if "toolConfig" not in antigravity_payload["request"]:
            antigravity_payload["request"]["toolConfig"] = {}
        if "functionCallingConfig" not in antigravity_payload["request"]["toolConfig"]:
            antigravity_payload["request"]["toolConfig"]["functionCallingConfig"] = {}
        antigravity_payload["request"]["toolConfig"]["functionCallingConfig"]["mode"] = "VALIDATED"
        
        # 5. Handle Gemini 3 specific thinking logic
        # For non-Gemini-3 models, convert thinkingLevel to thinkingBudget
        if not internal_model.startswith("gemini-3-"):
            gen_config = antigravity_payload["request"].get("generationConfig", {})
            thinking_config = gen_config.get("thinkingConfig", {})
            if "thinkingLevel" in thinking_config:
                # Remove thinkingLevel for non-Gemini-3 models
                del thinking_config["thinkingLevel"]
                # Set thinkingBudget to -1 (auto/dynamic)
                thinking_config["thinkingBudget"] = -1
        
        # 6. Preserve/add thoughtSignature to ALL function calls in model role content
        for content in antigravity_payload["request"].get("contents", []):
            if content.get("role") == "model":
                for part in content.get("parts", []):
                    # Add signature to function calls OR preserve if already exists
                    if "functionCall" in part and "thoughtSignature" not in part:
                        part["thoughtSignature"] = "skip_thought_signature_validator"
        
        # 7. CLAUDE-SPECIFIC TOOL SCHEMA TRANSFORMATION
        # Reference: Go implementation antigravity_executor.go lines 672-684
        # For Claude models: parametersJsonSchema → parameters, remove $schema
        if internal_model.startswith("claude-sonnet-"):
            lib_logger.debug(f"Applying Claude-specific tool schema transformation for {internal_model}")
            tools = antigravity_payload["request"].get("tools", [])
            
            for tool in tools:
                function_declarations = tool.get("functionDeclarations", [])
                for func_decl in function_declarations:
                    if "parametersJsonSchema" in func_decl:
                        params = func_decl["parametersJsonSchema"]
                        
                        # CRITICAL: Claude requires clean JSON Schema draft 2020-12
                        # Recursively remove ALL incompatible fields
                        def clean_claude_schema(schema):
                            """Recursively remove fields Claude doesn't support."""
                            if not isinstance(schema, dict):
                                return schema
                            
                            # Fields that break Claude's JSON Schema validation
                            incompatible = {'$schema', 'additionalProperties', 'minItems', 'maxItems', 'pattern'}
                            cleaned = {}
                            
                            for key, value in schema.items():
                                if key in incompatible:
                                    continue  # Skip incompatible fields
                                
                                if isinstance(value, dict):
                                    cleaned[key] = clean_claude_schema(value)
                                elif isinstance(value, list):
                                    cleaned[key] = [
                                        clean_claude_schema(item) if isinstance(item, dict) else item
                                        for item in value
                                    ]
                                else:
                                    cleaned[key] = value
                            
                            return cleaned
                        
                        # Clean the schema
                        params = clean_claude_schema(params) if isinstance(params, dict) else params
                        
                        # Rename parametersJsonSchema → parameters for Claude
                        func_decl["parameters"] = params
                        del func_decl["parametersJsonSchema"]
        
        return antigravity_payload

    #============================================================================
    # BASE URL FALLBACK LOGIC
    # ============================================================================

    def _get_current_base_url(self) -> str:
        """Get the current base URL from the fallback list."""
        return self._current_base_url

    def _try_next_base_url(self) -> bool:
        """
        Switch to the next base URL in the fallback list.
        
        Returns:
            True if successfully switched to next URL, False if no more URLs available
        """
        if self._base_url_index < len(BASE_URLS) - 1:
            self._base_url_index += 1
            self._current_base_url = BASE_URLS[self._base_url_index]
            lib_logger.info(f"Switching to fallback Antigravity base URL: {self._current_base_url}")
            return True
        return False

    def _reset_base_url(self):
        """Reset to the primary base URL (daily sandbox)."""
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

    # ============================================================================
    # RESPONSE TRANSFORMATION (Antigravity → OpenAI)
    # ============================================================================

    def _unwrap_antigravity_response(self, antigravity_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Gemini response from Antigravity envelope.
        
        Args:
            antigravity_response: Response from Antigravity API
            
        Returns:
            Gemini response (unwrapped)
        """
        # For both streaming and non-streaming, response is in 'response' field
        return antigravity_response.get("response", antigravity_response)

    def _gemini_to_openai_chunk(self, gemini_chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert a Gemini API response chunk to OpenAI format.
        
        UPDATED: Now preserves thoughtSignatures for Gemini 3 multi-turn conversations:
        - Stores signatures in server-side cache (if enabled)
        - Includes signatures in response (if client passthrough enabled)
        - Filters standalone signature parts (no functionCall/text)
        
        Args:
            gemini_chunk: Gemini API response chunk
            model: Model name for Gemini 3 detection
            
        Returns:
            OpenAI-compatible response chunk
        """
        # Extract the main response structure
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            return {}
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        content_parts = content.get("parts", [])
        
        # Build delta components
        text_content = ""
        reasoning_content = ""
        tool_calls = []
        
        # Track if we've seen a signature yet (for parallel tool call handling)
        # Per Gemini 3 spec: only FIRST tool call in parallel gets signature
        first_signature_seen = False
        tool_call_index = 0  # Track index for OpenAI streaming format
        
        for part in content_parts:
            has_function_call = "functionCall" in part
            has_text = "text" in part
            has_signature = "thoughtSignature" in part and part["thoughtSignature"]
            
            # FIXED: Only skip if ONLY signature (standalone encryption part)
            # Previously this filtered out ALL function calls with signatures!
            if has_signature and not has_function_call and not has_text:
                continue  # Skip standalone signature parts
            
            # Process text content
            if has_text:
                thought = part.get("thought")
                if thought is True or (isinstance(thought, str) and thought.lower() == 'true'):
                    reasoning_content += part["text"]
                else:
                    text_content += part["text"]
            
            # Process function calls (NOW WORKS with signatures!)
            if has_function_call:
                func_call = part["functionCall"]
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "index": tool_call_index,  # REQUIRED for OpenAI streaming format
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                }
                tool_call_index += 1  # Increment for next tool call
                
                # Handle thoughtSignature if present
                if has_signature and not first_signature_seen:
                    # Only first tool call gets signature (parallel call handling)
                    first_signature_seen = True
                    signature = part["thoughtSignature"]
                    
                    # Option 1: Store in server-side cache (if enabled)
                    if self._enable_signature_cache:
                        self._signature_cache.store(tool_call_id, signature)
                        lib_logger.debug(f"Stored thoughtSignature in cache for {tool_call_id}")
                    
                    # Option 2: Pass to client (if enabled) - INDEPENDENT of cache!
                    if self._preserve_signatures_in_client:
                        tool_call["thought_signature"] = signature
                
                tool_calls.append(tool_call)
        
        # Build delta
        delta = {}
        if text_content:
            delta["content"] = text_content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["role"] = "assistant"
        elif text_content or reasoning_content:
            delta["role"] = "assistant"
        
        # Handle finish reason
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            # Map Gemini finish reasons to OpenAI
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop"
            }
            finish_reason = finish_reason_map.get(finish_reason, "stop")
            if tool_calls:
                finish_reason = "tool_calls"
        
        # Build usage metadata
        usage = None
        usage_metadata = gemini_chunk.get("usageMetadata", {})
        if usage_metadata:
            prompt_tokens = usage_metadata.get("promptTokenCount", 0)
            thoughts_tokens = usage_metadata.get("thoughtsTokenCount", 0)
            completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
            
            usage = {
                "prompt_tokens": prompt_tokens + thoughts_tokens,  # Include thoughts in prompt
                "completion_tokens": completion_tokens,
                "total_tokens": usage_metadata.get("totalTokenCount", 0)
            }
            
            # Add reasoning tokens details if thinking was used
            if thoughts_tokens > 0:
                if "completion_tokens_details" not in usage:
                    usage["completion_tokens_details"] = {}
                usage["completion_tokens_details"]["reasoning_tokens"] = thoughts_tokens
        
        # Build final response
        response = {
            "id": gemini_chunk.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        
        if usage:
            response["usage"] = usage
        
        return response
            
    # ============================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # ============================================================================

    async def get_valid_token(self, credential_identifier: str) -> str:
        """
        Get a valid access token for the credential.
        
        Args:
            credential_identifier: Credential file path or "env"
            
        Returns:
            Access token string
        """
        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_identifier, creds)
        return creds['access_token']

    def has_custom_logic(self) -> bool:
        """Antigravity uses custom translation logic."""
        return True

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Get OAuth authorization header for Antigravity.
        
        Args:
            credential_identifier: Credential file path or "env"
            
        Returns:
            Dict with Authorization header
        """
        access_token = await self.get_valid_token(credential_identifier)
        return {"Authorization": f"Bearer {access_token}"}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from Antigravity.
        
        For Antigravity, we can optionally use the fetchAvailableModels endpoint and apply
        alias mapping to convert internal names to public names. However, this endpoint is
        often unavailable (404), so dynamic discovery is disabled by default.
        
        Set ANTIGRAVITY_ENABLE_DYNAMIC_MODELS=true to enable dynamic discovery.
        
        Args:
            api_key: Credential path (not a traditional API key)
            client: HTTP client
            
        Returns:
            List of public model names
        """
        # If dynamic discovery is disabled, immediately return hardcoded list
        if not self._enable_dynamic_model_discovery:
            lib_logger.debug("Using hardcoded Antigravity model list (dynamic discovery disabled)")
            return [f"antigravity/{m}" for m in HARDCODED_MODELS]
        
        # Dynamic discovery enabled - attempt to fetch from API
        credential_path = api_key  # For OAuth providers, this is the credential path
        
        try:
            access_token = await self.get_valid_token(credential_path)
            base_url = self._get_current_base_url()
            
            url = f"{base_url}/fetchAvailableModels"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "project": self.generate_project_id(),
                "requestId": self.generate_request_id(),
                "userAgent": "antigravity"
            }
            
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract model names and apply aliasing
            models = []
            if "models" in data:
                for model_info in data["models"]:
                    internal_name = model_info.get("name", "").replace("models/", "")
                    if internal_name:
                        public_name = self._model_name_to_alias(internal_name)
                        if public_name:  # Skip excluded models (empty string)
                            models.append(f"antigravity/{public_name}")
            
            if models:
                lib_logger.info(f"Discovered {len(models)} Antigravity models via dynamic discovery")
                return models
            else:
                lib_logger.warning("No models returned from Antigravity, using hardcoded list")
                return [f"antigravity/{m}" for m in HARDCODED_MODELS]
                
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Antigravity models: {e}, using hardcoded list")
            return [f"antigravity/{m}" for m in HARDCODED_MODELS]

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests for Antigravity.
        
        This is the main entry point that:
        1. Extracts the model and credential path
        2. Transforms OpenAI request → Gemini CLI → Antigravity format
        3. Makes the API call with fallback logic
        4. Transforms Antigravity response → Gemini → OpenAI format
        
        Args:
            client: HTTP client
            **kwargs: LiteLLM completion parameters
            
        Returns:
            ModelResponse (non-streaming) or AsyncGenerator (streaming)
        """
        # Extract key parameters
        model = kwargs.get("model", "gemini-2.5-pro")
        
        # Strip provider prefix from model name (e.g., "antigravity/claude-sonnet-4-5-thinking" -> "claude-sonnet-4-5-thinking")
        if "/" in model:
            model = model.split("/")[-1]
        
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("api_key", ""))
        tools = kwargs.get("tools")
        reasoning_effort = kwargs.get("reasoning_effort")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        max_tokens = kwargs.get("max_tokens")
        enable_request_logging = kwargs.pop("enable_request_logging", False)
        
        lib_logger.info(f"Antigravity completion: model={model}, stream={stream}, messages={len(messages)}")
        
        # Create file logger
        file_logger = _AntigravityFileLogger(
            model_name=model,
            enabled=enable_request_logging
        )
        
        # Step 1: Transform messages (OpenAI → Gemini CLI)
        system_instruction, gemini_contents = self._transform_messages(messages, model=model)
        
        # Apply tool response grouping
        gemini_contents = self._fix_tool_response_grouping(gemini_contents)
        
        # Step 2: Build Gemini CLI payload
        gemini_cli_payload = {
            "contents": gemini_contents
        }
        
        if system_instruction:
            gemini_cli_payload["system_instruction"] = system_instruction
        
        # Add generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        
        # Extract custom_reasoning_budget toggle
        # Check kwargs first, then headers if not found
        custom_reasoning_budget = kwargs.get("custom_reasoning_budget", False)
        
        # Handle thinking config
        thinking_config = self._map_reasoning_effort_to_thinking_config(
            reasoning_effort, 
            model,
            custom_reasoning_budget
        )
        if thinking_config:
            generation_config.setdefault("thinkingConfig", {}).update(thinking_config)
        
        if generation_config:
            gemini_cli_payload["generationConfig"] = generation_config
        
        # Add tools - using Go reference implementation approach
        # Go code (line 298-328): renames 'parameters' -> 'parametersJsonSchema' and removes 'strict'
        if tools:
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    
                    # Get parameters dict (may be missing)
                    parameters = func.get("parameters")
                    
                    # Build function declaration
                    func_decl = {
                        "name": func.get("name", ""),
                        "description": func.get("description", "")
                    }
                    
                    # Handle parameters -> parametersJsonSchema conversion (matching Go)
                    if parameters and isinstance(parameters, dict):
                        # Make a copy to avoid modifying original
                        schema = dict(parameters)
                        # Remove OpenAI-specific fields that Antigravity doesn't support
                        schema.pop("$schema", None)
                        schema.pop("strict", None)
                        # CRITICAL: Normalize type arrays for protobuf compatibility
                        # Converts ["string", "null"] → "string" to avoid "Proto field is not repeating" errors
                        schema = self._normalize_type_arrays(schema)
                        func_decl["parametersJsonSchema"] = schema
                    else:
                        # No parameters provided - set default empty schema (matching Go lines 318-323)
                        func_decl["parametersJsonSchema"] = {
                            "type": "object",
                            "properties": {}
                        }
                    
                    gemini_tools.append({
                        "functionDeclarations": [func_decl]
                    })
            
            if gemini_tools:
                gemini_cli_payload["tools"] = gemini_tools
        
        # Step 3: Transform to Antigravity format
        antigravity_payload = self._transform_to_antigravity_format(gemini_cli_payload, model)
        
        # Log the request
        file_logger.log_request(antigravity_payload)
        
        # Step 4: Make API call
        access_token = await self.get_valid_token(credential_path)
        base_url = self._get_current_base_url()
        
        endpoint = ":streamGenerateContent" if stream else ":generateContent"
        url = f"{base_url}{endpoint}"

        # Add query parameter for streaming (required by Antigravity API)
        if stream:
            url = f"{url}?alt=sse"

        # Extract host from base_url for Host header (required by Google's API)
        from urllib.parse import urlparse
        parsed_url = urlparse(base_url)
        host = parsed_url.netloc if parsed_url.netloc else base_url.replace("https://", "").replace("http://", "").rstrip("/")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Host": host,  # CRITICAL: Required by Antigravity API
            "User-Agent": "antigravity/1.11.5"  # Match Go implementation
        }

        if stream:
            headers["Accept"] = "text/event-stream"
        else:
            headers["Accept"] = "application/json"

        lib_logger.debug(f"Antigravity request to: {url}")
        
        try:
            if stream:
                return self._handle_streaming(client, url, headers, antigravity_payload, model, file_logger)
            else:
                return await self._handle_non_streaming(client, url, headers, antigravity_payload, model, file_logger)
        except Exception as e:
            # Try fallback URL if available
            if self._try_next_base_url():
                lib_logger.warning(f"Retrying Antigravity request with fallback URL: {e}")
                base_url = self._get_current_base_url()
                url = f"{base_url}{endpoint}"
                
                if stream:
                    return self._handle_streaming(client, url, headers, antigravity_payload, model)
                else:
                    return await self._handle_non_streaming(client, url, headers, antigravity_payload, model)
            else:
                raise

    async def _handle_non_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[_AntigravityFileLogger] = None
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        response = await client.post(url, headers=headers, json=payload, timeout=120.0)
        response.raise_for_status()
        
        antigravity_response = response.json()
        
        # Log response
        if file_logger:
            file_logger.log_final_response(antigravity_response)
        
        # Unwrap Antigravity envelope
        gemini_response = self._unwrap_antigravity_response(antigravity_response)
        
        # Convert to OpenAI format
        return self._gemini_to_openai_chunk(gemini_response, model)

    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[_AntigravityFileLogger] = None
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        async with client.stream("POST", url, headers=headers, json=payload, timeout=120.0) as response:
            # Log error response body for debugging if request failed
            if response.status_code >= 400:
                try:
                    error_body = await response.aread()
                    lib_logger.error(f"Antigravity API error {response.status_code}: {error_body.decode('utf-8', errors='replace')}")
                except Exception as e:
                    lib_logger.error(f"Failed to read error response body: {e}")
            
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                # Log raw chunk
                if file_logger:
                    file_logger.log_response_chunk(line)
                
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        antigravity_chunk = json.loads(data_str)
                        
                        # Unwrap Antigravity envelope
                        gemini_chunk = self._unwrap_antigravity_response(antigravity_chunk)
                        
                        # Convert to OpenAI format
                        openai_chunk = self._gemini_to_openai_chunk(gemini_chunk, model)
                        
                        # Convert dict to ModelResponse object
                        model_response = litellm.ModelResponse(**openai_chunk)
                        yield model_response
                    except json.JSONDecodeError:
                        if file_logger:
                            file_logger.log_error(f"Failed to parse chunk: {data_str[:100]}")
                        lib_logger.warning(f"Failed to parse Antigravity chunk: {data_str[:100]}")
                        continue

    # ============================================================================
    # TOKEN COUNTING
    # ============================================================================

    async def count_tokens(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        litellm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Counts tokens for the given prompt using the Antigravity :countTokens endpoint.
        
        Args:
            client: The HTTP client to use
            credential_path: Path to the credential file
            model: Model name to use for token counting
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions
            litellm_params: Optional additional parameters
        
        Returns:
            Dict with 'prompt_tokens' and 'total_tokens' counts
        """
        # Get auth token
        access_token = await self.get_valid_token(credential_path)
        
        # Convert public alias to internal name
        internal_model = self._alias_to_model_name(model)
        
        # Transform messages to Gemini format
        system_instruction, contents = self._transform_messages(messages, model=internal_model)
        
        # Build Gemini CLI payload
        gemini_cli_payload = {
            "contents": contents
        }
        
        if system_instruction:
            gemini_cli_payload["systemInstruction"] = system_instruction
        
        if tools:
            # Transform tools - same as in acompletion
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    parameters = func.get("parameters")
                    
                    func_decl = {
                        "name": func.get("name", ""),
                        "description": func.get("description", "")
                    }
                    
                    if parameters and isinstance(parameters, dict):
                        schema = dict(parameters)
                        schema.pop("$schema", None)
                        schema.pop("strict", None)
                        func_decl["parametersJsonSchema"] = schema
                    else:
                        func_decl["parametersJsonSchema"] = {
                            "type": "object",
                            "properties": {}
                        }
                    
                    gemini_tools.append({
                        "functionDeclarations": [func_decl]
                    })
            
            if gemini_tools:
                gemini_cli_payload["tools"] = gemini_tools
        
        # Wrap in Antigravity envelope
        antigravity_payload = {
            "project": self.generate_project_id(),
            "userAgent": "antigravity",
            "requestId": self.generate_request_id(),
            "model": internal_model,
            "request": gemini_cli_payload
        }
        
        # Make the request
        base_url = self._get_current_base_url()
        url = f"{base_url}:countTokens"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = await client.post(url, headers=headers, json=antigravity_payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Unwrap Antigravity response
            unwrapped = self._unwrap_antigravity_response(data)
            
            # Extract token counts from response
            total_tokens = unwrapped.get('totalTokens', 0)
            
            return {
                'prompt_tokens': total_tokens,
                'total_tokens': total_tokens,
            }
        
        except httpx.HTTPStatusError as e:
            lib_logger.error(f"Failed to count tokens: {e}")
            # Return 0 on error rather than raising
            return {'prompt_tokens': 0, 'total_tokens': 0}
