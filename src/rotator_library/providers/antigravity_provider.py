# src/rotator_library/providers/antigravity_provider_v2.py
"""
Antigravity Provider - Refactored Implementation

A clean, well-structured provider for Google's Antigravity API, supporting:
- Gemini 2.5 (Pro/Flash) with thinkingBudget
- Gemini 3 (Pro/Image) with thinkingLevel
- Claude (Sonnet 4.5) via Antigravity proxy

Key Features:
- Unified streaming/non-streaming handling
- Server-side thought signature caching
- Automatic base URL fallback
- Gemini 3 tool hallucination prevention
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import litellm

from .provider_interface import ProviderInterface
from .antigravity_auth_base import AntigravityAuthBase
from .provider_cache import ProviderCache
from ..model_definitions import ModelDefinitions


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

lib_logger = logging.getLogger('rotator_library')

# Antigravity base URLs with fallback order
# Priority: daily (sandbox) → autopush (sandbox) → production
BASE_URLS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://cloudcode-pa.googleapis.com/v1internal", # Production fallback
]

# Available models via Antigravity
AVAILABLE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-thinking",
]

# Default max output tokens (including thinking) - can be overridden per request
DEFAULT_MAX_OUTPUT_TOKENS = 16384

# Model alias mappings (internal ↔ public)
MODEL_ALIAS_MAP = {
    "rev19-uic3-1p": "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-pro-image": "gemini-3-pro-image-preview",
    "gemini-3-pro-high": "gemini-3-pro-preview",
}
MODEL_ALIAS_REVERSE = {v: k for k, v in MODEL_ALIAS_MAP.items()}

# Models to exclude from dynamic discovery
EXCLUDED_MODELS = {"chat_20706", "chat_23310", "gemini-2.5-flash-thinking", "gemini-3-pro-low", "gemini-2.5-pro"}

# Gemini finish reason mapping
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}

# Directory paths
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = _BASE_DIR / "logs" / "antigravity_logs"
CACHE_DIR = _BASE_DIR / "cache" / "antigravity"
GEMINI3_SIGNATURE_CACHE_FILE = CACHE_DIR / "gemini3_signatures.json"
CLAUDE_THINKING_CACHE_FILE = CACHE_DIR / "claude_thinking.json"

# Gemini 3 tool fix system instruction (prevents hallucination)
DEFAULT_GEMINI3_SYSTEM_INSTRUCTION = """CRITICAL TOOL USAGE INSTRUCTIONS:
You are operating in a custom environment where tool definitions differ from your training data.
You MUST follow these rules strictly:

1. DO NOT use your internal training data to guess tool parameters
2. ONLY use the exact parameter structure defined in the tool schema
3. If a tool takes a 'files' parameter, it is ALWAYS an array of objects with specific properties, NEVER a simple array of strings
4. If a tool edits code, it takes structured JSON objects with specific fields, NEVER raw diff strings or plain text
5. Parameter names in schemas are EXACT - do not substitute with similar names from your training (e.g., use 'follow_up' not 'suggested_answers')
6. Array parameters have specific item types - check the schema's 'items' field for the exact structure
7. When you see "STRICT PARAMETERS" in a tool description, those type definitions override any assumptions

If you are unsure about a tool's parameters, YOU MUST read the schema definition carefully. Your training data about common tool names like 'read_file' or 'apply_diff' does NOT apply here.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


def _generate_request_id() -> str:
    """Generate Antigravity request ID: agent-{uuid}"""
    return f"agent-{uuid.uuid4()}"


def _generate_session_id() -> str:
    """Generate Antigravity session ID: -{random_number}"""
    n = random.randint(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)
    return f"-{n}"


def _generate_project_id() -> str:
    """Generate fake project ID: {adj}-{noun}-{random}"""
    adjectives = ["useful", "bright", "swift", "calm", "bold"]
    nouns = ["fuze", "wave", "spark", "flow", "core"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:5]}"


def _normalize_type_arrays(schema: Any) -> Any:
    """
    Normalize type arrays in JSON Schema for Proto-based Antigravity API.
    Converts `"type": ["string", "null"]` → `"type": "string"`.
    """
    if isinstance(schema, dict):
        normalized = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                normalized[key] = non_null[0] if non_null else value[0]
            else:
                normalized[key] = _normalize_type_arrays(value)
        return normalized
    elif isinstance(schema, list):
        return [_normalize_type_arrays(item) for item in schema]
    return schema


def _recursively_parse_json_strings(obj: Any) -> Any:
    """
    Recursively parse JSON strings in nested data structures.
    
    Antigravity sometimes returns tool arguments with JSON-stringified values:
    {"files": "[{...}]"} instead of {"files": [{...}]}.
    """
    if isinstance(obj, dict):
        return {k: _recursively_parse_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursively_parse_json_strings(item) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try:
                parsed = json.loads(obj)
                return _recursively_parse_json_strings(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
    return obj


def _clean_claude_schema(schema: Any) -> Any:
    """Recursively remove fields that Claude's JSON Schema validation doesn't support."""
    if not isinstance(schema, dict):
        return schema
    
    incompatible = {'$schema', 'additionalProperties', 'minItems', 'maxItems', 'pattern'}
    cleaned = {}
    
    for key, value in schema.items():
        if key in incompatible:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_claude_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_claude_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    
    return cleaned


# =============================================================================
# FILE LOGGER
# =============================================================================

class AntigravityFileLogger:
    """Transaction file logger for debugging Antigravity requests/responses."""
    
    __slots__ = ('enabled', 'log_dir')
    
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        self.log_dir: Optional[Path] = None
        
        if not enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_model = model_name.replace('/', '_').replace(':', '_')
        self.log_dir = LOGS_DIR / f"{timestamp}_{safe_model}_{uuid.uuid4()}"
        
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create log directory: {e}")
            self.enabled = False
    
    def log_request(self, payload: Dict[str, Any]) -> None:
        """Log the request payload."""
        self._write_json("request_payload.json", payload)
    
    def log_response_chunk(self, chunk: str) -> None:
        """Append a raw chunk to the response stream log."""
        self._append_text("response_stream.log", chunk)
    
    def log_error(self, error_message: str) -> None:
        """Log an error message."""
        self._append_text("error.log", f"[{datetime.utcnow().isoformat()}] {error_message}")
    
    def log_final_response(self, response: Dict[str, Any]) -> None:
        """Log the final response."""
        self._write_json("final_response.json", response)
    
    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"Failed to write {filename}: {e}")
    
    def _append_text(self, filename: str, text: str) -> None:
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            lib_logger.error(f"Failed to append to {filename}: {e}")




# =============================================================================
# MAIN PROVIDER CLASS
# =============================================================================

class AntigravityProvider(AntigravityAuthBase, ProviderInterface):
    """
    Antigravity provider for Gemini and Claude models via Google's internal API.
    
    Supports:
    - Gemini 2.5 (Pro/Flash) with thinkingBudget
    - Gemini 3 (Pro/Image) with thinkingLevel  
    - Claude Sonnet 4.5 via Antigravity proxy
    
    Features:
    - Unified streaming/non-streaming handling
    - ThoughtSignature caching for multi-turn conversations
    - Automatic base URL fallback
    - Gemini 3 tool hallucination prevention
    """
    
    skip_cost_calculation = True
    
    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        
        # Base URL management
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]
        
        # Configuration from environment
        memory_ttl = _env_int("ANTIGRAVITY_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = _env_int("ANTIGRAVITY_SIGNATURE_DISK_TTL", 86400)
        
        # Initialize caches using shared ProviderCache
        self._signature_cache = ProviderCache(
            GEMINI3_SIGNATURE_CACHE_FILE, memory_ttl, disk_ttl,
            env_prefix="ANTIGRAVITY_SIGNATURE"
        )
        self._thinking_cache = ProviderCache(
            CLAUDE_THINKING_CACHE_FILE, memory_ttl, disk_ttl,
            env_prefix="ANTIGRAVITY_THINKING"
        )
        
        # Feature flags
        self._preserve_signatures_in_client = _env_bool("ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES", True)
        self._enable_signature_cache = _env_bool("ANTIGRAVITY_ENABLE_SIGNATURE_CACHE", True)
        self._enable_dynamic_models = _env_bool("ANTIGRAVITY_ENABLE_DYNAMIC_MODELS", False)
        self._enable_gemini3_tool_fix = _env_bool("ANTIGRAVITY_GEMINI3_TOOL_FIX", True)
        
        # Gemini 3 tool fix configuration
        self._gemini3_tool_prefix = os.getenv("ANTIGRAVITY_GEMINI3_TOOL_PREFIX", "gemini3_")
        self._gemini3_description_prompt = os.getenv(
            "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT",
            "\n\nSTRICT PARAMETERS: {params}."
        )
        self._gemini3_system_instruction = os.getenv(
            "ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION",
            DEFAULT_GEMINI3_SYSTEM_INSTRUCTION
        )
        
        # Log configuration
        self._log_config()
    
    def _log_config(self) -> None:
        """Log provider configuration."""
        lib_logger.debug(
            f"Antigravity config: signatures_in_client={self._preserve_signatures_in_client}, "
            f"cache={self._enable_signature_cache}, dynamic_models={self._enable_dynamic_models}, "
            f"gemini3_fix={self._enable_gemini3_tool_fix}"
        )
    
    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================
    
    def _alias_to_internal(self, alias: str) -> str:
        """Convert public alias to internal model name."""
        return MODEL_ALIAS_REVERSE.get(alias, alias)
    
    def _internal_to_alias(self, internal: str) -> str:
        """Convert internal model name to public alias."""
        if internal in EXCLUDED_MODELS:
            return ""
        return MODEL_ALIAS_MAP.get(internal, internal)
    
    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        internal = self._alias_to_internal(model)
        return internal.startswith("gemini-3-") or model.startswith("gemini-3-")
    
    def _is_claude(self, model: str) -> bool:
        """Check if model is Claude."""
        return "claude" in model.lower()
    
    def _strip_provider_prefix(self, model: str) -> str:
        """Strip provider prefix from model name."""
        return model.split("/")[-1] if "/" in model else model
    
    # =========================================================================
    # BASE URL MANAGEMENT
    # =========================================================================
    
    def _get_base_url(self) -> str:
        """Get current base URL."""
        return self._current_base_url
    
    def _try_next_base_url(self) -> bool:
        """Switch to next base URL in fallback list. Returns True if successful."""
        if self._base_url_index < len(BASE_URLS) - 1:
            self._base_url_index += 1
            self._current_base_url = BASE_URLS[self._base_url_index]
            lib_logger.info(f"Switching to fallback URL: {self._current_base_url}")
            return True
        return False
    
    def _reset_base_url(self) -> None:
        """Reset to primary base URL."""
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]
    
    # =========================================================================
    # THINKING CACHE KEY GENERATION
    # =========================================================================
    
    def _generate_thinking_cache_key(
        self,
        text_content: str,
        tool_calls: List[Dict]
    ) -> Optional[str]:
        """
        Generate stable cache key from response content for Claude thinking preservation.
        
        Uses composite key:
        - Tool call IDs (most stable)
        - Text hash (for text-only responses)
        """
        key_parts = []
        
        if tool_calls:
            first_id = tool_calls[0].get("id", "")
            if first_id:
                key_parts.append(f"tool_{first_id.replace('call_', '')}")
        
        if text_content:
            text_hash = hashlib.md5(text_content[:200].encode()).hexdigest()[:16]
            key_parts.append(f"text_{text_hash}")
        
        return "thinking_" + "_".join(key_parts) if key_parts else None
    
    # =========================================================================
    # REASONING CONFIGURATION
    # =========================================================================
    
    def _get_thinking_config(
        self,
        reasoning_effort: Optional[str],
        model: str,
        custom_budget: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration.
        
        - Gemini 2.5 & Claude: thinkingBudget (integer tokens)
        - Gemini 3: thinkingLevel (string: "low"/"high")
        """
        internal = self._alias_to_internal(model)
        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = internal.startswith("gemini-3-")
        is_claude = self._is_claude(model)
        
        if not (is_gemini_25 or is_gemini_3 or is_claude):
            return None
        
        # Gemini 3: String-based thinkingLevel
        if is_gemini_3:
            if reasoning_effort == "low":
                return {"thinkingLevel": "low", "include_thoughts": True}
            return {"thinkingLevel": "high", "include_thoughts": True}
        
        # Gemini 2.5 & Claude: Integer thinkingBudget
        if not reasoning_effort:
            return {"thinkingBudget": -1, "include_thoughts": True}  # Auto
        
        if reasoning_effort == "disable":
            return {"thinkingBudget": 0, "include_thoughts": False}
        
        # Model-specific budgets
        if "gemini-2.5-pro" in model or is_claude:
            budgets = {"low": 8192, "medium": 16384, "high": 32768}
        elif "gemini-2.5-flash" in model:
            budgets = {"low": 6144, "medium": 12288, "high": 24576}
        else:
            budgets = {"low": 1024, "medium": 2048, "high": 4096}
        
        budget = budgets.get(reasoning_effort, -1)
        if not custom_budget:
            budget = budget // 4  # Default to 25% of max output tokens
        
        return {"thinkingBudget": budget, "include_thoughts": True}
    
    # =========================================================================
    # MESSAGE TRANSFORMATION (OpenAI → Gemini)
    # =========================================================================
    
    def _transform_messages(
        self,
        messages: List[Dict[str, Any]],
        model: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.
        
        Handles:
        - System instruction extraction
        - Multi-part content (text, images)
        - Tool calls and responses
        - Claude thinking injection from cache
        - Gemini 3 thoughtSignature preservation
        """
        messages = copy.deepcopy(messages)
        system_instruction = None
        gemini_contents = []
        
        # Extract system prompt
        if messages and messages[0].get('role') == 'system':
            system_content = messages.pop(0).get('content', '')
            if system_content:
                system_parts = self._parse_content_parts(system_content, _strip_cache_control=True)
                if system_parts:
                    system_instruction = {"role": "user", "parts": system_parts}
        
        # Build tool_call_id → name mapping
        tool_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function":
                        tool_id_to_name[tc["id"]] = tc["function"]["name"]
        
        # Convert each message
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            
            if role == "user":
                parts = self._transform_user_message(content)
            elif role == "assistant":
                parts = self._transform_assistant_message(msg, model, tool_id_to_name)
            elif role == "tool":
                parts = self._transform_tool_message(msg, model, tool_id_to_name)
            
            if parts:
                gemini_role = "model" if role == "assistant" else "user" if role == "tool" else "user"
                gemini_contents.append({"role": gemini_role, "parts": parts})
        
        return system_instruction, gemini_contents
    
    def _parse_content_parts(
        self,
        content: Any,
        _strip_cache_control: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse content into Gemini parts format."""
        parts = []
        
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append({"text": text})
                elif item.get("type") == "image_url":
                    image_part = self._parse_image_url(item.get("image_url", {}))
                    if image_part:
                        parts.append(image_part)
        
        return parts
    
    def _parse_image_url(self, image_url: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse image URL into Gemini inlineData format."""
        url = image_url.get("url", "")
        if not url.startswith("data:"):
            return None
        
        try:
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inlineData": {"mimeType": mime_type, "data": data}}
        except Exception as e:
            lib_logger.warning(f"Failed to parse image URL: {e}")
            return None
    
    def _transform_user_message(self, content: Any) -> List[Dict[str, Any]]:
        """Transform user message content to Gemini parts."""
        return self._parse_content_parts(content)
    
    def _transform_assistant_message(
        self,
        msg: Dict[str, Any],
        model: str,
        _tool_id_to_name: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform assistant message including tool calls and thinking injection."""
        parts = []
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        
        # Try to inject cached thinking for Claude
        if self._is_claude(model) and self._enable_signature_cache:
            thinking_parts = self._get_cached_thinking(content, tool_calls)
            parts.extend(thinking_parts)
        
        # Add regular content
        if isinstance(content, str) and content:
            parts.append({"text": content})
        
        # Add tool calls
        for tc in tool_calls:
            if tc.get("type") != "function":
                continue
            
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            
            tool_id = tc.get("id", "")
            func_name = tc["function"]["name"]
            
            # Add prefix for Gemini 3
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                func_name = f"{self._gemini3_tool_prefix}{func_name}"
            
            func_part = {
                "functionCall": {
                    "name": func_name,
                    "args": args,
                    "id": tool_id
                }
            }
            
            # Add thoughtSignature for Gemini 3
            if self._is_gemini_3(model):
                sig = tc.get("thought_signature")
                if not sig and tool_id and self._enable_signature_cache:
                    sig = self._signature_cache.retrieve(tool_id)
                
                if sig:
                    func_part["thoughtSignature"] = sig
                else:
                    func_part["thoughtSignature"] = "skip_thought_signature_validator"
                    lib_logger.warning(f"Missing thoughtSignature for {tool_id}, using bypass")
            
            parts.append(func_part)
        
        return parts
    
    def _get_cached_thinking(
        self,
        content: Any,
        tool_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Retrieve and format cached thinking content for Claude."""
        parts = []
        msg_text = content if isinstance(content, str) else ""
        cache_key = self._generate_thinking_cache_key(msg_text, tool_calls)
        
        if not cache_key:
            return parts
        
        cached_json = self._thinking_cache.retrieve(cache_key)
        if not cached_json:
            return parts
        
        try:
            thinking_data = json.loads(cached_json)
            thinking_text = thinking_data.get("thinking_text", "")
            sig = thinking_data.get("thought_signature", "")
            
            if thinking_text:
                thinking_part = {
                    "text": thinking_text,
                    "thought": True,
                    "thoughtSignature": sig or "skip_thought_signature_validator"
                }
                parts.append(thinking_part)
                lib_logger.debug(f"Injected {len(thinking_text)} chars of thinking")
        except json.JSONDecodeError:
            lib_logger.warning(f"Failed to parse cached thinking: {cache_key}")
        
        return parts
    
    def _transform_tool_message(
        self,
        msg: Dict[str, Any],
        model: str,
        tool_id_to_name: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform tool response message."""
        tool_id = msg.get("tool_call_id", "")
        func_name = tool_id_to_name.get(tool_id, "unknown_function")
        content = msg.get("content", "{}")
        
        # Add prefix for Gemini 3
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
            func_name = f"{self._gemini3_tool_prefix}{func_name}"
        
        try:
            parsed_content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed_content = content
        
        return [{
            "functionResponse": {
                "name": func_name,
                "response": {"result": parsed_content},
                "id": tool_id
            }
        }]
    
    # =========================================================================
    # TOOL RESPONSE GROUPING
    # =========================================================================
    
    def _fix_tool_response_grouping(
        self,
        contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Antigravity compatibility.
        
        Converts linear format (call, response, call, response)
        to grouped format (model with calls, user with all responses).
        """
        new_contents = []
        pending_groups = []
        collected_responses = []
        
        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])
            
            response_parts = [p for p in parts if "functionResponse" in p]
            
            if response_parts:
                collected_responses.extend(response_parts)
                
                # Try to satisfy pending groups
                for i in range(len(pending_groups) - 1, -1, -1):
                    group = pending_groups[i]
                    if len(collected_responses) >= group["count"]:
                        group_responses = collected_responses[:group["count"]]
                        collected_responses = collected_responses[group["count"]:]
                        new_contents.append({"parts": group_responses, "role": "user"})
                        pending_groups.pop(i)
                        break
                continue
            
            if role == "model":
                func_calls = [p for p in parts if "functionCall" in p]
                new_contents.append(content)
                if func_calls:
                    pending_groups.append({"count": len(func_calls)})
            else:
                new_contents.append(content)
        
        # Handle remaining groups
        for group in pending_groups:
            if len(collected_responses) >= group["count"]:
                group_responses = collected_responses[:group["count"]]
                collected_responses = collected_responses[group["count"]:]
                new_contents.append({"parts": group_responses, "role": "user"})
        
        return new_contents
    
    # =========================================================================
    # GEMINI 3 TOOL TRANSFORMATIONS
    # =========================================================================
    
    def _apply_gemini3_namespace(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add namespace prefix to tool names for Gemini 3."""
        if not tools:
            return tools
        
        modified = copy.deepcopy(tools)
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                name = func_decl.get("name", "")
                if name:
                    func_decl["name"] = f"{self._gemini3_tool_prefix}{name}"
        
        return modified
    
    def _inject_signature_into_descriptions(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Inject parameter signatures into tool descriptions for Gemini 3."""
        if not tools:
            return tools
        
        modified = copy.deepcopy(tools)
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                schema = func_decl.get("parametersJsonSchema", {})
                if not schema:
                    continue
                
                required = schema.get("required", [])
                properties = schema.get("properties", {})
                
                if not properties:
                    continue
                
                param_list = []
                for prop_name, prop_data in properties.items():
                    if not isinstance(prop_data, dict):
                        continue
                    
                    type_hint = self._format_type_hint(prop_data)
                    is_required = prop_name in required
                    param_list.append(
                        f"{prop_name} ({type_hint}{', REQUIRED' if is_required else ''})"
                    )
                
                if param_list:
                    sig_str = self._gemini3_description_prompt.replace(
                        "{params}", ", ".join(param_list)
                    )
                    func_decl["description"] = func_decl.get("description", "") + sig_str
        
        return modified
    
    def _format_type_hint(self, prop_data: Dict[str, Any]) -> str:
        """Format a type hint for a property schema."""
        type_hint = prop_data.get("type", "unknown")
        
        if type_hint == "array":
            items = prop_data.get("items", {})
            if isinstance(items, dict):
                item_type = items.get("type", "unknown")
                if item_type == "object":
                    nested_props = items.get("properties", {})
                    nested_req = items.get("required", [])
                    if nested_props:
                        nested_list = []
                        for n, d in nested_props.items():
                            if isinstance(d, dict):
                                t = d.get("type", "unknown")
                                req = " REQUIRED" if n in nested_req else ""
                                nested_list.append(f"{n}: {t}{req}")
                        return f"ARRAY_OF_OBJECTS[{', '.join(nested_list)}]"
                    return "ARRAY_OF_OBJECTS"
                return f"ARRAY_OF_{item_type.upper()}"
            return "ARRAY"
        
        return type_hint
    
    def _strip_gemini3_prefix(self, name: str) -> str:
        """Strip the Gemini 3 namespace prefix from a tool name."""
        if name and name.startswith(self._gemini3_tool_prefix):
            return name[len(self._gemini3_tool_prefix):]
        return name
    
    # =========================================================================
    # REQUEST TRANSFORMATION
    # =========================================================================
    
    def _build_tools_payload(
        self,
        tools: Optional[List[Dict[str, Any]]],
        _model: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Build Gemini-format tools from OpenAI tools."""
        if not tools:
            return None
        
        gemini_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            
            func = tool.get("function", {})
            params = func.get("parameters")
            
            func_decl = {
                "name": func.get("name", ""),
                "description": func.get("description", "")
            }
            
            if params and isinstance(params, dict):
                schema = dict(params)
                schema.pop("$schema", None)
                schema.pop("strict", None)
                schema = _normalize_type_arrays(schema)
                func_decl["parametersJsonSchema"] = schema
            else:
                func_decl["parametersJsonSchema"] = {"type": "object", "properties": {}}
            
            gemini_tools.append({"functionDeclarations": [func_decl]})
        
        return gemini_tools or None
    
    def _transform_to_antigravity_format(
        self,
        gemini_payload: Dict[str, Any],
        model: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transform Gemini CLI payload to complete Antigravity format.
        
        Args:
            gemini_payload: Request in Gemini CLI format
            model: Model name (public alias)
            max_tokens: Max output tokens (including thinking)
        """
        internal_model = self._alias_to_internal(model)
        
        # Wrap in Antigravity envelope
        antigravity_payload = {
            "project": _generate_project_id(),
            "userAgent": "antigravity",
            "requestId": _generate_request_id(),
            "model": internal_model,
            "request": copy.deepcopy(gemini_payload)
        }
        
        # Add session ID
        antigravity_payload["request"]["sessionId"] = _generate_session_id()
        
        # Remove unsupported fields
        antigravity_payload["request"].pop("safetySettings", None)
        
        # Handle max_tokens - only apply to Claude, or if explicitly set for others
        gen_config = antigravity_payload["request"].get("generationConfig", {})
        is_claude = self._is_claude(model)
        
        if max_tokens is not None:
            # Explicitly set in request - apply to all models
            gen_config["maxOutputTokens"] = max_tokens
        elif is_claude:
            # Claude model without explicit max_tokens - use default
            gen_config["maxOutputTokens"] = DEFAULT_MAX_OUTPUT_TOKENS
        # For non-Claude models without explicit max_tokens, don't set it
        
        antigravity_payload["request"]["generationConfig"] = gen_config
        
        # Set toolConfig mode
        tool_config = antigravity_payload["request"].setdefault("toolConfig", {})
        func_config = tool_config.setdefault("functionCallingConfig", {})
        func_config["mode"] = "VALIDATED"
        
        # Handle Gemini 3 thinking logic
        if not internal_model.startswith("gemini-3-"):
            thinking_config = gen_config.get("thinkingConfig", {})
            if "thinkingLevel" in thinking_config:
                del thinking_config["thinkingLevel"]
                thinking_config["thinkingBudget"] = -1
        
        # Add thoughtSignature to function calls for Gemini 3
        if internal_model.startswith("gemini-3-"):
            for content in antigravity_payload["request"].get("contents", []):
                if content.get("role") == "model":
                    for part in content.get("parts", []):
                        if "functionCall" in part and "thoughtSignature" not in part:
                            part["thoughtSignature"] = "skip_thought_signature_validator"
        
        # Claude-specific tool schema transformation
        if internal_model.startswith("claude-sonnet-"):
            self._apply_claude_tool_transform(antigravity_payload)
        
        return antigravity_payload
    
    def _apply_claude_tool_transform(self, payload: Dict[str, Any]) -> None:
        """Apply Claude-specific tool schema transformations."""
        tools = payload["request"].get("tools", [])
        for tool in tools:
            for func_decl in tool.get("functionDeclarations", []):
                if "parametersJsonSchema" in func_decl:
                    params = func_decl["parametersJsonSchema"]
                    params = _clean_claude_schema(params) if isinstance(params, dict) else params
                    func_decl["parameters"] = params
                    del func_decl["parametersJsonSchema"]
    
    # =========================================================================
    # RESPONSE TRANSFORMATION
    # =========================================================================
    
    def _unwrap_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Gemini response from Antigravity envelope."""
        return response.get("response", response)
    
    def _gemini_to_openai_chunk(
        self,
        chunk: Dict[str, Any],
        model: str,
        accumulator: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert Gemini response chunk to OpenAI streaming format.
        
        Args:
            chunk: Gemini API response chunk
            model: Model name
            accumulator: Optional dict to accumulate data for post-processing
        """
        candidates = chunk.get("candidates", [])
        if not candidates:
            return {}
        
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        
        text_content = ""
        reasoning_content = ""
        tool_calls = []
        first_sig_seen = False
        tool_idx = 0
        
        for part in content_parts:
            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = part.get("thought") is True or str(part.get("thought")).lower() == 'true'
            
            # Accumulate signature for Claude caching
            if has_sig and is_thought and accumulator is not None:
                accumulator["thought_signature"] = part["thoughtSignature"]
            
            # Skip standalone signature parts
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue
            
            if has_text:
                text = part["text"]
                if is_thought:
                    reasoning_content += text
                    if accumulator is not None:
                        accumulator["reasoning_content"] += text
                else:
                    text_content += text
                    if accumulator is not None:
                        accumulator["text_content"] += text
            
            if has_func:
                tool_call = self._extract_tool_call(part, model, tool_idx, accumulator)
                
                if has_sig and not first_sig_seen:
                    first_sig_seen = True
                    self._handle_tool_signature(tool_call, part["thoughtSignature"])
                
                tool_calls.append(tool_call)
                tool_idx += 1
        
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
        finish_reason = self._map_finish_reason(candidate.get("finishReason"), bool(tool_calls))
        if finish_reason and accumulator is not None:
            accumulator["is_complete"] = True
        
        # Build usage
        usage = self._build_usage(chunk.get("usageMetadata", {}))
        
        response = {
            "id": chunk.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }
        
        if usage:
            response["usage"] = usage
        
        return response
    
    def _gemini_to_openai_non_streaming(
        self,
        response: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI non-streaming format."""
        candidates = response.get("candidates", [])
        if not candidates:
            return {}
        
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        
        text_content = ""
        reasoning_content = ""
        tool_calls = []
        thought_sig = ""
        first_sig_seen = False
        
        for part in content_parts:
            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = part.get("thought") is True or str(part.get("thought")).lower() == 'true'
            
            if has_sig and is_thought:
                thought_sig = part["thoughtSignature"]
            
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue
            
            if has_text:
                if is_thought:
                    reasoning_content += part["text"]
                else:
                    text_content += part["text"]
            
            if has_func:
                tool_call = self._extract_tool_call(part, model, len(tool_calls))
                
                if has_sig and not first_sig_seen:
                    first_sig_seen = True
                    self._handle_tool_signature(tool_call, part["thoughtSignature"])
                
                tool_calls.append(tool_call)
        
        # Cache Claude thinking
        if reasoning_content and self._is_claude(model) and self._enable_signature_cache:
            self._cache_thinking(reasoning_content, thought_sig, text_content, tool_calls)
        
        # Build message
        message = {"role": "assistant"}
        if text_content:
            message["content"] = text_content
        elif not tool_calls:
            message["content"] = ""
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
            message.pop("content", None)
        
        finish_reason = self._map_finish_reason(candidate.get("finishReason"), bool(tool_calls))
        usage = self._build_usage(response.get("usageMetadata", {}))
        
        result = {
            "id": response.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]
        }
        
        if usage:
            result["usage"] = usage
        
        return result
    
    def _extract_tool_call(
        self,
        part: Dict[str, Any],
        model: str,
        index: int,
        accumulator: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract and format a tool call from a response part."""
        func_call = part["functionCall"]
        tool_id = func_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        
        tool_name = func_call.get("name", "")
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
            tool_name = self._strip_gemini3_prefix(tool_name)
        
        raw_args = func_call.get("args", {})
        parsed_args = _recursively_parse_json_strings(raw_args)
        
        tool_call = {
            "id": tool_id,
            "type": "function",
            "index": index,
            "function": {
                "name": tool_name,
                "arguments": json.dumps(parsed_args)
            }
        }
        
        if accumulator is not None:
            accumulator["tool_calls"].append(tool_call)
        
        return tool_call
    
    def _handle_tool_signature(self, tool_call: Dict, signature: str) -> None:
        """Handle thoughtSignature for a tool call."""
        tool_id = tool_call["id"]
        
        if self._enable_signature_cache:
            self._signature_cache.store(tool_id, signature)
            lib_logger.debug(f"Stored signature for {tool_id}")
        
        if self._preserve_signatures_in_client:
            tool_call["thought_signature"] = signature
    
    def _map_finish_reason(
        self,
        gemini_reason: Optional[str],
        has_tool_calls: bool
    ) -> Optional[str]:
        """Map Gemini finish reason to OpenAI format."""
        if not gemini_reason:
            return None
        reason = FINISH_REASON_MAP.get(gemini_reason, "stop")
        return "tool_calls" if has_tool_calls else reason
    
    def _build_usage(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build usage dict from Gemini usage metadata."""
        if not metadata:
            return None
        
        prompt = metadata.get("promptTokenCount", 0)
        thoughts = metadata.get("thoughtsTokenCount", 0)
        completion = metadata.get("candidatesTokenCount", 0)
        
        usage = {
            "prompt_tokens": prompt + thoughts,
            "completion_tokens": completion,
            "total_tokens": metadata.get("totalTokenCount", 0)
        }
        
        if thoughts > 0:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}
        
        return usage
    
    def _cache_thinking(
        self,
        reasoning: str,
        signature: str,
        text: str,
        tool_calls: List[Dict]
    ) -> None:
        """Cache Claude thinking content."""
        cache_key = self._generate_thinking_cache_key(text, tool_calls)
        if not cache_key:
            return
        
        data = {
            "thinking_text": reasoning,
            "thought_signature": signature,
            "text_preview": text[:100] if text else "",
            "tool_ids": [tc.get("id", "") for tc in tool_calls],
            "timestamp": time.time()
        }
        
        self._thinking_cache.store(cache_key, json.dumps(data))
        lib_logger.info(f"Cached thinking: {cache_key[:50]}...")
    
    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================
    
    async def get_valid_token(self, credential_identifier: str) -> str:
        """Get a valid access token for the credential."""
        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_identifier, creds)
        return creds['access_token']
    
    def has_custom_logic(self) -> bool:
        """Antigravity uses custom translation logic."""
        return True
    
    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Get OAuth authorization header."""
        token = await self.get_valid_token(credential_identifier)
        return {"Authorization": f"Bearer {token}"}
    
    async def get_models(
        self,
        api_key: str,
        client: httpx.AsyncClient
    ) -> List[str]:
        """Fetch available models from Antigravity."""
        if not self._enable_dynamic_models:
            lib_logger.debug("Using hardcoded model list")
            return [f"antigravity/{m}" for m in AVAILABLE_MODELS]
        
        try:
            token = await self.get_valid_token(api_key)
            url = f"{self._get_base_url()}/fetchAvailableModels"
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "project": _generate_project_id(),
                "requestId": _generate_request_id(),
                "userAgent": "antigravity"
            }
            
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model_info in data.get("models", []):
                internal = model_info.get("name", "").replace("models/", "")
                if internal:
                    public = self._internal_to_alias(internal)
                    if public:
                        models.append(f"antigravity/{public}")
            
            if models:
                lib_logger.info(f"Discovered {len(models)} models")
                return models
        except Exception as e:
            lib_logger.warning(f"Dynamic model discovery failed: {e}")
        
        return [f"antigravity/{m}" for m in AVAILABLE_MODELS]
    
    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests for Antigravity.
        
        Main entry point that:
        1. Extracts parameters and transforms messages
        2. Builds Antigravity request payload
        3. Makes API call with fallback logic
        4. Transforms response to OpenAI format
        """
        # Extract parameters
        model = self._strip_provider_prefix(kwargs.get("model", "gemini-2.5-pro"))
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("api_key", ""))
        tools = kwargs.get("tools")
        reasoning_effort = kwargs.get("reasoning_effort")
        top_p = kwargs.get("top_p")
        max_tokens = kwargs.get("max_tokens")
        custom_budget = kwargs.get("custom_reasoning_budget", False)
        enable_logging = kwargs.pop("enable_request_logging", False)
        
        # Create logger
        file_logger = AntigravityFileLogger(model, enable_logging)
        
        # Transform messages
        system_instruction, gemini_contents = self._transform_messages(messages, model)
        gemini_contents = self._fix_tool_response_grouping(gemini_contents)
        
        # Build payload
        gemini_payload = {"contents": gemini_contents}
        
        if system_instruction:
            gemini_payload["system_instruction"] = system_instruction
        
        # Inject Gemini 3 system instruction
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix and tools:
            self._inject_gemini3_system_instruction(gemini_payload)
        
        # Add generation config
        gen_config = {}
        if top_p is not None:
            gen_config["topP"] = top_p
        
        thinking_config = self._get_thinking_config(reasoning_effort, model, custom_budget)
        if thinking_config:
            gen_config.setdefault("thinkingConfig", {}).update(thinking_config)
        
        if gen_config:
            gemini_payload["generationConfig"] = gen_config
        
        # Add tools
        gemini_tools = self._build_tools_payload(tools, model)
        if gemini_tools:
            gemini_payload["tools"] = gemini_tools
            
            # Apply Gemini 3 tool transformations
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                gemini_payload["tools"] = self._apply_gemini3_namespace(gemini_payload["tools"])
                gemini_payload["tools"] = self._inject_signature_into_descriptions(gemini_payload["tools"])
        
        # Transform to Antigravity format
        payload = self._transform_to_antigravity_format(gemini_payload, model, max_tokens)
        file_logger.log_request(payload)
        
        # Make API call
        token = await self.get_valid_token(credential_path)
        base_url = self._get_base_url()
        endpoint = ":streamGenerateContent" if stream else ":generateContent"
        url = f"{base_url}{endpoint}"
        
        if stream:
            url = f"{url}?alt=sse"
        
        parsed = urlparse(base_url)
        host = parsed.netloc or base_url.replace("https://", "").replace("http://", "").rstrip("/")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Host": host,
            "User-Agent": "antigravity/1.11.5",
            "Accept": "text/event-stream" if stream else "application/json"
        }
        
        try:
            if stream:
                return self._handle_streaming(client, url, headers, payload, model, file_logger)
            else:
                return await self._handle_non_streaming(client, url, headers, payload, model, file_logger)
        except Exception as e:
            if self._try_next_base_url():
                lib_logger.warning(f"Retrying with fallback URL: {e}")
                url = f"{self._get_base_url()}{endpoint}"
                if stream:
                    return self._handle_streaming(client, url, headers, payload, model, file_logger)
                else:
                    return await self._handle_non_streaming(client, url, headers, payload, model, file_logger)
            raise
    
    def _inject_gemini3_system_instruction(self, payload: Dict[str, Any]) -> None:
        """Inject Gemini 3 system instruction for tool fix."""
        if not self._gemini3_system_instruction:
            return
        
        instruction_part = {"text": self._gemini3_system_instruction}
        
        if "system_instruction" in payload:
            existing = payload["system_instruction"]
            if isinstance(existing, dict) and "parts" in existing:
                existing["parts"].insert(0, instruction_part)
            else:
                payload["system_instruction"] = {
                    "role": "user",
                    "parts": [instruction_part, {"text": str(existing)}]
                }
        else:
            payload["system_instruction"] = {"role": "user", "parts": [instruction_part]}
    
    async def _handle_non_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityFileLogger] = None
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        response = await client.post(url, headers=headers, json=payload, timeout=120.0)
        response.raise_for_status()
        
        data = response.json()
        if file_logger:
            file_logger.log_final_response(data)
        
        gemini_response = self._unwrap_response(data)
        openai_response = self._gemini_to_openai_non_streaming(gemini_response, model)
        
        return litellm.ModelResponse(**openai_response)
    
    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityFileLogger] = None
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        accumulator = {
            "reasoning_content": "",
            "thought_signature": "",
            "text_content": "",
            "tool_calls": [],
            "is_complete": False
        } if self._is_claude(model) and self._enable_signature_cache else None
        
        async with client.stream("POST", url, headers=headers, json=payload, timeout=120.0) as response:
            if response.status_code >= 400:
                try:
                    error_body = await response.aread()
                    lib_logger.error(f"API error {response.status_code}: {error_body.decode()}")
                except Exception:
                    pass
            
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if file_logger:
                    file_logger.log_response_chunk(line)
                
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        gemini_chunk = self._unwrap_response(chunk)
                        openai_chunk = self._gemini_to_openai_chunk(gemini_chunk, model, accumulator)
                        
                        yield litellm.ModelResponse(**openai_chunk)
                    except json.JSONDecodeError:
                        if file_logger:
                            file_logger.log_error(f"Parse error: {data_str[:100]}")
                        continue
        
        # Cache Claude thinking after stream completes
        if accumulator and accumulator.get("reasoning_content"):
            self._cache_thinking(
                accumulator["reasoning_content"],
                accumulator["thought_signature"],
                accumulator["text_content"],
                accumulator["tool_calls"]
            )
    
    async def count_tokens(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        _litellm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """Count tokens for the given prompt using Antigravity :countTokens endpoint."""
        try:
            token = await self.get_valid_token(credential_path)
            internal_model = self._alias_to_internal(model)
            
            system_instruction, contents = self._transform_messages(messages, internal_model)
            
            gemini_payload = {"contents": contents}
            if system_instruction:
                gemini_payload["systemInstruction"] = system_instruction
            
            gemini_tools = self._build_tools_payload(tools, model)
            if gemini_tools:
                gemini_payload["tools"] = gemini_tools
            
            antigravity_payload = {
                "project": _generate_project_id(),
                "userAgent": "antigravity",
                "requestId": _generate_request_id(),
                "model": internal_model,
                "request": gemini_payload
            }
            
            url = f"{self._get_base_url()}:countTokens"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = await client.post(url, headers=headers, json=antigravity_payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            unwrapped = self._unwrap_response(data)
            total = unwrapped.get('totalTokens', 0)
            
            return {'prompt_tokens': total, 'total_tokens': total}
        except Exception as e:
            lib_logger.error(f"Token counting failed: {e}")
            return {'prompt_tokens': 0, 'total_tokens': 0}