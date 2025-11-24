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
import tempfile
import shutil
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

# Cache configuration
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "cache"
ANTIGRAVITY_CACHE_DIR = CACHE_DIR / "antigravity"
ANTIGRAVITY_CACHE_FILE = ANTIGRAVITY_CACHE_DIR / "thought_signatures.json"


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
    - Dual-TTL system: 1hr memory, 24hr disk
    - Async disk persistence with batched writes
    - Background cleanup task for expired entries
    - Thread-safe for concurrent access
    - Fallback to disk when not in memory
    - High concurrency support with asyncio locks
    """
    
    def __init__(self, memory_ttl_seconds: int = 3600, disk_ttl_seconds: int = 86400):
        """
        Initialize the signature cache with disk persistence.
        
        Args:
            memory_ttl_seconds: Time-to-live for memory cache entries (default: 1 hour)
            disk_ttl_seconds: Time-to-live for disk cache entries (default: 24 hours)
        """
        # In-memory cache: {call_id: (signature, timestamp)}
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._memory_ttl = memory_ttl_seconds
        self._disk_ttl = disk_ttl_seconds
        self._lock = asyncio.Lock()
        self._disk_lock = asyncio.Lock()
        
        # Disk persistence configuration
        self._cache_file = ANTIGRAVITY_CACHE_FILE
        self._enable_disk_persistence = os.getenv(
            "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE",
            "true"
        ).lower() in ("true", "1", "yes")
        
        # Async write configuration
        self._dirty = False  # Flag for pending writes
        self._write_interval = int(os.getenv("ANTIGRAVITY_CACHE_WRITE_INTERVAL", "60"))
        self._cleanup_interval = int(os.getenv("ANTIGRAVITY_CACHE_CLEANUP_INTERVAL", "1800"))
        
        # Background tasks
        self._writer_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "writes": 0
        }
        
        # Initialize
        if self._enable_disk_persistence:
            lib_logger.debug(
                f"ThoughtSignatureCache: Disk persistence ENABLED "
                f"(memory_ttl={memory_ttl_seconds}s, disk_ttl={disk_ttl_seconds}s, "
                f"write_interval={self._write_interval}s)"
            )
            # Schedule async initialization
            asyncio.create_task(self._async_init())
        else:
            lib_logger.debug("ThoughtSignatureCache: Disk persistence DISABLED (memory-only mode)")
    
    async def _async_init(self):
        """Async initialization: load from disk and start background tasks."""
        try:
            await self._load_from_disk()
            await self._start_background_tasks()
        except Exception as e:
            lib_logger.error(f"ThoughtSignatureCache async init failed: {e}")
    
    async def _load_from_disk(self):
        """Load cache from disk file (with TTL validation)."""
        if not self._enable_disk_persistence:
            return
        
        if not self._cache_file.exists():
            lib_logger.debug("No existing cache file found, starting fresh")
            return
        
        try:
            async with self._disk_lock:
                # Read cache file
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate version
                if data.get("version") != "1.0":
                    lib_logger.warning(f"Cache file version mismatch, ignoring")
                    return
                
                # Load entries with disk TTL validation
                now = time.time()
                entries = data.get("entries", {})
                loaded = 0
                expired = 0
                
                for call_id, entry in entries.items():
                    timestamp = entry.get("timestamp", 0)
                    age = now - timestamp
                    
                    # Check against DISK TTL (24 hours)
                    if age <= self._disk_ttl:
                        signature = entry.get("signature", "")
                        if signature:
                            self._cache[call_id] = (signature, timestamp)
                            loaded += 1
                    else:
                        expired += 1
                
                lib_logger.debug(
                    f"ThoughtSignatureCache: Loaded {loaded} signatures from disk "
                    f"({expired} expired entries removed)"
                )
                
        except json.JSONDecodeError as e:
            lib_logger.warning(f"Cache file corrupted, starting fresh: {e}")
        except Exception as e:
            lib_logger.error(f"Failed to load cache from disk: {e}")
    
    async def _save_to_disk(self):
        """Persist cache to disk using atomic write."""
        if not self._enable_disk_persistence:
            return
        
        try:
            async with self._disk_lock:
                # Ensure cache directory exists
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Build cache data structure
                cache_data = {
                    "version": "1.0",
                    "memory_ttl_seconds": self._memory_ttl,
                    "disk_ttl_seconds": self._disk_ttl,
                    "entries": {
                        call_id: {
                            "signature": sig,
                            "timestamp": ts
                        }
                        for call_id, (sig, ts) in self._cache.items()
                    },
                    "statistics": {
                        "total_entries": len(self._cache),
                        "last_write": time.time(),
                        "memory_hits": self._stats["memory_hits"],
                        "disk_hits": self._stats["disk_hits"],
                        "misses": self._stats["misses"],
                        "writes": self._stats["writes"]
                    }
                }
                
                # Atomic write using tempfile pattern (same as OAuth credentials)
                parent_dir = self._cache_file.parent
                tmp_fd = None
                tmp_path = None
                
                try:
                    # Create temp file in same directory
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        dir=parent_dir,
                        prefix='.tmp_',
                        suffix='.json',
                        text=True
                    )
                    
                    # Write JSON to temp file
                    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                        tmp_fd = None  # fdopen closes the fd
                    
                    # Set secure permissions (owner read/write only)
                    try:
                        os.chmod(tmp_path, 0o600)
                    except (OSError, AttributeError):
                        # Windows may not support chmod, ignore
                        pass
                    
                    # Atomic move (overwrites target if exists)
                    shutil.move(tmp_path, self._cache_file)
                    tmp_path = None  # Successfully moved
                    
                    self._stats["writes"] += 1
                    lib_logger.debug(f"Saved {len(self._cache)} signatures to disk")
                    
                except Exception as e:
                    lib_logger.error(f"Failed to save cache to disk: {e}")
                    # Clean up temp file if it still exists
                    if tmp_fd is not None:
                        try:
                            os.close(tmp_fd)
                        except:
                            pass
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    raise
                    
        except Exception as e:
            lib_logger.error(f"Disk save operation failed: {e}")
    
    async def _start_background_tasks(self):
        """Start background writer and cleanup tasks."""
        if not self._enable_disk_persistence or self._running:
            return
        
        self._running = True
        
        # Start async writer task
        self._writer_task = asyncio.create_task(self._writer_loop())
        lib_logger.debug(f"Started background writer task (interval: {self._write_interval}s)")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        lib_logger.debug(f"Started background cleanup task (interval: {self._cleanup_interval}s)")
    
    async def _writer_loop(self):
        """Background task: periodically flush dirty cache to disk."""
        try:
            while self._running:
                await asyncio.sleep(self._write_interval)
                
                if self._dirty:
                    try:
                        await self._save_to_disk()
                        self._dirty = False
                    except Exception as e:
                        lib_logger.error(f"Background writer error: {e}")
        except asyncio.CancelledError:
            lib_logger.debug("Background writer task cancelled")
        except Exception as e:
            lib_logger.error(f"Background writer crashed: {e}")
    
    async def _cleanup_loop(self):
        """Background task: periodically clean up expired entries."""
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval)
                
                try:
                    await self._cleanup_expired()
                except Exception as e:
                    lib_logger.error(f"Background cleanup error: {e}")
        except asyncio.CancelledError:
            lib_logger.debug("Background cleanup task cancelled")
        except Exception as e:
            lib_logger.error(f"Background cleanup crashed: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries from memory cache (based on memory TTL)."""
        async with self._lock:
            now = time.time()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if now - ts > self._memory_ttl
            ]
            
            for k in expired:
                del self._cache[k]
            
            if expired:
                self._dirty = True  # Mark for disk save
                lib_logger.debug(f"Cleaned up {len(expired)} expired signatures from memory")
    
    def store(self, tool_call_id: str, signature: str):
        """
        Store a signature for a tool call ID (sync wrapper for async storage).
        
        Args:
            tool_call_id: Unique identifier for the tool call
            signature: Encrypted thoughtSignature from Antigravity API
        """
        # Create task for async storage
        asyncio.create_task(self._async_store(tool_call_id, signature))
    
    async def _async_store(self, tool_call_id: str, signature: str):
        """Async implementation of store."""
        async with self._lock:
            self._cache[tool_call_id] = (signature, time.time())
            self._dirty = True  # Mark for disk write
    
    def retrieve(self, tool_call_id: str) -> Optional[str]:
        """
        Retrieve signature for a tool call ID (sync method).
        
        Args:
            tool_call_id: Unique identifier for the tool call
            
        Returns:
            The signature if found and not expired, None otherwise
        """
        # Try memory cache first (sync access is safe for read)
        if tool_call_id in self._cache:
            signature, timestamp = self._cache[tool_call_id]
            if time.time() - timestamp <= self._memory_ttl:
                self._stats["memory_hits"] += 1
                return signature
            else:
                # Expired in memory, remove it
                del self._cache[tool_call_id]
                self._dirty = True
        
        # Not in memory - schedule async disk lookup
        # For now, return None (disk fallback happens on next request)
        # This is intentional to avoid blocking the sync caller
        self._stats["misses"] += 1
        
        # Schedule background disk check (non-blocking)
        if self._enable_disk_persistence:
            asyncio.create_task(self._check_disk_fallback(tool_call_id))
        
        return None
    
    async def _check_disk_fallback(self, tool_call_id: str):
        """Check disk for signature and load into memory if found."""
        try:
            # Reload from disk if file exists
            if self._cache_file.exists():
                async with self._disk_lock:
                    with open(self._cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    entries = data.get("entries", {})
                    if tool_call_id in entries:
                        entry = entries[tool_call_id]
                        timestamp = entry.get("timestamp", 0)
                        
                        # Check disk TTL (24 hours)
                        if time.time() - timestamp <= self._disk_ttl:
                            signature = entry.get("signature", "")
                            if signature:
                                # Load into memory cache
                                async with self._lock:
                                    self._cache[tool_call_id] = (signature, timestamp)
                                    self._stats["disk_hits"] += 1
                                lib_logger.debug(f"Loaded signature {tool_call_id} from disk")
        except Exception as e:
            lib_logger.debug(f"Disk fallback check failed: {e}")
    
    async def clear(self):
        """Clear all cached signatures (memory and disk)."""
        async with self._lock:
            self._cache.clear()
            self._dirty = True
        
        if self._enable_disk_persistence:
            await self._save_to_disk()
    
    async def shutdown(self):
        """Graceful shutdown: flush pending writes and stop background tasks."""
        lib_logger.info("ThoughtSignatureCache shutting down...")
        
        # Stop background tasks
        self._running = False
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Flush pending writes
        if self._dirty and self._enable_disk_persistence:
            lib_logger.info("Flushing pending cache writes...")
            await self._save_to_disk()
        
        lib_logger.info(
            f"ThoughtSignatureCache shutdown complete "
            f"(stats: mem_hits={self._stats['memory_hits']}, "
            f"disk_hits={self._stats['disk_hits']}, "
            f"misses={self._stats['misses']}, "
            f"writes={self._stats['writes']})"
        )


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
        memory_ttl = int(os.getenv("ANTIGRAVITY_SIGNATURE_CACHE_TTL", "3600"))
        disk_ttl = int(os.getenv("ANTIGRAVITY_SIGNATURE_DISK_TTL", "86400"))
        self._signature_cache = ThoughtSignatureCache(
            memory_ttl_seconds=memory_ttl,
            disk_ttl_seconds=disk_ttl
        )
        
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
            lib_logger.debug("Antigravity: thoughtSignature client passthrough ENABLED")
        else:
            lib_logger.debug("Antigravity: thoughtSignature client passthrough DISABLED")
        
        if self._enable_signature_cache:
            lib_logger.debug(f"Antigravity: thoughtSignature server-side cache ENABLED (memory_ttl={memory_ttl}s, disk_ttl={disk_ttl}s)")
        else:
            lib_logger.debug("Antigravity: thoughtSignature server-side cache DISABLED")
        
        if self._enable_dynamic_model_discovery:
            lib_logger.debug("Antigravity: Dynamic model discovery ENABLED (may fail if endpoint unavailable)")
        else:
            lib_logger.debug("Antigravity: Dynamic model discovery DISABLED (using hardcoded model list)")
        
        # Check if Gemini 3 tool fix is enabled (default: ON for testing)
        # This applies the "Quad-Lock" catch-all strategy to prevent tool hallucination
        self._enable_gemini3_tool_fix = os.getenv(
            "ANTIGRAVITY_GEMINI3_TOOL_FIX",
            "true"  # Default ON - applies namespace + signature injection
        ).lower() in ("true", "1", "yes")
        
        # Gemini 3 fix configuration - customize the fix components
        # Namespace prefix for tool names (Strategy 1)
        self._gemini3_tool_prefix = os.getenv(
            "ANTIGRAVITY_GEMINI3_TOOL_PREFIX",
            "gemini3_"  # Default prefix
        )
        
        # Description prompt format (Strategy 2)
        # Use {params} as placeholder for parameter list
        self._gemini3_description_prompt = os.getenv(
            "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT",
            "\n\nSTRICT PARAMETERS: {params}."  # Default format
        )
        
        # System instruction text (Strategy 3)
        # Set to empty string to disable system instruction injection
        self._gemini3_system_instruction = os.getenv(
            "ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION",
            # Default: comprehensive tool usage instructions
            """CRITICAL TOOL USAGE INSTRUCTIONS:
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
        )
        
        if self._enable_gemini3_tool_fix:
            lib_logger.debug(f"Antigravity: Gemini 3 tool fix ENABLED")
            lib_logger.debug(f"  - Namespace prefix: '{self._gemini3_tool_prefix}'")
            lib_logger.debug(f"  - Description prompt: '{self._gemini3_description_prompt[:50]}...'")
            lib_logger.debug(f"  - System instruction: {'ENABLED' if self._gemini3_system_instruction else 'DISABLED'} ({len(self._gemini3_system_instruction)} chars)")
        else:
            lib_logger.debug("Antigravity: Gemini 3 tool fix DISABLED (using default tool schemas)")



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
        Handles thoughtSignature preservation with 3-tier fallback (GEMINI 3 ONLY):
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
                            
                            # Get function name and add configured prefix if needed (Gemini 3 specific)
                            function_name = tool_call["function"]["name"]
                            if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix:
                                # Client sends original names, we need to prefix for API consistency
                                function_name = f"{self._gemini3_tool_prefix}{function_name}"
                            
                            func_call_part = {
                                "functionCall": {
                                    "name": function_name,
                                    "args": args_dict,
                                    "id": tool_call_id  # ← ADD THIS LINE - Antigravity needs it for Claude!
                                }
                            }
                            
                            # thoughtSignature handling (GEMINI 3 ONLY)
                            # Claude and other models don't support this field!
                            if self._is_gemini_3_model(model):
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
                
                # Add configured prefix to function response name if needed (Gemini 3 specific)
                if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix:
                    # Client sends responses for original names, we need to prefix for API consistency
                    function_name = f"{self._gemini3_tool_prefix}{function_name}"
                
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
                        },
                        "id": tool_call_id  # ← ADD THIS LINE - Antigravity needs it for Claude!
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
    # GEMINI 3 TOOL TRANSFORMATION (Catch-All Fix for Hallucination)
    # ============================================================================

    def _apply_gemini3_namespace_to_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply namespace prefix to all tool names for Gemini 3 (Strategy 1: Namespace).
        
        This breaks the model's association with training data by prepending 'gemini3_'
        to every tool name, forcing it to read the schema definition instead of using
        its internal knowledge.
        
        Args:
            tools: List of tool definitions (Gemini format with functionDeclarations)
            
        Returns:
            Modified tools with prefixed names
        """
        if not tools:
            return tools
            
        modified_tools = copy.deepcopy(tools)
        
        for tool in modified_tools:
            function_declarations = tool.get("functionDeclarations", [])
            for func_decl in function_declarations:
                # Prepend namespace to tool name
                original_name = func_decl.get("name", "")
                if original_name:
                    func_decl["name"] = f"{self._gemini3_tool_prefix}{original_name}"
                    #lib_logger.debug(f"Gemini 3 namespace: {original_name} -> {self._gemini3_tool_prefix}{original_name}")
        
        return modified_tools

    def _inject_signature_into_tool_descriptions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Inject parameter signatures into tool descriptions for Gemini 3 (Strategy 2: Signature Injection).
        
        This strategy appends the expected parameter structure into the description text,
        creating a natural language enforcement of the schema that models pay close attention to.
        
        Args:
            tools: List of tool definitions (Gemini format with functionDeclarations)
            
        Returns:
            Modified tools with enriched descriptions
        """
        if not tools:
            return tools
            
        modified_tools = copy.deepcopy(tools)
        
        for tool in modified_tools:
            function_declarations = tool.get("functionDeclarations", [])
            for func_decl in function_declarations:
                # Get parameter schema
                schema = func_decl.get("parametersJsonSchema", {})
                if not schema or not isinstance(schema, dict):
                    continue
                
                # Extract required parameters
                required_params = schema.get("required", [])
                properties = schema.get("properties", {})
                
                if not properties:
                    continue
                
                # Build parameter list with type hints
                param_list = []
                for prop_name, prop_data in properties.items():
                    if not isinstance(prop_data, dict):
                        continue
                        
                    type_hint = prop_data.get("type", "unknown")
                    
                    # Handle arrays specially (critical for read_file/apply_diff issues)
                    if type_hint == "array":
                        items_schema = prop_data.get("items", {})
                        if isinstance(items_schema, dict):
                            item_type = items_schema.get("type", "unknown")
                            
                            # Check if it's an array of objects - RECURSE into nested properties
                            if item_type == "object":
                                # Extract nested properties for explicit visibility
                                nested_props = items_schema.get("properties", {})
                                nested_required = items_schema.get("required", [])
                                
                                if nested_props:
                                    # Build nested property list with types
                                    nested_list = []
                                    for nested_name, nested_data in nested_props.items():
                                        if not isinstance(nested_data, dict):
                                            continue
                                        nested_type = nested_data.get("type", "unknown")
                                        
                                        # Mark nested required fields
                                        if nested_name in nested_required:
                                            nested_list.append(f"{nested_name}: {nested_type} REQUIRED")
                                        else:
                                            nested_list.append(f"{nested_name}: {nested_type}")
                                    
                                    # Format as ARRAY_OF_OBJECTS[key1: type1, key2: type2]
                                    nested_str = ", ".join(nested_list)
                                    type_hint = f"ARRAY_OF_OBJECTS[{nested_str}]"
                                else:
                                    # No properties defined - just generic objects
                                    type_hint = "ARRAY_OF_OBJECTS"
                            else:
                                type_hint = f"ARRAY_OF_{item_type.upper()}"
                        else:
                            type_hint = "ARRAY"
                    
                    # Mark required parameters
                    if prop_name in required_params:
                        param_list.append(f"{prop_name} ({type_hint}, REQUIRED)")
                    else:
                        param_list.append(f"{prop_name} ({type_hint})")
                
                # Create strict signature string using configurable template
                # Replace {params} placeholder with actual parameter list
                signature_str = self._gemini3_description_prompt.replace("{params}", ", ".join(param_list))
                
                # Inject into description
                description = func_decl.get("description", "")
                func_decl["description"] = description + signature_str
                
                #lib_logger.debug(f"Gemini 3 signature injection: {func_decl.get('name', '')} - {len(param_list)} params")
        
        return modified_tools

    def _strip_gemini3_namespace_from_name(self, tool_name: str) -> str:
        """
        Strip the configured namespace prefix from a tool name.
        
        This reverses the namespace transformation applied in the request,
        ensuring the client receives the original tool names.
        
        Args:
            tool_name: Tool name (possibly with configured prefix)
            
        Returns:
            Original tool name without prefix
        """
        if tool_name and tool_name.startswith(self._gemini3_tool_prefix):
            return tool_name[len(self._gemini3_tool_prefix):]
        return tool_name

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
        
        # 6. Preserve/add thoughtSignature to function calls in model role content (GEMINI 3 ONLY)
        # thoughtSignature is a Gemini 3 feature for preserving reasoning context in multi-turn conversations
        # DO NOT add this for Claude or other models - they don't support it!
        if internal_model.startswith("gemini-3-"):
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
                
                # Get tool name and strip gemini3_ namespace if present (Gemini 3 specific)
                tool_name = func_call.get("name", "")
                if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix:
                    tool_name = self._strip_gemini3_namespace_from_name(tool_name)
                
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "index": tool_call_index,  # REQUIRED for OpenAI streaming format
                    "function": {
                        "name": tool_name,
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
    
    def _gemini_to_openai_non_streaming(self, gemini_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert a Gemini API response to OpenAI non-streaming format.
        
        This is specifically for non-streaming completions where we need 'message' instead of 'delta'.
        
        Args:
            gemini_response: Gemini API response
            model: Model name for Gemini 3 detection
            
        Returns:
            OpenAI-compatible non-streaming response
        """
        # Extract the main response structure
        candidates = gemini_response.get("candidates", [])
        if not candidates:
            return {}
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        content_parts = content.get("parts", [])
        
        # Build message components
        text_content = ""
        reasoning_content = ""
        tool_calls = []
        
        # Track if we've seen a signature yet (for parallel tool call handling)
        first_signature_seen = False
        
        for part in content_parts:
            has_function_call = "functionCall" in part
            has_text = "text" in part
            has_signature = "thoughtSignature" in part and part["thoughtSignature"]
            
            # Skip standalone signature parts
            if has_signature and not has_function_call and not has_text:
                continue
            
            # Process text content
            if has_text:
                thought = part.get("thought")
                if thought is True or (isinstance(thought, str) and thought.lower() == 'true'):
                    reasoning_content += part["text"]
                else:
                    text_content += part["text"]
            
            # Process function calls
            if has_function_call:
                func_call = part["functionCall"]
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                # Get tool name and strip gemini3_ namespace if present
                tool_name = func_call.get("name", "")
                if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix:
                    tool_name = self._strip_gemini3_namespace_from_name(tool_name)
                
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                }
                
                # Handle thoughtSignature if present
                if has_signature and not first_signature_seen:
                    first_signature_seen = True
                    signature = part["thoughtSignature"]
                    
                    # Store in server-side cache
                    if self._enable_signature_cache:
                        self._signature_cache.store(tool_call_id, signature)
                        lib_logger.debug(f"Stored thoughtSignature in cache for {tool_call_id}")
                    
                    # Pass to client if enabled
                    if self._preserve_signatures_in_client:
                        tool_call["thought_signature"] = signature
                
                tool_calls.append(tool_call)
        
        # Build message object (not delta!)
        message = {"role": "assistant"}
        
        if text_content:
            message["content"] = text_content
        elif not tool_calls:
            # If no text and no tool calls, set content to empty string
            message["content"] = ""
        
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        
        if tool_calls:
            message["tool_calls"] = tool_calls
            # Don't set content if we have tool calls (OpenAI convention)
            if "content" in message:
                message.pop("content")
        
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
        usage_metadata = gemini_response.get("usageMetadata", {})
        if usage_metadata:
            prompt_tokens = usage_metadata.get("promptTokenCount", 0)
            thoughts_tokens = usage_metadata.get("thoughtsTokenCount", 0)
            completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
            
            usage = {
                "prompt_tokens": prompt_tokens + thoughts_tokens,
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
            "id": gemini_response.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",  # Non-streaming uses chat.completion, not chunk
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,  # message, not delta!
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
        
        #lib_logger.debug(f"Antigravity completion: model={model}, stream={stream}, messages={len(messages)}")
        
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
        
        # Apply Gemini 3 system instruction injection (Strategy 3) if fix is enabled
        # This prepends critical tool usage instructions to override model's training data
        if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix and tools:
            gemini3_instruction = self._gemini3_system_instruction
            
            if "system_instruction" in gemini_cli_payload:
                # Prepend to existing system instruction
                existing_instruction = gemini_cli_payload["system_instruction"]
                if isinstance(existing_instruction, dict) and "parts" in existing_instruction:
                    # System instruction with parts structure
                    gemini3_part = {"text": gemini3_instruction}
                    existing_instruction["parts"].insert(0, gemini3_part)
                else:
                    # Shouldn't happen, but handle gracefully
                    gemini_cli_payload["system_instruction"] = {
                        "role": "user",
                        "parts": [
                            {"text": gemini3_instruction},
                            {"text": str(existing_instruction)}
                        ]
                    }
            else:
                # Create new system instruction with Gemini 3 instructions
                gemini_cli_payload["system_instruction"] = {
                    "role": "user",
                    "parts": [{"text": gemini3_instruction}]
                }
            
            #lib_logger.debug("Gemini 3 system instruction injection applied")

        
        
        # Add generation config
        generation_config = {}
        
        # Temperature handling: Default to 1.0, override 0 to 1.0
        # Low temperature (especially 0) makes models deterministic and prone to following
        # training data patterns instead of actual schemas, which causes tool hallucination
        
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
                
                # Apply Gemini 3 specific tool transformations (ONLY for gemini-3-* models)
                # This implements the "Double-Lock" catch-all strategy to prevent tool hallucination
                if self._is_gemini_3_model(model) and self._enable_gemini3_tool_fix:
                    #lib_logger.debug(f"Applying Gemini 3 catch-all tool transformations for {model}")
                    
                    # Strategy 1: Namespace prefixing (breaks association with training data)
                    gemini_cli_payload["tools"] = self._apply_gemini3_namespace_to_tools(
                        gemini_cli_payload["tools"]
                    )
                    
                    # Strategy 2: Signature injection (natural language schema enforcement)
                    gemini_cli_payload["tools"] = self._inject_signature_into_tool_descriptions(
                        gemini_cli_payload["tools"]
                    )

        
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

        #lib_logger.debug(f"Antigravity request to: {url}")
        
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
        
        # Convert to OpenAI non-streaming format (returns dict with 'message' not 'delta')
        openai_response = self._gemini_to_openai_non_streaming(gemini_response, model)
        
        # Convert dict to ModelResponse object for non-streaming
        return litellm.ModelResponse(**openai_response)

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
