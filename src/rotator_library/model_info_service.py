"""
Unified Model Registry

Provides aggregated model metadata from external catalogs (OpenRouter, Models.dev)
for pricing calculations and the /v1/models endpoint.

Data retrieval happens asynchronously post-startup to keep initialization fast.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModelPricing:
    """Token-level pricing information."""
    prompt: Optional[float] = None
    completion: Optional[float] = None
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None


@dataclass
class ModelLimits:
    """Context and output token limits."""
    context_window: Optional[int] = None
    max_output: Optional[int] = None


@dataclass 
class ModelCapabilities:
    """Feature flags for model capabilities."""
    tools: bool = False
    functions: bool = False
    reasoning: bool = False
    vision: bool = False
    system_prompt: bool = True
    caching: bool = False
    prefill: bool = False


@dataclass
class ModelMetadata:
    """Complete model information record."""
    
    model_id: str
    display_name: str = ""
    provider: str = ""
    category: str = "chat"  # chat, embedding, image, audio
    
    pricing: ModelPricing = field(default_factory=ModelPricing)
    limits: ModelLimits = field(default_factory=ModelLimits)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    
    input_types: List[str] = field(default_factory=lambda: ["text"])
    output_types: List[str] = field(default_factory=lambda: ["text"])
    
    timestamp: int = field(default_factory=lambda: int(time.time()))
    origin: str = ""
    match_quality: str = "unknown"
    
    def as_api_response(self) -> Dict[str, Any]:
        """Format for OpenAI-compatible /v1/models response."""
        response = {
            "id": self.model_id,
            "object": "model",
            "created": self.timestamp,
            "owned_by": self.provider or "proxy",
        }
        
        # Pricing fields
        if self.pricing.prompt is not None:
            response["input_cost_per_token"] = self.pricing.prompt
        if self.pricing.completion is not None:
            response["output_cost_per_token"] = self.pricing.completion
        if self.pricing.cached_input is not None:
            response["cache_read_input_token_cost"] = self.pricing.cached_input
        if self.pricing.cache_write is not None:
            response["cache_creation_input_token_cost"] = self.pricing.cache_write
        
        # Limits
        if self.limits.context_window:
            response["max_input_tokens"] = self.limits.context_window
            response["context_window"] = self.limits.context_window
        if self.limits.max_output:
            response["max_output_tokens"] = self.limits.max_output
        
        # Category and modalities
        response["mode"] = self.category
        response["supported_modalities"] = self.input_types
        response["supported_output_modalities"] = self.output_types
        
        # Capability flags
        response["capabilities"] = {
            "tool_choice": self.capabilities.tools,
            "function_calling": self.capabilities.functions,
            "reasoning": self.capabilities.reasoning,
            "vision": self.capabilities.vision,
            "system_messages": self.capabilities.system_prompt,
            "prompt_caching": self.capabilities.caching,
            "assistant_prefill": self.capabilities.prefill,
        }
        
        # Debug metadata
        if self.origin:
            response["_sources"] = [self.origin]
            response["_match_type"] = self.match_quality
        
        return response
    
    def as_minimal(self) -> Dict[str, Any]:
        """Minimal OpenAI format."""
        return {
            "id": self.model_id,
            "object": "model", 
            "created": self.timestamp,
            "owned_by": self.provider or "proxy",
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Alias for as_api_response() - backward compatibility."""
        return self.as_api_response()
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Alias for as_minimal() - backward compatibility."""
        return self.as_minimal()
    
    # Backward-compatible property aliases
    @property
    def id(self) -> str:
        return self.model_id
    
    @property
    def name(self) -> str:
        return self.display_name
    
    @property
    def input_cost_per_token(self) -> Optional[float]:
        return self.pricing.prompt
    
    @property
    def output_cost_per_token(self) -> Optional[float]:
        return self.pricing.completion
    
    @property
    def cache_read_input_token_cost(self) -> Optional[float]:
        return self.pricing.cached_input
    
    @property
    def cache_creation_input_token_cost(self) -> Optional[float]:
        return self.pricing.cache_write
    
    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.limits.context_window
    
    @property
    def max_output_tokens(self) -> Optional[int]:
        return self.limits.max_output
    
    @property
    def mode(self) -> str:
        return self.category
    
    @property
    def supported_modalities(self) -> List[str]:
        return self.input_types
    
    @property
    def supported_output_modalities(self) -> List[str]:
        return self.output_types
    
    @property
    def supports_tool_choice(self) -> bool:
        return self.capabilities.tools
    
    @property
    def supports_function_calling(self) -> bool:
        return self.capabilities.functions
    
    @property
    def supports_reasoning(self) -> bool:
        return self.capabilities.reasoning
    
    @property
    def supports_vision(self) -> bool:
        return self.capabilities.vision
    
    @property
    def supports_system_messages(self) -> bool:
        return self.capabilities.system_prompt
    
    @property
    def supports_prompt_caching(self) -> bool:
        return self.capabilities.caching
    
    @property
    def supports_assistant_prefill(self) -> bool:
        return self.capabilities.prefill
    
    @property
    def litellm_provider(self) -> str:
        return self.provider
    
    @property
    def created(self) -> int:
        return self.timestamp
    
    @property
    def _sources(self) -> List[str]:
        return [self.origin] if self.origin else []
    
    @property
    def _match_type(self) -> str:
        return self.match_quality


# ============================================================================
# Data Source Adapters
# ============================================================================

class DataSourceAdapter:
    """Base interface for external data sources."""
    
    source_name: str = "unknown"
    endpoint: str = ""
    
    def fetch(self) -> Dict[str, Dict]:
        """Retrieve and normalize data. Returns {model_id: raw_data}."""
        raise NotImplementedError
    
    def _http_get(self, url: str, timeout: int = 30) -> Any:
        """Execute HTTP GET with standard headers."""
        req = Request(url, headers={"User-Agent": "ModelRegistry/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class OpenRouterAdapter(DataSourceAdapter):
    """Fetches model data from OpenRouter's public API."""
    
    source_name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/models"
    
    def fetch(self) -> Dict[str, Dict]:
        try:
            raw = self._http_get(self.endpoint)
            entries = raw.get("data", [])
            
            catalog = {}
            for entry in entries:
                mid = entry.get("id")
                if not mid:
                    continue
                
                full_id = f"openrouter/{mid}"
                catalog[full_id] = self._normalize(entry)
            
            return catalog
        except (URLError, json.JSONDecodeError, TimeoutError) as err:
            raise ConnectionError(f"OpenRouter unavailable: {err}") from err
    
    def _normalize(self, raw: Dict) -> Dict:
        """Transform OpenRouter schema to internal format."""
        prices = raw.get("pricing", {})
        arch = raw.get("architecture", {})
        top = raw.get("top_provider", {})
        params = raw.get("supported_parameters", [])
        
        tokenizer = arch.get("tokenizer", "")
        category = "embedding" if "embedding" in tokenizer.lower() else "chat"
        
        return {
            "name": raw.get("name", ""),
            "prompt_cost": float(prices.get("prompt", 0)),
            "completion_cost": float(prices.get("completion", 0)),
            "cache_read_cost": float(prices.get("input_cache_read", 0)) or None,
            "context": top.get("context_length", 0),
            "max_out": top.get("max_completion_tokens", 0),
            "category": category,
            "inputs": arch.get("input_modalities", ["text"]),
            "outputs": arch.get("output_modalities", ["text"]),
            "has_tools": "tool_choice" in params or "tools" in params,
            "has_functions": "tools" in params or "function_calling" in params,
            "has_reasoning": "reasoning" in params,
            "has_vision": "image" in arch.get("input_modalities", []),
            "provider": "openrouter",
            "source": "openrouter",
        }


class ModelsDevAdapter(DataSourceAdapter):
    """Fetches model data from Models.dev catalog."""
    
    source_name = "modelsdev"
    endpoint = "https://models.dev/api.json"
    
    def __init__(self, skip_providers: Optional[List[str]] = None):
        self.skip_providers = skip_providers or []
    
    def fetch(self) -> Dict[str, Dict]:
        try:
            raw = self._http_get(self.endpoint)
            
            catalog = {}
            for provider_key, provider_block in raw.items():
                if not isinstance(provider_block, dict):
                    continue
                if provider_key in self.skip_providers:
                    continue
                
                models_block = provider_block.get("models", {})
                if not isinstance(models_block, dict):
                    continue
                
                for model_key, model_data in models_block.items():
                    if not isinstance(model_data, dict):
                        continue
                    
                    full_id = f"{provider_key}/{model_key}"
                    catalog[full_id] = self._normalize(model_data, provider_key)
            
            return catalog
        except (URLError, json.JSONDecodeError, TimeoutError) as err:
            raise ConnectionError(f"Models.dev unavailable: {err}") from err
    
    def _normalize(self, raw: Dict, provider_key: str) -> Dict:
        """Transform Models.dev schema to internal format."""
        costs = raw.get("cost", {})
        mods = raw.get("modalities", {})
        lims = raw.get("limit", {})
        
        outputs = mods.get("output", ["text"])
        if "image" in outputs:
            category = "image"
        elif "audio" in outputs:
            category = "audio"
        else:
            category = "chat"
        
        # Models.dev uses per-million pricing, convert to per-token
        divisor = 1_000_000
        
        cache_read = costs.get("cache_read")
        cache_write = costs.get("cache_write")
        
        return {
            "name": raw.get("name", ""),
            "prompt_cost": float(costs.get("input", 0)) / divisor,
            "completion_cost": float(costs.get("output", 0)) / divisor,
            "cache_read_cost": float(cache_read) / divisor if cache_read else None,
            "cache_write_cost": float(cache_write) / divisor if cache_write else None,
            "context": lims.get("context", 0),
            "max_out": lims.get("output", 0),
            "category": category,
            "inputs": mods.get("input", ["text"]),
            "outputs": outputs,
            "has_tools": raw.get("tool_call", False),
            "has_functions": raw.get("tool_call", False),
            "has_reasoning": raw.get("reasoning", False),
            "has_vision": "image" in mods.get("input", []),
            "provider": provider_key,
            "source": "modelsdev",
        }


# ============================================================================
# Lookup Index
# ============================================================================

class ModelIndex:
    """Fast lookup structure for model ID resolution."""
    
    def __init__(self):
        self._by_full_id: Dict[str, str] = {}  # normalized_id -> canonical_id
        self._by_suffix: Dict[str, List[str]] = {}  # short_name -> [canonical_ids]
    
    def clear(self):
        """Reset the index."""
        self._by_full_id.clear()
        self._by_suffix.clear()
    
    def entry_count(self) -> int:
        """Return total number of suffix index entries."""
        return sum(len(v) for v in self._by_suffix.values())
    
    def add(self, canonical_id: str):
        """Index a canonical model ID for various lookup patterns."""
        self._by_full_id[canonical_id] = canonical_id
        
        segments = canonical_id.split("/")
        if len(segments) >= 2:
            # Index by everything after first segment
            partial = "/".join(segments[1:])
            self._by_suffix.setdefault(partial, []).append(canonical_id)
            
            # Index by final segment only
            if len(segments) >= 3:
                tail = segments[-1]
                self._by_suffix.setdefault(tail, []).append(canonical_id)
    
    def resolve(self, query: str) -> List[str]:
        """Find all canonical IDs matching a query."""
        # Direct match
        if query in self._by_full_id:
            return [self._by_full_id[query]]
        
        # Try with openrouter prefix
        prefixed = f"openrouter/{query}"
        if prefixed in self._by_full_id:
            return [self._by_full_id[prefixed]]
        
        # Extract search terms from query
        search_keys = []
        parts = query.split("/")
        if len(parts) >= 2:
            search_keys.append("/".join(parts[1:]))
            search_keys.append(parts[-1])
        else:
            search_keys.append(query)
        # Find matches
        matches = []
        seen = set()
        for key in search_keys:
            for cid in self._by_suffix.get(key, []):
                if cid not in seen:
                    seen.add(cid)
                    matches.append(cid)
        
        return matches


# ============================================================================
# Data Merger
# ============================================================================

class DataMerger:
    """Combines data from multiple sources into unified ModelMetadata."""
    
    @staticmethod
    def single(model_id: str, data: Dict, origin: str, quality: str) -> ModelMetadata:
        """Create ModelMetadata from a single source record."""
        return ModelMetadata(
            model_id=model_id,
            display_name=data.get("name", model_id),
            provider=data.get("provider", ""),
            category=data.get("category", "chat"),
            pricing=ModelPricing(
                prompt=data.get("prompt_cost"),
                completion=data.get("completion_cost"),
                cached_input=data.get("cache_read_cost"),
                cache_write=data.get("cache_write_cost"),
            ),
            limits=ModelLimits(
                context_window=data.get("context") or None,
                max_output=data.get("max_out") or None,
            ),
            capabilities=ModelCapabilities(
                tools=data.get("has_tools", False),
                functions=data.get("has_functions", False),
                reasoning=data.get("has_reasoning", False),
                vision=data.get("has_vision", False),
            ),
            input_types=data.get("inputs", ["text"]),
            output_types=data.get("outputs", ["text"]),
            origin=origin,
            match_quality=quality,
        )
    
    @staticmethod
    def combine(model_id: str, records: List[Tuple[Dict, str]], quality: str) -> ModelMetadata:
        """Merge multiple source records into one ModelMetadata."""
        if len(records) == 1:
            data, origin = records[0]
            return DataMerger.single(model_id, data, origin, quality)
        
        # Aggregate pricing - use average
        prompt_costs = [r[0]["prompt_cost"] for r in records if r[0].get("prompt_cost")]
        comp_costs = [r[0]["completion_cost"] for r in records if r[0].get("completion_cost")]
        cache_costs = [r[0]["cache_read_cost"] for r in records if r[0].get("cache_read_cost")]
        
        # Aggregate limits - use most common value
        contexts = [r[0]["context"] for r in records if r[0].get("context")]
        max_outs = [r[0]["max_out"] for r in records if r[0].get("max_out")]
        
        # Capabilities - OR logic (any source supporting = supported)
        has_tools = any(r[0].get("has_tools") for r in records)
        has_funcs = any(r[0].get("has_functions") for r in records)
        has_reason = any(r[0].get("has_reasoning") for r in records)
        has_vis = any(r[0].get("has_vision") for r in records)
        
        # Modalities - union
        all_inputs = set()
        all_outputs = set()
        for r in records:
            all_inputs.update(r[0].get("inputs", ["text"]))
            all_outputs.update(r[0].get("outputs", ["text"]))
        
        # Category - majority vote
        categories = [r[0].get("category", "chat") for r in records]
        category = max(set(categories), key=categories.count)
        
        # Name - first non-empty
        name = model_id
        for r in records:
            if r[0].get("name"):
                name = r[0]["name"]
                break
        
        origins = [r[1] for r in records]
        
        return ModelMetadata(
            model_id=model_id,
            display_name=name,
            provider=records[0][0].get("provider", ""),
            category=category,
            pricing=ModelPricing(
                prompt=sum(prompt_costs) / len(prompt_costs) if prompt_costs else None,
                completion=sum(comp_costs) / len(comp_costs) if comp_costs else None,
                cached_input=sum(cache_costs) / len(cache_costs) if cache_costs else None,
            ),
            limits=ModelLimits(
                context_window=DataMerger._mode(contexts),
                max_output=DataMerger._mode(max_outs),
            ),
            capabilities=ModelCapabilities(
                tools=has_tools,
                functions=has_funcs,
                reasoning=has_reason,
                vision=has_vis,
            ),
            input_types=list(all_inputs) or ["text"],
            output_types=list(all_outputs) or ["text"],
            origin=",".join(origins),
            match_quality=quality,
        )
    
    @staticmethod
    def _mode(values: List[int]) -> Optional[int]:
        """Return most frequent value."""
        if not values:
            return None
        return max(set(values), key=values.count)


# ============================================================================
# Main Registry Service
# ============================================================================

class ModelRegistry:
    """
    Central registry for model metadata from external catalogs.
    
    Manages background data refresh and provides lookup/pricing APIs.
    """
    
    REFRESH_INTERVAL_DEFAULT = 6 * 60 * 60  # 6 hours
    
    def __init__(
        self,
        refresh_seconds: Optional[int] = None,
        skip_modelsdev_providers: Optional[List[str]] = None,
    ):
        interval_env = os.getenv("MODEL_INFO_REFRESH_INTERVAL")
        self._refresh_interval = refresh_seconds or (
            int(interval_env) if interval_env else self.REFRESH_INTERVAL_DEFAULT
        )
        
        # Configure adapters
        self._adapters: List[DataSourceAdapter] = [
            OpenRouterAdapter(),
            ModelsDevAdapter(skip_providers=skip_modelsdev_providers or []),
        ]
        
        # Raw data stores
        self._openrouter_store: Dict[str, Dict] = {}
        self._modelsdev_store: Dict[str, Dict] = {}
        
        # Lookup infrastructure
        self._index = ModelIndex()
        self._result_cache: Dict[str, ModelMetadata] = {}
        
        # Async coordination
        self._ready = asyncio.Event()
        self._mutex = asyncio.Lock()
        self._worker: Optional[asyncio.Task] = None
        self._last_refresh: float = 0
    
    # ---------- Lifecycle ----------
    
    async def start(self):
        """Begin background refresh worker."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._refresh_worker())
            logger.info(
                "ModelRegistry started (refresh every %ds)", 
                self._refresh_interval
            )
    
    async def stop(self):
        """Halt background worker."""
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
            logger.info("ModelRegistry stopped")
    
    async def await_ready(self, timeout_secs: float = 30.0) -> bool:
        """Block until initial data load completes."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout_secs)
            return True
        except asyncio.TimeoutError:
            logger.warning("ModelRegistry ready timeout after %.1fs", timeout_secs)
            return False
    
    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()
    
    # ---------- Background Worker ----------
    
    async def _refresh_worker(self):
        """Periodic refresh loop."""
        await self._load_all_sources()
        self._ready.set()
        
        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
                logger.info("Scheduled registry refresh...")
                await self._load_all_sources()
                logger.info("Registry refresh complete")
            except asyncio.CancelledError:
                break
            except Exception as ex:
                logger.error("Registry refresh error: %s", ex)
    
    async def _load_all_sources(self):
        """Fetch from all adapters concurrently."""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(None, adapter.fetch)
            for adapter in self._adapters
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        async with self._mutex:
            for adapter, result in zip(self._adapters, results):
                if isinstance(result, Exception):
                    logger.error("%s fetch failed: %s", adapter.source_name, result)
                    continue
                
                if adapter.source_name == "openrouter":
                    self._openrouter_store = result
                    logger.info("OpenRouter: %d models loaded", len(result))
                elif adapter.source_name == "modelsdev":
                    self._modelsdev_store = result
                    logger.info("Models.dev: %d models loaded", len(result))
            
            self._rebuild_index()
            self._last_refresh = time.time()
    
    def _rebuild_index(self):
        """Reconstruct lookup index from current stores."""
        self._index.clear()
        self._result_cache.clear()
        
        for model_id in self._openrouter_store:
            self._index.add(model_id)
        
        for model_id in self._modelsdev_store:
            self._index.add(model_id)
    
    # ---------- Query API ----------
    
    def lookup(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Retrieve model metadata by ID.
        
        Matching strategy:
        1. Exact match against known IDs
        2. Fuzzy match by model name suffix
        3. Aggregate if multiple sources match
        """
        if model_id in self._result_cache:
            return self._result_cache[model_id]
        
        metadata = self._resolve_model(model_id)
        if metadata:
            self._result_cache[model_id] = metadata
        return metadata
    
    def _resolve_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Build ModelMetadata by matching source data."""
        records: List[Tuple[Dict, str]] = []
        quality = "none"
        
        # Check exact matches first
        or_key = f"openrouter/{model_id}" if not model_id.startswith("openrouter/") else model_id
        if or_key in self._openrouter_store:
            records.append((self._openrouter_store[or_key], f"openrouter:exact:{or_key}"))
            quality = "exact"
        
        if model_id in self._modelsdev_store:
            records.append((self._modelsdev_store[model_id], f"modelsdev:exact:{model_id}"))
            quality = "exact"
        
        # Fall back to index search
        if not records:
            candidates = self._index.resolve(model_id)
            for cid in candidates:
                if cid in self._openrouter_store:
                    records.append((self._openrouter_store[cid], f"openrouter:fuzzy:{cid}"))
                elif cid in self._modelsdev_store:
                    records.append((self._modelsdev_store[cid], f"modelsdev:fuzzy:{cid}"))
            
            if records:
                quality = "fuzzy"
        
        if not records:
            return None
        
        return DataMerger.combine(model_id, records, quality)
    
    def get_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        """Extract just pricing info for cost calculations."""
        meta = self.lookup(model_id)
        if not meta:
            return None
        
        result = {}
        if meta.pricing.prompt is not None:
            result["input_cost_per_token"] = meta.pricing.prompt
        if meta.pricing.completion is not None:
            result["output_cost_per_token"] = meta.pricing.completion
        if meta.pricing.cached_input is not None:
            result["cache_read_input_token_cost"] = meta.pricing.cached_input
        if meta.pricing.cache_write is not None:
            result["cache_creation_input_token_cost"] = meta.pricing.cache_write
        
        return result if result else None
    
    def compute_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> Optional[float]:
        """
        Calculate total request cost.
        
        Returns None if pricing unavailable.
        """
        pricing = self.get_pricing(model_id)
        if not pricing:
            return None
        
        in_rate = pricing.get("input_cost_per_token")
        out_rate = pricing.get("output_cost_per_token")
        
        if in_rate is None or out_rate is None:
            return None
        
        total = (input_tokens * in_rate) + (output_tokens * out_rate)
        
        cache_read_rate = pricing.get("cache_read_input_token_cost")
        if cache_read_rate and cache_hit_tokens:
            total += cache_hit_tokens * cache_read_rate
        
        cache_write_rate = pricing.get("cache_creation_input_token_cost")
        if cache_write_rate and cache_miss_tokens:
            total += cache_miss_tokens * cache_write_rate
        
        return total
    
    def enrich_models(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Attach metadata to a list of model IDs.
        
        Used by /v1/models endpoint.
        """
        enriched = []
        for mid in model_ids:
            meta = self.lookup(mid)
            if meta:
                enriched.append(meta.as_api_response())
            else:
                # Fallback minimal entry
                enriched.append({
                    "id": mid,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": mid.split("/")[0] if "/" in mid else "unknown",
                })
        return enriched
    
    def all_raw_models(self) -> Dict[str, Dict]:
        """Return all raw source data (for debugging)."""
        combined = {}
        combined.update(self._openrouter_store)
        combined.update(self._modelsdev_store)
        return combined
    
    def diagnostics(self) -> Dict[str, Any]:
        """Return service health/stats."""
        return {
            "ready": self._ready.is_set(),
            "last_refresh": self._last_refresh,
            "openrouter_count": len(self._openrouter_store),
            "modelsdev_count": len(self._modelsdev_store),
            "cached_lookups": len(self._result_cache),
            "index_entries": self._index.entry_count(),
            "refresh_interval": self._refresh_interval,
        }
    
    # ---------- Backward Compatibility Methods ----------
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Alias for lookup() - backward compatibility."""
        return self.lookup(model_id)
    
    def get_cost_info(self, model_id: str) -> Optional[Dict[str, float]]:
        """Alias for get_pricing() - backward compatibility."""
        return self.get_pricing(model_id)
    
    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> Optional[float]:
        """Alias for compute_cost() - backward compatibility."""
        return self.compute_cost(
            model_id, prompt_tokens, completion_tokens,
            cache_read_tokens, cache_creation_tokens
        )
    
    def enrich_model_list(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """Alias for enrich_models() - backward compatibility."""
        return self.enrich_models(model_ids)
    
    def get_all_source_models(self) -> Dict[str, Dict]:
        """Alias for all_raw_models() - backward compatibility."""
        return self.all_raw_models()
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for diagnostics() - backward compatibility."""
        return self.diagnostics()
    
    def wait_for_ready(self, timeout: float = 30.0):
        """Sync wrapper for await_ready() - for compatibility."""
        return self.await_ready(timeout)


# ============================================================================
# Backward Compatibility Layer
# ============================================================================

# Alias for backward compatibility
ModelInfo = ModelMetadata
ModelInfoService = ModelRegistry

# Global singleton
_registry_instance: Optional[ModelRegistry] = None


def get_model_info_service() -> ModelRegistry:
    """Get or create the global registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


async def init_model_info_service() -> ModelRegistry:
    """Initialize and start the global registry."""
    registry = get_model_info_service()
    await registry.start()
    return registry


# Compatibility shim - map old method names to new
class _CompatibilityWrapper:
    """Provides old API method names for gradual migration."""
    
    def __init__(self, registry: ModelRegistry):
        self._reg = registry
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        return self._reg.lookup(model_id)
    
    def get_cost_info(self, model_id: str) -> Optional[Dict[str, float]]:
        return self._reg.get_pricing(model_id)
    
    def calculate_cost(
        self, model_id: str, prompt_tokens: int, completion_tokens: int,
        cache_read_tokens: int = 0, cache_creation_tokens: int = 0
    ) -> Optional[float]:
        return self._reg.compute_cost(
            model_id, prompt_tokens, completion_tokens,
            cache_read_tokens, cache_creation_tokens
        )
    
    def enrich_model_list(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        return self._reg.enrich_models(model_ids)
    
    def get_all_source_models(self) -> Dict[str, Dict]:
        return self._reg.all_raw_models()
    
    def get_stats(self) -> Dict[str, Any]:
        return self._reg.diagnostics()
    
    async def start(self):
        await self._reg.start()
    
    async def stop(self):
        await self._reg.stop()
    
    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        return await self._reg.await_ready(timeout)
    
    def is_ready(self) -> bool:
        return self._reg.is_ready
