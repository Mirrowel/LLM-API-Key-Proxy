# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Slim RotatingClient facade.

This is a lightweight facade that delegates to extracted components:
- RequestExecutor: Unified retry/rotation logic
- CredentialFilter: Tier compatibility filtering
- ModelResolver: Model name resolution and filtering
- ProviderTransforms: Provider-specific request mutations
- StreamingHandler: Streaming response processing

The original client.py was ~3000 lines. This facade is ~300 lines,
with all complexity moved to specialized modules.
"""

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter

from ..core.types import RequestContext
from ..core.errors import NoAvailableKeysError, mask_credential
from ..core.config import ConfigLoader
from ..core.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
)

from .filters import CredentialFilter
from .models import ModelResolver
from .transforms import ProviderTransforms
from .executor import RequestExecutor

# Import providers and other dependencies
from ..providers import PROVIDER_PLUGINS
from ..cooldown_manager import CooldownManager
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..model_definitions import ModelDefinitions
from ..transaction_logger import TransactionLogger
from ..provider_config import ProviderConfig as LiteLLMProviderConfig
from ..utils.paths import get_default_root, get_logs_dir, get_oauth_dir
from ..utils.suppress_litellm_warnings import suppress_litellm_serialization_warnings
from ..failure_logger import configure_failure_logger

# Import new usage package
from ..usage import UsageManager as NewUsageManager
from ..usage.config import load_provider_usage_config

lib_logger = logging.getLogger("rotator_library")


class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.

    This is a slim facade that delegates to specialized components:
    - RequestExecutor: Handles retry/rotation logic
    - CredentialFilter: Filters credentials by tier
    - ModelResolver: Resolves model names
    - ProviderTransforms: Applies provider-specific transforms
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, List[str]]] = None,
        oauth_credentials: Optional[Dict[str, List[str]]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        usage_file_path: Optional[Union[str, Path]] = None,
        configure_logging: bool = True,
        global_timeout: int = DEFAULT_GLOBAL_TIMEOUT,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
        rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the RotatingClient.

        See original client.py for full parameter documentation.
        """
        # Resolve data directory
        self.data_dir = Path(data_dir).resolve() if data_dir else get_default_root()

        # Configure logging
        configure_failure_logger(get_logs_dir(self.data_dir))
        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        litellm.drop_params = True
        suppress_litellm_serialization_warnings()

        if configure_logging:
            lib_logger.propagate = True
            if lib_logger.hasHandlers():
                lib_logger.handlers.clear()
                lib_logger.addHandler(logging.NullHandler())
        else:
            lib_logger.propagate = False

        # Process credentials
        api_keys = api_keys or {}
        oauth_credentials = oauth_credentials or {}
        api_keys = {p: k for p, k in api_keys.items() if k}
        oauth_credentials = {p: c for p, c in oauth_credentials.items() if c}

        if not api_keys and not oauth_credentials:
            lib_logger.warning(
                "No provider credentials configured. Client will be unable to make requests."
            )

        # Discover OAuth credentials if not provided
        if oauth_credentials:
            self.oauth_credentials = oauth_credentials
        else:
            cred_manager = CredentialManager(
                os.environ, oauth_dir=get_oauth_dir(self.data_dir)
            )
            self.oauth_credentials = cred_manager.discover_and_prepare()

        # Build combined credentials
        self.all_credentials: Dict[str, List[str]] = {}
        for provider, keys in api_keys.items():
            self.all_credentials.setdefault(provider, []).extend(keys)
        for provider, paths in self.oauth_credentials.items():
            self.all_credentials.setdefault(provider, []).extend(paths)

        self.api_keys = api_keys
        self.oauth_providers = set(self.oauth_credentials.keys())

        # Store configuration
        self.max_retries = max_retries
        self.global_timeout = global_timeout
        self.abort_on_callback_error = abort_on_callback_error
        self.litellm_provider_params = litellm_provider_params or {}
        self.enable_request_logging = enable_request_logging
        self.max_concurrent_requests_per_key = max_concurrent_requests_per_key or {}

        # Validate concurrent requests config
        for provider, max_val in self.max_concurrent_requests_per_key.items():
            if max_val < 1:
                lib_logger.warning(
                    f"Invalid max_concurrent for '{provider}': {max_val}. Setting to 1."
                )
                self.max_concurrent_requests_per_key[provider] = 1

        # Initialize configuration loader
        self._config_loader = ConfigLoader(PROVIDER_PLUGINS)

        # Initialize components
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances: Dict[str, Any] = {}

        # Initialize managers
        self.cooldown_manager = CooldownManager()
        self.background_refresher = BackgroundRefresher(self)
        self.model_definitions = ModelDefinitions()
        self.provider_config = LiteLLMProviderConfig()
        self.http_client = httpx.AsyncClient()

        # Initialize extracted components
        self._credential_filter = CredentialFilter(PROVIDER_PLUGINS)
        self._model_resolver = ModelResolver(
            PROVIDER_PLUGINS,
            self.model_definitions,
            ignore_models or {},
            whitelist_models or {},
        )
        self._provider_transforms = ProviderTransforms(
            PROVIDER_PLUGINS,
            self.provider_config,
        )

        # Initialize UsageManagers (one per provider) using new usage package
        self._usage_managers: Dict[str, NewUsageManager] = {}

        # Resolve usage file path base
        if usage_file_path:
            self._usage_base_path = Path(usage_file_path).parent
        else:
            self._usage_base_path = self.data_dir

        # Build provider configs using ConfigLoader
        provider_configs = {}
        for provider in self.all_credentials.keys():
            provider_configs[provider] = self._config_loader.load_provider_config(
                provider
            )

        # Create UsageManager for each provider
        for provider, credentials in self.all_credentials.items():
            config = load_provider_usage_config(provider, PROVIDER_PLUGINS)
            # Override tolerance from constructor param
            config.rotation_tolerance = rotation_tolerance

            usage_file = self._usage_base_path / f"usage_{provider}.json"

            # Get max concurrent for this provider
            max_concurrent = self.max_concurrent_requests_per_key.get(provider)

            manager = NewUsageManager(
                provider=provider,
                file_path=usage_file,
                provider_plugins=PROVIDER_PLUGINS,
                config=config,
                max_concurrent_per_key=max_concurrent,
            )
            self._usage_managers[provider] = manager

        # Initialize executor with new usage managers
        self._executor = RequestExecutor(
            usage_managers=self._usage_managers,
            cooldown_manager=self.cooldown_manager,
            credential_filter=self._credential_filter,
            provider_transforms=self._provider_transforms,
            provider_plugins=PROVIDER_PLUGINS,
            http_client=self.http_client,
            max_retries=max_retries,
            global_timeout=global_timeout,
            abort_on_callback_error=abort_on_callback_error,
        )

        self._model_list_cache: Dict[str, List[str]] = {}

    async def __aenter__(self):
        # Initialize new usage managers
        for provider, manager in self._usage_managers.items():
            credentials = self.all_credentials.get(provider, [])
            await manager.initialize(credentials)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client and save usage data."""
        # Save and shutdown new usage managers
        for manager in self._usage_managers.values():
            await manager.shutdown()

        if hasattr(self, "http_client") and self.http_client:
            await self.http_client.aclose()

    def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Dispatcher for completion requests.

        Returns:
            Response object or async generator for streaming
        """
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""

        if not provider or provider not in self.all_credentials:
            raise ValueError(
                f"Invalid model format or no credentials for provider: {model}"
            )

        # Resolve model ID
        resolved_model = self._model_resolver.resolve_model_id(model, provider)
        kwargs["model"] = resolved_model

        # Create transaction logger if enabled
        transaction_logger = None
        if self.enable_request_logging:
            transaction_logger = TransactionLogger(
                provider=provider,
                model=resolved_model,
                enabled=True,
            )
            transaction_logger.log_request(kwargs)

        # Build request context
        context = RequestContext(
            model=resolved_model,
            provider=provider,
            kwargs=kwargs,
            streaming=kwargs.get("stream", False),
            credentials=self.all_credentials.get(provider, []),
            deadline=time.time() + self.global_timeout,
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
        )

        return self._executor.execute(context)

    def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """
        Execute an embedding request with retry logic.
        """
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""

        if not provider or provider not in self.all_credentials:
            raise ValueError(
                f"Invalid model format or no credentials for provider: {model}"
            )

        # Build request context (embeddings are never streaming)
        context = RequestContext(
            model=model,
            provider=provider,
            kwargs=kwargs,
            streaming=False,
            credentials=self.all_credentials.get(provider, []),
            deadline=time.time() + self.global_timeout,
            request=request,
            pre_request_callback=pre_request_callback,
        )

        return self._executor.execute(context)

    def token_count(self, **kwargs) -> int:
        """Calculate token count for text or messages.

        For Antigravity provider models, this also includes the preprompt tokens
        that get injected during actual API calls (agent instruction + identity override).
        This ensures token counts match actual usage.
        """
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is required")

        # Calculate base token count
        if messages:
            base_count = token_counter(model=model, messages=messages)
        elif text:
            base_count = token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided")

        # Add preprompt tokens for Antigravity provider
        # The Antigravity provider injects system instructions during actual API calls,
        # so we need to account for those tokens in the count
        provider = model.split("/")[0] if "/" in model else ""
        if provider == "antigravity":
            try:
                from ..providers.antigravity_provider import (
                    get_antigravity_preprompt_text,
                )

                preprompt_text = get_antigravity_preprompt_text()
                if preprompt_text:
                    preprompt_tokens = token_counter(model=model, text=preprompt_text)
                    base_count += preprompt_tokens
            except ImportError:
                # Provider not available, skip preprompt token counting
                pass

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider with caching."""
        if provider in self._model_list_cache:
            return self._model_list_cache[provider]

        credentials = self.all_credentials.get(provider, [])
        if not credentials:
            return []

        # Shuffle and try each credential
        shuffled = list(credentials)
        random.shuffle(shuffled)

        plugin = self._get_provider_instance(provider)
        if not plugin:
            return []

        for cred in shuffled:
            try:
                models = await plugin.get_models(cred, self.http_client)

                # Apply whitelist/blacklist
                final = [
                    m
                    for m in models
                    if self._model_resolver.is_model_allowed(m, provider)
                ]

                self._model_list_cache[provider] = final
                return final

            except Exception as e:
                lib_logger.debug(
                    f"Failed to get models for {provider} with {mask_credential(cred)}: {e}"
                )
                continue

        return []

    async def get_all_available_models(
        self,
        grouped: bool = True,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get all available models across all providers."""
        providers = list(self.all_credentials.keys())
        tasks = [self.get_available_models(p) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_models: Dict[str, List[str]] = {}
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                lib_logger.error(f"Failed to get models for {provider}: {result}")
                all_models[provider] = []
            else:
                all_models[provider] = result

        if grouped:
            return all_models
        else:
            flat = []
            for models in all_models.values():
                flat.extend(models)
            return flat

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get quota and usage stats for all credentials.

        Args:
            provider_filter: Optional provider name to filter results

        Returns:
            Dict with stats per provider
        """
        result = {}

        for provider, manager in self._usage_managers.items():
            if provider_filter and provider != provider_filter:
                continue
            result[provider] = await manager.get_stats_for_endpoint()

        return result

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        """Get discovered OAuth credentials."""
        return self.oauth_credentials

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """Get or create a provider plugin instance."""
        if provider not in self.all_credentials:
            return None

        if provider not in self._provider_instances:
            plugin_class = self._provider_plugins.get(provider)
            if plugin_class:
                self._provider_instances[provider] = plugin_class()
            else:
                return None

        return self._provider_instances[provider]

    def get_usage_manager(self, provider: str) -> Optional[NewUsageManager]:
        """
        Get the new UsageManager for a specific provider.

        Args:
            provider: Provider name

        Returns:
            UsageManager for the provider, or None if not found
        """
        return self._usage_managers.get(provider)

    @property
    def usage_managers(self) -> Dict[str, NewUsageManager]:
        """Get all new usage managers."""
        return self._usage_managers
