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

import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter

from ..core.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
)

from .filters import CredentialFilter
from .models import ModelResolver
from .transforms import ProviderTransforms
from .executor import RequestExecutor
from .anthropic import AnthropicHandler
from .scopes import ScopeManager
from .model_discovery import ModelDiscoveryService
from .usage_managers import UsageManagerRegistry
from .request_builder import RequestContextBuilder
from .quota import QuotaService
from ..session_tracking import SessionTracker

# Import providers and other dependencies
from ..providers import PROVIDER_PLUGINS
from ..cooldown_manager import CooldownManager
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..model_definitions import ModelDefinitions
from ..provider_config import ProviderConfig as LiteLLMProviderConfig
from ..utils.paths import get_default_root, get_logs_dir, get_oauth_dir
from ..utils.suppress_litellm_warnings import suppress_litellm_serialization_warnings
from ..failure_logger import configure_failure_logger

# Import new usage package
from ..usage import UsageManager as NewUsageManager

if TYPE_CHECKING:
    from ..anthropic_compat import AnthropicMessagesRequest, AnthropicCountTokensRequest

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
        max_concurrent_requests_per_key_by_mode: Optional[
            Dict[str, Dict[str, int]]
        ] = None,
        optimal_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
        optimal_concurrent_requests_per_key_by_mode: Optional[
            Dict[str, Dict[str, int]]
        ] = None,
        rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
        data_dir: Optional[Union[str, Path]] = None,
        session_stickiness_ttl_seconds: int = 3600,
        session_persistence_enabled: bool = False,
        session_persistence_flush_interval_seconds: float = 5.0,
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
        self._litellm_logger_fn = self._litellm_logger_callback
        self.enable_request_logging = enable_request_logging
        self.max_concurrent_requests_per_key = max_concurrent_requests_per_key or {}
        self.max_concurrent_requests_per_key_by_mode = (
            max_concurrent_requests_per_key_by_mode or {}
        )
        self.optimal_concurrent_requests_per_key = (
            optimal_concurrent_requests_per_key or {}
        )
        self.optimal_concurrent_requests_per_key_by_mode = (
            optimal_concurrent_requests_per_key_by_mode or {}
        )
        self._rotation_tolerance = rotation_tolerance

        # Validate concurrent requests config. None means unset/use provider
        # default; values <= 0 are normalized to -1 (unlimited).
        for provider, max_val in list(self.max_concurrent_requests_per_key.items()):
            if max_val is None:
                del self.max_concurrent_requests_per_key[provider]
                continue
            if max_val <= 0:
                self.max_concurrent_requests_per_key[provider] = -1

        for provider, optimal_val in list(
            self.optimal_concurrent_requests_per_key.items()
        ):
            if optimal_val is None:
                del self.optimal_concurrent_requests_per_key[provider]
                continue
            if optimal_val <= 0:
                self.optimal_concurrent_requests_per_key[provider] = -1

        self._normalize_mode_concurrency(self.max_concurrent_requests_per_key_by_mode)
        self._normalize_mode_concurrency(
            self.optimal_concurrent_requests_per_key_by_mode
        )

        # Initialize components
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances: Dict[str, Any] = {}

        # Initialize managers
        self.cooldown_manager = CooldownManager()
        self.background_refresher = BackgroundRefresher(self)
        self.model_definitions = ModelDefinitions()
        self.provider_config = LiteLLMProviderConfig()
        self.http_client = httpx.AsyncClient()
        self._session_tracker = SessionTracker(
            ttl_seconds=session_stickiness_ttl_seconds,
            persist_to_disk=session_persistence_enabled,
            persistence_path=self.data_dir / "session_stickiness.json",
            persistence_flush_interval_seconds=session_persistence_flush_interval_seconds,
        )

        # Initialize extracted components
        self._credential_filter = CredentialFilter(
            PROVIDER_PLUGINS,
            provider_instances=self._provider_instances,
        )
        self._model_resolver = ModelResolver(
            PROVIDER_PLUGINS,
            self.model_definitions,
            ignore_models or {},
            whitelist_models or {},
            provider_instances=self._provider_instances,
        )
        self._provider_transforms = ProviderTransforms(
            PROVIDER_PLUGINS,
            self.provider_config,
            provider_instances=self._provider_instances,
        )

        # Resolve usage file path base
        if usage_file_path:
            base_path = Path(usage_file_path)
            if base_path.suffix:
                base_path = base_path.parent
            self._usage_base_path = base_path / "usage"
        else:
            self._usage_base_path = self.data_dir / "usage"
        self._usage_base_path.mkdir(parents=True, exist_ok=True)

        self._usage_registry = UsageManagerRegistry(
            all_credentials=self.all_credentials,
            usage_base_path=self._usage_base_path,
            provider_plugins=PROVIDER_PLUGINS,
            max_concurrent_requests_per_key=self.max_concurrent_requests_per_key,
            max_concurrent_requests_per_key_by_mode=(
                self.max_concurrent_requests_per_key_by_mode
            ),
            optimal_concurrent_requests_per_key=self.optimal_concurrent_requests_per_key,
            optimal_concurrent_requests_per_key_by_mode=(
                self.optimal_concurrent_requests_per_key_by_mode
            ),
            rotation_tolerance=rotation_tolerance,
            get_provider_instance=self._get_provider_instance,
            scope_usage_key=self._scope_usage_key,
            scope_usage_file=self._scope_usage_file,
        )
        self._usage_registry.create_global_managers()
        self._usage_managers = self._usage_registry.managers

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
            litellm_provider_params=self.litellm_provider_params,
            litellm_logger_fn=self._litellm_logger_fn,
            provider_instances=self._provider_instances,
        )

        self._model_list_cache: Dict[str, List[str]] = {}
        fingerprint_key = os.environ.get("ROTATOR_LIBRARY_FINGERPRINT_KEY")
        if not fingerprint_key:
            fingerprint_key = str(self.data_dir)
        self._fingerprint_key = fingerprint_key.encode("utf-8")
        self._scope_manager = ScopeManager(
            all_credentials=self.all_credentials,
            usage_base_path=self._usage_base_path,
            fingerprint_key=self._fingerprint_key,
            model_list_cache=self._model_list_cache,
            ensure_scoped_usage_manager=self._ensure_scoped_usage_manager,
        )
        self._model_discovery = ModelDiscoveryService(
            all_credentials=self.all_credentials,
            model_list_cache=self._model_list_cache,
            normalize_provider_map=self._normalize_provider_map,
            normalize_api_key_map=self._normalize_api_key_map,
            scope_usage_key=self._scope_usage_key,
            get_registered_scope=self._get_registered_scope,
            resolve_scope_for_provider=self._resolve_scope_for_provider,
            get_provider_instance=self._get_provider_instance,
            get_http_client=lambda: self.http_client,
            provider_config=self.provider_config,
            model_resolver=self._model_resolver,
        )
        self._request_builder = RequestContextBuilder(
            resolve_scope_for_provider=self._resolve_scope_for_provider,
            model_resolver=self._model_resolver,
            session_tracker=self._session_tracker,
            get_global_timeout=lambda: self.global_timeout,
            get_enable_request_logging=lambda: self.enable_request_logging,
            get_provider_instance=self._get_provider_instance,
        )
        self._quota_service = QuotaService(
            usage_managers=self._usage_managers,
            all_credentials=self.all_credentials,
            provider_plugins=self._provider_plugins,
            safe_scope_name=self._safe_scope_name,
            get_provider_instance=self._get_provider_instance,
        )

        # Initialize Anthropic compatibility handler
        self._anthropic_handler = AnthropicHandler(self)

    @staticmethod
    def _normalize_mode_concurrency(values: Dict[str, Dict[str, int]]) -> None:
        for provider, mode_values in list(values.items()):
            if not mode_values:
                del values[provider]
                continue
            for mode, limit in list(mode_values.items()):
                if mode not in ("balanced", "sequential") or limit is None:
                    del mode_values[mode]
                    continue
                if limit <= 0:
                    mode_values[mode] = -1

    async def __aenter__(self):
        await self.initialize_usage_managers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize_usage_managers(self) -> None:
        await self._usage_registry.initialize_usage_managers()

        for provider, manager in self._usage_managers.items():
            instance = self._get_provider_instance(provider)
            if instance and hasattr(instance, "set_usage_manager"):
                instance.set_usage_manager(manager)

    async def close(self):
        """Close the HTTP client and save usage data."""
        # Save and shutdown new usage managers
        for manager in self._usage_managers.values():
            await manager.shutdown()

        self._session_tracker.flush()

        if hasattr(self, "http_client") and self.http_client:
            await self.http_client.aclose()

    @staticmethod
    def _safe_scope_name(classifier: str) -> str:
        return ScopeManager.safe_scope_name(classifier)

    @staticmethod
    def _normalize_provider_map(
        providers: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        return ScopeManager.normalize_provider_map(providers)

    @staticmethod
    def _normalize_api_key_map(
        api_keys: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        return ScopeManager.normalize_api_key_map(api_keys)

    def _fingerprint_credential(
        self,
        provider: str,
        classifier: Optional[str],
        secret: str,
    ) -> str:
        return self._scope_manager.fingerprint_credential(provider, classifier, secret)

    def _scope_usage_key(self, provider: str, classifier: Optional[str]) -> str:
        if classifier is None:
            return provider
        return f"classifier:{ScopeManager.safe_scope_name(classifier)}:{provider}"

    def _scope_usage_file(self, provider: str, classifier: Optional[str]) -> Path:
        if classifier is None:
            return self._usage_base_path / f"usage_{provider}.json"
        return (
            self._usage_base_path
            / "classifiers"
            / ScopeManager.safe_scope_name(classifier)
            / f"usage_{provider}.json"
        )

    def _get_concurrency_settings(
        self, provider: str, mode: str
    ) -> tuple[Optional[int], Optional[int]]:
        return self._usage_registry.get_concurrency_settings(provider, mode)

    async def _ensure_scoped_usage_manager(
        self,
        provider: str,
        classifier: Optional[str],
        credentials: Optional[List[str]] = None,
    ) -> str:
        return await self._usage_registry.ensure_scoped_usage_manager(
            provider, classifier, credentials
        )

    async def _get_registered_scope(self, classifier: str) -> Dict[str, Any]:
        return await self._scope_manager.get_registered_scope(classifier)

    async def _resolve_scope_for_provider(
        self,
        provider: str,
        classifier: Optional[str],
        request_api_keys: Optional[Dict[str, Any]],
        request_providers: Optional[Dict[str, Dict[str, Any]]],
        private: bool,
    ) -> Dict[str, Any]:
        return await self._scope_manager.resolve_scope_for_provider(
            provider,
            classifier,
            request_api_keys,
            request_providers,
            private,
        )

    async def register_scope(
        self,
        classifier: str,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        private: bool = True,
    ) -> Dict[str, Any]:
        return await self._scope_manager.register_scope(
            classifier, providers=providers, api_keys=api_keys, private=private
        )

    async def update_scope(
        self,
        classifier: str,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        private: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return await self._scope_manager.update_scope(
            classifier, providers=providers, api_keys=api_keys, private=private
        )

    async def get_scope(
        self,
        classifier: str,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        return await self._scope_manager.get_scope(
            classifier, include_secrets=include_secrets
        )

    async def remove_scope(self, classifier: str) -> None:
        await self._scope_manager.remove_scope(classifier)

    async def add_scope_provider(
        self,
        classifier: str,
        provider: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._scope_manager.add_scope_provider(classifier, provider, config)

    async def update_scope_provider(
        self,
        classifier: str,
        provider: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._scope_manager.update_scope_provider(
            classifier, provider, config
        )

    async def remove_scope_provider(self, classifier: str, provider: str) -> None:
        await self._scope_manager.remove_scope_provider(classifier, provider)

    async def list_scope_providers(self, classifier: str) -> Dict[str, Dict[str, Any]]:
        return await self._scope_manager.list_scope_providers(classifier)

    async def add_scope_credentials(
        self,
        classifier: str,
        provider: str,
        keys: Any,
        private: bool = True,
    ) -> Dict[str, Any]:
        return await self._scope_manager.add_scope_credentials(
            classifier, provider, keys, private=private
        )

    async def set_scope_credentials(
        self,
        classifier: str,
        provider: str,
        keys: Any,
        private: bool = True,
    ) -> Dict[str, Any]:
        return await self._scope_manager.set_scope_credentials(
            classifier, provider, keys, private=private
        )

    async def remove_scope_credentials(
        self,
        classifier: str,
        provider: str,
        credential_ids: Optional[List[str]] = None,
    ) -> None:
        await self._scope_manager.remove_scope_credentials(
            classifier, provider, credential_ids=credential_ids
        )

    async def list_scope_credentials(
        self,
        classifier: str,
        provider: Optional[str] = None,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        return await self._scope_manager.list_scope_credentials(
            classifier, provider=provider, include_secrets=include_secrets
        )

    async def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        context = await self._request_builder.build_completion_context(
            request, pre_request_callback, kwargs
        )
        return await self._executor.execute(context)

    async def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        context = await self._request_builder.build_embedding_context(
            request, pre_request_callback, kwargs
        )
        return await self._executor.execute(context)

    def token_count(self, **kwargs) -> int:
        """Calculate token count for text or messages."""
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

        return base_count

    def _model_cache_key(
        self,
        provider: str,
        classifier: Optional[str],
        provider_config: Optional[Dict[str, Any]],
        model_filters: Optional[Dict[str, Any]],
        credentials: Optional[List[str]] = None,
    ) -> str:
        return self._model_discovery.model_cache_key(
            provider,
            classifier,
            provider_config,
            model_filters,
            credentials,
        )

    @staticmethod
    def _filter_values(
        model_filters: Optional[Dict[str, Any]],
        provider: str,
        names: tuple[str, ...],
    ) -> List[str]:
        return ModelDiscoveryService.filter_values(model_filters, provider, names)

    def _is_scoped_model_allowed(
        self,
        model: str,
        provider: str,
        classifier: Optional[str],
        model_filters: Optional[Dict[str, Any]],
    ) -> bool:
        return self._model_discovery.is_scoped_model_allowed(
            model, provider, classifier, model_filters
        )

    async def _get_openai_compatible_models(
        self,
        provider: str,
        api_key: str,
        provider_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        return await self._model_discovery.get_openai_compatible_models(
            provider, api_key, provider_config
        )

    async def get_available_models(
        self,
        provider: str,
        classifier: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        private: bool = False,
        model_filters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        return await self._model_discovery.get_available_models(
            provider,
            classifier=classifier,
            api_keys=api_keys,
            providers=providers,
            private=private,
            model_filters=model_filters,
            force_refresh=force_refresh,
        )

    async def get_all_available_models(
        self,
        grouped: bool = True,
        classifier: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        private: bool = False,
        model_filters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Union[Dict[str, List[str]], List[str]]:
        return await self._model_discovery.get_all_available_models(
            grouped=grouped,
            classifier=classifier,
            api_keys=api_keys,
            providers=providers,
            private=private,
            model_filters=model_filters,
            force_refresh=force_refresh,
        )

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._quota_service.get_quota_stats(
            provider_filter=provider_filter,
            classifier=classifier,
        )

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        """Get discovered OAuth credentials."""
        return self.oauth_credentials

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """Get or create a provider plugin instance."""
        if provider not in self._provider_instances:
            plugin_class = self._provider_plugins.get(provider)
            if plugin_class:
                self._provider_instances[provider] = plugin_class()
            else:
                return None

        return self._provider_instances[provider]

    def _get_credential_metadata(
        self,
        provider: str,
        credentials: List[str],
    ) -> tuple[Dict[str, int], Dict[str, str]]:
        return self._usage_registry.get_credential_metadata(provider, credentials)

    def get_usage_manager(self, provider: str) -> Optional[NewUsageManager]:
        """
        Get the new UsageManager for a specific provider.

        Args:
            provider: Provider name

        Returns:
            UsageManager for the provider, or None if not found
        """
        return self._usage_registry.get_usage_manager(provider)

    @property
    def usage_managers(self) -> Dict[str, NewUsageManager]:
        """Get all new usage managers."""
        return self._usage_registry.managers

    def _apply_usage_reset_config(
        self,
        provider: str,
        credentials: List[str],
        config: Any,
    ) -> None:
        self._usage_registry.apply_usage_reset_config(provider, credentials, config)

    def _sanitize_litellm_log(self, log_data: dict) -> dict:
        """Remove large/sensitive fields from LiteLLM logs."""
        if not isinstance(log_data, dict):
            return log_data

        keys_to_pop = [
            "messages",
            "input",
            "response",
            "data",
            "api_key",
            "api_base",
            "original_response",
            "additional_args",
        ]
        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]

        clean_data = json.loads(json.dumps(log_data, default=str))

        def clean_recursively(data_dict: dict) -> None:
            for key in keys_to_pop:
                data_dict.pop(key, None)
            for key in nested_keys:
                if key in data_dict and isinstance(data_dict[key], dict):
                    clean_recursively(data_dict[key])
            for value in list(data_dict.values()):
                if isinstance(value, dict):
                    clean_recursively(value)

        clean_recursively(clean_data)
        return clean_data

    def _litellm_logger_callback(self, log_data: dict) -> None:
        """Redirect LiteLLM logs into rotator library logger."""
        log_event_type = log_data.get("log_event_type")
        if log_event_type in ["pre_api_call", "post_api_call"]:
            return

        if not log_data.get("exception"):
            sanitized_log = self._sanitize_litellm_log(log_data)
            lib_logger.debug(f"LiteLLM Log: {sanitized_log}")
            return

        model = log_data.get("model", "N/A")
        error_info = log_data.get("standard_logging_object", {}).get(
            "error_information", {}
        )
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get(
            "error_message", str(log_data.get("exception", ""))
        )
        error_message = " ".join(error_message.split())

        lib_logger.debug(
            f"LiteLLM Callback Handled Error: Model={model} | "
            f"Type={error_class} | Message='{error_message}'"
        )

    # =========================================================================
    # USAGE MANAGEMENT METHODS
    # =========================================================================

    async def reload_usage_from_disk(self) -> None:
        await self._quota_service.reload_usage_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._quota_service.force_refresh_quota(
            provider=provider,
            credential=credential,
        )

    # =========================================================================
    # ANTHROPIC API COMPATIBILITY METHODS
    # =========================================================================

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        return await self._anthropic_handler.messages(
            request=request,
            raw_request=raw_request,
            pre_request_callback=pre_request_callback,
        )

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        return await self._anthropic_handler.count_tokens(request=request)
